"""Attention-visualization helpers for pi0.5.

Everything here is *opt-in*: it relies on the optional ``attentions_out`` path
plumbed through :class:`PaliGemmaWithExpertModel.forward` and the
``sample_actions_with_attention`` method on :class:`PI0Pytorch`. Normal
inference / training paths are unaffected.

This file is responsible for:

  - Defining lightweight dataclasses describing the per-modality token spans
    (image / text / state / action) that the model's own
    ``embed_prefix`` / ``embed_suffix`` produce.
  - Computing those spans from a real ``Observation`` so callers do not have
    to hard-code anything (different camera counts, different ``action_horizon``,
    different image-patch grids etc. all just work).
  - Identifying subword token indices for a given target word inside the
    instruction.
  - Reducing the captured attention tensors into per-camera image-grid heatmaps
    and modality-level attention ratios.
  - Rendering RGB+heatmap overlays as PNGs.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from PIL import Image as PILImage

    from openpi.models import model as _model
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


# ---------------------------------------------------------------------------
# Token-span dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ImageSpan:
    """Span of one camera's image-patch tokens inside the prefix sequence.

    ``start`` / ``end`` are half-open token indices (``end - start == grid_h * grid_w``).
    ``grid_h`` / ``grid_w`` describe the patch grid laid out by the SigLIP backbone
    (typically 16x16 for a 224x224 input with a 14-pixel patch size).
    """

    name: str
    start: int
    end: int
    grid_h: int
    grid_w: int

    @property
    def num_tokens(self) -> int:
        return self.end - self.start


@dataclasses.dataclass(frozen=True)
class TokenSpans:
    """Per-modality token indices recovered from one Observation.

    Conventions
    -----------
    - ``image_spans`` is in the same order as ``observation.images`` iterated.
    - ``text_span`` is half-open. ``text_valid_mask`` is a bool array of length
      ``text_span[1] - text_span[0]`` indicating which positions inside the span
      contain a real token (vs. a pad slot).
    - ``state_span`` is ``None`` for pi0.5 (state is folded into the language
      tokens via ``discrete_state_input``); it is a length-1 span for pi0.
    - ``action_span`` is the suffix region. Length = ``action_horizon``.
    - ``prefix_len`` is the number of prefix tokens (image + text [+state for pi0]).
    - ``seq_len`` is the total length of the prefix+suffix sequence.
    """

    image_spans: tuple[ImageSpan, ...]
    text_span: tuple[int, int]
    text_valid_mask: np.ndarray  # shape (text_len,), bool
    state_span: tuple[int, int] | None
    action_span: tuple[int, int]
    prefix_len: int
    seq_len: int

    def to_metadata_dict(self) -> dict[str, Any]:
        """Plain-Python dict suitable for ``json.dump``."""
        return {
            "image_spans": [dataclasses.asdict(s) for s in self.image_spans],
            "text_span": list(self.text_span),
            "text_valid_mask": [bool(b) for b in self.text_valid_mask],
            "state_span": None if self.state_span is None else list(self.state_span),
            "action_span": list(self.action_span),
            "prefix_len": int(self.prefix_len),
            "seq_len": int(self.seq_len),
        }

    def context_indices_excluding_padding(self) -> np.ndarray:
        """Return a sorted array of all *valid* (non-pad) prefix indices.

        Useful when normalising attention over modalities.
        """
        idxs: list[int] = []
        for span in self.image_spans:
            idxs.extend(range(span.start, span.end))
        text_start = self.text_span[0]
        for offset, valid in enumerate(self.text_valid_mask):
            if bool(valid):
                idxs.append(text_start + offset)
        if self.state_span is not None:
            idxs.extend(range(self.state_span[0], self.state_span[1]))
        return np.asarray(sorted(idxs), dtype=np.int64)


# ---------------------------------------------------------------------------
# Compute spans from an Observation (re-uses model's own preprocessing/embeds)
# ---------------------------------------------------------------------------


def compute_token_spans(model, observation: "_model.Observation") -> TokenSpans:
    """Recover token spans from an Observation, backend-agnostically.

    The number of image-patch tokens per camera is determined by the SigLIP
    backbone ``So400m/14`` (patch size 14, square inputs) and is therefore
    derivable from the input image's spatial size alone — no need to run the
    image encoder. This also lets us avoid calling either backend's
    preprocessing helpers, which differ between PyTorch and JAX.
    """
    image_keys = list(observation.images.keys())
    patch_size = 14  # SigLIP So400m/14

    image_spans: list[ImageSpan] = []
    cursor = 0
    for name in image_keys:
        img = observation.images[name]
        shape = tuple(img.shape)
        # Accept (B, H, W, C), (B, C, H, W), (H, W, C), (C, H, W).
        if len(shape) == 4:
            # Heuristic: channels-last if last axis is small (3 / 4).
            if shape[-1] in (1, 3, 4):
                H, W = shape[1], shape[2]
            else:
                H, W = shape[2], shape[3]
        elif len(shape) == 3:
            if shape[-1] in (1, 3, 4):
                H, W = shape[0], shape[1]
            else:
                H, W = shape[1], shape[2]
        else:
            raise ValueError(f"Unexpected image shape for camera {name!r}: {shape}")
        grid_h = int(H) // patch_size
        grid_w = int(W) // patch_size
        num_patches = grid_h * grid_w
        if num_patches <= 0:
            raise ValueError(
                f"Image {name!r} of shape {shape} produced 0 patches with patch_size={patch_size}."
            )
        image_spans.append(
            ImageSpan(name=name, start=cursor, end=cursor + num_patches, grid_h=grid_h, grid_w=grid_w)
        )
        cursor += num_patches

    # Text span: full max_token_len region; ``tokenized_prompt_mask`` records
    # which positions are real (vs. pad) tokens.
    tokenized_prompt = observation.tokenized_prompt
    prompt_mask = observation.tokenized_prompt_mask
    if tokenized_prompt is None or prompt_mask is None:
        raise ValueError("Observation must include tokenized_prompt and tokenized_prompt_mask.")
    text_len = int(tokenized_prompt.shape[1])
    text_span = (cursor, cursor + text_len)
    # ``prompt_mask`` may be a torch.Tensor on CUDA (live PyTorch serving) or a
    # jax.Array (JAX backbone) or a numpy array (replay mode). Funnel everything
    # through CPU before np.asarray to avoid the "can't convert cuda tensor to
    # numpy" error from torch.Tensor.__array__.
    pm0 = prompt_mask[0]
    if hasattr(pm0, "detach"):  # torch.Tensor
        pm0 = pm0.detach().cpu()
    text_valid_mask = np.asarray(pm0).astype(bool)
    cursor += text_len

    prefix_len = cursor

    # pi05 attribute is defined on both JAX and PyTorch implementations.
    pi05 = bool(getattr(model, "pi05", True))
    # action_horizon lives on different attributes between backends.
    if hasattr(model, "config") and hasattr(model.config, "action_horizon"):
        action_horizon = int(model.config.action_horizon)
    else:
        action_horizon = int(getattr(model, "action_horizon"))

    if pi05:
        state_span: tuple[int, int] | None = None
    else:
        state_span = (cursor, cursor + 1)
        cursor += 1
    action_span = (cursor, cursor + action_horizon)
    seq_len = cursor + action_horizon

    return TokenSpans(
        image_spans=tuple(image_spans),
        text_span=tuple(text_span),
        text_valid_mask=text_valid_mask,
        state_span=state_span,
        action_span=tuple(action_span),
        prefix_len=int(prefix_len),
        seq_len=int(seq_len),
    )


# ---------------------------------------------------------------------------
# Subword-token lookup for a target word
# ---------------------------------------------------------------------------


def find_target_word_token_indices(
    tokenizer: "PaligemmaTokenizer",
    instruction: str,
    target_word: str,
    *,
    state: np.ndarray | None = None,
) -> list[int]:
    """Return the prompt-relative token indices that cover ``target_word``.

    The PaliGemma tokenizer is SentencePiece-based, so a word may be split into
    several subword tokens (e.g. ``"block"`` -> ``[" block"]`` is usually one
    token, but ``"strawberry"`` may split). We tokenize the *full* prompt the
    model would actually see, then re-tokenize a marker-padded variant to find
    the contiguous subword-token range that the target word occupies.

    Indices returned are local to the prompt (0-based, position 0 = ``<bos>``)
    and *not* yet shifted by the image-token offset.
    """
    if not target_word.strip():
        raise ValueError("target_word must be non-empty")

    # Match the same cleaning the tokenizer applies internally.
    cleaned = instruction.strip().replace("_", " ").replace("\n", " ")
    cleaned_target = target_word.strip().replace("_", " ").replace("\n", " ")

    # Try every case-insensitive occurrence; return the first one that round-trips.
    lower_clean = cleaned.lower()
    lower_target = cleaned_target.lower()
    candidates: list[tuple[int, str]] = []  # (char_offset, exact substring as it appears)
    start = 0
    while True:
        idx = lower_clean.find(lower_target, start)
        if idx < 0:
            break
        candidates.append((idx, cleaned[idx : idx + len(cleaned_target)]))
        start = idx + 1
    if not candidates:
        raise ValueError(
            f"target_word {target_word!r} not found in instruction {instruction!r} "
            "(case-insensitive substring match)."
        )

    # We tokenize the FULL constructed prompt (matching tokenize() exactly), find
    # how the target word's text range maps onto contiguous tokens by tokenizing
    # the prefix-only and prefix+target-only strings.
    sp = tokenizer._tokenizer  # type: ignore[attr-defined]

    if state is not None:
        # pi0.5 format
        discretized = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        state_str = " ".join(map(str, discretized))
        full_prefix = f"Task: {cleaned}, State: {state_str};\nAction: "

        def _build(replacement: str) -> str:
            return f"Task: {replacement}, State: {state_str};\nAction: "
    else:
        # pi0 format ("Task: <text>\n").
        full_prefix = cleaned + "\n"

        def _build(replacement: str) -> str:
            return replacement + "\n"

    # We want the indices, in tokens-of-full_prefix, that correspond to char_offset..char_offset+len(target).
    # Easiest robust approach: find the offset by tokenizing prefix substrings.
    # Use the FIRST candidate.
    char_offset, exact_text = candidates[0]
    before = cleaned[:char_offset]
    target_text = exact_text  # preserves original case
    # `cleaned` is what gets substituted into the template for the TASK slot.
    # Tokenise the full prompt + the variant where the target word is excluded.
    full_tokens = sp.encode(full_prefix, add_bos=True)
    before_only_text = _build(before)
    full_before_tokens = sp.encode(before_only_text, add_bos=True)
    # The target token range in the full prompt starts at len(full_before_tokens).
    target_start_in_prompt = len(full_before_tokens)
    # Tokenise prefix+target to find length of target in tokens.
    before_plus_target_text = _build(before + target_text)
    full_before_plus_target_tokens = sp.encode(before_plus_target_text, add_bos=True)
    target_end_in_prompt = len(full_before_plus_target_tokens)

    if target_end_in_prompt <= target_start_in_prompt:
        raise RuntimeError(
            f"Could not localise target {target_word!r} in prompt; "
            f"start={target_start_in_prompt} end={target_end_in_prompt}."
        )

    # Sanity check: ensure target tokens are within the un-padded portion of the prompt.
    if target_end_in_prompt > len(full_tokens):
        raise RuntimeError(
            f"Target tokens fall outside the tokenized prompt length "
            f"({target_end_in_prompt} > {len(full_tokens)}). Increase max_token_len."
        )

    return list(range(target_start_in_prompt, target_end_in_prompt))


# ---------------------------------------------------------------------------
# Attention reduction
# ---------------------------------------------------------------------------


def _select_layers_and_heads(
    attentions: Sequence[torch.Tensor],
    layers: Sequence[int] | None,
    heads: Sequence[int] | None,
) -> torch.Tensor:
    """Stack a list of per-layer attention tensors into one tensor with the requested
    layers/heads selected.

    Each tensor in ``attentions`` has shape ``(B, H, Q, K)``. The returned tensor
    has the same shape after taking the mean over the selected ``layers`` and
    ``heads`` axes -- i.e. ``(B, Q, K)``.
    """
    if not attentions:
        raise ValueError("attentions list is empty; was the model run with attentions_out=[]?")
    if layers is None:
        layers = range(len(attentions))
    if heads is None:
        # Use all heads from the first tensor; assume all layers share num_heads.
        heads = range(int(attentions[0].shape[1]))
    layers = list(layers)
    heads = list(heads)
    selected = []
    for layer_idx in layers:
        layer_attn = attentions[layer_idx]  # (B, H, Q, K)
        # Average over the selected heads only.
        head_avg = layer_attn[:, heads].mean(dim=1)  # (B, Q, K)
        selected.append(head_avg.to(torch.float32))
    stacked = torch.stack(selected, dim=0)  # (L_sel, B, Q, K)
    return stacked.mean(dim=0)  # (B, Q, K)


def language_to_image_heatmap(
    prefix_attentions: Sequence[torch.Tensor],
    spans: TokenSpans,
    image_span: ImageSpan,
    target_token_indices_in_prompt: Sequence[int],
    *,
    layers: Sequence[int] | None = None,
    heads: Sequence[int] | None = None,
) -> np.ndarray:
    """Reduce captured prefix attention into a single (grid_h, grid_w) heatmap.

    ``target_token_indices_in_prompt`` are the prompt-local positions returned
    by :func:`find_target_word_token_indices`. They are shifted by the prefix
    image offset internally.
    """
    text_offset = spans.text_span[0]
    target_global = [text_offset + i for i in target_token_indices_in_prompt]
    if any(idx < spans.text_span[0] or idx >= spans.text_span[1] for idx in target_global):
        raise ValueError(
            f"Target indices {target_token_indices_in_prompt} fall outside text span {spans.text_span}."
        )

    # (B, Q, K) - we have only batch=1 in inference.
    avg = _select_layers_and_heads(prefix_attentions, layers, heads)
    # Average over the selected (sub-)token rows.
    rows = avg[:, target_global, :]  # (B, n_target, K)
    rows = rows.mean(dim=1)  # (B, K)
    # Slice the image-key columns and reshape to grid.
    image_cols = rows[:, image_span.start : image_span.end]  # (B, num_patches)
    grid = image_cols[0].reshape(image_span.grid_h, image_span.grid_w).cpu().numpy()
    return grid


def whole_instruction_to_image_heatmap(
    prefix_attentions: Sequence[torch.Tensor],
    spans: TokenSpans,
    image_span: ImageSpan,
    *,
    layers: Sequence[int] | None = None,
    heads: Sequence[int] | None = None,
) -> np.ndarray:
    """Reduce attention from *every valid* instruction token onto the chosen camera.

    Unlike :func:`language_to_image_heatmap`, which restricts the query to the
    sub-tokens of one target word, this averages over the full prompt span,
    masking out padding tokens. The returned heatmap answers "given the entire
    instruction (not any single word), where in the image is the model
    attending the most?".
    """
    text_start, text_end = spans.text_span
    valid_local_indices = np.where(spans.text_valid_mask)[0]
    if valid_local_indices.size == 0:
        raise ValueError("No valid instruction tokens found in spans.text_valid_mask.")
    valid_global_indices = (text_start + valid_local_indices).tolist()

    avg = _select_layers_and_heads(prefix_attentions, layers, heads)  # (B, Q, K)
    rows = avg[:, valid_global_indices, :]  # (B, n_valid, K)
    rows = rows.mean(dim=1)  # (B, K)
    image_cols = rows[:, image_span.start : image_span.end]
    grid = image_cols[0].reshape(image_span.grid_h, image_span.grid_w).cpu().numpy()
    return grid


def action_to_image_heatmap(
    suffix_attentions_per_step: Sequence[Sequence[torch.Tensor]],
    spans: TokenSpans,
    image_span: ImageSpan,
    *,
    flow_steps: Sequence[int] | None = None,
    layers: Sequence[int] | None = None,
    heads: Sequence[int] | None = None,
    action_query_indices: Sequence[int] | None = None,
) -> np.ndarray:
    """Reduce action-expert attention to a single (grid_h, grid_w) heatmap.

    Action queries (suffix tokens) attend to ``[prefix-cache | suffix]``. We slice
    only the image-key range from the prefix portion.

    Parameters
    ----------
    suffix_attentions_per_step:
        Outer list = flow-matching steps. Inner list = per-layer attention tensors
        with shape ``(B, H, suffix_len, prefix_len + suffix_len)``.
    """
    if not suffix_attentions_per_step:
        raise ValueError("suffix_attentions_per_step is empty.")
    if flow_steps is None:
        flow_steps = range(len(suffix_attentions_per_step))
    flow_steps = list(flow_steps)

    per_step_grids: list[torch.Tensor] = []
    for step_idx in flow_steps:
        step_attns = suffix_attentions_per_step[step_idx]
        avg = _select_layers_and_heads(step_attns, layers, heads)  # (B, Q=suffix_len, K=full_len)
        if action_query_indices is None:
            # All suffix queries are action queries for pi0.5; for pi0 there is one
            # extra state token, but its row is dominated by self-attention so
            # averaging over the whole suffix is still a fine default.
            q_rows = avg.mean(dim=1)  # (B, K)
        else:
            q_rows = avg[:, list(action_query_indices), :].mean(dim=1)
        image_cols = q_rows[:, image_span.start : image_span.end]
        per_step_grids.append(image_cols[0].reshape(image_span.grid_h, image_span.grid_w))
    stacked = torch.stack(per_step_grids, dim=0).mean(dim=0)
    return stacked.cpu().numpy()


def compute_modality_attention_ratios(
    suffix_attentions_per_step: Sequence[Sequence[torch.Tensor]],
    spans: TokenSpans,
    *,
    flow_steps: Sequence[int] | None = None,
    layers: Sequence[int] | None = None,
    heads: Sequence[int] | None = None,
    action_query_indices: Sequence[int] | None = None,
) -> dict[str, float]:
    """Average action->context attention into per-modality fractions.

    The returned dict has keys ``image`` / ``text`` / ``state`` / ``action`` /
    ``padding`` / ``other`` and sums to 1.0 (within fp32 precision). Padding
    is counted *separately* so the user can see what fraction of probability
    mass leaks into pad slots.
    """
    if not suffix_attentions_per_step:
        raise ValueError("suffix_attentions_per_step is empty.")
    if flow_steps is None:
        flow_steps = range(len(suffix_attentions_per_step))
    flow_steps = list(flow_steps)

    image_idx = np.concatenate(
        [np.arange(s.start, s.end, dtype=np.int64) for s in spans.image_spans]
    ) if spans.image_spans else np.zeros(0, dtype=np.int64)
    text_start = spans.text_span[0]
    text_valid_idx = np.array(
        [text_start + i for i, v in enumerate(spans.text_valid_mask) if bool(v)],
        dtype=np.int64,
    )
    text_pad_idx = np.array(
        [text_start + i for i, v in enumerate(spans.text_valid_mask) if not bool(v)],
        dtype=np.int64,
    )
    state_idx = (
        np.arange(spans.state_span[0], spans.state_span[1], dtype=np.int64)
        if spans.state_span is not None
        else np.zeros(0, dtype=np.int64)
    )
    action_idx = np.arange(spans.action_span[0], spans.action_span[1], dtype=np.int64)
    accounted = np.unique(
        np.concatenate([image_idx, text_valid_idx, text_pad_idx, state_idx, action_idx])
    )

    total: dict[str, float] = {"image": 0.0, "text": 0.0, "state": 0.0, "action": 0.0, "padding": 0.0, "other": 0.0}
    for step_idx in flow_steps:
        avg = _select_layers_and_heads(suffix_attentions_per_step[step_idx], layers, heads)  # (B,Q,K)
        if action_query_indices is None:
            row = avg.mean(dim=1)  # (B, K)
        else:
            row = avg[:, list(action_query_indices), :].mean(dim=1)
        row_np = row[0].cpu().numpy().astype(np.float64)  # (K,)
        K = row_np.shape[0]
        total["image"] += float(row_np[image_idx].sum()) if image_idx.size else 0.0
        total["text"] += float(row_np[text_valid_idx].sum()) if text_valid_idx.size else 0.0
        total["padding"] += float(row_np[text_pad_idx].sum()) if text_pad_idx.size else 0.0
        total["state"] += float(row_np[state_idx].sum()) if state_idx.size else 0.0
        total["action"] += float(row_np[action_idx].sum()) if action_idx.size else 0.0
        # Anything not accounted for (e.g. positions past the suffix end if the
        # cache was over-allocated) goes into "other".
        all_idx = np.arange(K, dtype=np.int64)
        other_idx = np.setdiff1d(all_idx, accounted, assume_unique=False)
        if other_idx.size:
            total["other"] += float(row_np[other_idx].sum())

    n = max(1, len(flow_steps))
    return {k: v / n for k, v in total.items()}


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def _normalise_heatmap(grid: np.ndarray) -> np.ndarray:
    """Linearly rescale a 2-D heatmap to [0, 1] for plotting.

    Assumes ``grid`` already contains only valid (non-pad) attention entries -
    the caller is responsible for excluding pad columns before calling.
    """
    g = grid.astype(np.float32)
    lo, hi = float(g.min()), float(g.max())
    if hi <= lo:
        return np.zeros_like(g)
    return (g - lo) / (hi - lo)


def overlay_heatmap(
    image_rgb: np.ndarray,
    heatmap_grid: np.ndarray,
    *,
    alpha: float = 0.55,
    colormap_name: str = "turbo",
) -> "PILImage.Image":
    """Overlay a (small) heatmap grid on top of an RGB image.

    The heatmap is upsampled (PIL bilinear) to the image resolution and
    blended through ``alpha``.
    """
    from PIL import Image  # local import keeps PIL out of the import-time path

    if image_rgb.dtype != np.uint8:
        image_rgb = (np.clip(image_rgb, 0, 1) * 255.0).astype(np.uint8) if image_rgb.max() <= 1.0 else image_rgb.astype(np.uint8)
    if image_rgb.ndim == 2:
        image_rgb = np.stack([image_rgb] * 3, axis=-1)
    if image_rgb.shape[-1] == 4:
        image_rgb = image_rgb[..., :3]

    base = Image.fromarray(image_rgb).convert("RGB")
    norm = _normalise_heatmap(heatmap_grid)

    cmap = _get_colormap(colormap_name)
    rgba = (cmap(norm) * 255.0).astype(np.uint8)  # (H_grid, W_grid, 4)
    heat = Image.fromarray(rgba, mode="RGBA").resize(base.size, resample=Image.BILINEAR)
    heat_rgb = heat.convert("RGB")
    blended = Image.blend(base, heat_rgb, alpha=alpha)
    return blended


def _get_colormap(name: str):
    """Return a matplotlib colormap callable; falls back to a tiny built-in if
    matplotlib is missing.
    """
    try:
        import matplotlib

        return matplotlib.colormaps.get_cmap(name)
    except Exception:  # noqa: BLE001
        # Minimal fallback: simple red colormap. Returns RGBA in [0,1].
        def _fallback(x: np.ndarray) -> np.ndarray:
            x = np.clip(x, 0.0, 1.0)
            r = x
            g = np.zeros_like(x)
            b = 1.0 - x
            a = np.ones_like(x)
            return np.stack([r, g, b, a], axis=-1)

        return _fallback


def save_heatmap_artifacts(
    out_dir: Path,
    name: str,
    image_rgb: np.ndarray,
    heatmap_grid: np.ndarray,
    *,
    overlay_alpha: float = 0.55,
    colormap_name: str = "turbo",
) -> tuple[Path, Path]:
    """Save both the rendered overlay PNG and the raw NPY heatmap; return paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{name}.png"
    npy_path = out_dir / f"{name}.npy"
    overlay = overlay_heatmap(
        image_rgb, heatmap_grid, alpha=overlay_alpha, colormap_name=colormap_name
    )
    overlay.save(png_path)
    np.save(npy_path, heatmap_grid)
    return png_path, npy_path


# ---------------------------------------------------------------------------
# Helpers for the CLI
# ---------------------------------------------------------------------------


def composite_attention_frame(
    image_rgb_per_camera: dict[str, np.ndarray],
    heatmap_grid_per_camera: dict[str, np.ndarray],
    *,
    annotation: str | None = None,
    alpha: float = 0.55,
    colormap_name: str = "turbo",
    panel_height: int = 224,
) -> np.ndarray:
    """Render a single video frame: cameras side-by-side, original on top, overlay below.

    Layout (per camera column)::

        +--- original RGB (panel_height) ---+
        |     overlay heatmap (panel_height)|
        +-----------------------------------+

    Annotations (instruction / target word / step idx) are drawn at the bottom
    if ``annotation`` is provided. Returns a uint8 ``(H, W, 3)`` array suitable
    for handing to ``mediapy.write_video``.
    """
    from PIL import Image, ImageDraw, ImageFont

    if not image_rgb_per_camera:
        raise ValueError("image_rgb_per_camera is empty.")

    columns: list[Image.Image] = []
    for cam_name, raw in image_rgb_per_camera.items():
        # Top half: original RGB resized to panel_height x panel_height.
        if raw.dtype != np.uint8:
            raw = (
                (np.clip(raw, 0, 1) * 255.0).astype(np.uint8)
                if raw.max() <= 1.0
                else raw.astype(np.uint8)
            )
        if raw.ndim == 2:
            raw = np.stack([raw] * 3, axis=-1)
        if raw.shape[-1] == 4:
            raw = raw[..., :3]
        top = Image.fromarray(raw).convert("RGB").resize((panel_height, panel_height), Image.BILINEAR)
        # Bottom half: overlay.
        grid = heatmap_grid_per_camera.get(cam_name)
        if grid is None:
            bottom = Image.new("RGB", (panel_height, panel_height), (40, 40, 40))
        else:
            bottom = overlay_heatmap(raw, grid, alpha=alpha, colormap_name=colormap_name)
            bottom = bottom.resize((panel_height, panel_height), Image.BILINEAR)
        # Camera label band on top.
        label_band = Image.new("RGB", (panel_height, 18), (0, 0, 0))
        ImageDraw.Draw(label_band).text((4, 2), cam_name, fill=(220, 220, 220))
        col = Image.new("RGB", (panel_height, panel_height * 2 + 18), (0, 0, 0))
        col.paste(label_band, (0, 0))
        col.paste(top, (0, 18))
        col.paste(bottom, (0, panel_height + 18))
        columns.append(col)

    total_w = sum(c.width for c in columns) + (len(columns) - 1) * 4
    total_h = max(c.height for c in columns)
    annot_h = 28 if annotation else 0
    canvas = Image.new("RGB", (total_w, total_h + annot_h), (10, 10, 10))
    x = 0
    for c in columns:
        canvas.paste(c, (x, 0))
        x += c.width + 4

    if annotation:
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.load_default()
        except Exception:  # noqa: BLE001
            font = None
        draw.text((6, total_h + 4), annotation, fill=(230, 230, 230), font=font)

    arr = np.asarray(canvas, dtype=np.uint8)
    # H264 encoders require even dimensions; pad if needed.
    if arr.shape[0] % 2 == 1:
        arr = np.pad(arr, ((0, 1), (0, 0), (0, 0)), mode="edge")
    if arr.shape[1] % 2 == 1:
        arr = np.pad(arr, ((0, 0), (0, 1), (0, 0)), mode="edge")
    return arr


def parse_target_words(spec: str | Iterable[str] | None) -> list[str]:
    """Accept either a list of strings or a comma-separated CLI value."""
    if spec is None:
        return []
    if isinstance(spec, str):
        return [w.strip() for w in spec.split(",") if w.strip()]
    return [str(w).strip() for w in spec if str(w).strip()]


DEFAULT_TARGET_WORDS = (
    # color words
    "blue",
    "green",
    "red",
    # object words
    "block",
    "cube",
    "bowl",
    # spatial words
    "left",
    "right",
    "front",
    "behind",
)
