#!/usr/bin/env python3
"""Visualise pi0.5 attention.

Two visualisations are produced (each can be toggled independently):

1. ``--visualize-language``: instruction-token -> image-patch attention
   (PaliGemma prefix self-attention). Useful for asking "where in the image
   does the model look when it sees the word *blue*?"

2. ``--visualize-action``: action-token -> context-token attention
   (action expert attending to the cached prefix). Useful for asking "which
   inputs is the action expert reading from while it generates actions?"

Example:

    python third_party/openpi/scripts/visualize_pi05_attention.py \
        --checkpoint <path-or-gs-uri> \
        --config pi05_droid_jointpos_polaris \
        --instruction "pick up the blue block" \
        --image base_0_rgb=path/to/scene.png \
        --camera base_0_rgb \
        --output-dir attention_outputs/blue_block \
        --visualize-language \
        --visualize-action

The script is intentionally minimally invasive: it loads the model exactly the
way ``serve_policy.py`` does (so checkpoint format / preprocessing are
identical), runs ``PI0Pytorch.sample_actions_with_attention`` instead of the
normal ``sample_actions``, then slices the captured tensors using the spans
recovered from :func:`openpi.models_pytorch.attention_utils.compute_token_spans`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import openpi.models.model as _model
from openpi.models import pi0_config
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.models_pytorch import attention_utils as _au
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


logger = logging.getLogger("visualize_pi05_attention")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path or gs:// URI to a pi0.5 PyTorch checkpoint directory "
        "(must contain model.safetensors).",
    )
    p.add_argument(
        "--config",
        required=True,
        help="Training-config name registered in openpi (e.g. 'pi05_droid_jointpos_polaris').",
    )
    p.add_argument(
        "--instruction",
        required=True,
        help='Natural-language instruction to test, e.g. "pick up the blue block".',
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Where to write attention_outputs/{metadata.json, language_grounding/*, action_attention/*}.",
    )
    p.add_argument(
        "--image",
        action="append",
        default=[],
        metavar="CAMERA_KEY=PATH",
        help="Per-camera image path. Format: 'base_0_rgb=/path/to/img.png'. "
        "May be passed multiple times. Cameras not specified will be filled with zeros "
        "so they don't dominate the heatmap normalisation.",
    )
    p.add_argument(
        "--camera",
        default=None,
        help="Camera key whose image-patch grid is used for the heatmaps. "
        "Defaults to the first --image argument or the first camera in the model spec.",
    )
    p.add_argument(
        "--state",
        default=None,
        help="Comma-separated proprio state vector. Length must match model.action_dim "
        "(zeros are used when omitted).",
    )
    p.add_argument(
        "--target-words",
        default=None,
        help="Comma-separated target words for language-grounding heatmaps. "
        "Defaults to a small built-in list of color/object/spatial words.",
    )
    p.add_argument("--num-flow-steps", type=int, default=10, help="Number of flow-matching steps.")
    p.add_argument(
        "--layers",
        default=None,
        help="Comma-separated list of layer indices to average over (default: all).",
    )
    p.add_argument(
        "--heads",
        default=None,
        help="Comma-separated list of head indices to average over (default: all).",
    )
    p.add_argument(
        "--flow-steps",
        default=None,
        help="Comma-separated list of flow-matching step indices to average action attention over (default: all).",
    )
    p.add_argument("--visualize-language", action="store_true")
    p.add_argument("--visualize-action", action="store_true")
    p.add_argument("--device", default=None, help="torch device (default: cuda if available, else cpu).")
    p.add_argument("--alpha", type=float, default=0.55, help="Heatmap overlay alpha.")
    p.add_argument("--colormap", default="turbo", help="Matplotlib colormap name (e.g. turbo, jet, magma).")
    return p


def _parse_int_list(spec: str | None) -> list[int] | None:
    if spec is None:
        return None
    return [int(x) for x in spec.split(",") if x.strip()]


def _parse_image_args(image_args: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for spec in image_args:
        if "=" not in spec:
            raise ValueError(f"--image expects CAMERA_KEY=PATH, got {spec!r}")
        key, path = spec.split("=", 1)
        out[key.strip()] = Path(path.strip())
    return out


# ---------------------------------------------------------------------------
# Observation construction
# ---------------------------------------------------------------------------


def _load_image_uint8(path: Path) -> np.ndarray:
    """Load an image as uint8 RGB (H, W, 3)."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _resize_with_pad(img_uint8: np.ndarray, height: int, width: int) -> np.ndarray:
    """Match ``image_tools.resize_with_pad`` semantics for a single uint8 image."""
    from openpi_client import image_tools

    return image_tools.resize_with_pad(img_uint8, height, width)


def _build_observation(
    *,
    model_config: pi0_config.Pi0Config,
    image_keys: list[str],
    image_paths: dict[str, Path],
    state_vector: np.ndarray,
    instruction: str,
    device: torch.device,
) -> tuple[_model.Observation, dict[str, np.ndarray]]:
    """Construct a model-ready Observation. Also returns the per-camera resized
    uint8 RGB arrays (used for the visual overlays) so the caller does not have
    to re-load + re-resize.
    """
    raw_uint8: dict[str, np.ndarray] = {}
    images: dict[str, torch.Tensor] = {}
    image_masks: dict[str, torch.Tensor] = {}

    for key in image_keys:
        path = image_paths.get(key)
        if path is not None:
            arr = _load_image_uint8(path)
            arr = _resize_with_pad(arr, 224, 224)
            mask_value = True
        else:
            # All-zero placeholder so the slot exists but does not pull attention.
            arr = np.zeros((224, 224, 3), dtype=np.uint8)
            mask_value = False
        raw_uint8[key] = arr
        # The model expects floats in [-1, 1].
        f = (arr.astype(np.float32) / 127.5) - 1.0
        images[key] = torch.from_numpy(f).unsqueeze(0).to(device)  # (1,H,W,C)
        image_masks[key] = torch.tensor([mask_value], dtype=torch.bool, device=device)

    # State: pi0.5 uses discrete state, baked into the prompt. The model nevertheless
    # carries a state tensor of length action_dim through preprocessing.
    state = torch.from_numpy(state_vector.astype(np.float32)).unsqueeze(0).to(device)

    # Tokenise the prompt with the same tokenizer the data pipeline would use.
    tok = PaligemmaTokenizer(model_config.max_token_len)
    if model_config.discrete_state_input:
        tokens, mask = tok.tokenize(instruction, state_vector.astype(np.float32))
    else:
        tokens, mask = tok.tokenize(instruction)
    tokenized_prompt = torch.from_numpy(np.asarray(tokens, dtype=np.int64)).unsqueeze(0).to(device)
    tokenized_prompt_mask = torch.from_numpy(np.asarray(mask, dtype=bool)).unsqueeze(0).to(device)

    observation = _model.Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )
    return observation, raw_uint8


# ---------------------------------------------------------------------------
# Camera / span helpers
# ---------------------------------------------------------------------------


def _resolve_camera(
    spans: _au.TokenSpans, requested: str | None, raw_uint8: dict[str, np.ndarray]
) -> _au.ImageSpan:
    if requested is None:
        # Prefer the first camera that actually has a non-zero image.
        for span in spans.image_spans:
            arr = raw_uint8.get(span.name)
            if arr is not None and arr.any():
                return span
        return spans.image_spans[0]
    for span in spans.image_spans:
        if span.name == requested:
            return span
    raise ValueError(
        f"--camera {requested!r} not found among image keys {[s.name for s in spans.image_spans]}."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.visualize_language and not args.visualize_action:
        logger.error("Nothing to do: pass --visualize-language and/or --visualize-action.")
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")

    # 1. Resolve the training config so we know which model to instantiate.
    train_cfg = _config.get_config(args.config)
    if not isinstance(train_cfg.model, pi0_config.Pi0Config) or not train_cfg.model.pi05:
        logger.error(
            "Config %r is not a pi0.5 model (got %r); attention visualisation only supports pi0.5.",
            args.config,
            type(train_cfg.model).__name__,
        )
        return 2

    device_str = args.device
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # 2. Build the policy via the existing loader (uses the same transforms / weights
    #    that serve_policy.py would). We immediately pluck the wrapped PyTorch model
    #    out of it -- the Policy.infer() pipeline does not currently expose attentions.
    policy = _policy_config.create_trained_policy(
        train_cfg, args.checkpoint, pytorch_device=device_str
    )
    if not policy._is_pytorch_model:  # type: ignore[attr-defined]
        logger.error("Loaded checkpoint is not a PyTorch model (no model.safetensors). Aborting.")
        return 2
    model = policy._model  # type: ignore[attr-defined]
    model.eval()

    # 3. Build observation.
    image_paths = _parse_image_args(args.image)
    spec_obs, _ = train_cfg.model.inputs_spec(batch_size=1)
    image_keys = list(spec_obs.images.keys())
    if args.camera and args.camera not in image_keys:
        logger.warning(
            "Camera %r is not in the model's expected camera list %s; "
            "the camera will still be honoured if --image specifies it.",
            args.camera,
            image_keys,
        )

    state_vec = _parse_state(args.state, model_dim=train_cfg.model.action_dim)

    observation, raw_uint8 = _build_observation(
        model_config=train_cfg.model,
        image_keys=image_keys,
        image_paths=image_paths,
        state_vector=state_vec,
        instruction=args.instruction,
        device=device,
    )

    # 4. Capture attentions during inference.
    logger.info("Running sample_actions_with_attention (num_steps=%d, device=%s)…", args.num_flow_steps, device_str)
    captured = model.sample_actions_with_attention(device, observation, num_steps=args.num_flow_steps)

    # 5. Recover token spans from the *real* preprocessing pass.
    spans = _au.compute_token_spans(model, observation)

    layers = _parse_int_list(args.layers)
    heads = _parse_int_list(args.heads)
    flow_steps = _parse_int_list(args.flow_steps)

    # 6. Pick the camera whose grid we'll project onto.
    chosen_camera = _resolve_camera(spans, args.camera, raw_uint8)
    base_image = raw_uint8[chosen_camera.name]
    logger.info(
        "Using camera %r for visualisation. Token span = [%d, %d), grid = %dx%d.",
        chosen_camera.name,
        chosen_camera.start,
        chosen_camera.end,
        chosen_camera.grid_h,
        chosen_camera.grid_w,
    )

    # 7. Persist token-info / metadata first so partial failures still leave a trail.
    _save_token_info(args.output_dir, spans, train_cfg.model)
    _save_metadata(
        args.output_dir,
        instruction=args.instruction,
        chosen_camera=chosen_camera,
        layers=layers,
        heads=heads,
        flow_steps=flow_steps,
        num_flow_steps=args.num_flow_steps,
        prefix_attention_shapes=[tuple(t.shape) for t in captured["prefix_attentions"]],
        suffix_attention_shapes=[
            [tuple(t.shape) for t in step] for step in captured["suffix_attentions_per_step"]
        ],
        config_name=args.config,
        checkpoint=str(args.checkpoint),
        device=device_str,
        target_words=_au.parse_target_words(args.target_words) or list(_au.DEFAULT_TARGET_WORDS),
    )

    # 8. Language-grounding heatmaps.
    if args.visualize_language:
        _run_language_grounding(
            output_dir=args.output_dir / "language_grounding",
            instruction=args.instruction,
            target_words=_au.parse_target_words(args.target_words) or list(_au.DEFAULT_TARGET_WORDS),
            tokenizer=PaligemmaTokenizer(train_cfg.model.max_token_len),
            spans=spans,
            chosen_camera=chosen_camera,
            base_image=base_image,
            prefix_attentions=captured["prefix_attentions"],
            layers=layers,
            heads=heads,
            alpha=args.alpha,
            colormap=args.colormap,
            state_vector=state_vec if train_cfg.model.discrete_state_input else None,
        )

    # 9. Action-attention heatmap + modality ratios.
    if args.visualize_action:
        _run_action_attention(
            output_dir=args.output_dir / "action_attention",
            spans=spans,
            chosen_camera=chosen_camera,
            base_image=base_image,
            suffix_attentions_per_step=captured["suffix_attentions_per_step"],
            flow_steps=flow_steps,
            layers=layers,
            heads=heads,
            alpha=args.alpha,
            colormap=args.colormap,
        )

    logger.info("Done. Outputs at %s", args.output_dir.resolve())
    return 0


# ---------------------------------------------------------------------------
# Sub-pipelines
# ---------------------------------------------------------------------------


def _run_language_grounding(
    *,
    output_dir: Path,
    instruction: str,
    target_words: list[str],
    tokenizer: PaligemmaTokenizer,
    spans: _au.TokenSpans,
    chosen_camera: _au.ImageSpan,
    base_image: np.ndarray,
    prefix_attentions: list[torch.Tensor],
    layers: list[int] | None,
    heads: list[int] | None,
    alpha: float,
    colormap: str,
    state_vector: np.ndarray | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "instruction": instruction,
        "camera": chosen_camera.name,
        "grid_size": [chosen_camera.grid_h, chosen_camera.grid_w],
        "words": {},
    }
    for word in target_words:
        try:
            token_indices = _au.find_target_word_token_indices(
                tokenizer, instruction, word, state=state_vector
            )
        except ValueError as e:
            logger.info("Skipping target word %r: %s", word, e)
            summary["words"][word] = {"status": "not_in_instruction"}
            continue

        grid = _au.language_to_image_heatmap(
            prefix_attentions=prefix_attentions,
            spans=spans,
            image_span=chosen_camera,
            target_token_indices_in_prompt=token_indices,
            layers=layers,
            heads=heads,
        )
        png, npy = _au.save_heatmap_artifacts(
            output_dir,
            f"{_safe_filename(word)}_{chosen_camera.name}",
            base_image,
            grid,
            overlay_alpha=alpha,
            colormap_name=colormap,
        )
        summary["words"][word] = {
            "status": "ok",
            "token_indices_in_prompt": token_indices,
            "png": str(png.relative_to(output_dir.parent)),
            "npy": str(npy.relative_to(output_dir.parent)),
            "grid_min": float(grid.min()),
            "grid_max": float(grid.max()),
            "grid_mean": float(grid.mean()),
        }
        logger.info(
            "language_grounding[%s]: tokens=%s -> grid sum=%.4f", word, token_indices, float(grid.sum())
        )

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def _run_action_attention(
    *,
    output_dir: Path,
    spans: _au.TokenSpans,
    chosen_camera: _au.ImageSpan,
    base_image: np.ndarray,
    suffix_attentions_per_step: list[list[torch.Tensor]],
    flow_steps: list[int] | None,
    layers: list[int] | None,
    heads: list[int] | None,
    alpha: float,
    colormap: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = _au.action_to_image_heatmap(
        suffix_attentions_per_step=suffix_attentions_per_step,
        spans=spans,
        image_span=chosen_camera,
        flow_steps=flow_steps,
        layers=layers,
        heads=heads,
    )
    png, npy = _au.save_heatmap_artifacts(
        output_dir,
        f"action_{chosen_camera.name}",
        base_image,
        grid,
        overlay_alpha=alpha,
        colormap_name=colormap,
    )
    logger.info(
        "action_attention[%s]: grid sum=%.4f -> %s", chosen_camera.name, float(grid.sum()), png
    )

    ratios = _au.compute_modality_attention_ratios(
        suffix_attentions_per_step=suffix_attentions_per_step,
        spans=spans,
        flow_steps=flow_steps,
        layers=layers,
        heads=heads,
    )
    (output_dir / "modality_attention_ratios.json").write_text(json.dumps(ratios, indent=2))
    logger.info(
        "modality ratios: image=%.3f text=%.3f state=%.3f action=%.3f padding=%.3f other=%.3f",
        ratios["image"],
        ratios["text"],
        ratios["state"],
        ratios["action"],
        ratios["padding"],
        ratios["other"],
    )


# ---------------------------------------------------------------------------
# Saving metadata + token_info
# ---------------------------------------------------------------------------


def _save_token_info(out_dir: Path, spans: _au.TokenSpans, model_config: pi0_config.Pi0Config) -> None:
    info = {
        "max_token_len": model_config.max_token_len,
        "action_horizon": model_config.action_horizon,
        "action_dim": model_config.action_dim,
        "pi05": True,
        "discrete_state_input": model_config.discrete_state_input,
        **spans.to_metadata_dict(),
    }
    (out_dir / "token_info.json").write_text(json.dumps(info, indent=2))


def _save_metadata(
    out_dir: Path,
    *,
    instruction: str,
    chosen_camera: _au.ImageSpan,
    layers: list[int] | None,
    heads: list[int] | None,
    flow_steps: list[int] | None,
    num_flow_steps: int,
    prefix_attention_shapes: list[tuple[int, ...]],
    suffix_attention_shapes: list[list[tuple[int, ...]]],
    config_name: str,
    checkpoint: str,
    device: str,
    target_words: list[str],
) -> None:
    metadata = {
        "instruction": instruction,
        "config_name": config_name,
        "checkpoint": checkpoint,
        "device": device,
        "chosen_camera": {
            "name": chosen_camera.name,
            "start": chosen_camera.start,
            "end": chosen_camera.end,
            "grid_h": chosen_camera.grid_h,
            "grid_w": chosen_camera.grid_w,
        },
        "selected_layers": layers,
        "selected_heads": heads,
        "selected_flow_steps": flow_steps,
        "num_flow_steps": num_flow_steps,
        "prefix_attention_shapes": [list(s) for s in prefix_attention_shapes],
        "suffix_attention_shapes": [
            [list(s) for s in step] for step in suffix_attention_shapes
        ],
        "target_words": target_words,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _parse_state(spec: str | None, model_dim: int) -> np.ndarray:
    if spec is None:
        return np.zeros(model_dim, dtype=np.float32)
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    arr = np.asarray([float(p) for p in parts], dtype=np.float32)
    if arr.shape[0] != model_dim:
        # Pad or truncate to model_dim. Pad with zeros (most checkpoints expect this).
        if arr.shape[0] < model_dim:
            arr = np.pad(arr, (0, model_dim - arr.shape[0]))
        else:
            arr = arr[:model_dim]
    return arr


def _safe_filename(word: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in word).strip("_") or "word"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
