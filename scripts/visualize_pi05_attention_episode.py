#!/usr/bin/env python3
"""Render pi0.5 attention as MP4 videos over a full episode (per-step).

Two input modes are supported:

A. ``--obs-dump-dir <dir>``: replay a directory of ``chunk_<NNNNNN>.npz`` files
   that the polaris ``DroidJointPosClient`` writes when the env var
   ``POLARIS_PI05_OBS_DUMP_DIR`` is set. This is the recommended path: every
   chunk dumped is exactly the request the server saw, so the attention you
   visualise is the same attention the policy used during the rollout.

B. ``--video CAMERA_KEY=PATH`` (repeatable) + optional ``--state-trajectory PATH``:
   pre-recorded camera videos (one mp4 per camera). Useful when you want to
   re-run attention analysis on data that was not captured via the Droid
   client (e.g. raw camera streams).

For each input frame the script:

  - Builds an Observation in exactly the same way as the live serving stack
    (the same data_transforms / model_transforms registered for the
    training config).
  - Runs ``PI0Pytorch.sample_actions_with_attention`` (PyTorch checkpoints
    only).
  - Renders a single composite video frame with both cameras side by side,
    the original RGB on top and the attention overlay on the bottom.
  - Writes to one MP4 per visualisation mode:
      * ``language_grounding/<word>.mp4`` — instruction-token -> image-patch
        attention for each requested target word.
      * ``action_attention.mp4`` — action-expert -> image-patch attention.

Outputs also include ``token_info.json`` and ``metadata.json`` (single
snapshot for the first frame; spans are stable across frames so the same
metadata applies to every frame in every video).

Example::

    python third_party/openpi/scripts/visualize_pi05_attention_episode.py \\
        --checkpoint <pytorch-ckpt-dir> \\
        --config pi05_droid_jointpos_polaris \\
        --obs-dump-dir runs/pi05_lang_following/<case>/episode_0_obs/ \\
        --output-dir runs/pi05_lang_following/<case>/attention_episode/ \\
        --visualize-language --visualize-action \\
        --target-words "blue,green,block,left"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

import openpi.models.model as _model
from openpi.models import pi0_config
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.models_pytorch import attention_utils as _au
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


logger = logging.getLogger("visualize_pi05_attention_episode")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True, help="PyTorch pi0.5 checkpoint dir.")
    p.add_argument("--config", required=True, help="openpi training config name.")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--device", default=None)
    p.add_argument("--num-flow-steps", type=int, default=10)

    # Mode A: replay observations dumped by the polaris client.
    p.add_argument(
        "--obs-dump-dir",
        type=Path,
        default=None,
        help="Directory with chunk_*.npz files written by DroidJointPosClient "
        "(set env POLARIS_PI05_OBS_DUMP_DIR while running run_eval_policy.py).",
    )
    # Mode B: per-camera video files.
    p.add_argument(
        "--video",
        action="append",
        default=[],
        metavar="CAMERA_KEY=PATH",
        help="Per-camera input video. Repeatable. Example: 'base_0_rgb=ext.mp4'.",
    )
    p.add_argument(
        "--state-trajectory",
        type=Path,
        default=None,
        help="Optional .npy of shape (T, action_dim) with the proprio state per "
        "frame. Required for accurate pi0.5 attention (state is part of the "
        "discrete prompt). Falls back to zeros if missing.",
    )
    p.add_argument(
        "--instruction",
        default=None,
        help="Override / supply instruction. With --obs-dump-dir, the prompt is "
        "auto-loaded from each .npz; this flag forces a single instruction across "
        "all frames.",
    )

    p.add_argument("--visualize-language", action="store_true")
    p.add_argument("--visualize-action", action="store_true")
    p.add_argument(
        "--target-words",
        default=None,
        help="Comma-separated target words for language-grounding videos. "
        "Defaults to a built-in list.",
    )
    p.add_argument("--layers", default=None, help="Comma-separated layer indices to average.")
    p.add_argument("--heads", default=None, help="Comma-separated head indices to average.")
    p.add_argument("--flow-steps", default=None, help="Flow-matching steps to average action attention over.")
    p.add_argument("--alpha", type=float, default=0.55)
    p.add_argument("--colormap", default="turbo")
    p.add_argument("--fps", type=float, default=10.0, help="Output mp4 frame rate.")
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on number of frames to process (debug / smoke tests).",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every Nth input frame. Defaults to 1 (every frame).",
    )
    return p


def _parse_int_list(spec: str | None) -> list[int] | None:
    if spec is None:
        return None
    return [int(x) for x in spec.split(",") if x.strip()]


def _parse_video_args(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for spec in items:
        if "=" not in spec:
            raise ValueError(f"--video expects CAMERA_KEY=PATH, got {spec!r}")
        key, path = spec.split("=", 1)
        out[key.strip()] = Path(path.strip())
    return out


# ---------------------------------------------------------------------------
# Frame iterators
# ---------------------------------------------------------------------------


class FramePayload:
    """A single per-frame input bundle: raw camera images, state, instruction."""

    __slots__ = ("step", "raw_uint8", "state", "instruction")

    def __init__(
        self,
        *,
        step: int,
        raw_uint8: dict[str, np.ndarray],
        state: np.ndarray,
        instruction: str,
    ) -> None:
        self.step = step
        self.raw_uint8 = raw_uint8
        self.state = state
        self.instruction = instruction


def _iter_frames_from_obs_dump(
    dump_dir: Path,
    *,
    instruction_override: str | None,
    image_keys: list[str],
    droid_to_model: dict[str, str],
    model_dim: int,
    stride: int,
    max_frames: int | None,
) -> Iterator[FramePayload]:
    files = sorted(dump_dir.glob("chunk_*.npz"))
    if not files:
        raise FileNotFoundError(f"No chunk_*.npz under {dump_dir}.")
    out_count = 0
    for i, path in enumerate(files):
        if i % stride != 0:
            continue
        npz = np.load(path)
        # Recover the original DROID-namespaced keys (we replaced "/" with "__"
        # when dumping). Note: numpy 0-d str arrays survive as ``.item()``.
        ext = npz.get("observation__exterior_image_1_left")
        wrist = npz.get("observation__wrist_image_left")
        joint = npz.get("observation__joint_position")
        gripper = npz.get("observation__gripper_position")
        prompt_arr = npz.get("prompt")
        prompt = instruction_override if instruction_override is not None else (
            str(prompt_arr.item()) if prompt_arr is not None else ""
        )

        raw_uint8: dict[str, np.ndarray] = {}
        if ext is not None and "exterior_image_1_left" in droid_to_model:
            raw_uint8[droid_to_model["exterior_image_1_left"]] = np.asarray(ext, dtype=np.uint8)
        if wrist is not None and "wrist_image_left" in droid_to_model:
            raw_uint8[droid_to_model["wrist_image_left"]] = np.asarray(wrist, dtype=np.uint8)

        # Build a state vector of length model_dim from the 7-dof joint + 1 gripper.
        state = np.zeros(model_dim, dtype=np.float32)
        joint_len = 0
        if joint is not None:
            joint_arr = np.asarray(joint, dtype=np.float32).reshape(-1)
            joint_len = min(joint_arr.shape[0], state.shape[0])
            state[:joint_len] = joint_arr[:joint_len]
        if gripper is not None and joint_len < state.shape[0]:
            gripper_arr = np.asarray(gripper, dtype=np.float32).reshape(-1)
            state[joint_len] = float(gripper_arr[0])

        # Pad zero entries for any remaining model camera slots.
        for key in image_keys:
            raw_uint8.setdefault(key, np.zeros((224, 224, 3), dtype=np.uint8))

        yield FramePayload(step=i, raw_uint8=raw_uint8, state=state, instruction=prompt)
        out_count += 1
        if max_frames is not None and out_count >= max_frames:
            break


def _iter_frames_from_videos(
    video_paths: dict[str, Path],
    *,
    instruction: str,
    state_trajectory: np.ndarray | None,
    image_keys: list[str],
    model_dim: int,
    stride: int,
    max_frames: int | None,
) -> Iterator[FramePayload]:
    import mediapy

    readers = {key: mediapy.read_video(str(p)) for key, p in video_paths.items()}
    # min length across cameras
    n = min(len(v) for v in readers.values())
    out_count = 0
    for t in range(n):
        if t % stride != 0:
            continue
        raw_uint8 = {}
        for key in image_keys:
            v = readers.get(key)
            if v is None:
                raw_uint8[key] = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                raw_uint8[key] = np.asarray(v[t], dtype=np.uint8)
        if state_trajectory is not None and t < state_trajectory.shape[0]:
            state = state_trajectory[t].astype(np.float32)
            if state.shape[0] < model_dim:
                state = np.pad(state, (0, model_dim - state.shape[0]))
            elif state.shape[0] > model_dim:
                state = state[:model_dim]
        else:
            state = np.zeros(model_dim, dtype=np.float32)
        yield FramePayload(step=t, raw_uint8=raw_uint8, state=state, instruction=instruction)
        out_count += 1
        if max_frames is not None and out_count >= max_frames:
            break


# ---------------------------------------------------------------------------
# Per-frame inference + heatmap construction
# ---------------------------------------------------------------------------


def _build_observation(
    *,
    model_config: pi0_config.Pi0Config,
    image_keys: list[str],
    raw_uint8: dict[str, np.ndarray],
    state_vector: np.ndarray,
    instruction: str,
    device: torch.device | None,
    is_pytorch: bool,
) -> _model.Observation:
    """Construct a model-ready Observation for either JAX or PyTorch backbones.

    Both backbones expect the same image normalisation (-1, 1) and the same
    tokenized prompt. The only difference is the array library: torch on a
    specific device for PyTorch, jax.numpy (CPU/host) for JAX (the JAX model
    moves arrays to the accelerator internally).
    """
    from openpi_client import image_tools

    images: dict = {}
    image_masks: dict = {}
    for key in image_keys:
        arr = raw_uint8.get(key)
        if arr is None or not arr.any():
            arr = np.zeros((224, 224, 3), dtype=np.uint8)
            mask_value = False
        else:
            arr = image_tools.resize_with_pad(arr.astype(np.uint8), 224, 224)
            mask_value = True
        f = (arr.astype(np.float32) / 127.5) - 1.0
        if is_pytorch:
            images[key] = torch.from_numpy(f).unsqueeze(0).to(device)
            image_masks[key] = torch.tensor([mask_value], dtype=torch.bool, device=device)
        else:
            import jax.numpy as jnp_

            images[key] = jnp_.asarray(f)[None, ...]
            image_masks[key] = jnp_.asarray([mask_value], dtype=jnp_.bool_)

    tok = PaligemmaTokenizer(model_config.max_token_len)
    if model_config.discrete_state_input:
        tokens, mask = tok.tokenize(instruction, state_vector.astype(np.float32))
    else:
        tokens, mask = tok.tokenize(instruction)

    if is_pytorch:
        state = torch.from_numpy(state_vector.astype(np.float32)).unsqueeze(0).to(device)
        tokenized_prompt = torch.from_numpy(np.asarray(tokens, dtype=np.int64)).unsqueeze(0).to(device)
        tokenized_prompt_mask = torch.from_numpy(np.asarray(mask, dtype=bool)).unsqueeze(0).to(device)
    else:
        import jax.numpy as jnp_

        state = jnp_.asarray(state_vector.astype(np.float32))[None, ...]
        tokenized_prompt = jnp_.asarray(np.asarray(tokens, dtype=np.int32))[None, ...]
        tokenized_prompt_mask = jnp_.asarray(np.asarray(mask, dtype=bool))[None, ...]

    return _model.Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )


def _resized_uint8_per_camera(raw_uint8: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return resized 224x224 uint8 RGB frames keyed by camera name."""
    from openpi_client import image_tools

    out: dict[str, np.ndarray] = {}
    for key, arr in raw_uint8.items():
        if arr is None:
            out[key] = np.zeros((224, 224, 3), dtype=np.uint8)
            continue
        out[key] = image_tools.resize_with_pad(np.asarray(arr, dtype=np.uint8), 224, 224)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.visualize_language and not args.visualize_action:
        logger.error("Nothing to do: pass --visualize-language and/or --visualize-action.")
        return 2
    if not (args.obs_dump_dir or args.video):
        logger.error("Provide either --obs-dump-dir or one or more --video CAMERA_KEY=PATH.")
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")

    train_cfg = _config.get_config(args.config)
    if not isinstance(train_cfg.model, pi0_config.Pi0Config) or not train_cfg.model.pi05:
        logger.error("Config %r is not a pi0.5 model.", args.config)
        return 2

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    policy = _policy_config.create_trained_policy(train_cfg, args.checkpoint, pytorch_device=device_str)
    is_pytorch = bool(policy._is_pytorch_model)  # type: ignore[attr-defined]
    model = policy._model  # type: ignore[attr-defined]
    if is_pytorch:
        model.eval()
    if not is_pytorch:
        # JAX attention capture relies on flax sow + ``mutable=['intermediates']``,
        # which the current ``flax.nnx.bridge.ToNNX`` wrapper does NOT forward to
        # the underlying linen apply. As a result the attention probabilities
        # cannot be retrieved through this code path. Subtask decomposition still
        # works in JAX (it doesn't need per-layer attention), but per-layer
        # attention visualization is currently PyTorch-only.
        logger.error(
            "Per-layer attention capture is not available on JAX checkpoints in the current "
            "openpi build (flax-nnx ToNNX bridge does not forward mutable=['intermediates']). "
            "Convert your checkpoint to PyTorch with examples/convert_jax_model_to_pytorch.py "
            "and rerun this script with the converted directory:\n"
            "  cd third_party/openpi\n"
            "  uv run python examples/convert_jax_model_to_pytorch.py \\\n"
            "      --checkpoint_dir <jax-ckpt-dir> --config_name %s \\\n"
            "      --output_path <jax-ckpt-dir>_pt --precision bfloat16\n"
            "Then point --checkpoint at <jax-ckpt-dir>_pt.",
            args.config,
        )
        return 2
    if not hasattr(model, "sample_actions_with_attention"):
        logger.error(
            "Loaded model has no sample_actions_with_attention(). Rebuild openpi."
        )
        return 2

    # Spec image keys (e.g., base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb).
    spec_obs, _ = train_cfg.model.inputs_spec(batch_size=1)
    image_keys = list(spec_obs.images.keys())

    # DROID -> model-camera mapping (used only when reading from --obs-dump-dir).
    # The droid policy maps:
    #   exterior_image_1_left -> base_0_rgb
    #   wrist_image_left      -> right_wrist_0_rgb (or left_wrist_0_rgb on some configs)
    # We pick the first wrist key the model expects to keep this robust across
    # 1- vs 2-wrist setups.
    wrist_target = next((k for k in image_keys if "wrist" in k), image_keys[-1])
    droid_to_model = {
        "exterior_image_1_left": "base_0_rgb" if "base_0_rgb" in image_keys else image_keys[0],
        "wrist_image_left": wrist_target,
    }

    target_words = _au.parse_target_words(args.target_words) or list(_au.DEFAULT_TARGET_WORDS)
    layers = _parse_int_list(args.layers)
    heads = _parse_int_list(args.heads)
    flow_steps = _parse_int_list(args.flow_steps)

    # Build the frame iterator.
    if args.obs_dump_dir is not None:
        frames_iter = _iter_frames_from_obs_dump(
            args.obs_dump_dir,
            instruction_override=args.instruction,
            image_keys=image_keys,
            droid_to_model=droid_to_model,
            model_dim=train_cfg.model.action_dim,
            stride=args.stride,
            max_frames=args.max_frames,
        )
    else:
        video_paths = _parse_video_args(args.video)
        if args.instruction is None:
            logger.error("--instruction is required in --video mode.")
            return 2
        state_traj = (
            np.load(args.state_trajectory) if args.state_trajectory is not None else None
        )
        frames_iter = _iter_frames_from_videos(
            video_paths,
            instruction=args.instruction,
            state_trajectory=state_traj,
            image_keys=image_keys,
            model_dim=train_cfg.model.action_dim,
            stride=args.stride,
            max_frames=args.max_frames,
        )

    # Output writers. We open them lazily on the first frame.
    import mediapy

    lang_dir = args.output_dir / "language_grounding"
    action_dir = args.output_dir / "action_attention"
    if args.visualize_language:
        lang_dir.mkdir(parents=True, exist_ok=True)
    if args.visualize_action:
        action_dir.mkdir(parents=True, exist_ok=True)

    word_writers: dict[str, Any] = {}
    action_writer: Any = None
    whole_instruction_writer: Any = None

    spans_first: _au.TokenSpans | None = None
    instruction_first: str | None = None
    rendered_frames = 0
    skipped_words: dict[str, int] = {}

    tokenizer = PaligemmaTokenizer(train_cfg.model.max_token_len)

    for fp in frames_iter:
        observation = _build_observation(
            model_config=train_cfg.model,
            image_keys=image_keys,
            raw_uint8=fp.raw_uint8,
            state_vector=fp.state,
            instruction=fp.instruction,
            device=device,
            is_pytorch=True,
        )

        with torch.no_grad():
            captured = model.sample_actions_with_attention(
                device, observation, num_steps=args.num_flow_steps
            )

        spans = _au.compute_token_spans(model, observation)
        if spans_first is None:
            spans_first = spans
            instruction_first = fp.instruction
            _save_token_info(args.output_dir, spans, train_cfg.model)

        resized = _resized_uint8_per_camera(fp.raw_uint8)

        # ----- language grounding -----
        if args.visualize_language:
            # Whole-instruction view: query = every valid instruction token, averaged.
            # Answers "where does the *entire prompt* attend in the image?".
            whole_heatmaps = {}
            for cam_span in spans.image_spans:
                grid = _au.whole_instruction_to_image_heatmap(
                    prefix_attentions=captured["prefix_attentions"],
                    spans=spans,
                    image_span=cam_span,
                    layers=layers,
                    heads=heads,
                )
                whole_heatmaps[cam_span.name] = grid
            annotation = f"step={fp.step}  whole_instruction  '{fp.instruction[:80]}'"
            frame = _au.composite_attention_frame(
                image_rgb_per_camera={s.name: resized[s.name] for s in spans.image_spans},
                heatmap_grid_per_camera=whole_heatmaps,
                annotation=annotation,
                alpha=args.alpha,
                colormap_name=args.colormap,
            )
            if whole_instruction_writer is None:
                whole_path = lang_dir / "whole_instruction.mp4"
                whole_instruction_writer = mediapy.VideoWriter(
                    str(whole_path), shape=frame.shape[:2], fps=args.fps
                )
                whole_instruction_writer.__enter__()
            whole_instruction_writer.add_image(frame)

            # Per-target-word drill-down videos.
            for word in target_words:
                try:
                    token_idx = _au.find_target_word_token_indices(
                        tokenizer,
                        fp.instruction,
                        word,
                        state=fp.state if train_cfg.model.discrete_state_input else None,
                    )
                except ValueError:
                    skipped_words[word] = skipped_words.get(word, 0) + 1
                    continue

                heatmaps = {}
                for cam_span in spans.image_spans:
                    grid = _au.language_to_image_heatmap(
                        prefix_attentions=captured["prefix_attentions"],
                        spans=spans,
                        image_span=cam_span,
                        target_token_indices_in_prompt=token_idx,
                        layers=layers,
                        heads=heads,
                    )
                    heatmaps[cam_span.name] = grid
                annotation = f"step={fp.step}  word={word!r}  '{fp.instruction[:80]}'"
                frame = _au.composite_attention_frame(
                    image_rgb_per_camera={s.name: resized[s.name] for s in spans.image_spans},
                    heatmap_grid_per_camera=heatmaps,
                    annotation=annotation,
                    alpha=args.alpha,
                    colormap_name=args.colormap,
                )
                writer = word_writers.get(word)
                if writer is None:
                    out_path = lang_dir / f"{_safe_filename(word)}.mp4"
                    writer = mediapy.VideoWriter(str(out_path), shape=frame.shape[:2], fps=args.fps)
                    writer.__enter__()
                    word_writers[word] = writer
                writer.add_image(frame)

        # ----- action attention -----
        if args.visualize_action:
            heatmaps = {}
            for cam_span in spans.image_spans:
                grid = _au.action_to_image_heatmap(
                    suffix_attentions_per_step=captured["suffix_attentions_per_step"],
                    spans=spans,
                    image_span=cam_span,
                    flow_steps=flow_steps,
                    layers=layers,
                    heads=heads,
                )
                heatmaps[cam_span.name] = grid
            annotation = f"step={fp.step}  action_attention  '{fp.instruction[:80]}'"
            frame = _au.composite_attention_frame(
                image_rgb_per_camera={s.name: resized[s.name] for s in spans.image_spans},
                heatmap_grid_per_camera=heatmaps,
                annotation=annotation,
                alpha=args.alpha,
                colormap_name=args.colormap,
            )
            if action_writer is None:
                out_path = action_dir / "action.mp4"
                action_writer = mediapy.VideoWriter(str(out_path), shape=frame.shape[:2], fps=args.fps)
                action_writer.__enter__()
            action_writer.add_image(frame)

        # Free per-frame attention tensors immediately to keep peak memory low.
        del captured
        if device.type == "cuda":
            torch.cuda.empty_cache()
        rendered_frames += 1
        if rendered_frames % 10 == 0:
            logger.info("rendered %d frames", rendered_frames)

    # Close writers.
    for writer in word_writers.values():
        writer.__exit__(None, None, None)
    if action_writer is not None:
        action_writer.__exit__(None, None, None)
    if whole_instruction_writer is not None:
        whole_instruction_writer.__exit__(None, None, None)

    if rendered_frames == 0:
        logger.error("No frames produced — check --obs-dump-dir / --video paths.")
        return 1

    _save_metadata(
        args.output_dir,
        instruction=instruction_first or args.instruction or "",
        spans=spans_first,  # type: ignore[arg-type]
        layers=layers,
        heads=heads,
        flow_steps=flow_steps,
        num_flow_steps=args.num_flow_steps,
        config_name=args.config,
        checkpoint=str(args.checkpoint),
        device=device_str,
        target_words=target_words,
        rendered_frames=rendered_frames,
        skipped_words=skipped_words,
        videos_written={
            "language_grounding_whole_instruction": (
                str((lang_dir / "whole_instruction.mp4").relative_to(args.output_dir))
                if whole_instruction_writer is not None else None
            ),
            "language_grounding_per_word": [
                str((lang_dir / f"{_safe_filename(w)}.mp4").relative_to(args.output_dir))
                for w in word_writers
            ],
            "action_attention": (
                str((action_dir / "action.mp4").relative_to(args.output_dir))
                if action_writer is not None else None
            ),
        },
    )
    logger.info("Done. %d frames rendered. Outputs at %s", rendered_frames, args.output_dir.resolve())
    return 0


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
    spans: _au.TokenSpans,
    layers: list[int] | None,
    heads: list[int] | None,
    flow_steps: list[int] | None,
    num_flow_steps: int,
    config_name: str,
    checkpoint: str,
    device: str,
    target_words: list[str],
    rendered_frames: int,
    skipped_words: dict[str, int],
    videos_written: dict[str, Any],
) -> None:
    metadata = {
        "instruction_first_frame": instruction,
        "config_name": config_name,
        "checkpoint": checkpoint,
        "device": device,
        "image_spans": [
            {"name": s.name, "start": s.start, "end": s.end, "grid_h": s.grid_h, "grid_w": s.grid_w}
            for s in spans.image_spans
        ],
        "selected_layers": layers,
        "selected_heads": heads,
        "selected_flow_steps": flow_steps,
        "num_flow_steps": num_flow_steps,
        "target_words": target_words,
        "rendered_frames": rendered_frames,
        "skipped_words_frames_per_word": skipped_words,
        "videos": videos_written,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def _safe_filename(word: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in word).strip("_") or "word"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
