from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


def _aggregate_attention_payload(
    *,
    model: Any,
    observation: _model.Observation,
    captured: dict[str, Any],
    layers: Sequence[int] | None,
    heads: Sequence[int] | None,
    flow_steps: Sequence[int] | None,
) -> dict[str, Any]:
    """Reduce raw attention tensors into a small wire-friendly payload.

    Returns one (grid_h, grid_w) numpy array per camera per mode plus the token
    spans needed to redraw the same heatmaps offline. The full per-layer / per-head
    tensors stay on the server — only the aggregated grids and per-modality
    ratios cross the websocket.
    """
    # Imported lazily so the JAX-only path never pays the import cost.
    from openpi.models_pytorch import attention_utils as _au

    spans = _au.compute_token_spans(model, observation)

    # Drop cameras whose ``image_mask`` is False — those slots are filled with
    # zeros by the data transform (e.g. the right_wrist_0_rgb slot when only an
    # exterior + left_wrist feed is available) and any heatmap drawn on them
    # would just be the model's response to an all-zero image, which is
    # misleading. We still keep the offset cursor right because token-grid math
    # in the heatmap helpers is keyed on absolute prefix indices in ``spans``.
    valid_image_spans = []
    for s in spans.image_spans:
        mask_val = None
        if observation.image_masks is not None:
            m = observation.image_masks.get(s.name)
            if m is not None:
                if hasattr(m, "detach"):
                    m = m.detach().cpu()
                m_arr = np.asarray(m).reshape(-1)
                mask_val = bool(m_arr[0]) if m_arr.size else None
        # Default-include when no mask information is available — preserves the
        # behaviour of older callers that don't populate image_masks at all.
        if mask_val is False:
            continue
        valid_image_spans.append(s)

    image_spans_meta = [
        {
            "name": s.name,
            "start": int(s.start),
            "end": int(s.end),
            "grid_h": int(s.grid_h),
            "grid_w": int(s.grid_w),
        }
        for s in valid_image_spans
    ]

    whole_instruction: dict[str, np.ndarray] = {}
    action: dict[str, np.ndarray] = {}
    # Note: ``spans`` (passed below) is the FULL spans object from
    # compute_token_spans — the heatmap helpers use it to know absolute prefix
    # token offsets, which include the masked-out cameras we filtered above.
    # We're only suppressing the *output* heatmaps for those cameras, not
    # changing the model's input or the token layout.
    for cam_span in valid_image_spans:
        whole_instruction[cam_span.name] = _au.whole_instruction_to_image_heatmap(
            prefix_attentions=captured["prefix_attentions"],
            spans=spans,
            image_span=cam_span,
            layers=layers,
            heads=heads,
        ).astype(np.float32)
        action[cam_span.name] = _au.action_to_image_heatmap(
            suffix_attentions_per_step=captured["suffix_attentions_per_step"],
            spans=spans,
            image_span=cam_span,
            flow_steps=flow_steps,
            layers=layers,
            heads=heads,
        ).astype(np.float32)

    modality_ratios = _au.compute_modality_attention_ratios(
        suffix_attentions_per_step=captured["suffix_attentions_per_step"],
        spans=spans,
        flow_steps=flow_steps,
        layers=layers,
        heads=heads,
    )

    return {
        "image_spans": image_spans_meta,
        "whole_instruction_heatmap": whole_instruction,
        "action_heatmap": action,
        "modality_ratios": {k: float(v) for k, v in modality_ratios.items()},
    }


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        attention_num_flow_steps: int = 10,
        attention_layers: Sequence[int] | None = None,
        attention_heads: Sequence[int] | None = None,
        attention_flow_steps: Sequence[int] | None = None,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

        # Live attention recording (PyTorch pi0.5 only). When enabled, infer()
        # routes through ``sample_actions_with_attention`` (which already returns
        # actions, so there is no double-inference cost) and aggregates per-camera
        # heatmaps that ride along in the response.
        self._record_attention: bool = False
        self._attention_num_flow_steps = int(attention_num_flow_steps)
        self._attention_layers = list(attention_layers) if attention_layers is not None else None
        self._attention_heads = list(attention_heads) if attention_heads is not None else None
        self._attention_flow_steps = list(attention_flow_steps) if attention_flow_steps is not None else None

    @property
    def record_attention(self) -> bool:
        """When True, ``infer`` aggregates per-camera attention heatmaps into the response."""
        return self._record_attention

    @record_attention.setter
    def record_attention(self, value: bool) -> None:
        if value and not self.attention_supported:
            logging.warning(
                "Policy.record_attention was set to True but the loaded model does not support "
                "attention capture. PyTorch pi0.5 with sample_actions_with_attention() is required. "
                "infer() will return an explanatory sentinel under the 'attention' key instead."
            )
        self._record_attention = bool(value)

    @property
    def attention_supported(self) -> bool:
        # Per-layer attention capture is currently PyTorch pi0.5 only — the JAX
        # branch's flax-nnx ToNNX bridge does not forward mutable=['intermediates'],
        # so prefix/suffix attention probabilities cannot be recovered there.
        return self._is_pytorch_model and hasattr(self._model, "sample_actions_with_attention")

    def _attention_unsupported_reason(self) -> str:
        reasons: list[str] = []
        if not self._is_pytorch_model:
            reasons.append("JAX backbone (convert checkpoint to PyTorch)")
        if not hasattr(self._model, "sample_actions_with_attention"):
            reasons.append("model has no sample_actions_with_attention() (rebuild openpi?)")
        return "; ".join(reasons) or "unknown"

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)

        # Decide whether this call must capture attention. We only take the
        # capture path if (a) the user opted in, and (b) the backend can
        # actually provide attention. Otherwise we fall back to plain
        # sample_actions and mark the attention payload as unsupported below.
        capture_attention = self._record_attention and self.attention_supported
        attention_payload: dict[str, Any] | None = None
        attention_error: str | None = None

        start_time = time.monotonic()
        if capture_attention:
            # sample_actions_with_attention already returns the same actions
            # ``sample_actions`` would have produced (with deterministic noise
            # passed through), so there is no double-inference cost. The extra
            # cost is only the per-layer attention tensors held in memory for
            # the duration of this call.
            try:
                captured = self._model.sample_actions_with_attention(
                    self._pytorch_device,
                    observation,
                    noise=sample_kwargs.get("noise"),
                    num_steps=self._attention_num_flow_steps,
                )
                actions_tensor = captured["actions"]
                attention_payload = _aggregate_attention_payload(
                    model=self._model,
                    observation=observation,
                    captured=captured,
                    layers=self._attention_layers,
                    heads=self._attention_heads,
                    flow_steps=self._attention_flow_steps,
                )
            except Exception as e:  # noqa: BLE001
                logging.exception("Attention capture failed; falling back to plain sample_actions.")
                attention_error = f"<attention capture error: {e}>"
                actions_tensor = self._sample_actions(
                    sample_rng_or_pytorch_device, observation, **sample_kwargs
                )
            outputs = {"state": inputs["state"], "actions": actions_tensor}
        else:
            outputs = {
                "state": inputs["state"],
                "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
            }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        if self._record_attention:
            if attention_payload is not None:
                outputs["attention"] = attention_payload
            elif attention_error is not None:
                outputs["attention"] = {"error": attention_error}
            else:
                outputs["attention"] = {
                    "error": f"<unsupported: {self._attention_unsupported_reason()}>"
                }
        return outputs

    def batch_infer(self, obs_list: Sequence[dict], *, noise: np.ndarray | None = None) -> list[dict]:
        """Run inference on multiple observations in one batched forward pass."""
        if len(obs_list) == 0:
            return []

        transformed = [self._input_transform(jax.tree.map(lambda x: x, o)) for o in obs_list]
        bsize = len(transformed)

        if not self._is_pytorch_model:
            inputs = jax.tree.map(
                lambda *xs: jnp.stack([jnp.asarray(x) for x in xs], axis=0),
                *transformed,
            )
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            inputs = jax.tree.map(
                lambda *xs: torch.stack(
                    [torch.from_numpy(np.array(x)) for x in xs], dim=0
                ).to(self._pytorch_device),
                *transformed,
            )
            sample_rng_or_pytorch_device = self._pytorch_device

        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            if self._is_pytorch_model:
                noise = torch.from_numpy(noise).to(self._pytorch_device)
            else:
                noise = jnp.asarray(noise)
            if noise.ndim == 2:
                # (H, D) shared across the batch.
                if self._is_pytorch_model:
                    noise = noise[None, ...].expand(bsize, -1, -1).contiguous()
                else:
                    noise = jnp.broadcast_to(noise[None, ...], (bsize, *noise.shape))
            elif noise.shape[0] != bsize:
                raise ValueError(
                    f"batch_infer got noise with batch dim {noise.shape[0]} but obs_list has {bsize} samples"
                )
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)

        start_time = time.monotonic()
        actions_tensor = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        model_time = time.monotonic() - start_time

        outputs_batched = {"state": inputs["state"], "actions": actions_tensor}
        if self._is_pytorch_model:
            outputs_batched = jax.tree.map(lambda x: np.asarray(x.detach().cpu()), outputs_batched)
        else:
            outputs_batched = jax.tree.map(lambda x: np.asarray(x), outputs_batched)

        avg_ms = model_time * 1000 / bsize
        results: list[dict] = []
        for i in range(bsize):
            per_sample = jax.tree.map(lambda x, _i=i: x[_i], outputs_batched)
            per_sample = self._output_transform(per_sample)
            per_sample["policy_timing"] = {"infer_ms": avg_ms}
            results.append(per_sample)
        return results

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
