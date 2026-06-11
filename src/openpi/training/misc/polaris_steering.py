"""Steerable-policy prompt sampling for the Polaris SFT (arXiv:2602.13193).

Loads the steering annotations produced by
``scripts/imitation_learning/generate_steering_annotations.py`` (polaris_real2sim)
and swaps the episode-level prompt for a randomly sampled steering command at
training time. Architecture and action contract are untouched — only the input
language changes, exactly as in the reference steerable-policies-bridge repo.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np

import openpi.transforms as _transforms

# (frame_start, frame_end_exclusive, commands)
Segment = tuple[int, int, tuple[str, ...]]

# Per-process RNG: dataloader workers are spawned, so each process re-seeds
# from OS entropy and commands are re-sampled on every dataset pass.
_rng = np.random.default_rng()


def load_steering_segments(path: str | pathlib.Path) -> dict[int, tuple[Segment, ...]]:
    payload = json.loads(pathlib.Path(path).read_text())
    table: dict[int, tuple[Segment, ...]] = {}
    for episode in payload["episodes"]:
        segments = tuple(
            (
                int(segment["frame_start"]),
                int(segment["frame_end"]),
                tuple(str(command["text"]) for command in segment["commands"]),
            )
            for segment in episode["segments"]
        )
        table[int(episode["episode_index"])] = segments
    return table


@dataclasses.dataclass(frozen=True)
class SteeringPromptSampler(_transforms.DataTransformFn):
    """Replace ``prompt`` with a sampled steering command for covered frames.

    With probability ``steer_prob`` (and only when the frame falls inside an
    annotated segment) the prompt becomes one of the segment's commands,
    sampled uniformly; otherwise the original task instruction is kept. Frames
    outside any segment always keep the task instruction.
    """

    segments: dict[int, tuple[Segment, ...]]
    steer_prob: float = 0.5

    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        episode_index = data.get("episode_index")
        frame_index = data.get("frame_index")
        if episode_index is None or frame_index is None:
            return data
        episode_segments = self.segments.get(int(episode_index))
        if not episode_segments:
            return data
        frame = int(frame_index)
        commands = next(
            (cmds for start, end, cmds in episode_segments if start <= frame < end),
            None,
        )
        if not commands or float(_rng.random()) >= self.steer_prob:
            return data
        data = dict(data)
        data["prompt"] = str(commands[int(_rng.integers(len(commands)))])
        return data
