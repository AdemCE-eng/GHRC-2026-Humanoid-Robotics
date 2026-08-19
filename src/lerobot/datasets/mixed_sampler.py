#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Weighted sampler for mixing two LeRobotDataset sources at a controlled,
per-draw sampling ratio -- independent of their raw episode/frame counts.

Used for fine-tuning runs that combine a large nominal dataset with a much
smaller auxiliary dataset (e.g. a recovery-behavior dataset) without
physically duplicating either on disk, and without relying on
MultiLeRobotDataset (unsupported in this codebase; see
src/lerobot/datasets/factory.py::make_dataset).

Intended usage: wrap the two datasets in torch.utils.data.ConcatDataset([
primary_dataset, secondary_dataset]) and pass that to the DataLoader together
with the sampler returned by build_mixed_sampler().
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.lerobot.datasets.lerobot_dataset import LeRobotDataset


def episode_valid_indices(
    dataset: LeRobotDataset, drop_n_first_frames: int = 0, drop_n_last_frames: int = 0
) -> list[int]:
    """Boundary-trimmed valid frame-start indices for one dataset.

    Mirrors EpisodeAwareSampler's own per-episode trimming exactly (see
    src/lerobot/datasets/sampler.py) so that, per source dataset, no sampled
    window can read past that episode's end (or before its start).
    """
    from_indices = dataset.meta.episodes["dataset_from_index"]
    to_indices = dataset.meta.episodes["dataset_to_index"]
    episodes_to_use = dataset.episodes
    indices: list[int] = []
    for episode_idx, (start_index, end_index) in enumerate(zip(from_indices, to_indices, strict=True)):
        if episodes_to_use is None or episode_idx in episodes_to_use:
            indices.extend(range(start_index + drop_n_first_frames, end_index - drop_n_last_frames))
    return indices


@dataclass
class MixedSamplingReport:
    """Bookkeeping about a constructed mixed sampler, for logging and tests."""

    n_primary_valid: int
    n_secondary_valid: int
    primary_ratio: float
    secondary_ratio: float


class WeightedConcatSampler(torch.utils.data.Sampler[int]):
    """Draws indices, with replacement, over a torch.utils.data.ConcatDataset of
    exactly two sources, at a fixed expected sampling ratio between them --
    regardless of the two sources' relative sizes.

    `combined_indices[i]` is the valid ConcatDataset index that weight
    `weights[i]` applies to. Replacement is required: the smaller source is
    typically far too small to draw from without replacement across a
    multi-thousand-step run.
    """

    def __init__(
        self,
        combined_indices: list[int],
        weights: list[float],
        num_samples: int,
        generator: torch.Generator | None = None,
    ):
        if len(combined_indices) != len(weights):
            raise ValueError("combined_indices and weights must have the same length")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        self.combined_indices = combined_indices
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.generator = generator

    def __iter__(self):
        drawn_positions = torch.multinomial(
            self.weights, self.num_samples, replacement=True, generator=self.generator
        )
        for position in drawn_positions.tolist():
            yield self.combined_indices[position]

    def __len__(self) -> int:
        return self.num_samples


def build_mixed_sampler(
    primary_dataset: LeRobotDataset,
    secondary_dataset: LeRobotDataset,
    primary_ratio: float,
    num_samples: int,
    drop_n_last_frames: int,
    drop_n_first_frames: int = 0,
    generator: torch.Generator | None = None,
) -> tuple[WeightedConcatSampler, MixedSamplingReport]:
    """Builds a sampler over torch.utils.data.ConcatDataset([primary_dataset,
    secondary_dataset]) that draws from `primary_dataset` with expected
    probability `primary_ratio` per individual sample, and from
    `secondary_dataset` with expected probability `1 - primary_ratio` --
    regardless of the two datasets' relative episode/frame counts.

    Episode-boundary trimming (no window reads past an episode's end) is
    computed independently per source dataset before combining, exactly
    matching EpisodeAwareSampler's own single-dataset behavior.

    Args:
        primary_dataset: The first dataset passed to ConcatDataset. Its valid
            indices are NOT offset.
        secondary_dataset: The second dataset passed to ConcatDataset. Its
            valid indices are offset by len(primary_dataset), matching
            torch.utils.data.ConcatDataset's own indexing convention.
        primary_ratio: Expected fraction of draws from primary_dataset, in
            (0, 1). secondary_dataset gets 1 - primary_ratio.
        num_samples: Total number of draws the returned sampler will produce
            (typically cfg.steps * cfg.batch_size, so exactly one pass through
            the sampler covers the whole training run).
        drop_n_last_frames: Frames to drop from the end of each episode in
            BOTH datasets (typically cfg.policy.drop_n_last_frames).
        drop_n_first_frames: Frames to drop from the start of each episode in
            BOTH datasets. Default 0.
        generator: Optional torch.Generator for reproducible draws.

    Returns:
        (sampler, report) -- report.n_primary_valid / n_secondary_valid are
        the boundary-trimmed valid-index counts actually used to compute
        per-sample weights, useful for logging and tests.

    Raises:
        ValueError: if primary_ratio is not in (0, 1), or either dataset has
            zero valid (boundary-trimmed) indices.
    """
    if not (0.0 < primary_ratio < 1.0):
        raise ValueError(f"primary_ratio must be in (0, 1), got {primary_ratio}")

    primary_valid = episode_valid_indices(primary_dataset, drop_n_first_frames, drop_n_last_frames)
    secondary_valid_local = episode_valid_indices(secondary_dataset, drop_n_first_frames, drop_n_last_frames)
    if not primary_valid:
        raise ValueError("primary_dataset has zero valid (boundary-trimmed) indices")
    if not secondary_valid_local:
        raise ValueError("secondary_dataset has zero valid (boundary-trimmed) indices")

    # ConcatDataset offsets the second dataset by len(first dataset) (total
    # frame count), not by the count of boundary-trimmed valid indices.
    offset = len(primary_dataset)
    secondary_valid = [i + offset for i in secondary_valid_local]

    secondary_ratio = 1.0 - primary_ratio
    primary_weight = primary_ratio / len(primary_valid)
    secondary_weight = secondary_ratio / len(secondary_valid)

    combined_indices = primary_valid + secondary_valid
    weights = [primary_weight] * len(primary_valid) + [secondary_weight] * len(secondary_valid)

    sampler = WeightedConcatSampler(combined_indices, weights, num_samples, generator=generator)
    report = MixedSamplingReport(
        n_primary_valid=len(primary_valid),
        n_secondary_valid=len(secondary_valid),
        primary_ratio=primary_ratio,
        secondary_ratio=secondary_ratio,
    )
    return sampler, report
