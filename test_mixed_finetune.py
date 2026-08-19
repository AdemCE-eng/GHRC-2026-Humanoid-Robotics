"""Tests for the mixed nominal+recovery dataset sampling infrastructure
(src/lerobot/datasets/mixed_sampler.py, factory.make_mixed_secondary_dataset,
and their wiring into lerobot_train.py).

Covers the 10 pre-training checks required before any mixed-data fine-tune is
launched: legacy single-dataset path unchanged, 80/20 and 90/10 sampler
ratios, index-to-dataset mapping, feature-contract match between the two
source datasets, real batching, exact 200K normalizer preservation, exact
200K weight loading, no NaN/Inf on a real mixed batch, and one dry
forward/backward step (no checkpoint written).

Ratio-accuracy tests (2, 3) deliberately never call dataset[idx] -- they only
exercise sampler index/weight bookkeeping, so no video is decoded. Tests that
need real batches (6, 9, 10) do decode video, unavoidably.
"""

from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import load_file

from src.lerobot.configs.policies import PreTrainedConfig
from src.lerobot.configs.train import TrainPipelineConfig
from src.lerobot.datasets.factory import make_mixed_secondary_dataset, resolve_delta_timestamps
from src.lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from src.lerobot.datasets.mixed_sampler import build_mixed_sampler, episode_valid_indices
from src.lerobot.policies.factory import get_policy_class, make_pre_post_processors
from src.lerobot.processor import StateSlicerProcessorStep, load_normalizer_stats_from_pretrained, slice_stats_for_state

CHECKPOINT_200K = Path("/mnt/d/guedr/Projects/GHRC2026/checkpoints/part_sorting_act_200k")
NOMINAL_REPO = "local/Part_Sorting"
NOMINAL_ROOT = Path("datasets/Part_Sorting")
RECOVERY_REPO = "local/part_sorting_recovery_v2_pilot50"
RECOVERY_ROOT = Path("/mnt/d/guedr/Projects/GHRC2026/datasets/Part_Sorting_Recovery_V2_Pilot50")

DROP_N_LAST_FRAMES = 49  # chunk_size - 1, matching ACTConfig(chunk_size=50)

CAMERA_KEYS = (
    "observation.images.head_left",
    "observation.images.head_right",
    "observation.images.wrist_left",
    "observation.images.wrist_right",
)
ALL_NORMALIZED_FEATURES = ("observation.state", "action") + CAMERA_KEYS


def _skip_if_missing():
    if not CHECKPOINT_200K.is_dir():
        raise unittest.SkipTest(f"checkpoint not available at {CHECKPOINT_200K}")
    if not (NOMINAL_ROOT / "meta" / "info.json").is_file():
        raise unittest.SkipTest(f"nominal dataset not available at {NOMINAL_ROOT}")
    if not (RECOVERY_ROOT / "meta" / "info.json").is_file():
        raise unittest.SkipTest(f"recovery dataset not available at {RECOVERY_ROOT}")


class LegacySingleDatasetPathTests(unittest.TestCase):
    """1. Proves the default (mixed_dataset_repo_id unset) config produces no
    secondary dataset, so lerobot_train.py's dataloader-construction branch
    is untouched and behaves exactly as before this feature existed."""

    def test_default_config_has_no_mixed_dataset(self):
        cfg = TrainPipelineConfig.__new__(TrainPipelineConfig)
        cfg.mixed_dataset_repo_id = None
        cfg.mixed_dataset_root = None
        cfg.mixed_dataset_ratio = None
        self.assertIsNone(make_mixed_secondary_dataset(cfg))

    def test_ratio_without_repo_id_is_a_noop(self):
        # Setting only mixed_dataset_ratio (no repo_id) must still be a no-op --
        # repo_id is the sole gate, matching normalization_stats_pretrained_path's
        # own opt-in pattern.
        cfg = TrainPipelineConfig.__new__(TrainPipelineConfig)
        cfg.mixed_dataset_repo_id = None
        cfg.mixed_dataset_root = None
        cfg.mixed_dataset_ratio = 0.8
        self.assertIsNone(make_mixed_secondary_dataset(cfg))

    def test_repo_id_without_ratio_raises(self):
        cfg = TrainPipelineConfig.__new__(TrainPipelineConfig)
        cfg.mixed_dataset_repo_id = RECOVERY_REPO
        cfg.mixed_dataset_root = RECOVERY_ROOT
        cfg.mixed_dataset_ratio = None
        cfg.dataset = type("D", (), {"streaming": False, "image_transforms": type("T", (), {"enable": False})()})()
        with self.assertRaises(ValueError):
            make_mixed_secondary_dataset(cfg)


class MixedSamplerRatioTests(unittest.TestCase):
    """2, 3, 4. Proves the sampler's realized draw ratio matches the configured
    target regardless of the 3000:43 raw episode-count imbalance, and that
    every drawn index resolves to the correct source dataset. No video is
    decoded -- only sampler index/weight bookkeeping is exercised."""

    @classmethod
    def setUpClass(cls):
        _skip_if_missing()
        cls.nominal = LeRobotDataset(NOMINAL_REPO, root=NOMINAL_ROOT, video_backend="pyav")
        cls.recovery = LeRobotDataset(RECOVERY_REPO, root=RECOVERY_ROOT, video_backend="pyav")

    def _measure_ratio(self, primary_ratio: float, num_samples: int = 200_000):
        sampler, report = build_mixed_sampler(
            primary_dataset=self.nominal,
            secondary_dataset=self.recovery,
            primary_ratio=primary_ratio,
            num_samples=num_samples,
            drop_n_last_frames=DROP_N_LAST_FRAMES,
            generator=torch.Generator().manual_seed(1234),
        )
        primary_boundary = len(self.nominal)
        drawn = list(iter(sampler))
        self.assertEqual(len(drawn), num_samples)
        primary_count = sum(1 for idx in drawn if idx < primary_boundary)
        realized_ratio = primary_count / num_samples
        return realized_ratio, report

    def test_80_20_ratio(self):
        realized, report = self._measure_ratio(0.80)
        self.assertAlmostEqual(realized, 0.80, delta=0.01, msg=f"realized primary ratio={realized}")
        self.assertGreater(report.n_primary_valid, 1_000_000)  # sanity: nominal set is large
        self.assertGreater(report.n_secondary_valid, 10_000)  # sanity: recovery set is non-trivial

    def test_90_10_ratio(self):
        realized, _ = self._measure_ratio(0.90)
        self.assertAlmostEqual(realized, 0.90, delta=0.01, msg=f"realized primary ratio={realized}")

    def test_out_of_range_ratio_rejected(self):
        with self.assertRaises(ValueError):
            build_mixed_sampler(self.nominal, self.recovery, primary_ratio=1.5, num_samples=10, drop_n_last_frames=DROP_N_LAST_FRAMES)
        with self.assertRaises(ValueError):
            build_mixed_sampler(self.nominal, self.recovery, primary_ratio=0.0, num_samples=10, drop_n_last_frames=DROP_N_LAST_FRAMES)

    def test_indices_map_to_correct_dataset(self):
        """4. Every drawn index resolves through ConcatDataset's own offset
        convention to a genuinely valid, boundary-trimmed frame in the
        dataset it claims to come from."""
        sampler, _ = build_mixed_sampler(
            primary_dataset=self.nominal,
            secondary_dataset=self.recovery,
            primary_ratio=0.8,
            num_samples=50_000,
            drop_n_last_frames=DROP_N_LAST_FRAMES,
            generator=torch.Generator().manual_seed(7),
        )
        concat = torch.utils.data.ConcatDataset([self.nominal, self.recovery])
        primary_valid = set(episode_valid_indices(self.nominal, drop_n_last_frames=DROP_N_LAST_FRAMES))
        recovery_valid_local = set(episode_valid_indices(self.recovery, drop_n_last_frames=DROP_N_LAST_FRAMES))
        offset = len(self.nominal)

        seen_primary = seen_secondary = 0
        for combined_idx in list(iter(sampler))[:5000]:
            dataset_idx, sample_idx = self._concat_lookup(concat, combined_idx)
            if dataset_idx == 0:
                self.assertIn(combined_idx, primary_valid)
                seen_primary += 1
            else:
                self.assertEqual(dataset_idx, 1)
                self.assertIn(combined_idx - offset, recovery_valid_local)
                seen_secondary += 1
        self.assertGreater(seen_primary, 0)
        self.assertGreater(seen_secondary, 0)

    @staticmethod
    def _concat_lookup(concat: torch.utils.data.ConcatDataset, idx: int) -> tuple[int, int]:
        import bisect

        dataset_idx = bisect.bisect_right(concat.cumulative_sizes, idx)
        sample_idx = idx if dataset_idx == 0 else idx - concat.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx


class FeatureContractTests(unittest.TestCase):
    """5. Proves both source datasets present an identical feature contract to
    the DataLoader/collate machinery (same keys, dtypes, shapes)."""

    def test_features_match_between_datasets(self):
        _skip_if_missing()
        nominal_meta = LeRobotDatasetMetadata(NOMINAL_REPO, root=NOMINAL_ROOT)
        recovery_meta = LeRobotDatasetMetadata(RECOVERY_REPO, root=RECOVERY_ROOT)
        self.assertEqual(set(nominal_meta.features.keys()), set(recovery_meta.features.keys()))
        for key, nominal_feat in nominal_meta.features.items():
            recovery_feat = recovery_meta.features[key]
            self.assertEqual(nominal_feat.get("dtype"), recovery_feat.get("dtype"), msg=key)
            self.assertEqual(nominal_feat.get("shape"), recovery_feat.get("shape"), msg=key)
        self.assertEqual(nominal_meta.fps, recovery_meta.fps)
        self.assertEqual(nominal_meta.robot_type, recovery_meta.robot_type)


class MixedBatchTests(unittest.TestCase):
    """6, 9, 10. Real DataLoader batching over the mixed ConcatDataset, real
    200K-sourced normalizer, no NaN/Inf, and one dry forward/backward step.
    These necessarily decode a small number of real video frames."""

    @classmethod
    def setUpClass(cls):
        _skip_if_missing()
        # delta_timestamps must be resolved from the real policy config, exactly as
        # make_dataset()/make_mixed_secondary_dataset() do -- without it, "action"
        # comes back as a single frame instead of the chunk_size-length sequence
        # ACT's training forward() requires (this bit ACT's training forward, not
        # inference, since predict_action()/select_action() never need "action").
        policy_cfg = PreTrainedConfig.from_pretrained(str(CHECKPOINT_200K))
        nominal_meta = LeRobotDatasetMetadata(NOMINAL_REPO, root=NOMINAL_ROOT)
        recovery_meta = LeRobotDatasetMetadata(RECOVERY_REPO, root=RECOVERY_ROOT)
        delta_timestamps = resolve_delta_timestamps(policy_cfg, nominal_meta)
        cls.nominal = LeRobotDataset(
            NOMINAL_REPO, root=NOMINAL_ROOT, video_backend="pyav", delta_timestamps=delta_timestamps
        )
        cls.recovery = LeRobotDataset(
            RECOVERY_REPO, root=RECOVERY_ROOT, video_backend="pyav", delta_timestamps=delta_timestamps
        )
        cls.concat = torch.utils.data.ConcatDataset([cls.nominal, cls.recovery])
        cls.sampler, cls.report = build_mixed_sampler(
            primary_dataset=cls.nominal,
            secondary_dataset=cls.recovery,
            primary_ratio=0.5,  # heavier recovery weight so a tiny batch is likely to include both sources
            num_samples=64,
            drop_n_last_frames=DROP_N_LAST_FRAMES,
            generator=torch.Generator().manual_seed(42),
        )

    def _make_loader(self, batch_size=8):
        return torch.utils.data.DataLoader(
            self.concat, batch_size=batch_size, sampler=self.sampler, num_workers=0, drop_last=False
        )

    def test_batching_works(self):
        """6. Real batches collate cleanly from the mixed ConcatDataset."""
        loader = self._make_loader(batch_size=8)
        batch = next(iter(loader))
        self.assertEqual(batch["observation.state"].shape[0], 8)
        for key in CAMERA_KEYS:
            self.assertEqual(batch[key].shape[0], 8)

    def test_no_nan_inf_on_real_mixed_batch(self):
        """9. A real mixed batch, run through the exact 200K-sourced
        preprocessor, is entirely finite."""
        policy_cfg = PreTrainedConfig.from_pretrained(str(CHECKPOINT_200K))
        loaded_stats = load_normalizer_stats_from_pretrained(str(CHECKPOINT_200K))
        truncated_stats = slice_stats_for_state(loaded_stats, n_dims=20)
        preprocessor, _ = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=str(CHECKPOINT_200K),
            preprocessor_overrides={
                "device_processor": {"device": "cpu"},
                "normalizer_processor": {
                    "stats": truncated_stats,
                    "features": {**policy_cfg.input_features, **policy_cfg.output_features},
                    "norm_map": policy_cfg.normalization_mapping,
                },
            },
        )
        preprocessor.steps.insert(0, StateSlicerProcessorStep(n_dims=20))

        loader = self._make_loader(batch_size=8)
        batch = next(iter(loader))
        processed = preprocessor(batch)
        for key in ("observation.state", *CAMERA_KEYS):
            self.assertTrue(torch.isfinite(processed[key]).all(), msg=f"{key} has non-finite values")

    def test_one_dry_forward_backward_step(self):
        """10. One real forward+backward pass on a real mixed batch, with the
        real 200K weights and normalizer. No optimizer.step(), no checkpoint
        written."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = get_policy_class("act").from_pretrained(str(CHECKPOINT_200K))
        policy.to(device)
        policy.train()

        policy_cfg = policy.config
        loaded_stats = load_normalizer_stats_from_pretrained(str(CHECKPOINT_200K))
        truncated_stats = slice_stats_for_state(loaded_stats, n_dims=20)
        preprocessor, _ = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=str(CHECKPOINT_200K),
            preprocessor_overrides={
                "device_processor": {"device": device.type},
                "normalizer_processor": {
                    "stats": truncated_stats,
                    "features": {**policy_cfg.input_features, **policy_cfg.output_features},
                    "norm_map": policy_cfg.normalization_mapping,
                },
                "rename_observations_processor": {"rename_map": {}},
            },
        )
        preprocessor.steps.insert(0, StateSlicerProcessorStep(n_dims=20))
        preprocessor.reset()
        policy.reset()

        loader = self._make_loader(batch_size=8)
        batch = next(iter(loader))
        batch = preprocessor(batch)

        loss, _output_dict = policy.forward(batch)
        self.assertTrue(torch.isfinite(loss), msg=f"loss={loss.item()}")
        loss.backward()

        any_finite_grad = False
        for p in policy.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all(), msg="non-finite gradient found")
                any_finite_grad = True
        self.assertTrue(any_finite_grad, "no gradients were populated by backward()")
        # Deliberately no optimizer.step() and no checkpoint write.


class MixedNormalizerAndWeightTests(unittest.TestCase):
    """7, 8. Regression tests tying the mixed-training scenario explicitly to
    the two guarantees it must not break: exact 200K normalizer preservation
    (for every feature) and exact 200K weight loading."""

    @classmethod
    def setUpClass(cls):
        _skip_if_missing()

    def test_normalizer_matches_200k_for_every_feature_in_mixed_scenario(self):
        """7. Constructs the preprocessor exactly as the mixed lerobot_train.py
        code path will (load_normalizer_stats_from_pretrained + overrides),
        and checks it against the checkpoint's own saved normalizer for state,
        action, and all four image keys."""
        import glob

        matches = glob.glob(str(CHECKPOINT_200K / "policy_preprocessor*normalizer*.safetensors"))
        saved = load_file(matches[0])

        loaded = load_normalizer_stats_from_pretrained(str(CHECKPOINT_200K))
        truncated = slice_stats_for_state(loaded, n_dims=20)
        for key in ALL_NORMALIZED_FEATURES:
            for stat_name in ("mean", "std"):
                saved_val = saved[f"{key}.{stat_name}"]
                loaded_val = torch.as_tensor(truncated[key][stat_name], dtype=torch.float32)
                torch.testing.assert_close(saved_val, loaded_val, atol=1e-6, rtol=0, msg=f"{key}.{stat_name}")

    def test_weights_load_exactly_from_200k(self):
        """8. Zero missing/unexpected keys, and every tensor matches the saved
        checkpoint's model.safetensors exactly."""
        saved_weights = load_file(str(CHECKPOINT_200K / "model.safetensors"))
        policy = get_policy_class("act").from_pretrained(str(CHECKPOINT_200K))
        state_dict = policy.state_dict()

        saved_keys = set(saved_weights.keys())
        loaded_keys = set(state_dict.keys())
        missing = saved_keys - loaded_keys
        unexpected = loaded_keys - saved_keys
        self.assertEqual(missing, set(), msg=f"missing keys: {missing}")
        self.assertEqual(unexpected, set(), msg=f"unexpected keys: {unexpected}")
        for key, saved_tensor in saved_weights.items():
            torch.testing.assert_close(state_dict[key].cpu(), saved_tensor, atol=1e-6, rtol=0, msg=key)


if __name__ == "__main__":
    unittest.main()
