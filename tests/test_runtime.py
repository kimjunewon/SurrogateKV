from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import torch

from surrogatekv import SURROGATEKV_METHOD_TO_MODE, SurKVCluster
from surrogatekv.registry import MODE_TO_SPEC
from surrogatekv.schedule import method_capacity_profile, surkv_method_family


class RegistryTests(unittest.TestCase):
    def test_public_aliases(self) -> None:
        expected = {
            "surrogatekv": "surrogate_kv",
            "surrogatekv-snap": "surrogate_kv",
            "surrogatekv-ada": "surrogate_kv_ada",
            "surrogatekv-dynamic": "surrogate_kv_dynamic_layer",
        }
        for alias, mode in expected.items():
            self.assertEqual(SURROGATEKV_METHOD_TO_MODE[alias], mode)

    def test_registered_modes_are_unique(self) -> None:
        names = [spec.name for spec in MODE_TO_SPEC.values()]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(
            set(MODE_TO_SPEC),
            {"surrogate_kv", "surrogate_kv_ada", "surrogate_kv_dynamic_layer"},
        )
        for mode, spec in MODE_TO_SPEC.items():
            self.assertEqual(mode, spec.mode)


class RuntimeSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.key_states = torch.randn(1, 2, 32, 8)
        self.query_states = torch.randn(1, 2, 32, 8)
        self.value_states = torch.randn(1, 2, 32, 8)

    def test_shared_token_variants_respect_capacity(self) -> None:
        modes = ("surrogate_kv", "surrogate_kv_dynamic_layer")
        for mode in modes:
            with self.subTest(mode=mode):
                cluster = SurKVCluster(
                    mode=mode,
                    window_size=8,
                    max_capacity_prompt=16,
                    kernel_size=3,
                    chunk_size=4,
                )
                compressed_k, compressed_v = cluster.update_kv(
                    self.key_states,
                    self.query_states,
                    self.value_states,
                    attention_mask=None,
                    num_key_value_groups=1,
                )
                self.assertEqual(compressed_k.shape, compressed_v.shape)
                self.assertEqual(compressed_k.shape[-2], 16)

    def test_ada_rejects_shared_token_entry_point(self) -> None:
        cluster = SurKVCluster(
            mode="surrogate_kv_ada",
            window_size=8,
            max_capacity_prompt=16,
            kernel_size=3,
            chunk_size=4,
        )
        with self.assertRaisesRegex(RuntimeError, "update_kv_headwise"):
            cluster.update_kv(
                self.key_states,
                self.query_states,
                self.value_states,
                attention_mask=None,
                num_key_value_groups=1,
            )

    def test_ada_headwise_entry_point_preserves_budget(self) -> None:
        for prototype_mode in ("peak", "mean"):
            with (
                self.subTest(prototype_mode=prototype_mode),
                patch.dict(
                    os.environ,
                    {"SURKV_HEADWISE_SURROGATE_PROTO": prototype_mode},
                ),
            ):
                cluster = SurKVCluster(
                    mode="surrogate_kv_ada",
                    window_size=8,
                    max_capacity_prompt=16,
                    kernel_size=3,
                    chunk_size=4,
                )
                compressed_k, compressed_v = cluster.update_kv_headwise(
                    self.key_states,
                    self.query_states,
                    self.value_states,
                    attention_mask=None,
                    num_key_value_groups=1,
                )
                self.assertEqual(compressed_k.shape, compressed_v.shape)
                self.assertEqual(compressed_k.ndim, 2)
                self.assertEqual(compressed_k.shape[-1], 8)
                self.assertLessEqual(compressed_k.shape[0], 2 * 16)
                self.assertEqual(cluster.last_stats["surrogate_kv_headwise_budget_preserved"], 1)


class ScheduleTests(unittest.TestCase):
    def test_method_families(self) -> None:
        self.assertEqual(surkv_method_family("SurrogateKV-Snap"), "snap")
        self.assertEqual(surkv_method_family("SurrogateKV-Ada"), "ada")
        self.assertEqual(surkv_method_family("SurrogateKV-Dynamic"), "dynamic")

    def test_profile_has_one_entry_per_layer(self) -> None:
        profile = method_capacity_profile(
            method="SurrogateKV-Snap",
            num_layers=4,
            prompt_tokens=1024,
            base_capacity=128,
            scheduler="uniform",
            keep_high=-1.0,
            keep_mid=-1.0,
            keep_low=-1.0,
            r_max=-1.0,
            r_min=-1.0,
            hparam_profile="niah",
        )
        self.assertEqual(profile["capacities"], [128] * 4)
        self.assertEqual(profile["window_sizes"], [32] * 4)


if __name__ == "__main__":
    unittest.main()
