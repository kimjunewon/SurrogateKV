#!/usr/bin/env python3

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BUDGETS = {64, 128, 256, 512, 1024, 2048}
COMPRESSED_METHODS = {
    "H2O",
    "SnapKV",
    "PyramidKV",
    "DynamicKV",
    "Ada-KV",
    "SurrogateKV-Snap",
    "SurrogateKV-Dynamic",
    "SurrogateKV-Ada",
}
DATASET_GROUPS = {
    "single_document_qa": ("narrativeqa", "qasper", "multifieldqa_en"),
    "multi_document_qa": ("hotpotqa", "2wikimqa", "musique"),
    "summarization": ("gov_report", "qmsum", "multi_news"),
    "few_shot_learning": ("trec", "triviaqa", "samsum"),
    "synthetic": ("passage_count", "passage_retrieval_en"),
    "code": ("lcc", "repobench_p"),
}


def read_rows(relative_path: str) -> list[dict[str, str]]:
    path = DATA / relative_path
    if not path.is_file():
        raise AssertionError(f"missing data file: {path.relative_to(ROOT)}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def close(actual: float, expected: float, tolerance: float = 0.011) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"expected {expected}, found {actual}")


def validate_longbench() -> None:
    prefix = "longbench/llama3_8b_instruct"
    budgets = read_rows(f"{prefix}/budget_scores.csv")
    assert len(budgets) == len(BUDGETS) * len(COMPRESSED_METHODS)
    assert {row["method"] for row in budgets} == COMPRESSED_METHODS
    assert {int(row["budget"]) for row in budgets} == BUDGETS

    budget_lookup = {(row["method"], row["budget"]): float(row["average"]) for row in budgets}
    close(budget_lookup[("SnapKV", "512")], 40.26)
    close(budget_lookup[("SurrogateKV-Snap", "512")], 40.8819)
    close(budget_lookup[("DynamicKV", "512")], 40.6050)
    close(budget_lookup[("SurrogateKV-Dynamic", "512")], 40.8088)
    close(budget_lookup[("Ada-KV", "512")], 40.7744)
    close(budget_lookup[("SurrogateKV-Ada", "512")], 41.2613)

    per_dataset = read_rows(f"{prefix}/per_dataset_scores.csv")
    assert len(per_dataset) == 49
    assert {row["method"] for row in per_dataset} == COMPRESSED_METHODS | {"FullKV"}
    fullkv = [row for row in per_dataset if row["method"] == "FullKV"]
    assert len(fullkv) == 1
    close(float(fullkv[0]["average"]), 41.92)
    for row in per_dataset:
        if row["method"] == "FullKV":
            continue
        close(float(row["average"]), budget_lookup[(row["method"], row["budget"])])

    per_dataset_lookup = {(row["method"], row["budget"]): row for row in per_dataset}
    categories = read_rows(f"{prefix}/category_scores.csv")
    category_method_keys = {("FullKV", "")}
    category_method_keys.update(
        (method, str(budget))
        for method in ("H2O", "SnapKV", "PyramidKV", "DynamicKV", "SurrogateKV")
        for budget in BUDGETS
    )
    assert len(categories) == len(category_method_keys) * (len(DATASET_GROUPS) + 1)
    for row in categories:
        category_key = (row["method"], row["target_budget_tokens"])
        assert category_key in category_method_keys
        source_method = "SurrogateKV-Snap" if row["method"] == "SurrogateKV" else row["method"]
        source = per_dataset_lookup[(source_method, row["target_budget_tokens"])]
        if row["type_slug"] == "average":
            expected = float(source["average"])
            expected_count = sum(len(columns) for columns in DATASET_GROUPS.values())
        else:
            columns = DATASET_GROUPS[row["type_slug"]]
            expected = sum(float(source[column]) for column in columns) / len(columns)
            expected_count = len(columns)
        close(float(row["score"]), expected)
        assert int(row["num_datasets"]) == expected_count

    allocations = read_rows(f"{prefix}/allocation_by_budget.csv")
    assert len(allocations) == 18
    assert {row["method"] for row in allocations} == {
        "SurrogateKV-Snap",
        "SurrogateKV-Dynamic",
        "SurrogateKV-Ada",
    }
    for row in allocations:
        coverage = sum(float(row[key]) for key in ("raw_region_pct", "surrogate_coverage_pct", "drop_region_pct"))
        close(coverage, 100.0, tolerance=0.001)


def heatmap_mean(path: Path) -> tuple[float, int]:
    values: list[float] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            values.extend(float(value) for key, value in row.items() if key != "depth_percent")
    if not values:
        raise AssertionError(f"empty heatmap: {path.relative_to(ROOT)}")
    return sum(values) / len(values), len(values)


def validate_niah() -> None:
    root = DATA / "niah/mistral_7b_instruct_v02"
    file_names = {
        "SnapKV": "snapkv",
        "SurrogateKV-Snap": "surrogatekv-snap",
        "DynamicKV": "dynamickv",
        "SurrogateKV-Dynamic": "surrogatekv-dynamic",
        "Ada-KV": "adakv",
        "SurrogateKV-Ada": "surrogatekv-ada",
    }
    for directory in ("k64_ctx1000_32000_step200", "k128_ctx1000_32000_step200"):
        rows = read_rows(f"niah/mistral_7b_instruct_v02/{directory}/niah_average_table.csv")
        assert {row["method"] for row in rows} == set(file_names)
        for row in rows:
            heatmap = root / directory / f"niah_heatmap_{file_names[row['method']]}.csv"
            actual, count = heatmap_mean(heatmap)
            assert count == int(row["examples"]) == 1560
            close(actual, float(row["average"]))


def validate_analysis() -> None:
    motivation = read_rows("analysis/motivation_summary.csv")
    expected_counts = {
        "attention_shift": 4,
        "attention_support_gap": 4,
        "top5_attention_mass_gap_pp": 5,
        "snapkv_removed_attention": 6,
        "chunk_coherence": 4,
    }
    assert len(motivation) == sum(expected_counts.values())
    assert len({(row["section"], row["name"]) for row in motivation}) == len(motivation)
    for section, expected_count in expected_counts.items():
        assert sum(row["section"] == section for row in motivation) == expected_count

    motivation_lookup = {(row["section"], row["name"]): float(row["value"]) for row in motivation}
    top5_values = [
        value for (section, _), value in motivation_lookup.items() if section == "top5_attention_mass_gap_pp"
    ]
    close(min(top5_values), 3.3468, tolerance=1e-6)
    close(max(top5_values), 5.9626, tolerance=1e-6)
    for space in ("key", "value"):
        random_value = motivation_lookup[("chunk_coherence", f"Random dropped tokens:{space}_mean_cosine")]
        local_value = motivation_lookup[
            ("chunk_coherence", f"SnapKV-dropped local region:{space}_mean_cosine")
        ]
        assert local_value > random_value

    attention = read_rows("attention/k64_workload_summary.csv")
    methods = {
        "SnapKV",
        "SurrogateKV-Snap",
        "DynamicKV",
        "SurrogateKV-Dynamic",
        "Ada-KV",
        "SurrogateKV-Ada",
    }
    workloads = {"Single QA", "Multi QA", "Summ.", "Few-shot", "Synthetic", "Code"}
    base_metrics = {"entropy_shift", "top5_shift", "top1_shift", "support_rel_shift"}
    extra_metrics = {"js_to_fullkv", "l1_to_fullkv"}
    assert len(attention) == 180
    assert {row["budget"] for row in attention} == {"k64"}
    assert {row["method"] for row in attention} == methods
    assert {row["workload"] for row in attention} == workloads
    assert len({(row["workload"], row["method"], row["metric"]) for row in attention}) == len(attention)
    for row in attention:
        expected_metrics = (
            base_metrics | extra_metrics
            if row["method"].startswith("SurrogateKV")
            else base_metrics
        )
        assert row["metric"] in expected_metrics
        p25 = float(row["p25"])
        median = float(row["median"])
        p75 = float(row["p75"])
        assert 0.0 <= p25 <= median <= p75
        assert float(row["mean"]) >= 0.0
        assert int(row["n"]) > 0


def validate_tables() -> None:
    ablations = read_rows("ablations/longbench_b512.csv")
    ablation_lookup = {row["setting"]: row for row in ablations}
    assert set(ablation_lookup) == {
        "full_method",
        "raw_drop_only",
        "fixed_quota_0.25",
        "fixed_quota_0.50",
        "fixed_quota_0.75",
        "mean_only",
        "attention_weighted_mean",
        "salience_pivot",
        "cosine_medoid",
        "c2",
        "c4",
        "c8",
        "c16",
    }
    assert {int(row["workloads"]) for row in ablations} == {16}
    assert {int(row["examples_per_workload"]) for row in ablations} == {20}
    close(float(ablation_lookup["full_method"]["longbench_score"]), 42.21)
    close(float(ablation_lookup["raw_drop_only"]["longbench_score"]), 40.44)
    assert int(ablation_lookup["c4"]["mean_surrogate_entries"]) == 397

    merging = read_rows("comparisons/merging_longbench_summary.csv")
    merging_lookup = {(row["method"], int(row["budget"])): float(row["average"]) for row in merging}
    close(merging_lookup[("SurrogateKV-Snap", 128)], 37.99)
    close(merging_lookup[("D2O", 512)], 39.50)
    merging_per_dataset = read_rows("comparisons/merging_longbench_per_dataset.csv")
    for row in merging_per_dataset:
        close(float(row["average"]), merging_lookup[(row["method"], int(row["budget"]))])

    efficiency = read_rows("efficiency/llama3_8b_instruct_b128.csv")
    assert len(efficiency) == 21
    efficiency_methods = {
        "FullKV",
        "SnapKV",
        "SurrogateKV-Snap",
        "DynamicKV",
        "SurrogateKV-Dynamic",
        "Ada-KV",
        "SurrogateKV-Ada",
    }
    assert {(int(row["input_tokens"]), row["method"]) for row in efficiency} == {
        (input_tokens, method)
        for input_tokens in (8192, 16384, 32768)
        for method in efficiency_methods
    }
    for row in efficiency:
        assert int(row["output_tokens"]) == int(row["input_tokens"]) // 4
        assert all(
            float(row[key]) > 0.0
            for key in (
                "ttft_s",
                "decode_tokens_per_s",
                "end_to_end_latency_s",
                "prefill_resident_kv_gb",
                "decode_resident_kv_gb",
            )
        )
    main = {row["method"]: row for row in efficiency if int(row["input_tokens"]) == 32768}
    close(float(main["SurrogateKV-Ada"]["ttft_s"]), 4.44)
    for base, variant in (
        ("SnapKV", "SurrogateKV-Snap"),
        ("DynamicKV", "SurrogateKV-Dynamic"),
        ("Ada-KV", "SurrogateKV-Ada"),
    ):
        assert main[base]["prefill_resident_kv_gb"] == main[variant]["prefill_resident_kv_gb"]
        assert main[base]["decode_resident_kv_gb"] == main[variant]["decode_resident_kv_gb"]

    scaling = read_rows("scaling/qwen25_longbench.csv")
    assert len(scaling) == 4
    scaling_lookup = {(row["model"], row["budget"]): row for row in scaling}
    expected_scaling = {
        "Qwen2.5-14B": (6, 20),
        "Qwen2.5-72B": (4, 10),
    }
    expected_scaling_tasks = {
        "Qwen2.5-14B": {
            "narrativeqa",
            "qasper",
            "hotpotqa",
            "gov_report",
            "passage_retrieval_en",
            "lcc",
        },
        "Qwen2.5-72B": {"qasper", "hotpotqa", "passage_retrieval_en", "lcc"},
    }
    for row in scaling:
        expected_tasks, expected_examples = expected_scaling[row["model"]]
        assert int(row["tasks"]) == expected_tasks
        assert int(row["examples_per_task"]) == expected_examples
        delta = float(row["surrogate_score"]) - float(row["base_score"])
        close(delta, float(row["delta"]))

    scaling_tasks = read_rows("scaling/qwen25_longbench_per_task.csv")
    assert len(scaling_tasks) == 20
    task_groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in scaling_tasks:
        assert row["base_method"] == "SnapKV"
        assert row["surrogate_method"] == "SurrogateKV-Snap"
        close(
            float(row["surrogate_score"]) - float(row["base_score"]),
            float(row["delta"]),
            tolerance=1e-5,
        )
        task_groups.setdefault((row["model"], row["budget"]), []).append(row)

    assert set(task_groups) == set(scaling_lookup)
    for key, rows in task_groups.items():
        summary = scaling_lookup[key]
        assert len(rows) == int(summary["tasks"])
        assert {row["dataset"] for row in rows} == expected_scaling_tasks[key[0]]
        assert {int(row["examples"]) for row in rows} == {int(summary["examples_per_task"])}
        close(sum(float(row["base_score"]) for row in rows) / len(rows), float(summary["base_score"]))
        close(sum(float(row["surrogate_score"]) for row in rows) / len(rows), float(summary["surrogate_score"]))


def validate_figure_assets() -> None:
    required = (
        "images/longbench_budget_sweep.pdf",
        "images/longbench_budget_sweep.svg",
        "images/longbench_budget_sweep-dark.svg",
        "images/mistral_niah_k128_method_comparison.pdf",
        "images/mistral_niah_k128_method_comparison.svg",
        "images/mistral_niah_k128_method_comparison-dark.svg",
        "images/surrogatekv_overview.pdf",
        "images/surrogatekv_overview.svg",
        "images/surrogatekv_overview-dark.svg",
        "images/source/surrogatekv_overview.drawio",
    )
    for relative_path in required:
        path = DATA / relative_path
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"missing or empty figure asset: {path.relative_to(ROOT)}")


def validate_no_local_paths() -> None:
    extensions = {".cff", ".csv", ".drawio", ".md", ".py", ".svg", ".toml", ".txt", ".yaml", ".yml"}
    forbidden = ("/home/", "/mnt/", "SurKV-camera-ready/artifacts")
    for path in ROOT.rglob("*"):
        if (
            ".git" in path.parts
            or path == Path(__file__).resolve()
            or not path.is_file()
            or path.suffix not in extensions
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for marker in forbidden:
            if marker in text:
                raise AssertionError(f"local path marker {marker!r} in {path.relative_to(ROOT)}")


def main() -> None:
    validate_longbench()
    validate_niah()
    validate_analysis()
    validate_tables()
    validate_figure_assets()
    validate_no_local_paths()
    print("Release data validation passed.")


if __name__ == "__main__":
    main()
