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

    budget_lookup = {
        (row["method"], row["budget"]): float(row["average"])
        for row in budgets
    }
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

    allocations = read_rows(f"{prefix}/allocation_by_budget.csv")
    assert len(allocations) == 18
    assert {row["method"] for row in allocations} == {
        "SurrogateKV-Snap",
        "SurrogateKV-Dynamic",
        "SurrogateKV-Ada",
    }
    for row in allocations:
        coverage = sum(
            float(row[key])
            for key in ("raw_region_pct", "surrogate_coverage_pct", "drop_region_pct")
        )
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


def validate_tables() -> None:
    ablations = read_rows("ablations/longbench_b512.csv")
    ablation_lookup = {row["setting"]: row for row in ablations}
    close(float(ablation_lookup["full_method"]["longbench_score"]), 42.21)
    close(float(ablation_lookup["raw_drop_only"]["longbench_score"]), 40.44)
    assert int(ablation_lookup["c4"]["mean_surrogate_entries"]) == 397

    merging = read_rows("comparisons/merging_longbench_summary.csv")
    merging_lookup = {
        (row["method"], int(row["budget"])): float(row["average"])
        for row in merging
    }
    close(merging_lookup[("SurrogateKV-Snap", 128)], 37.99)
    close(merging_lookup[("D2O", 512)], 39.50)

    efficiency = read_rows("efficiency/llama3_8b_instruct_b128.csv")
    assert len(efficiency) == 21
    main = {
        row["method"]: row
        for row in efficiency
        if int(row["input_tokens"]) == 32768
    }
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
    for row in scaling:
        delta = float(row["surrogate_score"]) - float(row["base_score"])
        close(delta, float(row["delta"]))


def validate_no_local_paths() -> None:
    extensions = {".csv", ".md", ".toml", ".cff", ".py", ".txt"}
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
    validate_tables()
    validate_no_local_paths()
    print("Release data validation passed.")


if __name__ == "__main__":
    main()
