#!/usr/bin/env python3

# Evaluation flow and dataset-to-metric mapping adapted from THUDM/LongBench
# (MIT License). Copyright (c) 2023 THU-KEG & Zhipu AI.
# See THIRD_PARTY_NOTICES.md.

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from metrics import (
    classification_score,
    code_sim_score,
    count_score,
    qa_f1_score,
    qa_f1_zh_score,
    retrieval_score,
    retrieval_zh_score,
    rouge_score,
    rouge_zh_score,
)

DATASET2METRIC = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_count": count_score,
    "passage_retrieval_en": retrieval_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


def parse_csv_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def scorer(dataset: str, rows: list[dict[str, object]]) -> float:
    if not rows:
        raise ValueError(f"No predictions found for dataset {dataset!r}.")
    metric = DATASET2METRIC[dataset]
    total = 0.0
    for row in rows:
        prediction = str(row.get("pred", ""))
        if dataset in {"trec", "triviaqa", "samsum", "lsht"}:
            prediction = prediction.lstrip("\n").split("\n")[0]
        answers = row.get("answers") or []
        if not answers:
            raise ValueError(f"Prediction row for dataset {dataset!r} has no reference answers.")
        all_classes = row.get("all_classes") or []
        total += max(metric(prediction, str(answer), all_classes=all_classes) for answer in answers)
    return round(100.0 * total / len(rows), 2)


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SurrogateKV LongBench predictions.")
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--datasets", type=str, default=",".join(DATASET2METRIC))
    parser.add_argument("--methods", type=str, default="SurrogateKV,SurrogateKV-Dynamic,SurrogateKV-Ada")
    args = parser.parse_args()

    datasets = parse_csv_list(args.datasets)
    methods = parse_csv_list(args.methods)
    unknown_datasets = sorted(set(datasets) - DATASET2METRIC.keys())
    if unknown_datasets:
        parser.error(f"unsupported dataset(s): {', '.join(unknown_datasets)}")
    result_rows = [["dataset", *datasets]]
    evaluated = 0

    for method in methods:
        method_scores = [method]
        for dataset in datasets:
            prediction_path = args.results_dir / dataset / f"{method}.json"
            if not prediction_path.exists():
                method_scores.append(-1)
                continue
            rows = load_jsonl(prediction_path)
            score = scorer(dataset, rows)
            evaluated += 1
            method_scores.append(score)
            with (prediction_path.parent / f"{method}.metrics.json").open("w", encoding="utf-8") as handle:
                json.dump({dataset: score}, handle, ensure_ascii=False, indent=2)
            print(f"dataset {dataset} method {method} score {score}")
        result_rows.append(method_scores)

    if evaluated == 0:
        raise FileNotFoundError(
            f"No prediction files found under {args.results_dir}. "
            "Pass one <model>_budget_<B> result directory."
        )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    with (args.results_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(result_rows)


if __name__ == "__main__":
    main()
