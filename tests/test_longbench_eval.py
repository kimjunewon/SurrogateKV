from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = REPO_ROOT / "run" / "longbench" / "eval.py"


class LongBenchEvaluatorTests(unittest.TestCase):
    def test_writes_method_specific_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = Path(tmpdir)
            dataset_dir = result_dir / "qasper"
            dataset_dir.mkdir()
            prediction = {
                "pred": "the answer",
                "answers": ["answer"],
                "all_classes": [],
            }
            (dataset_dir / "SurrogateKV.json").write_text(json.dumps(prediction) + "\n", encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(EVALUATOR),
                    "--results_dir",
                    str(result_dir),
                    "--datasets",
                    "qasper",
                    "--methods",
                    "SurrogateKV",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            with (result_dir / "results.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows, [["dataset", "qasper"], ["SurrogateKV", "100.0"]])
            self.assertTrue((dataset_dir / "SurrogateKV.metrics.json").is_file())
            self.assertFalse((dataset_dir / "metrics.json").exists())

    def test_rejects_a_result_root_without_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(EVALUATOR),
                    "--results_dir",
                    tmpdir,
                    "--datasets",
                    "qasper",
                    "--methods",
                    "SurrogateKV",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("No prediction files found", completed.stderr)


if __name__ == "__main__":
    unittest.main()
