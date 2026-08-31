# Metric definitions adapted from THUDM/LongBench (MIT License).
# Copyright (c) 2023 THU-KEG & Zhipu AI. See THIRD_PARTY_LICENSES.md.

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(text.split())


def normalize_zh_answer(text: str) -> str:
    cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    punctuation = set(string.punctuation + cn_punctuation)
    return "".join(ch for ch in text.lower() if ch not in punctuation and not ch.isspace())


def f1_score(prediction_tokens: list[str], ground_truth_tokens: list[str], **_: object) -> float:
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / max(len(prediction_tokens), 1)
    recall = num_same / max(len(ground_truth_tokens), 1)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str, **kwargs: object) -> float:
    return f1_score(normalize_answer(prediction).split(), normalize_answer(ground_truth).split(), **kwargs)


def qa_f1_zh_score(prediction: str, ground_truth: str, **kwargs: object) -> float:
    try:
        import jieba

        prediction_tokens = [normalize_zh_answer(token) for token in jieba.cut(prediction, cut_all=False)]
        ground_truth_tokens = [normalize_zh_answer(token) for token in jieba.cut(ground_truth, cut_all=False)]
    except ImportError:
        prediction_tokens = list(normalize_zh_answer(prediction))
        ground_truth_tokens = list(normalize_zh_answer(ground_truth))
    prediction_tokens = [token for token in prediction_tokens if token]
    ground_truth_tokens = [token for token in ground_truth_tokens if token]
    return f1_score(prediction_tokens, ground_truth_tokens, **kwargs)


def rouge_score(prediction: str, ground_truth: str, **_: object) -> float:
    try:
        from rouge import Rouge

        return Rouge().get_scores([prediction], [ground_truth], avg=True)["rouge-l"]["f"]
    except Exception:
        return 0.0


def rouge_zh_score(prediction: str, ground_truth: str, **kwargs: object) -> float:
    try:
        import jieba

        prediction = " ".join(jieba.cut(prediction, cut_all=False))
        ground_truth = " ".join(jieba.cut(ground_truth, cut_all=False))
    except ImportError:
        pass
    return rouge_score(prediction, ground_truth, **kwargs)


def classification_score(prediction: str, ground_truth: str, **kwargs: object) -> float:
    all_classes = kwargs.get("all_classes") or []
    matched = [class_name for class_name in all_classes if class_name in prediction]
    matched = [class_name for class_name in matched if class_name == ground_truth or class_name not in ground_truth]
    if ground_truth not in matched:
        return 0.0
    return 1.0 / max(len(matched), 1)


def retrieval_score(prediction: str, ground_truth: str, **_: object) -> float:
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    if not matches:
        return 0.0
    gold = matches[0]
    numbers = re.findall(r"\d+", prediction)
    return 0.0 if not numbers else sum(number == gold for number in numbers) / len(numbers)


def retrieval_zh_score(prediction: str, ground_truth: str, **_: object) -> float:
    matches = re.findall(r"段落(\d+)", ground_truth)
    if not matches:
        return 0.0
    gold = matches[0]
    numbers = re.findall(r"\d+", prediction)
    return 0.0 if not numbers else sum(number == gold for number in numbers) / len(numbers)


def count_score(prediction: str, ground_truth: str, **_: object) -> float:
    numbers = re.findall(r"\d+", prediction)
    return 0.0 if not numbers else sum(number == str(ground_truth) for number in numbers) / len(numbers)


def code_sim_score(prediction: str, ground_truth: str, **_: object) -> float:
    for line in prediction.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            candidate = line
            break
    else:
        candidate = prediction
    try:
        from fuzzywuzzy import fuzz

        return fuzz.ratio(candidate, ground_truth) / 100.0
    except ImportError:
        return 1.0 if candidate.strip() == ground_truth.strip() else 0.0
