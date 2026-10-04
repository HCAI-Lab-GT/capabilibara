"""SocialIQA ATOMIC reasoning type classifier.

Classifies SocialIQA questions into one of 9 ATOMIC commonsense relation
types (Sap et al. 2019) using regex pattern matching, context-based x/o
disambiguation, and LLM-verified correction overrides.
"""

from __future__ import annotations

import re

from data_attribution.evaluation.socialiqa_patterns import (
    O_PATTERNS,
    SIQA_NAMES,
    X_PATTERNS,
)
from data_attribution.evaluation.socialiqa_verified_labels import (
    VERIFIED_CORRECTIONS,
    VERIFIED_CORRECTIONS_BY_TEXT,
)

MANUAL_OVERRIDES: dict[str, str] = {
    "socialiqa:20835": "oWant",
    "socialiqa:11721": "oReact",
    "socialiqa:12574": "oReact",
    "socialiqa:33303": "oWant",
    "socialiqa:1323": "xIntent",
    "socialiqa:26591": "xIntent",
    "socialiqa:18251": "xWant",
    "socialiqa:339": "xWant",
    "socialiqa:1190": "xWant",
    "socialiqa:27936": "xWant",
    "socialiqa:1785": "xAttr",
    "socialiqa:26842": "xEffect",
    "socialiqa:10161": "xEffect",
    "socialiqa:20785": "xEffect",
    "socialiqa:3011": "xEffect",
    "socialiqa:11755": "xEffect",
    "socialiqa:11045": "xEffect",
    "socialiqa:10009": "xEffect",
    "socialiqa:14637": "xEffect",
    "socialiqa:269": "xEffect",
    "socialiqa:21663": "xEffect",
    "socialiqa:18182": "xReact",
    "socialiqa:21653": "xReact",
    "socialiqa:1482": "oReact",
    "socialiqa:10018": "xEffect",
    "socialiqa:21257": "xEffect",
    "socialiqa:3052": "xEffect",
    "socialiqa:21595": "xEffect",
    "socialiqa:32082": "xEffect",
    "socialiqa:28540": "xWant",
    "socialiqa:24065": "xEffect",
    "socialiqa:27697": "oReact",
    "socialiqa:13147": "xWant",
    "socialiqa:253": "xEffect",
}

_X_TO_O: dict[str, str] = {
    "xReact": "oReact",
    "xWant": "oWant",
    "xEffect": "oEffect",
}


def _extract_question_sentence(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) <= 1:
        return text
    return sentences[-1]


def _extract_context_actor(context: str) -> str | None:
    first = re.split(r"(?<=[.!?])\s+", context.strip())[0]
    for token in first.split():
        name = token.strip(",.;:'\"").lower()
        if name in SIQA_NAMES:
            return name
    return None


def _extract_question_subject(question: str, label: str) -> str | None:
    patterns: list[re.Pattern[str]] = []
    if label == "xReact":
        patterns = [
            re.compile(r"\bhow (?:would|will|does|might) (\w+)\b", re.IGNORECASE),
            re.compile(r"\bwhat (?:would|will) (\w+) feel\b", re.IGNORECASE),
            re.compile(r"\bhow will (\w+) (?:react|respond)\b", re.IGNORECASE),
        ]
    elif label == "xWant":
        patterns = [
            re.compile(
                r"\bwhat (?:will|would|does) (\w+) (?:want|do)\b", re.IGNORECASE
            ),
            re.compile(r"\bwhat will (\w+) (?:likely |probably )?do\b", re.IGNORECASE),
            re.compile(r"\bwill (\w+) (?:want|do)\b", re.IGNORECASE),
        ]
    elif label == "xEffect":
        patterns = [
            re.compile(r"\bhappen(?:s|ed)? to (\w+)\b", re.IGNORECASE),
        ]
    for pat in patterns:
        m = pat.search(question)
        if m:
            return m.group(1).lower()
    return None


def classify_socialiqa_reasoning_type(
    question: str, *, query_id: str | None = None, context: str | None = None
) -> str | None:
    """Classify a SocialIQA question into an ATOMIC reasoning type.

    Classification layers:
    1. o-type regex patterns on the question sentence
    2. x-type regex patterns with context-based x/o disambiguation
    3. Manual overrides for unclassifiable questions (fallback)
    4. Verified corrections from LLM review of all 10K questions
    """
    question_sentence = _extract_question_sentence(question)

    label: str | None = None

    for lbl, pattern in O_PATTERNS:
        if pattern.search(question_sentence):
            label = lbl
            break

    if label is None:
        for lbl, pattern in X_PATTERNS:
            if pattern.search(question_sentence):
                label = lbl
                if context is not None and lbl in _X_TO_O:
                    actor = _extract_context_actor(context)
                    subject = _extract_question_subject(question_sentence, lbl)
                    if actor and subject and subject in SIQA_NAMES and subject != actor:
                        label = _X_TO_O[lbl]
                break

    if label is None and query_id and query_id in MANUAL_OVERRIDES:
        label = MANUAL_OVERRIDES[query_id]

    if query_id:
        correction = VERIFIED_CORRECTIONS.get(query_id)
        if correction is not None and label != correction:
            return correction
        text_corrections = VERIFIED_CORRECTIONS_BY_TEXT.get(query_id)
        if text_corrections:
            for substr, corr in text_corrections:
                if substr in question and label != corr:
                    return corr

    return label
