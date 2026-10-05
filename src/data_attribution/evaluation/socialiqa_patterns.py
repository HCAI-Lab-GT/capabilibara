"""Regex patterns for SocialIQA ATOMIC reasoning type classification.

o-type patterns detect questions about others (bystanders, recipients).
x-type patterns detect questions about the actor (subject of the event).
"""

from __future__ import annotations

import re

_RELATION_NOUNS = (
    r"brother|sister|mother|father|mom|dad|mum|parents|family|husband|wife|"
    r"boss|friend|neighbor|children|son|daughter|girlfriend|boyfriend|"
    r"teacher|nurse|doctor|coach|waitress|landlord|bully|director|"
    r"uncle|aunt|grandmother|grandfather|partner|person"
)
_GROUP_NOUNS = (
    r"friends|students|coworkers|colleagues|classmates|teammates|"
    r"neighbors|cops|police|spectators|pedestrians|customers|employees|"
    r"bystanders|opponents|siblings|managers|tutors|girls|women|kids|"
    r"children|parents|members|congregation|audience|family"
)

O_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("oReact", re.compile(r"\bothers?\b.*\bfeel\b", re.IGNORECASE)),
    ("oReact", re.compile(r"\bfeel\b.*\bothers?\b", re.IGNORECASE)),
    (
        "oReact",
        re.compile(
            r"\bothers?\b.*\b(?:think|say|observe|respond|react)\b", re.IGNORECASE
        ),
    ),
    ("oReact", re.compile(r"\bmake others feel\b", re.IGNORECASE)),
    ("oWant", re.compile(r"\bothers?\b.*\b(?:want|do|going to do)\b", re.IGNORECASE)),
    (
        "oEffect",
        re.compile(r"\b(?:happen|result|effect)\b.*\bothers?\b", re.IGNORECASE),
    ),
    ("oReact", re.compile(r"\b(?:someone|everyone)\b.*\bfeel\b", re.IGNORECASE)),
    ("oWant", re.compile(r"\b(?:someone|everyone)\b.*\b(?:want|do)\b", re.IGNORECASE)),
    (
        "oReact",
        re.compile(
            rf"\b(?:his|her|their|the)\s+(?:{_RELATION_NOUNS})\b.*\bfeel\b",
            re.IGNORECASE,
        ),
    ),
    (
        "oReact",
        re.compile(rf"\b\w+'s\s+(?:{_RELATION_NOUNS})\b.*\bfeel\b", re.IGNORECASE),
    ),
    (
        "oWant",
        re.compile(
            rf"\b(?:his|her|their|the)\s+(?:{_RELATION_NOUNS})\b.*\b(?:want|do)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "oWant",
        re.compile(
            rf"\b\w+'s\s+(?:{_RELATION_NOUNS})\b.*\b(?:want|do)\b", re.IGNORECASE
        ),
    ),
    (
        "oEffect",
        re.compile(
            rf"\b(?:happen|result|effect)\b.*\b(?:his|her|their|the)\s+"
            rf"(?:{_RELATION_NOUNS})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "oEffect",
        re.compile(
            rf"\b(?:happen|result|effect)\b.*\b\w+'s\s+(?:{_RELATION_NOUNS})\b",
            re.IGNORECASE,
        ),
    ),
    ("oReact", re.compile(rf"\b(?:{_GROUP_NOUNS})\b.*\bfeel\b", re.IGNORECASE)),
    (
        "oWant",
        re.compile(rf"\b(?:{_GROUP_NOUNS})\b.*\b(?:want|do)\b", re.IGNORECASE),
    ),
    (
        "oEffect",
        re.compile(
            rf"\b(?:happen|result|effect)\b.*\b(?:{_GROUP_NOUNS})\b", re.IGNORECASE
        ),
    ),
]

X_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("xReact", re.compile(r"\bhow\b.*\bfeel\b", re.IGNORECASE)),
    ("xReact", re.compile(r"\bwhat\b.*\b\w+\b\s+feel\b", re.IGNORECASE)),
    ("xReact", re.compile(r"\bfeel\s+(?:how|what)\b", re.IGNORECASE)),
    ("xReact", re.compile(r"\bfeeling\s+afterwards\b", re.IGNORECASE)),
    ("xReact", re.compile(r"\bhow will \w+ react\b", re.IGNORECASE)),
    ("xReact", re.compile(r"\bhow will \w+ respond\b", re.IGNORECASE)),
    ("xAttr", re.compile(r"\bhow\b.*\b(?:describ\w*|seen|perceived)\b", re.IGNORECASE)),
    ("xAttr", re.compile(r"\bcould be described\b", re.IGNORECASE)),
    ("xAttr", re.compile(r"\bwhat kind of person\b", re.IGNORECASE)),
    ("xAttr", re.compile(r"\bis \w+ \w+ or \w+\?", re.IGNORECASE)),
    ("xIntent", re.compile(r"\bwhy\b.*\b(?:did|do|does|would|is)\b", re.IGNORECASE)),
    ("xIntent", re.compile(r"\bwhat (?:made|led|caused)\b", re.IGNORECASE)),
    ("xIntent", re.compile(r"\bbecause\??\s*$", re.IGNORECASE)),
    ("xWant", re.compile(r"\bwhat\b.*\bwant\b", re.IGNORECASE)),
    ("xNeed", re.compile(r"\bwhat\b.*\bneed\b.*\bbefore\b", re.IGNORECASE)),
    (
        "xNeed",
        re.compile(r"\bbefore\b.*\bwhat\b.*\b(?:did|do|have to)\b", re.IGNORECASE),
    ),
    ("xNeed", re.compile(r"\bwhat\b.*\bbefore\b", re.IGNORECASE)),
    ("xNeed", re.compile(r"\bshould\b.*\bbefore\b", re.IGNORECASE)),
    ("xWant", re.compile(r"\bwhat will \w+ do (?:next|after)\b", re.IGNORECASE)),
    (
        "xWant",
        re.compile(
            r"\bwhat (?:will|would|does|is|are|should|may|can)\b.*\bdo\b", re.IGNORECASE
        ),
    ),
    ("xWant", re.compile(r"\bcourse of action\b", re.IGNORECASE)),
    ("xWant", re.compile(r"\bgoing to do\b", re.IGNORECASE)),
    ("xWant", re.compile(r"\blikely to do\b", re.IGNORECASE)),
    ("xEffect", re.compile(r"\bwhat\b.*\b(?:happen|result|effect)\b", re.IGNORECASE)),
    ("xEffect", re.compile(r"\bwhat did \w+ (?:do|say|create)\b", re.IGNORECASE)),
]

SIQA_NAMES: frozenset[str] = frozenset(
    {
        "addison",
        "alex",
        "ash",
        "aubrey",
        "austin",
        "avery",
        "bailey",
        "blake",
        "bobby",
        "brett",
        "cameron",
        "carson",
        "casey",
        "charlie",
        "chris",
        "danny",
        "drew",
        "dylan",
        "emerson",
        "evan",
        "finley",
        "fred",
        "hayden",
        "hunter",
        "jan",
        "jason",
        "jayden",
        "jesse",
        "jill",
        "jordan",
        "jules",
        "kai",
        "kendall",
        "lee",
        "logan",
        "lynn",
        "morgan",
        "nico",
        "peyton",
        "quinn",
        "remy",
        "riley",
        "robin",
        "sam",
        "sasha",
        "skylar",
        "sydney",
        "taylor",
        "terona",
        "tim",
        "tracy",
    }
)
