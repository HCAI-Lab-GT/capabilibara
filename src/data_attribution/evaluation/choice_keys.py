"""Canonical answer keys for fixed-choice evaluation rows."""

from __future__ import annotations


def _choice_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError("choice_count must be an integer of at least 2")
    return value


def choice_keys(choice_count: int) -> tuple[str, ...]:
    """Return deterministic answer keys without extending letters past ``Z``."""

    count = _choice_count(choice_count)
    if count <= 26:
        return tuple(chr(ord("A") + index) for index in range(count))
    width = max(3, len(str(count - 1)))
    return tuple(f"choice_{index:0{width}d}" for index in range(count))


def choice_key(index: int, choice_count: int) -> str:
    """Return the canonical answer key at ``index`` for one choice set."""

    count = _choice_count(choice_count)
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError("choice index must be an integer")
    if index < 0 or index >= count:
        raise ValueError("choice index must be within the choice set")
    return choice_keys(count)[index]
