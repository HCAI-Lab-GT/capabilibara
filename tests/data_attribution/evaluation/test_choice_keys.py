from __future__ import annotations

import pytest

from data_attribution.evaluation.choice_keys import choice_key, choice_keys


@pytest.mark.parametrize("count", [2, 3, 4, 10, 26])
def test_choice_keys_preserve_historical_letters_through_26(count: int) -> None:
    expected = tuple(chr(ord("A") + index) for index in range(count))
    assert choice_keys(count) == expected
    assert tuple(choice_key(index, count) for index in range(count)) == expected


def test_choice_keys_use_cardinality_safe_indices_above_26() -> None:
    keys = choice_keys(77)
    assert len(keys) == 77
    assert len(set(keys)) == 77
    assert keys[0] == "choice_000"
    assert keys[25] == "choice_025"
    assert keys[26] == "choice_026"
    assert keys[76] == "choice_076"
    assert all(key.isprintable() and key.isascii() for key in keys)


@pytest.mark.parametrize("count", [True, False, -1, 0, 1, 2.5, "2"])
def test_choice_keys_reject_invalid_cardinality(count: object) -> None:
    with pytest.raises(ValueError, match="choice_count"):
        choice_keys(count)  # type: ignore[arg-type]


@pytest.mark.parametrize("index", [True, -1, 77, 2.5, "1"])
def test_choice_key_rejects_invalid_index(index: object) -> None:
    with pytest.raises(ValueError, match="index"):
        choice_key(index, 77)  # type: ignore[arg-type]
