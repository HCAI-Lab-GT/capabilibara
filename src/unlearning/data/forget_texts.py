"""Load fixed text forget sets for unlearning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TEXT_COL = "text"
SPLIT_COL = "split"
DOC_ID_COLS = ("doc_id", "id")


@dataclass(frozen=True)
class TextForgetSet:
    texts: list[str]
    doc_ids: list[str]
    rows_read: int
    rows_kept: int


@dataclass(frozen=True)
class TextTrainSet:
    forget_texts: list[str]
    retain_texts: list[str]
    forget_doc_ids: list[str]
    retain_doc_ids: list[str]
    rows_read: int


def load_text_forget_set(path: str | Path) -> TextForgetSet:
    """Read a parquet, CSV, or JSONL forget set with a required text column."""
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"forget text set not found: {source}")
    frame = _read_frame(source)
    if TEXT_COL not in frame.columns:
        raise ValueError(
            f"forget text set must contain a '{TEXT_COL}' column: {source}"
        )
    doc_id_col = next((col for col in DOC_ID_COLS if col in frame.columns), None)
    rows_read = len(frame)
    if doc_id_col is not None:
        frame = frame.drop_duplicates(subset=[doc_id_col], keep="first")
    text = frame[TEXT_COL].fillna("").astype(str)
    keep = text.str.strip() != ""
    frame = frame.loc[keep].copy()
    texts = frame[TEXT_COL].astype(str).tolist()
    doc_ids = (
        frame[doc_id_col].fillna("").astype(str).tolist()
        if doc_id_col is not None
        else []
    )
    if not texts:
        raise ValueError(f"forget text set has no non-empty texts: {source}")
    return TextForgetSet(
        texts=texts,
        doc_ids=doc_ids,
        rows_read=rows_read,
        rows_kept=len(texts),
    )


def load_text_train_set(path: str | Path) -> TextTrainSet:
    """Read a materialized forget/retain text set."""
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"train text set not found: {source}")
    frame = _read_frame(source)
    missing = {SPLIT_COL, TEXT_COL} - set(frame.columns)
    if missing:
        raise ValueError(f"train text set missing columns {sorted(missing)}: {source}")
    doc_id_col = next((col for col in DOC_ID_COLS if col in frame.columns), None)
    frame = frame.copy()
    frame[SPLIT_COL] = frame[SPLIT_COL].astype(str)
    frame[TEXT_COL] = frame[TEXT_COL].fillna("").astype(str)
    frame = frame.loc[frame[TEXT_COL].str.strip() != ""]
    forget = frame.loc[frame[SPLIT_COL] == "forget"]
    retain = frame.loc[frame[SPLIT_COL] == "retain"]
    if forget.empty or retain.empty:
        raise ValueError(
            f"train text set must contain non-empty forget and retain splits: {source}"
        )
    return TextTrainSet(
        forget_texts=forget[TEXT_COL].astype(str).tolist(),
        retain_texts=retain[TEXT_COL].astype(str).tolist(),
        forget_doc_ids=_doc_ids(forget, doc_id_col),
        retain_doc_ids=_doc_ids(retain, doc_id_col),
        rows_read=len(frame),
    )


def _doc_ids(frame: pd.DataFrame, doc_id_col: str | None) -> list[str]:
    if doc_id_col is None:
        return []
    return frame[doc_id_col].fillna("").astype(str).tolist()


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".json"}:
        return _read_json_lines(path)
    raise ValueError(f"unsupported forget text set format: {path}")


def _read_json_lines(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)
