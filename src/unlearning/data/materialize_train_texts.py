"""Materialize unlearning train text sets from a TSV manifest."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import pandas as pd

from unlearning.data.forget_texts import load_text_forget_set
from unlearning.data.materialize_manifest import Row
from unlearning.data.sampling import DOC_ID_COL, TEXT_COL, TOPIC_COL, filter_num_proc
from unlearning.data.sampling import sample_forget

LOG = logging.getLogger(__name__)


def normalize_topics(value: str) -> tuple[bool, list[str]]:
    if value in {"", "null", "None"}:
        return True, []
    return False, [part.strip() for part in value.split(",") if part.strip()]


def load_dataset(cache: Path):
    from datasets import disable_caching, load_from_disk

    disable_caching()
    LOG.info("Loading corpus cache from %s", cache)
    return load_from_disk(str(cache))


def sample_retain(
    ds,
    topics: list[str],
    max_docs: int,
    rng: random.Random,
    exclude_doc_ids: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    topics_set = set(topics)
    retain_ds = _filter_retain_by_topic(ds, topics_set)
    if exclude_doc_ids and DOC_ID_COL in retain_ds.column_names:
        retain_ds = retain_ds.filter(
            lambda x, banned=exclude_doc_ids: str(x[DOC_ID_COL]) not in banned,
            num_proc=filter_num_proc(),
            desc="filter:retain:exclude_forget_doc_ids",
        )
    indices = list(range(len(retain_ds)))
    rng.shuffle(indices)
    selected = retain_ds.select(indices[:max_docs])
    texts = [text for text in selected[TEXT_COL] if text and str(text).strip()]
    doc_ids = (
        [str(doc_id) for doc_id in selected[DOC_ID_COL]]
        if DOC_ID_COL in selected.column_names
        else []
    )
    return texts, doc_ids


def _filter_retain_by_topic(ds, topics_set: set[str]):
    if not topics_set:
        return ds
    return ds.filter(
        lambda x, s=topics_set: x[TOPIC_COL] not in s,
        num_proc=filter_num_proc(),
        desc="filter:retain",
    )


def materialize_row(
    row: Row,
    ds,
    out_dir: Path,
    max_forget_docs: int,
    max_retain_docs: int,
    overwrite: bool,
) -> dict[str, object]:
    out_path = out_dir / f"{row.label}.parquet"
    if out_path.exists() and not overwrite:
        LOG.info("Skipping existing %s", out_path)
        return {"label": row.label, "path": str(out_path), "status": "exists"}
    is_null, topics = normalize_topics(row.topic_bin)
    rng = random.Random(row.seed)
    forget_texts, forget_doc_ids = _forget_texts(
        row, ds, topics, is_null, rng, max_forget_docs
    )
    retain_texts, retain_doc_ids = sample_retain(
        ds,
        [] if is_null else topics,
        max_retain_docs,
        rng,
        exclude_doc_ids=set(forget_doc_ids),
    )
    _write_train_frame(
        out_path, forget_texts, forget_doc_ids, retain_texts, retain_doc_ids
    )
    LOG.info(
        "%s: wrote %d forget + %d retain rows to %s",
        row.label,
        len(forget_texts),
        len(retain_texts),
        out_path,
    )
    return {
        "label": row.label,
        "path": str(out_path),
        "status": "written",
        "forget_rows": len(forget_texts),
        "retain_rows": len(retain_texts),
    }


def _forget_texts(
    row: Row,
    ds,
    topics: list[str],
    is_null: bool,
    rng: random.Random,
    max_forget_docs: int,
) -> tuple[list[str], list[str]]:
    if row.forget_texts:
        fixed = load_text_forget_set(row.forget_texts)
        return fixed.texts, fixed.doc_ids
    if is_null:
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        selected = ds.select(indices[:max_forget_docs])
        texts = [text for text in selected[TEXT_COL] if text and str(text).strip()]
        doc_ids = (
            [str(doc_id) for doc_id in selected[DOC_ID_COL]]
            if DOC_ID_COL in selected.column_names
            else []
        )
        return texts, doc_ids
    return sample_forget(ds, topics, max_forget_docs, rng)


def _write_train_frame(
    out_path: Path,
    forget_texts: list[str],
    forget_doc_ids: list[str],
    retain_texts: list[str],
    retain_doc_ids: list[str],
) -> None:
    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "split": "forget",
                    "doc_id": pad_doc_ids(forget_doc_ids, len(forget_texts)),
                    "text": forget_texts,
                }
            ),
            pd.DataFrame(
                {
                    "split": "retain",
                    "doc_id": pad_doc_ids(retain_doc_ids, len(retain_texts)),
                    "text": retain_texts,
                }
            ),
        ],
        ignore_index=True,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)


def pad_doc_ids(doc_ids: list[str], size: int) -> list[str]:
    return doc_ids if len(doc_ids) == size else [""] * size
