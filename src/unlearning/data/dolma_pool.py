# pyright: reportPrivateImportUsage=false, reportAttributeAccessIssue=false, reportArgumentType=false
"""
Data loader for the 6T-filtered Arrow cache of dolma3_pool_stratified.

Only documents that appear in the OLMo-3-7B 6T training mix are included
(~5.5M of the original 39.7M, ~14%).

  FILTERED_CACHE_DIR - pre-built filtered Arrow cache
"""

from __future__ import annotations

import logging
import math
import os
import random

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from unlearning.data.forget_texts import load_text_forget_set, load_text_train_set

logger = logging.getLogger(__name__)

FILTERED_CACHE_DIR = os.environ.get(
    "DOLMA_CACHE",
    os.path.join(
        os.path.expanduser("~"), "scratch", "hf_cache", "datasets", "dolma3_6t_filtered"
    ),
)

TOPIC_COL = "weborganizer_topic"
TEXT_COL = "text"
DOC_ID_COL = "doc_id"

MAX_LENGTH = 2048
PAD_TO_MULTIPLE = 8
RETAIN_DOCS_DEFAULT = 9_000


def _load_hf_dataset():
    """Load the 6T-filtered dataset from local Arrow cache."""
    from datasets import disable_caching, load_from_disk

    if os.environ.get("UNLEARNING_DISABLE_DATASET_CACHE", "0") == "1":
        disable_caching()
    logger.info("Loading 6T-filtered Arrow cache from %s ...", FILTERED_CACHE_DIR)
    return load_from_disk(FILTERED_CACHE_DIR)


def _tokenize(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> list[dict]:
    samples = []
    for text in texts:
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            padding=False,
        )
        iids = enc["input_ids"].squeeze(0)
        mask = enc["attention_mask"].squeeze(0)
        samples.append(
            {
                "input_ids": iids,
                "attention_mask": mask,
                "labels": iids.clone(),
            }
        )
    return samples


class ForgetRetainDataset(Dataset):
    def __init__(self, forget_samples: list[dict], retain_samples: list[dict]):
        self.forget = forget_samples
        self.retain = retain_samples

    def __len__(self) -> int:
        return len(self.forget)

    def __getitem__(self, idx: int) -> dict:
        return {
            "forget": self.forget[idx],
            "retain": self.retain[idx % len(self.retain)],
        }


def _pad(samples: list[dict]) -> dict:
    L = (
        math.ceil(
            max(s["input_ids"].size(0) for s in samples) / PAD_TO_MULTIPLE,
        )
        * PAD_TO_MULTIPLE
    )
    B = len(samples)
    iids = torch.zeros(B, L, dtype=torch.long)
    mask = torch.zeros(B, L, dtype=torch.long)
    lbls = torch.full((B, L), -100, dtype=torch.long)
    for i, s in enumerate(samples):
        n = s["input_ids"].size(0)
        iids[i, :n] = s["input_ids"]
        mask[i, :n] = s["attention_mask"]
        lbls[i, :n] = s["labels"]
    return {"input_ids": iids, "attention_mask": mask, "labels": lbls}


def _collate_fn(batch: list[dict]) -> dict:
    return {
        "forget": _pad([b["forget"] for b in batch]),
        "retain": _pad([b["retain"] for b in batch]),
    }


def _normalize_topics(target_topic) -> tuple[bool, list[str]]:
    """Normalize target_topic to (is_null, topics_list)."""
    if target_topic is None or target_topic == "null":
        return True, []
    if isinstance(target_topic, str):
        return False, [t.strip() for t in target_topic.split(",")]
    return False, [str(t).strip() for t in target_topic]


def _sample_null_bin(ds, max_forget, max_retain, rng):
    """Null-bin control: random sample from all topics."""
    logger.info(
        "Null bin: sampling %d forget + %d retain from all topics ...",
        max_forget,
        max_retain,
    )
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    forget_texts = ds.select(indices[:max_forget])[TEXT_COL]
    retain_texts = ds.select(indices[max_forget : max_forget + max_retain])[TEXT_COL]
    return (
        [t for t in forget_texts if t and t.strip()],
        [t for t in retain_texts if t and t.strip()],
    )


def build_forget_retain_loaders(
    target_topic: str | list[str],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 4,
    max_forget_docs: int | None = None,
    max_retain_docs: int = RETAIN_DOCS_DEFAULT,
    docs_per_retain_bin: int | None = None,
    min_tokens: int = 0,
    max_length: int = MAX_LENGTH,
    num_workers: int = 0,
    seed: int = 42,
    retain_topics: str | list[str] | None = None,
    output_dir: str | None = None,
    forget_manifest_path: str | None = None,
    forget_texts_path: str | None = None,
    train_texts_path: str | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, object]:
    """Build train, forget-eval, and retain-eval DataLoaders.

    Returns (train_loader, forget_eval_loader, retain_eval_loader, filtered_ds).
    filtered_ds is the min-token-filtered HF dataset for RetainPool.

    Args:
        forget_manifest_path: Optional path to a JSON manifest containing
            forget document IDs. When set, the forget set is loaded by filtering
            the corpus to exactly those doc_ids instead of using the normal
            random/topic-based sampling.
        forget_texts_path: Optional path to a parquet, CSV, or JSONL file with
            `text` and optional `doc_id` columns. This bypasses corpus lookup for
            fixed forget sets whose doc IDs are outside the local Arrow cache.
        train_texts_path: Optional path to a materialized parquet, CSV, or JSONL
            file with `split`, `text`, and optional `doc_id` columns. This bypasses
            corpus lookup for both forget and retain sets.
    """
    from unlearning.data.sampling import (
        filter_by_min_tokens,
        filter_num_proc,
        sample_forget,
        sample_retain_stratified,
        save_sampling_manifest,
    )

    max_forget = max_forget_docs if max_forget_docs is not None else 2_000
    is_null, topics = _normalize_topics(target_topic)
    label = "+".join(topics) if topics else "null"
    rng = random.Random(seed)
    filtered_ds = None

    if train_texts_path is not None:
        train_set = load_text_train_set(train_texts_path)
        forget_texts = train_set.forget_texts
        retain_texts = train_set.retain_texts
        forget_doc_ids = train_set.forget_doc_ids
        retain_doc_ids = train_set.retain_doc_ids
        logger.info(
            "Loading materialized train texts: %d forget + %d retain from %s",
            len(forget_texts),
            len(retain_texts),
            train_texts_path,
        )
        ds = None
    else:
        ds = _load_hf_dataset()
        ds_raw = ds  # unfiltered; used for manifest forget lookup

    if ds is not None and min_tokens > 0:
        ds, coverage = filter_by_min_tokens(ds, min_tokens=min_tokens)
        filtered_ds = ds

    if train_texts_path is not None:
        pass
    elif forget_texts_path is not None:
        fixed_forget = load_text_forget_set(forget_texts_path)
        forget_texts = fixed_forget.texts
        forget_doc_ids = fixed_forget.doc_ids
        logger.info(
            "Loading forget set from texts: %d/%d rows kept from %s",
            fixed_forget.rows_kept,
            fixed_forget.rows_read,
            forget_texts_path,
        )
        if is_null:
            r_idx = list(range(len(ds)))
            rng.shuffle(r_idx)
            retain_texts = ds.select(r_idx[:max_retain_docs])[TEXT_COL]
            retain_texts = [t for t in retain_texts if t and t.strip()]
            retain_doc_ids = []
        elif docs_per_retain_bin is not None:
            exclude = set(topics) if not is_null else set()
            retain_texts, retain_doc_ids, _ = sample_retain_stratified(
                ds,
                exclude,
                docs_per_retain_bin,
                rng,
            )
        else:
            retain_ds = ds.filter(
                lambda x: x[TOPIC_COL] not in set(topics) if not is_null else True,
                num_proc=filter_num_proc(),
                desc="filter:retain",
            )
            r_idx = list(range(len(retain_ds)))
            rng.shuffle(r_idx)
            retain_texts = retain_ds.select(r_idx[:max_retain_docs])[TEXT_COL]
            retain_texts = [t for t in retain_texts if t and t.strip()]
            retain_doc_ids = []
    elif forget_manifest_path is not None:
        import json as _json

        with open(forget_manifest_path) as _f:
            _manifest = _json.load(_f)
        _manifest_ids = set(_manifest["doc_ids"])
        logger.info(
            "Loading forget set from manifest: %d doc_ids (condition=%s, bin=%s)",
            len(_manifest_ids),
            _manifest.get("condition", "unknown"),
            _manifest.get("target_bin", "unknown"),
        )
        # Use unfiltered ds_raw so pre-selected manifest docs are not
        # dropped by the word-count proxy in filter_by_min_tokens.
        forget_ds = ds_raw.filter(
            lambda x: x[DOC_ID_COL] in _manifest_ids,
            num_proc=filter_num_proc(),
            desc="filter:forget:manifest",
        )
        forget_texts = forget_ds[TEXT_COL]
        forget_texts = [t for t in forget_texts if t and t.strip()]
        forget_doc_ids = (
            list(forget_ds[DOC_ID_COL]) if DOC_ID_COL in forget_ds.column_names else []
        )
        logger.info(
            "Manifest forget set: %d docs loaded from corpus", len(forget_texts)
        )
        if not forget_texts:
            raise ValueError(
                f"No documents from manifest found in corpus. "
                f"Manifest has {len(_manifest_ids)} doc_ids but none matched."
            )
        # Retain set: same logic as non-null case
        if docs_per_retain_bin is not None:
            exclude = set(topics) if not is_null else set()
            retain_texts, retain_doc_ids, _ = sample_retain_stratified(
                ds,
                exclude,
                docs_per_retain_bin,
                rng,
            )
        else:
            retain_ds = ds.filter(
                lambda x: x[TOPIC_COL] not in set(topics) if not is_null else True,
                num_proc=filter_num_proc(),
                desc="filter:retain",
            )
            r_idx = list(range(len(retain_ds)))
            rng.shuffle(r_idx)
            retain_texts = retain_ds.select(r_idx[:max_retain_docs])[TEXT_COL]
            retain_texts = [t for t in retain_texts if t and t.strip()]
            retain_doc_ids = []
    elif is_null:
        forget_texts, retain_texts = _sample_null_bin(
            ds,
            max_forget,
            max_retain_docs,
            rng,
        )
        forget_doc_ids, retain_doc_ids = [], []
    else:
        forget_texts, forget_doc_ids = sample_forget(
            ds,
            topics,
            max_forget,
            rng,
        )

        if docs_per_retain_bin is not None:
            exclude = set(topics)
            if retain_topics is not None:
                rt = (
                    [t.strip() for t in retain_topics.split(",")]
                    if isinstance(retain_topics, str)
                    else list(retain_topics)
                )
                all_t = rt
            else:
                all_t = None
            retain_texts, retain_doc_ids, _ = sample_retain_stratified(
                ds,
                exclude,
                docs_per_retain_bin,
                rng,
                all_topics=all_t,
            )
        else:
            topics_set = set(topics)
            if retain_topics is not None:
                rt = (
                    [t.strip() for t in retain_topics.split(",")]
                    if isinstance(retain_topics, str)
                    else list(retain_topics)
                )
                retain_ds = ds.filter(
                    lambda x, s=set(rt): x[TOPIC_COL] in s,
                    num_proc=filter_num_proc(),
                    desc="filter:retain",
                )
            else:
                retain_ds = ds.filter(
                    lambda x: x[TOPIC_COL] not in topics_set,
                    num_proc=filter_num_proc(),
                    desc="filter:retain",
                )
            r_idx = list(range(len(retain_ds)))
            rng.shuffle(r_idx)
            retain_texts = retain_ds.select(r_idx[:max_retain_docs])[TEXT_COL]
            retain_texts = [t for t in retain_texts if t and t.strip()]
            retain_doc_ids = []

    if not forget_texts:
        raise ValueError(f"No documents found for topic(s) '{label}'.")
    if not retain_texts:
        raise ValueError("No retain documents found.")

    if output_dir:
        save_sampling_manifest(
            output_dir,
            forget_doc_ids,
            retain_doc_ids,
            seed,
            topics,
            config_snapshot={
                "max_forget_docs": max_forget,
                "docs_per_retain_bin": docs_per_retain_bin,
                "min_tokens": min_tokens,
            },
            forget_text_count=len(forget_texts),
            retain_text_count=len(retain_texts),
        )

    logger.info(
        "Tokenizing %d forget + %d retain docs ...",
        len(forget_texts),
        len(retain_texts),
    )
    forget_samples = _tokenize(forget_texts, tokenizer, max_length)
    retain_samples = _tokenize(retain_texts, tokenizer, max_length)

    train_ds = ForgetRetainDataset(forget_samples, retain_samples)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    ev = min(50, len(forget_samples))
    eval_batch_size = 1
    forget_eval_loader = DataLoader(
        forget_samples[:ev],
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=_pad,
    )
    retain_eval_loader = DataLoader(
        retain_samples[:ev],
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=_pad,
    )

    if filtered_ds is None:
        filtered_ds = ds

    return train_loader, forget_eval_loader, retain_eval_loader, filtered_ds
