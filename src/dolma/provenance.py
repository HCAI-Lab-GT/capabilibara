"""Provenance utilities for 6T deduplication and materialization."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

ID_PATTERN = re.compile(r'"id"\s*:\s*"([^"]+)"')
_BLOOM_MAGIC = b"DOLMA_BLOOM_V1\n"


def extract_doc_id(line: str) -> str | None:
    match = ID_PATTERN.search(line)
    if not match:
        return None
    return match.group(1)


def shard_folder_name(shard_path: str) -> str:
    if shard_path.startswith("data/"):
        parts = shard_path.split("/", 2)
        if len(parts) >= 2:
            return parts[1]
    return ""


def source_family(shard_path: str) -> str:
    folder = shard_folder_name(shard_path)
    if folder.startswith("common_crawl-"):
        return "common_crawl"
    if folder.startswith("olmocr_science_pdfs-"):
        return "olmocr_science_pdfs"
    if folder.startswith("stack_edu-"):
        return "stack_edu"
    if folder.startswith("finemath-"):
        return "finemath"
    if folder:
        return folder
    return "unknown"


def is_pool_family(shard_path: str) -> bool:
    return source_family(shard_path) in {"common_crawl", "olmocr_science_pdfs"}


def source_category(shard_path: str) -> str:
    folder = shard_folder_name(shard_path)
    return folder or "unknown"


def stable_bucket(doc_id: str, bucket_count: int) -> int:
    if bucket_count <= 0:
        raise ValueError("bucket_count must be > 0")
    digest = hashlib.blake2b(doc_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False) % bucket_count


def normalize_hf_shard_path(path: str, dataset_id: str) -> str:
    prefix = f"datasets/{dataset_id}/"
    if path.startswith(prefix):
        return path[len(prefix) :]
    if path.startswith("datasets/"):
        parts = path.split("/", 3)
        if len(parts) == 4:
            return parts[3]
    return path


@dataclass(slots=True)
class BloomIndex:
    bit_count: int
    hash_count: int
    expected_items: int
    false_positive_rate: float
    bit_array: bytearray

    @classmethod
    def from_capacity(
        cls, expected_items: int, false_positive_rate: float = 1e-3
    ) -> BloomIndex:
        if expected_items <= 0:
            raise ValueError("expected_items must be > 0")
        if not (0 < false_positive_rate < 1):
            raise ValueError("false_positive_rate must be between 0 and 1")
        ln2 = math.log(2.0)
        bit_count = math.ceil(
            -(expected_items * math.log(false_positive_rate)) / (ln2 * ln2)
        )
        hash_count = max(1, round((bit_count / expected_items) * ln2))
        byte_count = (bit_count + 7) // 8
        return cls(
            bit_count=bit_count,
            hash_count=hash_count,
            expected_items=expected_items,
            false_positive_rate=false_positive_rate,
            bit_array=bytearray(byte_count),
        )

    @classmethod
    def load(cls, path: Path) -> BloomIndex:
        payload = path.read_bytes()
        if not payload.startswith(_BLOOM_MAGIC):
            raise ValueError(f"Invalid bloom header in {path}")
        header_end = payload.find(b"\n", len(_BLOOM_MAGIC))
        if header_end < 0:
            raise ValueError(f"Missing bloom JSON header in {path}")
        header_bytes = payload[len(_BLOOM_MAGIC) : header_end]
        header = json.loads(header_bytes.decode("utf-8"))
        bit_array = bytearray(payload[header_end + 1 :])
        expected_bytes = (int(header["bit_count"]) + 7) // 8
        if len(bit_array) != expected_bytes:
            raise ValueError(
                f"Bloom payload byte length mismatch: {len(bit_array)} vs {expected_bytes}"
            )
        return cls(
            bit_count=int(header["bit_count"]),
            hash_count=int(header["hash_count"]),
            expected_items=int(header["expected_items"]),
            false_positive_rate=float(header["false_positive_rate"]),
            bit_array=bit_array,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        header = json.dumps(
            {
                "bit_count": self.bit_count,
                "hash_count": self.hash_count,
                "expected_items": self.expected_items,
                "false_positive_rate": self.false_positive_rate,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        path.write_bytes(_BLOOM_MAGIC + header + b"\n" + bytes(self.bit_array))

    def add(self, value: str) -> None:
        for bit_idx in self._hash_positions(value):
            byte_idx = bit_idx // 8
            bit_mask = 1 << (bit_idx % 8)
            self.bit_array[byte_idx] |= bit_mask

    def contains(self, value: str) -> bool:
        for bit_idx in self._hash_positions(value):
            byte_idx = bit_idx // 8
            bit_mask = 1 << (bit_idx % 8)
            if (self.bit_array[byte_idx] & bit_mask) == 0:
                return False
        return True

    def _hash_positions(self, value: str) -> list[int]:
        digest = hashlib.blake2b(value.encode("utf-8"), digest_size=16).digest()
        h1 = int.from_bytes(digest[:8], "big", signed=False)
        h2 = int.from_bytes(digest[8:], "big", signed=False) | 1
        return [int((h1 + i * h2) % self.bit_count) for i in range(self.hash_count)]

    def __contains__(self, value: str) -> bool:
        return self.contains(value)


__all__ = [
    "BloomIndex",
    "extract_doc_id",
    "is_pool_family",
    "normalize_hf_shard_path",
    "shard_folder_name",
    "source_category",
    "source_family",
    "stable_bucket",
]
