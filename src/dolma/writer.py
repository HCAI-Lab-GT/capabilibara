"""Streaming JSONL writer with stats and resume helpers."""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO


@dataclass
class WriterStats:
    start_time: float
    num_docs: int = 0
    num_missing_url: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)

    def record(self, max_label: str, missing_url: bool) -> None:
        self.num_docs += 1
        if missing_url:
            self.num_missing_url += 1
        self.label_counts[max_label] = self.label_counts.get(max_label, 0) + 1

    def finalize(self) -> dict[str, object]:
        elapsed = max(time.monotonic() - self.start_time, 1e-6)
        return {
            "num_docs": self.num_docs,
            "num_missing_url": self.num_missing_url,
            "label_histogram": self.label_counts,
            "docs_per_second": self.num_docs / elapsed,
        }


class JsonlWriter:
    def __init__(
        self, path: Path, compress: str | None = "zst", level: int = 3
    ) -> None:
        self.path = path
        self.compress = compress
        self.level = level
        self.stats = WriterStats(start_time=time.monotonic())
        self._stream: IO[str] | None = None
        self._compressor: object | None = None

    def __enter__(self) -> JsonlWriter:
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.compress == "zst":
            import zstandard as zstd

            self._compressor = zstd.ZstdCompressor(level=self.level)
            binary_stream = self.path.open("wb")
            writer = self._compressor.stream_writer(binary_stream)
            self._stream = io.TextIOWrapper(writer, encoding="utf-8")
        else:
            self._stream = self.path.open("w", encoding="utf-8")

    def write(self, record: dict[str, object]) -> None:
        if self._stream is None:
            raise RuntimeError("Writer is not open")
        json.dump(record, self._stream)
        self._stream.write("\n")

    def close(self) -> None:
        if self._stream is None:
            return
        self._stream.flush()
        self._stream.close()
        self._stream = None

    def write_stats(self) -> Path:
        stats_path = self.stats_path(self.path)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w", encoding="utf-8") as stream:
            json.dump(self.stats.finalize(), stream, indent=2)
            stream.write("\n")
        return stats_path

    def write_done(self) -> Path:
        done_path = self.done_path(self.path)
        done_path.write_text("done\n", encoding="utf-8")
        return done_path

    @staticmethod
    def stats_path(path: Path) -> Path:
        return Path(f"{path}.stats.json")

    @staticmethod
    def done_path(path: Path) -> Path:
        return Path(f"{path}.done")


def is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    if path.stat().st_size == 0:
        return False
    return (
        JsonlWriter.stats_path(path).exists() and JsonlWriter.done_path(path).exists()
    )


__all__ = ["JsonlWriter", "WriterStats", "is_complete"]
