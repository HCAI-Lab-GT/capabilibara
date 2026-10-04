"""Sharded JSONL writer that rotates output files at a configurable line count."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .writer import JsonlWriter


@dataclass
class ShardedWriterStats:
    total_docs: int = 0
    total_tokens: int = 0
    shard_count: int = 0
    shard_paths: list[str] = field(default_factory=list)


class ShardedJsonlWriter:
    def __init__(
        self,
        output_dir: Path,
        lines_per_shard: int = 10_000_000,
        compress: str | None = "zst",
    ) -> None:
        self.output_dir = output_dir
        self.lines_per_shard = lines_per_shard
        self.compress = compress
        self.stats = ShardedWriterStats()
        self._current_writer: JsonlWriter | None = None
        self._current_lines = 0

    def __enter__(self) -> ShardedJsonlWriter:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._close_current()

    def _shard_name(self, index: int) -> str:
        suffix = ".jsonl.zst" if self.compress == "zst" else ".jsonl"
        return f"train-{index:05d}{suffix}"

    def _open_next(self) -> None:
        name = self._shard_name(self.stats.shard_count)
        path = self.output_dir / name
        self._current_writer = JsonlWriter(path, compress=self.compress)
        self._current_writer.open()
        self._current_lines = 0
        self.stats.shard_count += 1
        self.stats.shard_paths.append(str(path))

    def _close_current(self) -> None:
        if self._current_writer is None:
            return
        self._current_writer.close()
        self._current_writer.write_done()
        self._current_writer = None

    def write(self, record: dict[str, object], tokens: int = 0) -> None:
        if self._current_writer is None or self._current_lines >= self.lines_per_shard:
            self._close_current()
            self._open_next()
        self._current_writer.write(record)
        self._current_lines += 1
        self.stats.total_docs += 1
        self.stats.total_tokens += tokens


__all__ = ["ShardedJsonlWriter", "ShardedWriterStats"]
