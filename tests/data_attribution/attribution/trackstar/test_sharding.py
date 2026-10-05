import json
from pathlib import Path

import pytest
import zstandard as zstd

from data_attribution.attribution.trackstar.sharding import (
    _open_lines,
    split_jsonl,
    split_multi_jsonl,
)


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )
    return path


def _write_jsonl_zst(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(r) for r in records) + "\n"
    cctx = zstd.ZstdCompressor()
    with open(path, "wb") as f:
        f.write(cctx.compress(text.encode("utf-8")))
    return path


def _read_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines]


class TestOpenLines:
    def test_reads_plain_jsonl(self, tmp_path: Path):
        records = [{"id": i} for i in range(5)]
        path = _write_jsonl(tmp_path / "test.jsonl", records)

        lines = list(_open_lines(path))

        assert len(lines) == 5
        assert json.loads(lines[0])["id"] == 0

    def test_reads_zst_jsonl(self, tmp_path: Path):
        records = [{"id": i} for i in range(5)]
        path = _write_jsonl_zst(tmp_path / "test.jsonl.zst", records)

        lines = list(_open_lines(path))

        assert len(lines) == 5
        assert json.loads(lines[0])["id"] == 0

    def test_empty_zst_raises_no_error(self, tmp_path: Path):
        path = _write_jsonl_zst(tmp_path / "empty.jsonl.zst", [])
        lines = list(_open_lines(path))
        assert lines == ["\n"]


class TestSplitJsonl:
    def test_round_robin_distribution(self, tmp_path: Path):
        records = [{"id": i} for i in range(10)]
        source = _write_jsonl(tmp_path / "source.jsonl", records)

        shards = split_jsonl(source, 3, tmp_path / "shards")

        assert len(shards) == 3
        shard_contents = [_read_jsonl(s) for s in shards]
        all_ids = sorted(r["id"] for sc in shard_contents for r in sc)
        assert all_ids == list(range(10))

    def test_single_shard(self, tmp_path: Path):
        records = [{"id": i} for i in range(5)]
        source = _write_jsonl(tmp_path / "source.jsonl", records)

        shards = split_jsonl(source, 1, tmp_path / "shards")

        assert len(shards) == 1
        assert len(_read_jsonl(shards[0])) == 5

    def test_more_shards_than_lines(self, tmp_path: Path):
        records = [{"id": 0}, {"id": 1}]
        source = _write_jsonl(tmp_path / "source.jsonl", records)

        shards = split_jsonl(source, 10, tmp_path / "shards")

        assert len(shards) == 2
        all_ids = sorted(r["id"] for s in shards for r in _read_jsonl(s))
        assert all_ids == [0, 1]

    def test_empty_source_exits(self, tmp_path: Path):
        source = tmp_path / "empty.jsonl"
        source.write_text("", encoding="utf-8")

        with pytest.raises(SystemExit):
            split_jsonl(source, 3, tmp_path / "shards")


class TestSplitMultiJsonl:
    def test_consolidates_multiple_plain_files(self, tmp_path: Path):
        src_a = _write_jsonl(tmp_path / "a.jsonl", [{"id": i} for i in range(5)])
        src_b = _write_jsonl(tmp_path / "b.jsonl", [{"id": i} for i in range(5, 10)])

        shards = split_multi_jsonl([src_a, src_b], 3, tmp_path / "shards")

        assert len(shards) == 3
        all_ids = sorted(r["id"] for s in shards for r in _read_jsonl(s))
        assert all_ids == list(range(10))

    def test_consolidates_zst_files(self, tmp_path: Path):
        src_a = _write_jsonl_zst(
            tmp_path / "a.jsonl.zst", [{"id": i} for i in range(5)]
        )
        src_b = _write_jsonl_zst(
            tmp_path / "b.jsonl.zst", [{"id": i} for i in range(5, 10)]
        )

        shards = split_multi_jsonl([src_a, src_b], 4, tmp_path / "shards")

        assert len(shards) == 4
        all_ids = sorted(r["id"] for s in shards for r in _read_jsonl(s))
        assert all_ids == list(range(10))

    def test_mixes_plain_and_zst(self, tmp_path: Path):
        src_plain = _write_jsonl(tmp_path / "a.jsonl", [{"id": 0}])
        src_zst = _write_jsonl_zst(tmp_path / "b.jsonl.zst", [{"id": 1}])

        shards = split_multi_jsonl([src_plain, src_zst], 2, tmp_path / "shards")

        all_ids = sorted(r["id"] for s in shards for r in _read_jsonl(s))
        assert all_ids == [0, 1]

    def test_empty_sources_exits(self, tmp_path: Path):
        src = _write_jsonl(tmp_path / "empty.jsonl", [])

        with pytest.raises(SystemExit):
            split_multi_jsonl([src], 3, tmp_path / "shards")

    def test_skips_blank_lines(self, tmp_path: Path):
        path = tmp_path / "with_blanks.jsonl"
        path.write_text('{"id": 0}\n\n\n{"id": 1}\n\n', encoding="utf-8")

        shards = split_multi_jsonl([path], 2, tmp_path / "shards")

        all_ids = sorted(r["id"] for s in shards for r in _read_jsonl(s))
        assert all_ids == [0, 1]

    def test_output_shards_are_plain_jsonl(self, tmp_path: Path):
        src = _write_jsonl_zst(
            tmp_path / "compressed.jsonl.zst", [{"id": i} for i in range(3)]
        )

        shards = split_multi_jsonl([src], 2, tmp_path / "shards")

        for shard in shards:
            assert shard.suffix == ".jsonl"
            content = shard.read_text(encoding="utf-8")
            for line in content.strip().splitlines():
                json.loads(line)

    def test_many_small_files_consolidated(self, tmp_path: Path):
        sources = []
        for i in range(50):
            src = _write_jsonl_zst(
                tmp_path / f"src_{i:03d}.jsonl.zst", [{"id": i}]
            )
            sources.append(src)

        shards = split_multi_jsonl(sources, 4, tmp_path / "shards")

        assert len(shards) == 4
        all_ids = sorted(r["id"] for s in shards for r in _read_jsonl(s))
        assert all_ids == list(range(50))
