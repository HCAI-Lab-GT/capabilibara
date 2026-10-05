import json

import pytest

from data_attribution.scoring.io import (
    build_record,
    load_id_field,
    load_id_field_from_jsonl,
    materialize_ids,
    resolve_query_ids,
)


def test_materialize_ids_defaults() -> None:
    assert materialize_ids(None, 3) == [0, 1, 2]


def test_materialize_ids_mismatch() -> None:
    with pytest.raises(ValueError):
        materialize_ids(["a"], 2)


def test_load_id_field_from_jsonl(tmp_path) -> None:
    path = tmp_path / "ids.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"query_id": "a"}) + "\n")
        handle.write(json.dumps({"query_id": "b"}) + "\n")
    assert load_id_field_from_jsonl(path, "query_id") == ["a", "b"]


def test_resolve_query_ids_manifest(tmp_path) -> None:
    path = tmp_path / "manifest.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"query_id": "x"}) + "\n")
        handle.write(json.dumps({"query_id": "y"}) + "\n")
    resolved = resolve_query_ids(tmp_path, "query_id", path, 2)
    assert resolved == ["x", "y"]


def test_load_id_field_resolves_relative_dataset(tmp_path) -> None:
    index_path = tmp_path / "index"
    data_dir = index_path / "data"
    data_dir.mkdir(parents=True)
    dataset_path = data_dir / "ids.jsonl"
    dataset_path.write_text(
        json.dumps({"query_id": "a"}) + "\n" + json.dumps({"query_id": "b"}) + "\n",
        encoding="utf-8",
    )

    config = {"data": {"dataset": "data/ids.jsonl"}}
    (index_path / "index_config.json").write_text(json.dumps(config), encoding="utf-8")

    assert load_id_field(index_path, "query_id") == ["a", "b"]


def test_load_id_field_falls_back_when_data_hf_is_unreadable(tmp_path) -> None:
    index_path = tmp_path / "index"
    data_hf = index_path / "data.hf"
    data_hf.mkdir(parents=True)
    (data_hf / "dataset_info.json").write_text("{}", encoding="utf-8")

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps({"query_id": "a"}) + "\n" + json.dumps({"query_id": "b"}) + "\n",
        encoding="utf-8",
    )
    config = {"data": {"dataset": str(manifest_path)}}
    (index_path / "index_config.json").write_text(json.dumps(config), encoding="utf-8")

    assert load_id_field(index_path, "query_id") == ["a", "b"]


def test_build_record_with_text() -> None:
    record = build_record("q", [1], [0.5], ["g"], {1: "text"})
    documents = record["documents"]
    assert isinstance(documents, list)
    assert isinstance(documents[0], dict)
    assert documents[0]["text"] == "text"
