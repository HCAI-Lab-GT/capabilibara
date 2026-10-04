"""Model-content evidence, strict loading, file evidence, and Hub snapshots."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from pathlib import Path, PurePosixPath
from typing import BinaryIO, cast

from huggingface_hub import HfApi, hf_hub_download

from data_attribution.attribution.trackstar.lineage_records import (
    ModelContentAttestation,
    ModelContentRequest,
    NormalizedLoadingFile,
    RepoFile,
    RepoSnapshot,
    RepoSnapshots,
    TotalParametersCorrection,
    validate_hex,
    validate_model_request,
)
from data_attribution.artifact_integrity import (
    canonical_json_bytes,
    jsonable,
    parse_json_object,
    sha256_json,
)

SAFETENSORS_INDEX_PATH = "model.safetensors.index.json"
HUB_REQUEST_TIMEOUT_SECONDS = 30.0
_DOCUMENTATION_PATHS = frozenset({"README.md"})


def build_model_content_attestation(
    request: ModelContentRequest, snapshots: RepoSnapshots
) -> ModelContentAttestation:
    basis = validate_model_request(request)
    build, requested, ignored = normalize_snapshots(request, snapshots)
    normalized, corrections = _compare_snapshots(request, build, requested, ignored)
    result = ModelContentAttestation(
        request.artifact_id,
        request.artifact_uri,
        request.artifact_sha256,
        request.build_timestamp,
        request.model_id,
        request.raw_revision,
        request.build_revision,
        request.requested_revision,
        basis,
        build,
        requested,
        ignored,
        corrections,
        normalized,
        sha256_json(normalized),
        "",
    )
    return replace(result, attestation_sha256=model_attestation_sha256(result))


def model_attestation_sha256(attestation: ModelContentAttestation) -> str:
    payload = jsonable(attestation)
    if not isinstance(payload, dict):
        raise TypeError("Model attestation must serialize to an object")
    payload.pop("attestation_sha256")
    return sha256_json(payload)


def validate_model_attestation(attestation: ModelContentAttestation) -> None:
    normalizations = attestation.metadata_normalizations
    if len(normalizations) > 1 or any(
        type(item) is not TotalParametersCorrection for item in normalizations
    ):
        raise ValueError("Model attestation has unknown metadata normalization")
    request = ModelContentRequest(
        attestation.artifact_id,
        attestation.artifact_uri,
        attestation.artifact_sha256,
        attestation.build_timestamp,
        attestation.model_id,
        attestation.raw_revision,
        attestation.build_revision,
        attestation.requested_revision,
        attestation.ignored_documentation_paths,
        normalizations[0] if normalizations else None,
    )
    rebuilt = build_model_content_attestation(
        request,
        RepoSnapshots(attestation.build_snapshot, attestation.requested_snapshot),
    )
    if rebuilt != attestation:
        raise ValueError("Model attestation semantic validation failed")


def _compare_snapshots(
    request: ModelContentRequest,
    build: RepoSnapshot,
    requested: RepoSnapshot,
    ignored: tuple[str, ...],
) -> tuple[tuple[NormalizedLoadingFile, ...], tuple[TotalParametersCorrection, ...]]:
    left = {item.path: item for item in build.files if item.path not in ignored}
    right = {item.path: item for item in requested.files if item.path not in ignored}
    if set(left) != set(right):
        raise ValueError("Loading-relevant file roster differs between snapshots")
    normalized: list[NormalizedLoadingFile] = []
    corrections: tuple[TotalParametersCorrection, ...] = ()
    for path in sorted(left):
        item, correction = _compare_file(request, left[path], right[path], left, right)
        normalized.append(item)
        if correction:
            corrections = (*corrections, correction)
    return tuple(normalized), corrections


def _compare_file(
    request: ModelContentRequest,
    build: RepoFile,
    requested: RepoFile,
    left: Mapping[str, RepoFile],
    right: Mapping[str, RepoFile],
) -> tuple[NormalizedLoadingFile, TotalParametersCorrection | None]:
    if build.path == SAFETENSORS_INDEX_PATH:
        payload, correction = _compare_index(request, build, requested, left, right)
        size = len(canonical_json_bytes(payload))
        return NormalizedLoadingFile(
            build.path, size, "safetensors_index", sha256_json(payload)
        ), correction
    if build.path.endswith(".safetensors"):
        if build.size != requested.size or build.lfs_sha256 != requested.lfs_sha256:
            raise ValueError("safetensors shard content differs between snapshots")
        if build.lfs_sha256 is None:
            raise ValueError("Safetensors shard requires an LFS SHA-256")
        return NormalizedLoadingFile(
            build.path, build.size, "lfs_object", build.lfs_sha256
        ), None
    if (build.size, build.git_oid) != (requested.size, requested.git_oid):
        raise ValueError(f"loading-relevant file differs: {build.path}")
    return NormalizedLoadingFile(
        build.path, build.size, "git_blob", build.git_oid
    ), None


def _compare_index(
    request: ModelContentRequest,
    build: RepoFile,
    requested: RepoFile,
    left: Mapping[str, RepoFile],
    right: Mapping[str, RepoFile],
) -> tuple[dict[str, object], TotalParametersCorrection | None]:
    left_metadata, left_map = index_parts(build.index_payload)
    right_metadata, right_map = index_parts(requested.index_payload)
    if left_map != right_map:
        raise ValueError("Safetensors weight map differs between snapshots")
    validate_referenced_shards(left_map, left, right)
    differences = _metadata_differences(left_metadata, right_metadata)
    if differences != {"total_parameters"}:
        if differences:
            raise ValueError("Unsupported safetensors index metadata difference")
        if request.total_parameters_correction is not None:
            raise ValueError("declared total_parameters correction was not observed")
        return {"metadata": left_metadata, "weight_map": left_map}, None
    actual = _total_parameters_correction(left_metadata, right_metadata)
    if request.total_parameters_correction != actual:
        raise ValueError(
            "declared total_parameters correction does not match snapshots"
        )
    metadata = dict(left_metadata)
    metadata.pop("total_parameters")
    return {"metadata": metadata, "weight_map": left_map}, actual


def _metadata_differences(
    left: Mapping[str, object], right: Mapping[str, object]
) -> set[str]:
    left_keys, right_keys = set(left), set(right)
    changed = {key for key in left_keys & right_keys if left[key] != right[key]}
    return left_keys ^ right_keys | changed


def _total_parameters_correction(
    left: Mapping[str, object], right: Mapping[str, object]
) -> TotalParametersCorrection:
    left_value = left.get("total_parameters")
    right_value = right.get("total_parameters")
    if any(type(value) is not int or value < 0 for value in (left_value, right_value)):
        raise ValueError("Invalid total_parameters correction values")
    return TotalParametersCorrection(left_value, right_value)  # type: ignore[arg-type]


def require_keys(value: dict[str, object], cls: type, label: str) -> None:
    expected = {item.name for item in fields(cls)}
    if set(value) != expected:
        raise ValueError(f"Invalid {label} fields")


def require_text(value: dict[str, object], key: str) -> str:
    item = value[key]
    if type(item) is not str:
        raise ValueError(f"Invalid string field: {key}")
    return item


def optional_text(value: dict[str, object], key: str) -> str | None:
    item = value[key]
    if item is not None and type(item) is not str:
        raise ValueError(f"Invalid optional string field: {key}")
    return item


def require_integer(value: dict[str, object], key: str) -> int:
    item = value[key]
    if type(item) is not int:
        raise ValueError(f"Invalid integer field: {key}")
    return item


def optional_integer(value: dict[str, object], key: str) -> int | None:
    item = value[key]
    if item is not None and type(item) is not int:
        raise ValueError(f"Invalid optional integer field: {key}")
    return item


def require_float(value: dict[str, object], key: str) -> float:
    item = value[key]
    if type(item) is not float:
        raise ValueError(f"Invalid float field: {key}")
    return item


def require_bool(value: dict[str, object], key: str) -> bool:
    item = value[key]
    if type(item) is not bool:
        raise ValueError(f"Invalid boolean field: {key}")
    return item


def require_object(value: dict[str, object], key: str) -> dict[str, object]:
    item = value[key]
    if type(item) is not dict:
        raise ValueError(f"Invalid object field: {key}")
    return item


def require_list(value: dict[str, object], key: str) -> list[object]:
    item = value[key]
    if type(item) is not list:
        raise ValueError(f"Invalid list field: {key}")
    return item


def require_strings(value: dict[str, object], key: str) -> tuple[str, ...]:
    items = require_list(value, key)
    if any(type(item) is not str for item in items):
        raise ValueError(f"Invalid string-list field: {key}")
    return tuple(items)  # type: ignore[arg-type]


def require_choice[Choice: str](
    value: dict[str, object], key: str, choices: tuple[Choice, ...]
) -> Choice:
    item = require_text(value, key)
    if item not in choices:
        raise ValueError(f"Invalid choice field: {key}")
    return cast(Choice, item)


def has_symlink_component(path: Path) -> bool:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            return True
    return False


@dataclass(frozen=True)
class StableFileDigest:
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class StableJsonlEvidence:
    byte_count: int
    sha256: str
    row_count: int


@contextmanager
def open_stable_regular_file(path: Path) -> Iterator[BinaryIO]:
    before_path = _lstat_regular(path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"Unable to open stable regular file: {path}") from error
    try:
        stream = os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
    try:
        before_file = os.fstat(stream.fileno())
        _require_same_state(before_path, before_file, path)
        try:
            yield stream
        finally:
            after_file = os.fstat(stream.fileno())
            after_path = _lstat_regular(path)
            _require_same_state(before_file, after_file, path)
            _require_same_state(after_file, after_path, path)
    finally:
        stream.close()


def digest_stable_file(path: Path) -> StableFileDigest:
    digest = hashlib.sha256()
    byte_count = 0
    with open_stable_regular_file(path) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            byte_count += len(chunk)
    return StableFileDigest(byte_count, digest.hexdigest())


def inspect_stable_jsonl(path: Path) -> StableJsonlEvidence:
    digest = hashlib.sha256()
    byte_count = 0
    row_count = 0
    with open_stable_regular_file(path) as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            digest.update(raw_line)
            byte_count += len(raw_line)
            _validate_jsonl_row(raw_line, path, line_number)
            row_count += 1
    return StableJsonlEvidence(byte_count, digest.hexdigest(), row_count)


def _validate_jsonl_row(raw_line: bytes, path: Path, line_number: int) -> None:
    try:
        line = raw_line.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"Invalid UTF-8 at {path}:{line_number}") from error
    if not line.strip():
        raise ValueError("Strict source JSONL contains a blank row")
    parse_json_object(line, path, line_number)


def _lstat_regular(path: Path) -> os.stat_result:
    try:
        value = path.lstat()
    except OSError as error:
        raise ValueError(f"Stable regular file is unavailable: {path}") from error
    if has_symlink_component(path) or not stat.S_ISREG(value.st_mode):
        raise ValueError(f"Stable evidence path is not a regular file: {path}")
    return value


def _require_same_state(
    expected: os.stat_result, actual: os.stat_result, path: Path
) -> None:
    if _file_state(expected) != _file_state(actual):
        raise ValueError(f"Stable evidence file changed or was replaced: {path}")


def _file_state(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def normalize_snapshots(
    request: ModelContentRequest, snapshots: RepoSnapshots
) -> tuple[RepoSnapshot, RepoSnapshot, tuple[str, ...]]:
    build = _validate_snapshot(snapshots.build_time, request.build_revision)
    requested = _validate_snapshot(snapshots.requested, request.requested_revision)
    ignored = _validate_ignored(request, build, requested)
    return build, requested, ignored


def _validate_snapshot(snapshot: RepoSnapshot, revision: str) -> RepoSnapshot:
    if type(snapshot) is not RepoSnapshot or snapshot.revision != revision:
        raise ValueError("Repository snapshot revision does not match request")
    files = tuple(sorted(snapshot.files, key=lambda item: item.path))
    if any(type(item) is not RepoFile for item in files):
        raise ValueError("Repository snapshot contains an invalid file")
    paths = [item.path for item in files]
    if len(paths) != len(set(paths)):
        raise ValueError("Repository snapshot contains duplicate paths")
    for item in files:
        _validate_repo_file(item)
    return RepoSnapshot(snapshot.revision, files)


def _validate_repo_file(item: RepoFile) -> None:
    path = PurePosixPath(item.path)
    invalid_path = (
        not item.path
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != item.path
    )
    if invalid_path:
        raise ValueError("Repository file has an invalid path")
    if type(item.size) is not int or item.size < 0:
        raise ValueError("Repository file has an invalid size")
    validate_hex(item.git_oid, 40, "Git object ID")
    if item.lfs_sha256 is not None:
        validate_hex(item.lfs_sha256, 64, "LFS SHA-256")
    is_index = item.path == SAFETENSORS_INDEX_PATH
    if is_index != (item.index_payload is not None):
        raise ValueError("Safetensors index payload is missing or misplaced")


def _validate_ignored(
    request: ModelContentRequest, build: RepoSnapshot, requested: RepoSnapshot
) -> tuple[str, ...]:
    values = request.ignored_documentation_paths
    if type(values) is not tuple or len(values) != len(set(values)):
        raise ValueError("Ignored documentation paths must be a unique tuple")
    available = {item.path for item in build.files} & {
        item.path for item in requested.files
    }
    for value in values:
        if type(value) is not str or value not in available:
            raise ValueError("Ignored documentation path is absent or invalid")
        if value not in _DOCUMENTATION_PATHS:
            raise ValueError("Ignored path is not recognized documentation")
    return tuple(sorted(values))


def index_parts(payload: object) -> tuple[dict[str, object], dict[str, str]]:
    if not isinstance(payload, Mapping) or set(payload) != {"metadata", "weight_map"}:
        raise ValueError("Safetensors index must contain metadata and weight_map")
    metadata, weight_map = payload["metadata"], payload["weight_map"]
    if not isinstance(metadata, Mapping) or not isinstance(weight_map, Mapping):
        raise ValueError("Safetensors index metadata and weight map must be objects")
    if any(type(key) is not str for key in metadata):
        raise ValueError("Safetensors metadata keys must be strings")
    invalid_map = any(
        type(key) is not str or type(value) is not str or not value
        for key, value in weight_map.items()
    )
    if invalid_map or not weight_map:
        raise ValueError("Safetensors weight map must contain string paths")
    return dict(metadata), dict(weight_map)


def validate_referenced_shards(
    weight_map: Mapping[str, str],
    left: Mapping[str, RepoFile],
    right: Mapping[str, RepoFile],
) -> None:
    referenced = set(weight_map.values())
    shards = {path for path in left if path.endswith(".safetensors")}
    if shards != referenced:
        raise ValueError("Safetensors snapshot contains an unreferenced shard")
    for shard in referenced:
        if shard not in left or shard not in right:
            raise ValueError(f"Safetensors index referenced shard is missing: {shard}")
        if left[shard].lfs_sha256 is None or right[shard].lfs_sha256 is None:
            raise ValueError("Safetensors referenced shard lacks LFS identity")


def load_model_request(path: Path) -> ModelContentRequest:
    payload = parse_json_object(path.read_text(encoding="utf-8"), path, 1)
    require_keys(payload, ModelContentRequest, "model request")
    result = ModelContentRequest(
        artifact_id=require_text(payload, "artifact_id"),
        artifact_uri=require_text(payload, "artifact_uri"),
        artifact_sha256=require_text(payload, "artifact_sha256"),
        build_timestamp=require_text(payload, "build_timestamp"),
        model_id=require_text(payload, "model_id"),
        raw_revision=optional_text(payload, "raw_revision"),
        build_revision=require_text(payload, "build_revision"),
        requested_revision=require_text(payload, "requested_revision"),
        ignored_documentation_paths=require_strings(
            payload, "ignored_documentation_paths"
        ),
        total_parameters_correction=_decode_correction(
            payload["total_parameters_correction"]
        ),
    )
    validate_model_request(result)
    return result


def load_repo_snapshots(path: Path) -> RepoSnapshots:
    payload = parse_json_object(path.read_text(encoding="utf-8"), path, 1)
    require_keys(payload, RepoSnapshots, "repository snapshots")
    return RepoSnapshots(
        build_time=_decode_snapshot(require_object(payload, "build_time")),
        requested=_decode_snapshot(require_object(payload, "requested")),
    )


def fetch_repo_snapshots(request: ModelContentRequest) -> RepoSnapshots:
    validate_model_request(request)
    api = HfApi()
    snapshots = RepoSnapshots(
        _fetch_snapshot(api, request.model_id, request.build_revision),
        _fetch_snapshot(api, request.model_id, request.requested_revision),
    )
    build, requested, _ = normalize_snapshots(request, snapshots)
    return RepoSnapshots(build, requested)


def _decode_correction(value: object) -> TotalParametersCorrection | None:
    if value is None:
        return None
    if type(value) is not dict:
        raise ValueError("Invalid total_parameters correction")
    require_keys(value, TotalParametersCorrection, "total_parameters correction")
    return TotalParametersCorrection(
        require_integer(value, "build_value"),
        require_integer(value, "requested_value"),
    )


def _decode_snapshot(payload: dict[str, object]) -> RepoSnapshot:
    require_keys(payload, RepoSnapshot, "repository snapshot")
    files = require_list(payload, "files")
    return RepoSnapshot(
        require_text(payload, "revision"), tuple(_decode_file(item) for item in files)
    )


def _decode_file(value: object) -> RepoFile:
    if type(value) is not dict:
        raise ValueError("Repository file must be an object")
    require_keys(value, RepoFile, "repository file")
    payload = value["index_payload"]
    if payload is not None and type(payload) is not dict:
        raise ValueError("Repository index payload must be an object or null")
    return RepoFile(
        path=require_text(value, "path"),
        size=require_integer(value, "size"),
        git_oid=require_text(value, "git_oid"),
        lfs_sha256=optional_text(value, "lfs_sha256"),
        index_payload=payload,
    )


def _fetch_snapshot(api: HfApi, repo_id: str, revision: str) -> RepoSnapshot:
    info = api.model_info(
        repo_id,
        revision=revision,
        timeout=HUB_REQUEST_TIMEOUT_SECONDS,
        files_metadata=True,
        token=False,
    )
    if getattr(info, "sha", None) != revision:
        raise ValueError("Hub resolved revision does not match requested revision")
    siblings = getattr(info, "siblings", None)
    if type(siblings) is not list:
        raise ValueError("Hub repository file metadata is missing")
    files = tuple(_hub_file(item, repo_id, revision) for item in siblings)
    return RepoSnapshot(revision, files)


def _hub_file(value: object, repo_id: str, revision: str) -> RepoFile:
    path = getattr(value, "rfilename", None)
    size = getattr(value, "size", None)
    git_oid = getattr(value, "blob_id", None)
    if type(path) is not str or type(size) is not int or type(git_oid) is not str:
        raise ValueError("Hub repository file metadata is incomplete")
    lfs = getattr(value, "lfs", None)
    if isinstance(lfs, Mapping):
        lfs_sha256 = lfs.get("sha256")
    else:
        lfs_sha256 = None if lfs is None else getattr(lfs, "sha256", None)
    if lfs_sha256 is not None and type(lfs_sha256) is not str:
        raise ValueError("Hub LFS metadata is incomplete")
    payload = _load_index(repo_id, revision) if path == SAFETENSORS_INDEX_PATH else None
    return RepoFile(path, size, git_oid, lfs_sha256, payload)


def _load_index(repo_id: str, revision: str) -> dict[str, object]:
    local = hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        filename=SAFETENSORS_INDEX_PATH,
        revision=revision,
        etag_timeout=HUB_REQUEST_TIMEOUT_SECONDS,
        token=False,
    )
    path = Path(local)
    return parse_json_object(path.read_text(encoding="utf-8"), path, 1)


def decode_model_attestation(
    payload: dict[str, object],
) -> ModelContentAttestation:
    require_keys(payload, ModelContentAttestation, "model attestation")
    return ModelContentAttestation(
        artifact_id=require_text(payload, "artifact_id"),
        artifact_uri=require_text(payload, "artifact_uri"),
        artifact_sha256=require_text(payload, "artifact_sha256"),
        build_timestamp=require_text(payload, "build_timestamp"),
        model_id=require_text(payload, "model_id"),
        raw_revision=optional_text(payload, "raw_revision"),
        build_revision=require_text(payload, "build_revision"),
        requested_revision=require_text(payload, "requested_revision"),
        lineage_basis=require_choice(
            payload, "lineage_basis", ("pinned_revision", "loading_content_attestation")
        ),
        build_snapshot=_attestation_snapshot(payload, "build_snapshot"),
        requested_snapshot=_attestation_snapshot(payload, "requested_snapshot"),
        ignored_documentation_paths=require_strings(
            payload, "ignored_documentation_paths"
        ),
        metadata_normalizations=_metadata_corrections(payload),
        normalized_files=_normalized_files(payload),
        loading_content_sha256=require_text(payload, "loading_content_sha256"),
        attestation_sha256=require_text(payload, "attestation_sha256"),
        schema_version=require_text(payload, "schema_version"),
        kind=require_choice(payload, "kind", ("model_content",)),
    )


def _attestation_snapshot(payload: dict[str, object], key: str) -> RepoSnapshot:
    value = require_object(payload, key)
    require_keys(value, RepoSnapshot, "repository snapshot")
    files = require_list(value, "files")
    return RepoSnapshot(
        require_text(value, "revision"), tuple(_repo_file(item) for item in files)
    )


def _repo_file(value: object) -> RepoFile:
    if type(value) is not dict:
        raise ValueError("Repository file must be an object")
    require_keys(value, RepoFile, "repository file")
    index = value["index_payload"]
    if index is not None and type(index) is not dict:
        raise ValueError("Repository index payload must be an object or null")
    return RepoFile(
        require_text(value, "path"),
        require_integer(value, "size"),
        require_text(value, "git_oid"),
        optional_text(value, "lfs_sha256"),
        index,
    )


def _normalized_files(payload: dict[str, object]) -> tuple[NormalizedLoadingFile, ...]:
    result: list[NormalizedLoadingFile] = []
    for value in require_list(payload, "normalized_files"):
        if type(value) is not dict:
            raise ValueError("Normalized loading file must be an object")
        require_keys(value, NormalizedLoadingFile, "normalized loading file")
        result.append(
            NormalizedLoadingFile(
                require_text(value, "path"),
                require_integer(value, "size"),
                require_choice(
                    value,
                    "kind",
                    ("git_blob", "lfs_object", "safetensors_index"),
                ),
                require_text(value, "content_id"),
            )
        )
    return tuple(result)


def _metadata_corrections(
    payload: dict[str, object],
) -> tuple[TotalParametersCorrection, ...]:
    result: list[TotalParametersCorrection] = []
    for value in require_list(payload, "metadata_normalizations"):
        if type(value) is not dict:
            raise ValueError("Metadata correction must be an object")
        require_keys(value, TotalParametersCorrection, "metadata correction")
        result.append(
            TotalParametersCorrection(
                require_integer(value, "build_value"),
                require_integer(value, "requested_value"),
            )
        )
    return tuple(result)
