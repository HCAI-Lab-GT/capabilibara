"""Build, load, hash, and validate TrackStar preconditioner lineage."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Literal, cast

import torch

type SourceRole = Literal["value", "query"]


@dataclass(frozen=True)
class ArtifactRef:
    role: SourceRole
    artifact_id: str
    path: str
    durable_uri: str
    sha256: str
    byte_count: int
    row_count: int


@dataclass(frozen=True)
class ArtifactSourceManifest:
    schema_version: str
    sources: tuple[ArtifactRef, ...]


@dataclass(frozen=True)
class PreconditionerBuildRecord:
    artifact_id: str
    artifact_uri: str
    build_timestamp: str
    model_id: str
    raw_model_revision: str | None
    bergson_version: str
    transformers_version: str
    bergson_code_revision: str | None
    schema_version: str = "1"
    kind: Literal["trackstar_preconditioner_build"] = "trackstar_preconditioner_build"


@dataclass(frozen=True)
class RootArtifactFile:
    path: str
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class ModuleShape:
    module_name: str
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))


@dataclass(frozen=True)
class PreconditionerSettings:
    precision: str
    loss_fn: str
    loss_reduction: str
    label_smoothing: float
    truncation: bool
    normalizer: str
    token_batch_size: int
    projection_dim: int | None
    projection_type: Literal["normal", "rademacher"]
    reshape_to_square: bool
    include_bias: bool
    unit_normalize: bool
    aggregation: Literal["none"]
    normalize_aggregated_grad: bool


@dataclass(frozen=True)
class PreconditionerAttestation:
    artifact_id: str
    artifact_uri: str
    artifact_root: str
    artifact_sha256: str
    invariant_config_sha256: str
    invariant_build_contract_sha256: str
    build_timestamp: str
    bergson_version: str
    transformers_version: str
    bergson_code_revision: str | None
    model_id: str
    raw_model_revision: str | None
    lineage_basis: LineageBasis
    model_attestation_sha256: str
    source_refs: tuple[ArtifactRef, ...]
    settings: PreconditionerSettings
    mixing_policy: Literal["bergson.utils.math.compute_lambda"]
    target_downweight_components: int
    mixing_coefficient: float
    module_shapes: tuple[ModuleShape, ...]
    value_tensor_sha256: str
    query_tensor_sha256: str
    mixed_tensor_sha256: str
    files: tuple[RootArtifactFile, ...]
    attestation_sha256: str
    schema_version: str = "1"
    kind: Literal["preconditioner"] = "preconditioner"

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_refs", tuple(self.source_refs))
        object.__setattr__(self, "module_shapes", tuple(self.module_shapes))
        object.__setattr__(self, "files", tuple(self.files))


from data_attribution.attribution.trackstar.lineage_records import (  # noqa: E402
    LineageBasis,
    ModelContentAttestation,
    validate_hex,
)
from data_attribution.attribution.trackstar.lineage_model import (  # noqa: E402
    digest_stable_file,
    has_symlink_component,
    inspect_stable_jsonl,
    model_attestation_sha256,
    open_stable_regular_file,
    optional_integer,
    optional_text,
    require_bool,
    require_choice,
    require_float,
    require_integer,
    require_keys,
    require_list,
    require_object,
    require_text,
    validate_model_attestation,
)
from data_attribution.artifact_integrity import (  # noqa: E402
    canonical_json_bytes,
    jsonable,
    parse_json_object,
    sha256_json,
)

PROCESSOR_KEYS = {
    "projection_dim",
    "reshape_to_square",
    "projection_type",
    "include_bias",
}
CONTROLLED_PREPROCESS = {
    "unit_normalize": False,
    "preconditioner_path": None,
    "aggregation": "none",
    "normalize_aggregated_grad": False,
}
MIX_KEYS = {
    "query_path",
    "index_path",
    "mixing_coefficient",
    "target_downweight_components",
}
PIPELINE_REQUIRED = {
    "model",
    "value_data",
    "query_data",
    "target_downweight_components",
    "value_preconditioner",
    "query_preconditioner",
    "mixed_preconditioner",
}
INDEX_REQUIRED = {
    "run_path",
    "data",
    "model",
    "revision",
    "precision",
    "projection_dim",
    "token_batch_size",
    "normalizer",
    "skip_preconditioners",
    "skip_index",
    "loss_fn",
    "loss_reduction",
    "label_smoothing",
}
FINAL_RECORD_NAMES = ("lineage_build.json", "pipeline_config.json")
REQUIRED_PRECONDITIONER_FILES = tuple(
    f"{role}_preconditioner/{name}"
    for role, names in {
        "value": (
            "index_config.json",
            "preprocess_config.json",
            "processor_config.json",
            "normalizers.pth",
            "preconditioners.pth",
            "preconditioners_eigen.pth",
        ),
        "query": (
            "index_config.json",
            "preprocess_config.json",
            "processor_config.json",
            "normalizers.pth",
            "preconditioners.pth",
            "preconditioners_eigen.pth",
        ),
        "mixed": (
            "processor_config.json",
            "normalizers.pth",
            "preconditioners.pth",
            "preconditioners_eigen.pth",
            "mix_config.json",
        ),
    }.items()
    for name in names
)


def build_preconditioner_settings(index, processor) -> PreconditionerSettings:
    data = index["data"]
    if type(data) is not dict:
        raise ValueError("Invalid index data config")
    projection = processor["projection_type"]
    if projection not in ("normal", "rademacher"):
        raise ValueError("Invalid processor projection type")
    values = PreconditionerSettings(
        precision=index["precision"],
        loss_fn=index["loss_fn"],
        loss_reduction=index["loss_reduction"],
        label_smoothing=index["label_smoothing"],
        truncation=data["truncation"],
        normalizer=index["normalizer"],
        token_batch_size=index["token_batch_size"],
        projection_dim=processor["projection_dim"],
        projection_type=projection,
        reshape_to_square=processor["reshape_to_square"],
        include_bias=processor["include_bias"],
        unit_normalize=False,
        aggregation="none",
        normalize_aggregated_grad=False,
    )
    _validate_controlled_settings(index, values)
    return cast(PreconditionerSettings, values)


def _validate_controlled_settings(index, values: PreconditionerSettings) -> None:
    projection = values.projection_dim
    exact = (
        type(values.precision) is str,
        type(values.loss_fn) is str,
        type(values.loss_reduction) is str,
        type(values.label_smoothing) is float,
        type(values.truncation) is bool,
        type(values.token_batch_size) is int and values.token_batch_size > 0,
        values.normalizer == "none",
        type(projection) is int and projection > 0,
        index["projection_dim"] == values.projection_dim,
    )
    if not all(exact):
        raise ValueError("Invalid controlled preconditioner settings")
    for key in ("projection_type", "reshape_to_square", "include_bias"):
        if key in index and index[key] != getattr(values, key):
            raise ValueError("Processor and index settings are inconsistent")


def normalized_config_sha256(
    pipeline: dict[str, object],
    indices: dict[str, dict[str, object]],
    preprocess: dict[str, object],
    processors: dict[str, dict[str, object]],
    mix: dict[str, object],
) -> str:
    if not indices_match(indices["value"], indices["query"]):
        raise ValueError("Index build settings differ across preconditioners")
    normalized_pipeline = dict(pipeline)
    normalized_pipeline.update(
        model="$MODEL", value_data="$SOURCE/value", query_data="$SOURCE/query"
    )
    for role in ("value", "query", "mixed"):
        normalized_pipeline[f"{role}_preconditioner"] = f"$ROOT/{role}_preconditioner"
    normalized_mix = {
        **mix,
        "query_path": "$ROOT/query_preconditioner",
        "index_path": "$ROOT/value_preconditioner",
    }
    normalized_mix.pop("mixing_coefficient")
    payload = {
        "pipeline": normalized_pipeline,
        "index": _normalize_index(indices["value"]),
        "preprocess": preprocess,
        "processors": processors,
        "mix": normalized_mix,
    }
    return sha256_json(payload)


def indices_match(left: dict[str, object], right: dict[str, object]) -> bool:
    if left.get("tokenizer") != right.get("tokenizer"):
        return False
    return _normalize_index(left) == _normalize_index(right)


def _normalize_index(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result.update(run_path="$ROOT", model="$MODEL", revision="$REVISION")
    if "tokenizer" in result:
        result["tokenizer"] = "$TOKENIZER"
    data = payload["data"]
    if type(data) is not dict:
        raise ValueError("Invalid index data config")
    result["data"] = {**data, "data_args": "$SOURCE"}
    return result


@dataclass(frozen=True)
class RootArtifactReader:
    root: Path
    files: tuple[RootArtifactFile, ...]

    @contextmanager
    def open_verified(self, relative: str) -> Iterator[BinaryIO]:
        expected = self._expected(relative)
        path = self.root / relative
        with open_stable_regular_file(path) as stream:
            byte_count, sha256 = _digest_stream(stream)
            if (byte_count, sha256) != (expected.byte_count, expected.sha256):
                raise ValueError(
                    f"Root artifact bytes differ from initial inventory: {relative}"
                )
            stream.seek(0)
            yield stream

    def read_text(self, relative: str) -> str:
        with self.open_verified(relative) as stream:
            content = stream.read()
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"Root artifact is not UTF-8: {relative}") from error

    def _expected(self, relative: str) -> RootArtifactFile:
        matches = tuple(item for item in self.files if item.path == relative)
        if len(matches) != 1:
            raise ValueError(
                f"Root artifact is absent from initial inventory: {relative}"
            )
        return matches[0]


def _digest_stream(stream: BinaryIO) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_count = 0
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
        byte_count += len(chunk)
    return byte_count, digest.hexdigest()


def load_preconditioner_config(reader, sources, build):
    root = reader.root
    by_role = {source.role: source for source in sources}
    pipeline = _required_json(
        reader, _record_relative(reader, "pipeline_config.json"), PIPELINE_REQUIRED
    )
    processors = {
        role: _exact_json(
            reader, f"{role}_preconditioner/processor_config.json", PROCESSOR_KEYS
        )
        for role in ("value", "query", "mixed")
    }
    if len({sha256_json(value) for value in processors.values()}) != 1:
        raise ValueError("Processor settings differ across preconditioners")
    indices = {
        role: _load_index(reader, role, by_role[role], build)
        for role in ("value", "query")
    }
    if not indices_match(indices["value"], indices["query"]):
        raise ValueError("Index build settings differ across preconditioners")
    preprocesses = {role: _load_preprocess(reader, role) for role in ("value", "query")}
    mix = _exact_json(reader, "mixed_preconditioner/mix_config.json", MIX_KEYS)
    settings = build_preconditioner_settings(indices["value"], processors["value"])
    target, coefficient = _validate_paths(root, pipeline, mix, by_role, build)
    config_hash = normalized_config_sha256(
        pipeline, indices, preprocesses["value"], processors, mix
    )
    return settings, target, coefficient, config_hash


def _load_index(
    reader: RootArtifactReader,
    role: str,
    source: ArtifactRef,
    build: PreconditionerBuildRecord,
) -> dict[str, object]:
    root = reader.root
    relative = f"{role}_preconditioner/index_config.json"
    payload = _required_json(reader, relative, INDEX_REQUIRED)
    data = payload["data"]
    if type(data) is not dict or not {"truncation", "data_args"} <= set(data):
        raise ValueError("Invalid index data config")
    expected = (
        str(root / f"{role}_preconditioner"),
        build.model_id,
        build.raw_model_revision,
        f"data_files={source.path}",
    )
    actual = (
        payload["run_path"],
        payload["model"],
        payload["revision"],
        data["data_args"],
    )
    if actual != expected or payload["skip_index"] is not True:
        raise ValueError("Index source or build identity is inconsistent")
    if payload["skip_preconditioners"] is not False:
        raise ValueError("Index preconditioner build settings are inconsistent")
    return payload


def _load_preprocess(reader: RootArtifactReader, role: str) -> dict[str, object]:
    relative = f"{role}_preconditioner/preprocess_config.json"
    payload = _exact_json(reader, relative, set(CONTROLLED_PREPROCESS))
    for key, expected in CONTROLLED_PREPROCESS.items():
        observed = payload[key]
        if type(observed) is not type(expected) or observed != expected:
            raise ValueError("Preprocess settings are not controlled")
    return payload


def _validate_paths(root, pipeline, mix, sources, build) -> tuple[int, float]:
    expected = {
        "model": build.model_id,
        "value_data": sources["value"].path,
        "query_data": sources["query"].path,
        **{
            f"{role}_preconditioner": str(root / f"{role}_preconditioner")
            for role in ("value", "query", "mixed")
        },
    }
    if any(pipeline[key] != value for key, value in expected.items()):
        raise ValueError("Pipeline paths or model are inconsistent")
    paths = (mix["query_path"], mix["index_path"])
    if paths != (expected["query_preconditioner"], expected["value_preconditioner"]):
        raise ValueError("Mix paths are inconsistent")
    target = _integer(mix, "target_downweight_components")
    pipeline_target = pipeline["target_downweight_components"]
    if type(pipeline_target) is not int or pipeline_target <= 0:
        raise ValueError("Invalid pipeline target downweight components")
    if pipeline_target != target:
        raise ValueError("Mix target is inconsistent")
    return target, _coefficient(mix)


def _exact_json(
    reader: RootArtifactReader, relative: str, keys: set[str]
) -> dict[str, object]:
    path = reader.root / relative
    payload = parse_json_object(reader.read_text(relative), path, 1)
    if set(payload) != keys:
        raise ValueError(f"Invalid {path.name} fields")
    return payload


def _required_json(
    reader: RootArtifactReader, relative: str, keys: set[str]
) -> dict[str, object]:
    path = reader.root / relative
    payload = parse_json_object(reader.read_text(relative), path, 1)
    if not keys <= set(payload):
        raise ValueError(f"Invalid {path.name} fields")
    return payload


def _integer(payload: dict[str, object], key: str) -> int:
    value = payload[key]
    if type(value) is not int or value <= 0:
        raise ValueError(f"Invalid integer field: {key}")
    return value


def _coefficient(payload: dict[str, object]) -> float:
    value = payload["mixing_coefficient"]
    if type(value) is not float or not 0.0 <= value <= 1.0:
        raise ValueError("Invalid mixing coefficient")
    return value


def build_root_inventory(root: Path) -> tuple[RootArtifactFile, ...]:
    files: list[RootArtifactFile] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Preconditioner root contains a special file: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Preconditioner root contains a special file: {path}")
        evidence = digest_stable_file(path)
        files.append(
            RootArtifactFile(
                path.relative_to(root).as_posix(),
                evidence.byte_count,
                evidence.sha256,
            )
        )
    return tuple(files)


def root_artifact_sha256(root: str, files: tuple[RootArtifactFile, ...]) -> str:
    digest = hashlib.sha256()
    _update_part(digest, b"trackstar-preconditioner-tree-v1")
    _update_part(digest, root.encode())
    for item in files:
        _update_part(digest, item.path.encode())
        _update_part(digest, str(item.byte_count).encode())
        _update_part(digest, bytes.fromhex(item.sha256))
    return digest.hexdigest()


def invariant_contract_sha256(
    build: PreconditionerBuildRecord,
    sources: tuple[ArtifactRef, ...],
    settings: PreconditionerSettings,
    target: int,
    shapes: tuple[ModuleShape, ...],
    config_sha256: str,
) -> str:
    payload = {
        "bergson_version": build.bergson_version,
        "transformers_version": build.transformers_version,
        "bergson_code_revision": build.bergson_code_revision,
        "sources": _source_claims(sources),
        "settings": settings,
        "mixing_policy": "bergson.utils.math.compute_lambda",
        "target_downweight_components": target,
        "module_shapes": shapes,
        "normalized_config_sha256": config_sha256,
    }
    return sha256_json(payload)


def state_sha256(value: object) -> str:
    digest = hashlib.sha256()
    _update_state(digest, value)
    return digest.hexdigest()


def _source_claims(sources: tuple[ArtifactRef, ...]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "role": item.role,
            "artifact_id": item.artifact_id,
            "durable_uri": item.durable_uri,
            "sha256": item.sha256,
            "byte_count": item.byte_count,
            "row_count": item.row_count,
        }
        for item in sources
    )


def _update_state(digest, value: object) -> None:
    if isinstance(value, torch.Tensor):
        _update_tensor(digest, value)
    elif isinstance(value, Mapping):
        _update_mapping(digest, value)
    elif isinstance(value, (tuple, list)):
        _update_part(digest, b"sequence")
        for item in value:
            _update_state(digest, item)
    else:
        _update_part(digest, canonical_json_bytes(value))


def _update_tensor(digest, value: torch.Tensor) -> None:
    if not value.isfinite().all().item():
        raise ValueError("Tensor state must be finite")
    _update_part(digest, b"tensor")
    _update_part(digest, str(value.dtype).encode())
    _update_part(digest, canonical_json_bytes(tuple(value.shape)))
    data = bytes(value.detach().cpu().contiguous().clone().untyped_storage())
    _update_part(digest, data)


def _update_mapping(digest, value: Mapping[object, object]) -> None:
    _update_part(digest, b"mapping")
    if any(type(key) is not str for key in value):
        raise ValueError("Tensor state mapping keys must be strings")
    keys = cast(list[str], list(value))
    for key in sorted(keys):
        _update_part(digest, key.encode())
        _update_state(digest, value[key])


def _update_part(digest, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


_EIGEN_RTOL = 1e-4
_EIGEN_ATOL = 1e-6


@dataclass(frozen=True)
class TensorEvidence:
    module_shapes: tuple[ModuleShape, ...]
    value_sha256: str
    query_sha256: str
    mixed_sha256: str


def verify_preconditioner_tensors(
    reader: RootArtifactReader, target: int, recorded_coefficient: float
) -> TensorEvidence:
    raw = {
        role: _load_mapping(reader, f"{role}_preconditioner/preconditioners.pth")
        for role in ("value", "query", "mixed")
    }
    shapes = _validate_raw_mappings(raw)
    eigen = {
        role: _load_mapping(reader, f"{role}_preconditioner/preconditioners_eigen.pth")
        for role in ("value", "query", "mixed")
    }
    value_eigen = _validate_eigen(eigen["value"], raw["value"], "value")
    query_eigen = _validate_eigen(eigen["query"], raw["query"], "query")
    if eigen["mixed"]:
        raise ValueError("Mixed preconditioner eigensystem must be empty")
    _validate_coefficient(value_eigen, query_eigen, target, recorded_coefficient)
    _validate_mixed(raw, recorded_coefficient)
    _validate_normalizers(reader)
    return TensorEvidence(
        shapes,
        state_sha256(raw["value"]),
        state_sha256(raw["query"]),
        state_sha256(raw["mixed"]),
    )


def _load_mapping(reader: RootArtifactReader, relative: str) -> dict[str, object]:
    with reader.open_verified(relative) as stream:
        value = torch.load(stream, map_location="cpu", weights_only=True)
    path = reader.root / relative
    if not isinstance(value, Mapping):
        raise ValueError(f"Tensor artifact must contain a mapping: {path}")
    if any(type(key) is not str or not key for key in value):
        raise ValueError(f"Tensor artifact has invalid module keys: {path}")
    return dict(value)


def _validate_raw_mappings(
    values: dict[str, dict[str, object]],
) -> tuple[ModuleShape, ...]:
    keys = {role: set(mapping) for role, mapping in values.items()}
    if not keys["value"] or len({frozenset(item) for item in keys.values()}) != 1:
        raise ValueError("Preconditioner module keys differ or are empty")
    shapes: list[ModuleShape] = []
    for name in sorted(keys["value"]):
        tensors = tuple(values[role][name] for role in ("value", "query", "mixed"))
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise ValueError("Preconditioner entries must be tensors")
        typed = cast(tuple[torch.Tensor, ...], tensors)
        _validate_tensor_group(typed)
        tensor = typed[0]
        shapes.append(ModuleShape(name, tuple(tensor.shape), str(tensor.dtype)))
    return tuple(shapes)


def _validate_tensor_group(tensors: tuple[torch.Tensor, ...]) -> None:
    first = tensors[0]
    valid = (
        first.ndim == 2
        and first.shape[0] == first.shape[1]
        and first.is_floating_point()
    )
    if not valid:
        raise ValueError("Preconditioner tensors must be square floating matrices")
    if any(
        tensor.shape != first.shape or tensor.dtype != first.dtype for tensor in tensors
    ):
        raise ValueError("Preconditioner module tensor shape or dtype differs")
    if any(not tensor.isfinite().all().item() for tensor in tensors):
        raise ValueError("Preconditioner tensors must be finite")


def _validate_eigen(
    eigen: dict[str, object], raw: dict[str, object], role: str
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    if set(eigen) != set(raw):
        raise ValueError(f"{role} eigensystem module keys differ")
    result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, value in eigen.items():
        result[name] = _eigen_pair(value, raw[name], role)
    return result


def _eigen_pair(
    value: object, raw: object, role: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(raw, torch.Tensor):
        raise ValueError("Preconditioner entries must be tensors")
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"Malformed {role} eigensystem")
    eigenvalues, eigenvectors = value
    if not isinstance(eigenvalues, torch.Tensor) or not isinstance(
        eigenvectors, torch.Tensor
    ):
        raise ValueError(f"Malformed {role} eigensystem")
    if eigenvalues.shape != (raw.shape[0],) or eigenvectors.shape != raw.shape:
        raise ValueError(f"Malformed {role} eigensystem shape")
    if not eigenvalues.isfinite().all() or not eigenvectors.isfinite().all():
        raise ValueError(f"Nonfinite {role} eigensystem")
    if not eigenvalues.is_floating_point() or not eigenvectors.is_floating_point():
        raise ValueError(f"Malformed {role} eigensystem dtype")
    _validate_eigenbasis(eigenvalues, eigenvectors, role)
    reconstructed = (eigenvectors * eigenvalues.unsqueeze(0)) @ eigenvectors.mT
    if not reconstructed.isclose(raw, rtol=_EIGEN_RTOL, atol=_EIGEN_ATOL).all().item():
        raise ValueError(f"{role} eigensystem does not reconstruct raw preconditioner")
    return eigenvalues, eigenvectors


def _validate_eigenbasis(
    eigenvalues: torch.Tensor, eigenvectors: torch.Tensor, role: str
) -> None:
    scale = max(eigenvalues.abs().max().item(), 1.0)
    tolerance = _EIGEN_ATOL * scale
    if eigenvalues.min().item() < -tolerance:
        raise ValueError(f"{role} eigensystem has negative eigenvalues")
    dimension = eigenvalues.numel()
    identity = eigenvalues.new_zeros((dimension, dimension))
    identity.diagonal().fill_(1)
    gram = eigenvectors.mT @ eigenvectors
    if not gram.isclose(identity, rtol=_EIGEN_RTOL, atol=_EIGEN_ATOL).all().item():
        raise ValueError(f"{role} eigensystem vectors are not orthonormal")


def _validate_coefficient(value_eigen, query_eigen, target, recorded) -> None:
    from bergson.utils.math import compute_lambda  # pyright: ignore[reportMissingImports]

    total = sum(pair[0].numel() for pair in query_eigen.values())
    if type(target) is not int or not 1 <= target <= total:
        raise ValueError("Invalid target downweight component count")
    computed = compute_lambda(
        query_eigen=query_eigen,
        index_eigen=value_eigen,
        target_components=target,
    )
    if not math.isfinite(recorded) or recorded != computed:
        raise ValueError("Recorded mixing coefficient does not match compute_lambda")


def _validate_mixed(values, coefficient: float) -> None:
    for name, mixed in values["mixed"].items():
        expected = values["query"][name] * coefficient + values["value"][name] * (
            1 - coefficient
        )
        if not mixed.equal(expected):
            raise ValueError("Mixed preconditioner tensor does not match coefficient")


def _validate_normalizers(reader: RootArtifactReader) -> None:
    states = {
        role: _load_mapping(reader, f"{role}_preconditioner/normalizers.pth")
        for role in ("value", "query", "mixed")
    }
    query_hash = state_sha256(states["query"])
    if state_sha256(states["mixed"]) != query_hash:
        raise ValueError("Mixed normalizers do not match query normalizers")
    state_sha256(states["value"])


def load_artifact_refs(path: Path) -> tuple[ArtifactRef, ...]:
    payload = parse_json_object(path.read_text(encoding="utf-8"), path, 1)
    require_keys(payload, ArtifactSourceManifest, "artifact source manifest")
    if require_text(payload, "schema_version") != "1":
        raise ValueError("Unsupported artifact source manifest schema")
    sources = tuple(
        _decode_source(value, path.parent) for value in require_list(payload, "sources")
    )
    if len(sources) != 2 or {item.role for item in sources} != {"value", "query"}:
        raise ValueError("Artifact source manifest requires value and query sources")
    return tuple(sorted(sources, key=lambda item: item.role))


def decode_preconditioner_attestation(
    payload: dict[str, object],
) -> PreconditionerAttestation:
    require_keys(payload, PreconditionerAttestation, "preconditioner attestation")
    values = _decode_identity(payload)
    values.update(_decode_evidence(payload))
    return PreconditionerAttestation(**values)  # type: ignore[arg-type]


def _decode_identity(payload: dict[str, object]) -> dict[str, object]:
    basis = cast(
        LineageBasis,
        require_choice(
            payload,
            "lineage_basis",
            ("pinned_revision", "loading_content_attestation"),
        ),
    )
    return {
        "artifact_id": require_text(payload, "artifact_id"),
        "artifact_uri": require_text(payload, "artifact_uri"),
        "artifact_root": require_text(payload, "artifact_root"),
        "artifact_sha256": require_text(payload, "artifact_sha256"),
        "invariant_config_sha256": require_text(payload, "invariant_config_sha256"),
        "invariant_build_contract_sha256": require_text(
            payload, "invariant_build_contract_sha256"
        ),
        "build_timestamp": require_text(payload, "build_timestamp"),
        "bergson_version": require_text(payload, "bergson_version"),
        "transformers_version": require_text(payload, "transformers_version"),
        "bergson_code_revision": optional_text(payload, "bergson_code_revision"),
        "model_id": require_text(payload, "model_id"),
        "raw_model_revision": optional_text(payload, "raw_model_revision"),
        "lineage_basis": basis,
        "model_attestation_sha256": require_text(payload, "model_attestation_sha256"),
    }


def _decode_evidence(payload: dict[str, object]) -> dict[str, object]:
    policy = cast(
        Literal["bergson.utils.math.compute_lambda"],
        require_choice(
            payload,
            "mixing_policy",
            ("bergson.utils.math.compute_lambda",),
        ),
    )
    kind = cast(
        Literal["preconditioner"],
        require_choice(payload, "kind", ("preconditioner",)),
    )
    return {
        "source_refs": _decode_sources(payload),
        "settings": _decode_settings(require_object(payload, "settings")),
        "mixing_policy": policy,
        "target_downweight_components": require_integer(
            payload, "target_downweight_components"
        ),
        "mixing_coefficient": require_float(payload, "mixing_coefficient"),
        "module_shapes": _decode_shapes(payload),
        "value_tensor_sha256": require_text(payload, "value_tensor_sha256"),
        "query_tensor_sha256": require_text(payload, "query_tensor_sha256"),
        "mixed_tensor_sha256": require_text(payload, "mixed_tensor_sha256"),
        "files": _decode_files(payload),
        "attestation_sha256": require_text(payload, "attestation_sha256"),
        "schema_version": require_text(payload, "schema_version"),
        "kind": kind,
    }


def _decode_sources(payload: dict[str, object]) -> tuple[ArtifactRef, ...]:
    return tuple(
        _decode_source(value) for value in require_list(payload, "source_refs")
    )


def _decode_source(value: object, base: Path | None = None) -> ArtifactRef:
    if type(value) is not dict:
        raise ValueError("Artifact source must be an object")
    require_keys(value, ArtifactRef, "artifact source")
    raw_path = Path(require_text(value, "path"))
    path = raw_path if base is None or raw_path.is_absolute() else (base / raw_path)
    return ArtifactRef(
        role=cast(SourceRole, require_choice(value, "role", ("value", "query"))),
        artifact_id=require_text(value, "artifact_id"),
        path=str(path.absolute()),
        durable_uri=require_text(value, "durable_uri"),
        sha256=require_text(value, "sha256"),
        byte_count=require_integer(value, "byte_count"),
        row_count=require_integer(value, "row_count"),
    )


def _decode_settings(payload: dict[str, object]) -> PreconditionerSettings:
    require_keys(payload, PreconditionerSettings, "preconditioner settings")
    return PreconditionerSettings(
        precision=require_text(payload, "precision"),
        loss_fn=require_text(payload, "loss_fn"),
        loss_reduction=require_text(payload, "loss_reduction"),
        label_smoothing=require_float(payload, "label_smoothing"),
        truncation=require_bool(payload, "truncation"),
        normalizer=require_text(payload, "normalizer"),
        token_batch_size=require_integer(payload, "token_batch_size"),
        projection_dim=optional_integer(payload, "projection_dim"),
        projection_type=cast(
            Literal["normal", "rademacher"],
            require_choice(payload, "projection_type", ("normal", "rademacher")),
        ),
        reshape_to_square=require_bool(payload, "reshape_to_square"),
        include_bias=require_bool(payload, "include_bias"),
        unit_normalize=require_bool(payload, "unit_normalize"),
        aggregation=cast(
            Literal["none"], require_choice(payload, "aggregation", ("none",))
        ),
        normalize_aggregated_grad=require_bool(payload, "normalize_aggregated_grad"),
    )


def _decode_shapes(payload: dict[str, object]) -> tuple[ModuleShape, ...]:
    result: list[ModuleShape] = []
    for value in require_list(payload, "module_shapes"):
        if type(value) is not dict:
            raise ValueError("Module shape must be an object")
        require_keys(value, ModuleShape, "module shape")
        shape = require_list(value, "shape")
        if any(type(item) is not int for item in shape):
            raise ValueError("Module shape dimensions must be integers")
        result.append(
            ModuleShape(
                require_text(value, "module_name"),
                tuple(shape),  # type: ignore[arg-type]
                require_text(value, "dtype"),
            )
        )
    return tuple(result)


def _decode_files(payload: dict[str, object]) -> tuple[RootArtifactFile, ...]:
    result: list[RootArtifactFile] = []
    for value in require_list(payload, "files"):
        if type(value) is not dict:
            raise ValueError("Root artifact file must be an object")
        require_keys(value, RootArtifactFile, "root artifact file")
        result.append(
            RootArtifactFile(
                require_text(value, "path"),
                require_integer(value, "byte_count"),
                require_text(value, "sha256"),
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class VerifiedPreconditionerContract:
    root: Path
    root_files: tuple[RootArtifactFile, ...]
    reader: RootArtifactReader
    sources: tuple[ArtifactRef, ...]
    build: PreconditionerBuildRecord
    settings: PreconditionerSettings
    target_components: int
    mixing_coefficient: float
    normalized_config_sha256: str


def verify_preconditioner_contract(
    root: Path,
    source_refs: tuple[ArtifactRef, ...],
    model: ModelContentAttestation,
) -> VerifiedPreconditionerContract:
    root = _validate_root(root)
    root_files = build_root_inventory(root)
    reader = RootArtifactReader(root, root_files)
    sources = _validate_contract_sources(source_refs)
    build = _load_build(reader, model)
    settings, target, coefficient, config_hash = load_preconditioner_config(
        reader, sources, build
    )
    return VerifiedPreconditionerContract(
        root,
        root_files,
        reader,
        sources,
        build,
        settings,
        target,
        coefficient,
        config_hash,
    )


def _validate_root(root: Path) -> Path:
    if has_symlink_component(root) or not root.is_dir():
        raise ValueError("Preconditioner root must be a regular directory")
    root = root.resolve()
    for item in root.rglob("*"):
        if item.is_symlink() or not (item.is_dir() or item.is_file()):
            raise ValueError(f"Preconditioner root contains a special file: {item}")
    missing = [
        name for name in REQUIRED_PRECONDITIONER_FILES if not (root / name).is_file()
    ]
    if missing:
        raise ValueError(f"Missing required preconditioner artifact: {missing[0]}")
    _record_prefix(root)
    return root


def _validate_contract_sources(
    refs: tuple[ArtifactRef, ...],
) -> tuple[ArtifactRef, ...]:
    if type(refs) is not tuple or any(type(ref) is not ArtifactRef for ref in refs):
        raise ValueError("Preconditioner sources require immutable ArtifactRef records")
    if len(refs) != 2 or {ref.role for ref in refs} != {"value", "query"}:
        raise ValueError("Preconditioner requires one value and one query source")
    if len({ref.artifact_id for ref in refs}) != 2:
        raise ValueError("Preconditioner source artifact IDs must be unique")
    for ref in refs:
        _validate_source(ref)
    return tuple(sorted(refs, key=lambda ref: ref.role))


def _validate_source(ref: ArtifactRef) -> None:
    text_values = (ref.artifact_id, ref.path, ref.durable_uri)
    if any(type(value) is not str or not value for value in text_values):
        raise ValueError("Preconditioner source requires string identity and path")
    path = Path(ref.path)
    if not path.is_absolute():
        raise ValueError("Preconditioner source path must be absolute")
    if has_symlink_component(path) or not path.is_file():
        raise ValueError("Preconditioner source must be a regular non-symlink file")
    if not ref.artifact_id.strip() or "://" not in ref.durable_uri:
        raise ValueError("Preconditioner source requires durable identity")
    validate_hex(ref.sha256, 64, "source SHA-256")
    evidence = inspect_stable_jsonl(path)
    if type(ref.byte_count) is not int or ref.byte_count != evidence.byte_count:
        raise ValueError("Invalid source byte count")
    if evidence.sha256 != ref.sha256:
        raise ValueError("Invalid source SHA-256")
    if (
        type(ref.row_count) is not int
        or ref.row_count != evidence.row_count
        or evidence.row_count == 0
    ):
        raise ValueError("Invalid source row count")


def _load_build(
    reader: RootArtifactReader,
    model: ModelContentAttestation,
) -> PreconditionerBuildRecord:
    relative = _record_relative(reader, "lineage_build.json")
    path = reader.root / relative
    payload = parse_json_object(reader.read_text(relative), path, 1)
    expected = {item.name for item in fields(PreconditionerBuildRecord)}
    if set(payload) != expected:
        raise ValueError("Invalid lineage_build.json fields")
    build = PreconditionerBuildRecord(**payload)  # type: ignore[arg-type]
    _validate_build(build, model)
    return build


def _record_prefix(root: Path) -> str:
    legacy = tuple((root / name).is_file() for name in FINAL_RECORD_NAMES)
    bundled = tuple(
        (root / "mixed_preconditioner" / name).is_file() for name in FINAL_RECORD_NAMES
    )
    if all(legacy) and not any(bundled):
        return ""
    if all(bundled) and not any(legacy):
        return "mixed_preconditioner/"
    raise ValueError(
        "Preconditioner required final records are incomplete or ambiguous"
    )


def _record_relative(reader: RootArtifactReader, name: str) -> str:
    prefix = _record_prefix(reader.root)
    relative = f"{prefix}{name}"
    if not any(item.path == relative for item in reader.files):
        raise ValueError(f"Missing required preconditioner artifact: {relative}")
    return relative


def _validate_build(
    build: PreconditionerBuildRecord, model: ModelContentAttestation
) -> None:
    values = (
        build.artifact_id,
        build.artifact_uri,
        build.model_id,
        build.bergson_version,
        build.transformers_version,
    )
    if any(type(value) is not str or not value.strip() for value in values):
        raise ValueError("Invalid lineage build identity")
    if "://" not in build.artifact_uri:
        raise ValueError("Lineage build requires a durable artifact URI")
    if build.schema_version != "1" or build.kind != "trackstar_preconditioner_build":
        raise ValueError("Invalid lineage build schema or kind")
    _validate_model_binding(build, model)
    _validate_revisions(build)
    _validate_build_timestamp(build.build_timestamp)
    installed = (version("bergson"), version("transformers"))
    if (build.bergson_version, build.transformers_version) != installed:
        raise ValueError("Lineage build versions do not match verifier environment")


def _validate_model_binding(
    build: PreconditionerBuildRecord, model: ModelContentAttestation
) -> None:
    if build.model_id != model.model_id:
        raise ValueError("Lineage build model identity is inconsistent")
    if (
        model.lineage_basis == "pinned_revision"
        and build.raw_model_revision != model.build_revision
    ):
        raise ValueError(
            "Pinned model lineage requires the exact pinned build revision"
        )
    if build.raw_model_revision not in (None, model.build_revision):
        raise ValueError("Lineage build model revision is inconsistent")


def _validate_revisions(build: PreconditionerBuildRecord) -> None:
    for revision, label in (
        (build.raw_model_revision, "raw model revision"),
        (build.bergson_code_revision, "Bergson code revision"),
    ):
        if revision is not None:
            validate_hex(revision, 40, label)


def _validate_build_timestamp(value: object) -> None:
    if type(value) is not str or not value.endswith("Z"):
        raise ValueError("Build timestamp must be UTC ISO-8601")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("Build timestamp must be UTC ISO-8601") from error
    if parsed.tzinfo != UTC:
        raise ValueError("Build timestamp must be UTC ISO-8601")


def validate_preconditioner_records(value: PreconditionerAttestation) -> None:
    if value.lineage_basis not in ("pinned_revision", "loading_content_attestation"):
        raise ValueError("Invalid inherited model lineage basis")
    if value.mixing_policy != "bergson.utils.math.compute_lambda":
        raise ValueError("Invalid preconditioner mixing policy")
    if (
        type(value.target_downweight_components) is not int
        or value.target_downweight_components <= 0
    ):
        raise ValueError("Invalid target downweight components")
    coefficient = value.mixing_coefficient
    if not math.isfinite(coefficient) or not 0 <= coefficient <= 1:
        raise ValueError("Invalid mixing coefficient")
    _validate_identity(value)
    _validate_record_sources(value)
    _validate_shapes(value)
    _validate_files(value)
    _validate_attestation_settings(value)
    _validate_sorted_records(value)


def _validate_identity(value: PreconditionerAttestation) -> None:
    identities = (
        value.artifact_id,
        value.artifact_uri,
        value.artifact_root,
        value.model_id,
        value.bergson_version,
        value.transformers_version,
    )
    if any(type(item) is not str or not item.strip() for item in identities):
        raise ValueError("Invalid preconditioner identity")
    root = PurePosixPath(value.artifact_root)
    if "://" not in value.artifact_uri or not root.is_absolute():
        raise ValueError("Invalid preconditioner artifact location")
    if value.lineage_basis == "pinned_revision" and value.raw_model_revision is None:
        raise ValueError("Pinned lineage requires a raw model revision")
    for revision, label in (
        (value.raw_model_revision, "raw model revision"),
        (value.bergson_code_revision, "Bergson code revision"),
    ):
        if revision is not None:
            validate_hex(revision, 40, label)
    _validate_record_timestamp(value.build_timestamp)


def _validate_record_sources(value: PreconditionerAttestation) -> None:
    sources = value.source_refs
    if len(sources) != 2 or {item.role for item in sources} != {"value", "query"}:
        raise ValueError("Invalid preconditioner source roles")
    if len({item.artifact_id for item in sources}) != 2:
        raise ValueError("Invalid preconditioner source identities")
    for item in sources:
        valid = (
            bool(item.artifact_id.strip())
            and "://" in item.durable_uri
            and bool(item.path)
            and type(item.byte_count) is int
            and item.byte_count >= 0
            and type(item.row_count) is int
            and item.row_count > 0
        )
        if not valid:
            raise ValueError("Invalid preconditioner source record")
        validate_hex(item.sha256, 64, "source SHA-256")


def _validate_shapes(value: PreconditionerAttestation) -> None:
    shapes = value.module_shapes
    if not shapes or len({item.module_name for item in shapes}) != len(shapes):
        raise ValueError("Invalid preconditioner module shapes")
    for item in shapes:
        valid = (
            bool(item.module_name)
            and bool(item.shape)
            and all(type(size) is int and size > 0 for size in item.shape)
            and item.dtype.startswith("torch.")
        )
        if not valid:
            raise ValueError("Invalid preconditioner module shape")


def _validate_files(value: PreconditionerAttestation) -> None:
    files = value.files
    if not files or len({item.path for item in files}) != len(files):
        raise ValueError("Invalid preconditioner file inventory")
    for item in files:
        path = PurePosixPath(item.path)
        valid = (
            bool(item.path)
            and not path.is_absolute()
            and ".." not in path.parts
            and path.as_posix() == item.path
            and type(item.byte_count) is int
            and item.byte_count >= 0
        )
        if not valid:
            raise ValueError("Invalid preconditioner file inventory")
        validate_hex(item.sha256, 64, "root file SHA-256")


def _validate_attestation_settings(value: PreconditionerAttestation) -> None:
    settings = value.settings
    projection = settings.projection_dim
    valid = (
        type(settings.label_smoothing) is float
        and math.isfinite(settings.label_smoothing)
        and type(settings.truncation) is bool
        and type(settings.token_batch_size) is int
        and settings.token_batch_size > 0
        and type(projection) is int
        and projection > 0
        and settings.projection_type in ("normal", "rademacher")
        and type(settings.reshape_to_square) is bool
        and type(settings.include_bias) is bool
        and settings.normalizer == "none"
        and settings.aggregation == "none"
    )
    if not valid:
        raise ValueError("Invalid preconditioner settings")


def _validate_sorted_records(value: PreconditionerAttestation) -> None:
    if (
        tuple(sorted(value.source_refs, key=lambda item: item.role))
        != value.source_refs
    ):
        raise ValueError("Preconditioner source records are not sorted")
    expected_shapes = tuple(
        sorted(value.module_shapes, key=lambda item: item.module_name)
    )
    if expected_shapes != value.module_shapes:
        raise ValueError("Preconditioner module records are not sorted")
    if tuple(sorted(value.files, key=lambda item: item.path)) != value.files:
        raise ValueError("Preconditioner file inventory is not sorted")
    if any(type(item) is not ArtifactRef for item in value.source_refs):
        raise ValueError("Invalid preconditioner source record")
    if any(type(item) is not ModuleShape for item in value.module_shapes):
        raise ValueError("Invalid preconditioner module record")
    if any(type(item) is not RootArtifactFile for item in value.files):
        raise ValueError("Invalid preconditioner file record")


def _validate_record_timestamp(value: str) -> None:
    if not value.endswith("Z"):
        raise ValueError("Build timestamp must be UTC ISO-8601")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("Build timestamp must be UTC ISO-8601") from error
    if parsed.tzinfo != UTC:
        raise ValueError("Build timestamp must be UTC ISO-8601")


def preconditioner_attestation_sha256(value: PreconditionerAttestation) -> str:
    payload = jsonable(value)
    if type(payload) is not dict:
        raise TypeError("Preconditioner attestation must serialize to an object")
    payload.pop("attestation_sha256")
    return sha256_json(payload)


def validate_preconditioner_attestation(value: PreconditionerAttestation) -> None:
    if type(value) is not PreconditionerAttestation:
        raise ValueError("Invalid preconditioner attestation type")
    if value.schema_version != "1" or value.kind != "preconditioner":
        raise ValueError("Unsupported preconditioner attestation schema or kind")
    _validate_hashes(value)
    validate_preconditioner_records(value)
    expected_artifact = root_artifact_sha256(value.artifact_root, value.files)
    if value.artifact_sha256 != expected_artifact:
        raise ValueError("Invalid preconditioner artifact hash")
    expected_invariant = invariant_contract_sha256(
        _build_record(value),
        value.source_refs,
        value.settings,
        value.target_downweight_components,
        value.module_shapes,
        value.invariant_config_sha256,
    )
    if value.invariant_build_contract_sha256 != expected_invariant:
        raise ValueError("Invalid invariant preconditioner contract hash")


def _validate_hashes(value: PreconditionerAttestation) -> None:
    values = (
        value.artifact_sha256,
        value.invariant_config_sha256,
        value.invariant_build_contract_sha256,
        value.model_attestation_sha256,
        value.value_tensor_sha256,
        value.query_tensor_sha256,
        value.mixed_tensor_sha256,
        value.attestation_sha256,
    )
    for item in values:
        validate_hex(item, 64, "preconditioner SHA-256")
    if value.attestation_sha256 != preconditioner_attestation_sha256(value):
        raise ValueError("Invalid preconditioner attestation hash")


def _build_record(value: PreconditionerAttestation) -> PreconditionerBuildRecord:
    return PreconditionerBuildRecord(
        value.artifact_id,
        value.artifact_uri,
        value.build_timestamp,
        value.model_id,
        value.raw_model_revision,
        value.bergson_version,
        value.transformers_version,
        value.bergson_code_revision,
    )


def build_preconditioner_attestation(
    root: Path,
    source_refs: tuple[ArtifactRef, ...],
    model_attestation: ModelContentAttestation,
) -> PreconditionerAttestation:
    validate_model_attestation(model_attestation)
    contract = verify_preconditioner_contract(root, source_refs, model_attestation)
    files = contract.root_files
    tensors = verify_preconditioner_tensors(
        contract.reader, contract.target_components, contract.mixing_coefficient
    )
    _require_stable_root(contract.root, files)
    artifact_hash = root_artifact_sha256(str(contract.root), files)
    invariant_hash = invariant_contract_sha256(
        contract.build,
        contract.sources,
        contract.settings,
        contract.target_components,
        tensors.module_shapes,
        contract.normalized_config_sha256,
    )
    result = _make_attestation(
        contract, tensors, files, artifact_hash, invariant_hash, model_attestation
    )
    signed = replace(
        result, attestation_sha256=preconditioner_attestation_sha256(result)
    )
    validate_preconditioner_attestation(signed)
    return signed


def _require_stable_root(root: Path, expected: tuple[RootArtifactFile, ...]) -> None:
    if build_root_inventory(root) != expected:
        raise ValueError("Preconditioner root changed during attestation build")


def _make_attestation(
    contract: VerifiedPreconditionerContract,
    tensors: TensorEvidence,
    files: tuple[RootArtifactFile, ...],
    artifact_hash: str,
    invariant_hash: str,
    model: ModelContentAttestation,
) -> PreconditionerAttestation:
    values = _identity_values(contract, artifact_hash, invariant_hash, model)
    values.update(_evidence_values(contract, tensors, files))
    return PreconditionerAttestation(**values)  # type: ignore[arg-type]


def _identity_values(contract, artifact_hash, invariant_hash, model):
    return {
        "artifact_id": contract.build.artifact_id,
        "artifact_uri": contract.build.artifact_uri,
        "artifact_root": str(contract.root),
        "artifact_sha256": artifact_hash,
        "invariant_config_sha256": contract.normalized_config_sha256,
        "invariant_build_contract_sha256": invariant_hash,
        "build_timestamp": contract.build.build_timestamp,
        "bergson_version": contract.build.bergson_version,
        "transformers_version": contract.build.transformers_version,
        "bergson_code_revision": contract.build.bergson_code_revision,
        "model_id": contract.build.model_id,
        "raw_model_revision": contract.build.raw_model_revision,
        "lineage_basis": model.lineage_basis,
        "model_attestation_sha256": model_attestation_sha256(model),
    }


def _evidence_values(contract, tensors, files):
    return {
        "source_refs": contract.sources,
        "settings": contract.settings,
        "mixing_policy": "bergson.utils.math.compute_lambda",
        "target_downweight_components": contract.target_components,
        "mixing_coefficient": contract.mixing_coefficient,
        "module_shapes": tensors.module_shapes,
        "value_tensor_sha256": tensors.value_sha256,
        "query_tensor_sha256": tensors.query_sha256,
        "mixed_tensor_sha256": tensors.mixed_sha256,
        "files": files,
        "attestation_sha256": "",
    }


__all__ = [
    "ArtifactRef",
    "ArtifactSourceManifest",
    "ModuleShape",
    "PreconditionerAttestation",
    "PreconditionerBuildRecord",
    "PreconditionerSettings",
    "RootArtifactFile",
    "SourceRole",
    "build_preconditioner_attestation",
    "build_root_inventory",
    "decode_preconditioner_attestation",
    "load_artifact_refs",
    "preconditioner_attestation_sha256",
    "root_artifact_sha256",
    "state_sha256",
    "validate_preconditioner_attestation",
]
