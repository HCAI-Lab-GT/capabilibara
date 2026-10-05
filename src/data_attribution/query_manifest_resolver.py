"""Resolve strict query-manifest pointers to immutable generation members."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
import stat
from typing import Any

STRICT_MANIFEST_SUFFIX = "strict-manifest.json"
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "study_id",
        "ecosystem",
        "profile",
        "file_prefix",
        "generation",
        "files",
    }
)
_MEMBER_KEYS = frozenset({"path", "result_group", "rows", "sha256"})


class StrictQueryManifestError(ValueError):
    """A strict query-manifest pointer or generation is invalid."""


class _StrictJsonError(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _StrictJsonError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise _StrictJsonError(f"non-finite JSON number {value!r}")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise _StrictJsonError(f"non-finite JSON number {value!r}")
    return parsed


def _exact_keys(
    value: dict[str, object], expected: frozenset[str], context: str
) -> None:
    if set(value) != expected:
        raise StrictQueryManifestError(f"{context} has invalid fields")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_all(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _read_regular_file(path: Path, *, context: str) -> bytes:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise StrictQueryManifestError(
            f"{context} is not a readable regular file"
        ) from error
    if not stat.S_ISREG(mode):
        raise StrictQueryManifestError(f"{context} is not a readable regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise StrictQueryManifestError(
            f"{context} is not a readable regular file"
        ) from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise StrictQueryManifestError(f"{context} is not a regular file")
        return _read_all(descriptor)
    finally:
        os.close(descriptor)


def _matching_pointer(direct_path: Path) -> tuple[Path, str] | None:
    parent = direct_path.parent
    if not parent.is_dir():
        return None
    matches: list[tuple[int, Path, str]] = []
    for candidate in parent.iterdir():
        if not candidate.name.endswith(STRICT_MANIFEST_SUFFIX):
            continue
        file_prefix = candidate.name.removesuffix(STRICT_MANIFEST_SUFFIX)
        if file_prefix and direct_path.name.startswith(file_prefix):
            matches.append((len(file_prefix), candidate, file_prefix))
    if not matches:
        return None
    _length, pointer_path, file_prefix = max(matches, key=lambda item: item[0])
    return pointer_path, file_prefix


def _load_pointer(pointer_path: Path) -> dict[str, Any]:
    context = f"strict query manifest pointer {pointer_path}"
    try:
        raw = _read_regular_file(pointer_path, context=context).decode("utf-8")
        payload = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, _StrictJsonError) as error:
        raise StrictQueryManifestError(f"{context} is unreadable: {error}") from error
    if not isinstance(payload, dict):
        raise StrictQueryManifestError(f"{context} has invalid schema")
    _exact_keys(payload, _MANIFEST_KEYS, context)
    schema_version = payload["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 2
    ):
        raise StrictQueryManifestError(f"{context} must use schema version 2")
    for key in ("study_id", "ecosystem", "profile", "file_prefix"):
        value = payload[key]
        if not isinstance(value, str) or not value.strip():
            raise StrictQueryManifestError(f"{context} has invalid {key}")
    if not _is_sha256(payload["generation"]):
        raise StrictQueryManifestError(f"{context} has invalid generation")
    return payload


def _validated_members(
    payload: dict[str, Any],
    *,
    pointer_path: Path,
    file_prefix: str,
) -> dict[str, tuple[str, int, str]]:
    context = f"strict query manifest pointer {pointer_path}"
    generation = str(payload["generation"])
    generation_relative = (
        Path(f".{file_prefix}{STRICT_MANIFEST_SUFFIX}.generations") / generation
    )
    raw_files = payload["files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise StrictQueryManifestError(f"{context} has invalid files")
    members: dict[str, tuple[str, int, str]] = {}
    result_groups: set[str] = set()
    for index, entry in enumerate(raw_files):
        entry_context = f"{context} file entry {index}"
        if not isinstance(entry, dict):
            raise StrictQueryManifestError(f"{entry_context} is invalid")
        _exact_keys(entry, _MEMBER_KEYS, entry_context)
        raw_path = entry["path"]
        result_group = entry["result_group"]
        rows = entry["rows"]
        sha256 = entry["sha256"]
        if not isinstance(raw_path, str):
            raise StrictQueryManifestError(
                f"{entry_context} has invalid generation path"
            )
        relative_path = Path(raw_path)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.parent != generation_relative
        ):
            raise StrictQueryManifestError(
                f"{entry_context} has invalid generation path"
            )
        if (
            not isinstance(result_group, str)
            or not result_group
            or result_group in result_groups
            or relative_path.name != f"{file_prefix}{result_group}.jsonl"
        ):
            raise StrictQueryManifestError(f"{entry_context} has invalid result_group")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
            raise StrictQueryManifestError(f"{entry_context} has invalid rows")
        if not _is_sha256(sha256) or relative_path.name in members:
            raise StrictQueryManifestError(
                f"{entry_context} has invalid sha256 or path"
            )
        members[relative_path.name] = (raw_path, rows, str(sha256))
        result_groups.add(result_group)
    return members


def _open_directory(path: Path, *, context: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise StrictQueryManifestError(f"{context} is not a directory") from error
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise StrictQueryManifestError(f"{context} is not a directory")
    return descriptor


def _open_directory_at(parent: int, name: str, *, context: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent)
    except OSError as error:
        raise StrictQueryManifestError(f"{context} is not a directory") from error
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise StrictQueryManifestError(f"{context} is not a directory")
    return descriptor


def _read_generation_member(
    generation_descriptor: int,
    name: str,
    *,
    context: str,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=generation_descriptor)
    except OSError as error:
        raise StrictQueryManifestError(f"{context} is not a regular file") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise StrictQueryManifestError(f"{context} is not a regular file")
        return _read_all(descriptor)
    finally:
        os.close(descriptor)


def _validate_generation(
    pointer_path: Path,
    *,
    file_prefix: str,
    generation: str,
    members: dict[str, tuple[str, int, str]],
) -> None:
    root_name = f".{file_prefix}{STRICT_MANIFEST_SUFFIX}.generations"
    root_descriptor = _open_directory(
        pointer_path.parent / root_name,
        context="strict query manifest generation root",
    )
    try:
        generation_descriptor = _open_directory_at(
            root_descriptor,
            generation,
            context="strict query manifest generation directory",
        )
        try:
            observed_names = set(os.listdir(generation_descriptor))
            if observed_names != set(members):
                raise StrictQueryManifestError(
                    "strict query manifest generation has unexpected directory entries"
                )
            for name, (_raw_path, rows, expected_sha256) in members.items():
                data = _read_generation_member(
                    generation_descriptor,
                    name,
                    context=f"strict query manifest generation member {name}",
                )
                observed_sha256 = hashlib.sha256(data).hexdigest()
                if observed_sha256 != expected_sha256:
                    raise StrictQueryManifestError(
                        f"strict query manifest generation member {name} hash mismatch"
                    )
                if not data.endswith(b"\n") or len(data.splitlines()) != rows:
                    raise StrictQueryManifestError(
                        f"strict query manifest generation member {name} row count mismatch"
                    )
        finally:
            os.close(generation_descriptor)
    finally:
        os.close(root_descriptor)


def validate_strict_query_manifest_pointer(pointer_path: Path) -> dict[str, Path]:
    """Validate one strict pointer and return its immutable members by basename."""

    payload = _load_pointer(pointer_path)
    file_prefix = str(payload["file_prefix"])
    if pointer_path.name != f"{file_prefix}{STRICT_MANIFEST_SUFFIX}":
        raise StrictQueryManifestError(
            f"strict query manifest pointer {pointer_path} has a different file_prefix"
        )
    members = _validated_members(
        payload,
        pointer_path=pointer_path,
        file_prefix=file_prefix,
    )
    generation = str(payload["generation"])
    _validate_generation(
        pointer_path,
        file_prefix=file_prefix,
        generation=generation,
        members=members,
    )
    return {
        name: pointer_path.parent / raw_path
        for name, (raw_path, _rows, _sha256) in members.items()
    }


def resolve_query_manifest_member(direct_path: Path) -> Path:
    """Return an immutable strict member, or the direct legacy path if unmanaged."""

    matched = _matching_pointer(direct_path)
    if matched is None:
        return direct_path
    pointer_path, _file_prefix = matched
    members = validate_strict_query_manifest_pointer(pointer_path)
    member = members.get(direct_path.name)
    if member is None:
        raise StrictQueryManifestError(
            f"strict query manifest pointer {pointer_path} does not contain member "
            f"{direct_path.name}"
        )
    return member


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve a query JSONL through an adjacent strict manifest pointer."
    )
    parser.add_argument("path", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    print(resolve_query_manifest_member(args.path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
