"""Atomically publish one staged preconditioner artifact without replacement."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import stat
from collections.abc import Callable
from uuid import uuid4

from data_attribution.attribution.trackstar.preconditioner_promotion_kernel import (
    EntryIdentity,
    entry_identity,
    open_directory,
    open_stage as _open_stage,
    quarantine_entry,
    rename_noreplace as _rename_noreplace,
    unlink_stage,
)


def promote_absent(
    stage: Path,
    target: Path,
    *,
    expected_identity: EntryIdentity | None = None,
    final_mode: int | None = None,
) -> None:
    """Move a same-parent stage only when the target name is absent."""
    _validate_paths(stage, target)
    directory = open_directory(stage.parent)
    descriptor: int | None = None
    try:
        descriptor = _open_stage(directory, stage.name, final_mode is not None)
        source = os.fstat(descriptor)
        source_identity = entry_identity(source)
        if expected_identity is not None and source_identity != expected_identity:
            raise ValueError("Promotion stage changed before publication")
        _require_stage_entry(source)
        regular = stat.S_ISREG(source.st_mode)
        if regular and final_mode is not None:
            os.fchmod(descriptor, final_mode)
        if regular:
            _link_noreplace(directory, stage.name, target.name)
        else:
            _rename_noreplace(directory, stage.name, target.name)
        published = os.stat(target.name, dir_fd=directory, follow_symlinks=False)
        published_identity = entry_identity(published)
        try:
            staged_identity = entry_identity(
                os.stat(stage.name, dir_fd=directory, follow_symlinks=False)
            )
        except FileNotFoundError:
            staged_identity = None
        expected_stage = source_identity if regular else None
        if source_identity != published_identity or staged_identity != expected_stage:
            if regular and published_identity == staged_identity:
                rejected = f".part-rejected-{target.name}-{uuid4().hex}"
                _quarantine_linked_file(
                    directory, target.name, rejected, published_identity
                )
            elif not regular:
                rejected = f".part-rejected-{target.name}-{uuid4().hex}"
                quarantine_entry(directory, target.name, rejected, published_identity)
            raise ValueError("Promotion stage changed before publication")
        if regular:
            os.close(descriptor)
            descriptor = None
            unlink_stage(directory, stage.name)
        elif final_mode is not None:
            os.fchmod(descriptor, final_mode)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)


def publish_bytes_absent(path: Path, content: bytes, mode: int = 0o444) -> None:
    """Write complete bytes to a stage and publish them without replacement."""
    _validate_leaf(path)
    stage = path.with_name(f".part-{path.name}-{uuid4().hex}")
    directory = open_directory(path.parent)
    expected_identity: EntryIdentity
    try:
        descriptor = os.open(
            stage.name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            mode,
            dir_fd=directory,
        )
        try:
            with os.fdopen(descriptor, "w+b") as stream:
                stream.write(content)
                stream.flush()
                os.fchmod(stream.fileno(), mode)
                os.fsync(stream.fileno())
                before = os.fstat(stream.fileno())
                stream.seek(0)
                observed = stream.read()
                after = os.fstat(stream.fileno())
                before_state = (
                    before.st_dev,
                    before.st_ino,
                    before.st_mode,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                )
                after_state = (
                    after.st_dev,
                    after.st_ino,
                    after.st_mode,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                )
                if observed != content or before_state != after_state:
                    raise ValueError("Publication stage bytes changed before promotion")
                expected_identity = entry_identity(after)
        except BaseException:
            unlink_stage(directory, stage.name)
            raise
    finally:
        os.close(directory)
    promote_absent(stage, path, expected_identity=expected_identity)


def stage_identity(stage: Path) -> EntryIdentity:
    """Read one anchored stage identity without following links."""
    _validate_leaf(stage)
    directory = open_directory(stage.parent)
    descriptor: int | None = None
    try:
        descriptor = _open_stage(directory, stage.name)
        source = os.fstat(descriptor)
        _require_stage_entry(source)
        return entry_identity(source)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)


def validate_and_promote(
    stage: Path, target: Path, validator: Callable[[Path], object]
) -> None:
    """Bind validation to a stage identity and revalidate its final path."""
    expected = stage_identity(stage)
    validator(stage)
    promote_absent(stage, target, expected_identity=expected)
    validator(target)


def _validate_paths(stage: Path, target: Path) -> None:
    _validate_leaf(stage)
    _validate_leaf(target)
    if stage.parent != target.parent:
        raise ValueError("Promotion stage and target must use the same parent")
    if stage.name == target.name:
        raise ValueError("Promotion stage and target must differ")


def _validate_leaf(path: Path) -> None:
    if not isinstance(path, Path) or not path.is_absolute() or ".." in path.parts:
        raise ValueError("Promotion paths must be absolute and canonical")
    if not path.name:
        raise ValueError("Promotion paths require a leaf name")


def _require_stage_entry(entry: os.stat_result) -> None:
    if stat.S_ISLNK(entry.st_mode) or not (
        stat.S_ISREG(entry.st_mode) or stat.S_ISDIR(entry.st_mode)
    ):
        raise ValueError("Promotion stage must be a regular file or directory")


def _link_noreplace(directory: int, source: str, target: str) -> None:
    os.link(
        source,
        target,
        src_dir_fd=directory,
        dst_dir_fd=directory,
        follow_symlinks=False,
    )


def _quarantine_linked_file(
    directory: int,
    source: str,
    target: str,
    expected_identity: EntryIdentity,
) -> None:
    _link_noreplace(directory, source, target)
    observed = os.stat(target, dir_fd=directory, follow_symlinks=False)
    if entry_identity(observed) != expected_identity:
        raise ValueError("Rejected publication entry changed during quarantine")
    current = os.stat(source, dir_fd=directory, follow_symlinks=False)
    if entry_identity(current) != expected_identity:
        raise ValueError("Rejected publication entry changed during quarantine")
    os.unlink(source, dir_fd=directory)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        promote_absent(args.stage, args.target)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
