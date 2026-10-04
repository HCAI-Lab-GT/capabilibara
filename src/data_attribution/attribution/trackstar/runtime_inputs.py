"""Stable, content-bound readers for attribution runtime inputs."""

from __future__ import annotations

import hashlib
import argparse
import json
import os
from pathlib import Path
import stat


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def stable_file_bytes(path: Path, *, expected_sha256: str | None = None) -> bytes:
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise ValueError(f"runtime input must be a regular file: {path}")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        with os.fdopen(descriptor, "rb") as stream:
            if _identity(os.fstat(stream.fileno())) != _identity(before):
                raise ValueError(f"runtime input changed while opening: {path}")
            content = stream.read()
            descriptor_after = os.fstat(stream.fileno())
    except BaseException:
        raise
    after = path.lstat()
    if _identity(before) != _identity(descriptor_after) or _identity(
        before
    ) != _identity(after):
        raise ValueError(f"runtime input changed while reading: {path}")
    observed = hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError(f"runtime input sha256 mismatch: {path}")
    return content


def verify_query_build_artifacts(
    query_build_dir: Path, probes: dict[str, dict[str, object]]
) -> None:
    root = Path(query_build_dir).resolve()
    observed = {
        child.name
        for child in root.iterdir()
        if child.is_dir() and (child / "gradients.bin").exists()
    }
    if observed != set(probes):
        raise ValueError("runtime query-build groups do not match attribution policy")
    for group, policy in probes.items():
        verify_query_build_artifact(root, group, policy)


def verify_query_build_artifact(
    query_build_dir: Path, group: str, policy: dict[str, object]
) -> None:
    """Recheck one query index immediately before it is loaded for scoring."""
    root = Path(query_build_dir).resolve()
    group_root = root / group
    for filename, field in (
        ("gradients.bin", "gradients_sha256"),
        ("info.json", "info_sha256"),
        ("index_config.json", "index_config_sha256"),
    ):
        expected = policy.get(field)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"{group}: policy lacks {field}")
        stable_file_bytes(group_root / filename, expected_sha256=expected)


def verify_allocation_inputs(
    *,
    policy_path: Path,
    policy_sha256: str,
    query_build_dir: Path,
    source_shards: Path,
    manifest: Path,
) -> None:
    """Perform the expensive complete preflight once per Slurm allocation."""
    policy = json.loads(stable_file_bytes(policy_path, expected_sha256=policy_sha256))
    if not isinstance(policy, dict) or not isinstance(policy.get("probes"), dict):
        raise ValueError("attribution policy lacks query artifact bindings")
    runtime = policy.get("runtime_inputs")
    if not isinstance(runtime, dict):
        raise ValueError("attribution policy lacks corpus bindings")
    source_digest = runtime.get("source_shards_sha256")
    manifest_digest = runtime.get("manifest_sha256")
    if not isinstance(source_digest, str) or not isinstance(manifest_digest, str):
        raise ValueError("attribution policy corpus bindings are invalid")
    from data_attribution.experiments.financial_tda.paths import (
        stable_directory_sha256,
    )

    if stable_directory_sha256(source_shards) != source_digest:
        raise ValueError("source shard tree does not match attribution policy")
    stable_file_bytes(manifest, expected_sha256=manifest_digest)
    verify_query_build_artifacts(query_build_dir, policy["probes"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--query-build-dir", required=True, type=Path)
    parser.add_argument("--source-shards", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    verify_allocation_inputs(
        policy_path=args.policy,
        policy_sha256=args.policy_sha256,
        query_build_dir=args.query_build_dir,
        source_shards=args.source_shards,
        manifest=args.manifest,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
