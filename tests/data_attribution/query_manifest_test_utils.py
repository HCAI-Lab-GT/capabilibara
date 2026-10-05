from __future__ import annotations

import hashlib
import json
from pathlib import Path


def write_strict_query_pointer(
    output_dir: Path,
    *,
    member_name: str,
    immutable_rows: list[dict[str, object]],
    compatibility_rows: list[dict[str, object]],
    file_prefix: str = "olmes_instruct_",
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    immutable_text = "".join(json.dumps(row) + "\n" for row in immutable_rows)
    compatibility_text = "".join(json.dumps(row) + "\n" for row in compatibility_rows)
    generation = hashlib.sha256(immutable_text.encode("utf-8")).hexdigest()
    generation_relative = (
        Path(f".{file_prefix}strict-manifest.json.generations") / generation
    )
    generation_dir = output_dir / generation_relative
    generation_dir.mkdir(parents=True)
    immutable_path = generation_dir / member_name
    immutable_path.write_text(immutable_text, encoding="utf-8")
    compatibility_path = output_dir / member_name
    compatibility_path.write_text(compatibility_text, encoding="utf-8")
    pointer_path = output_dir / f"{file_prefix}strict-manifest.json"
    pointer_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "study_id": "test-study",
                "ecosystem": "test-ecosystem",
                "profile": "base",
                "file_prefix": file_prefix,
                "generation": generation,
                "files": [
                    {
                        "path": str(generation_relative / member_name),
                        "result_group": member_name.removeprefix(
                            file_prefix
                        ).removesuffix(".jsonl"),
                        "rows": len(immutable_rows),
                        "sha256": hashlib.sha256(
                            immutable_text.encode("utf-8")
                        ).hexdigest(),
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return compatibility_path, immutable_path, pointer_path
