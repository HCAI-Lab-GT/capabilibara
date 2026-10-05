"""Write machine-readable metadata for WebOrganizer report outputs."""

from __future__ import annotations

import json
from pathlib import Path


def write_report_metadata(output_dir: Path, summary: dict[str, object]) -> Path:
    path = Path(output_dir) / "report_metadata.json"
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


__all__ = ["write_report_metadata"]
