# Software Source Guide

This is a reading guide for the curated public source snapshot. The historical
guide filenames below reflect their original review-bundle purpose.

Start here:

1. Read [docs/ANONYMOUS_SOFTWARE_GUIDE.md](docs/ANONYMOUS_SOFTWARE_GUIDE.md).
2. Use [docs/ANONYMOUS_SOFTWARE_WORKFLOWS.md](docs/ANONYMOUS_SOFTWARE_WORKFLOWS.md) for command maps and workflow-specific code paths.
3. Inspect `pyproject.toml` for package metadata and console scripts.
4. Use `src/` for source code, `tests/` for expected behavior, and `configs/` for analysis settings.
5. Read `artifacts/zscored_bin_scores/README.md` for the four published aggregate results.

The snapshot excludes internal run outputs, document-level influence scores,
query records, and raw training text.

The dependency setup recorded in the guide has not been verified from a fresh
public clone. No production experiments were rerun for this source release.
