# The tests import shared helpers as ``tests.data_attribution...``. Because
# tests/ is a package, pytest puts its parent (the repository root) on
# sys.path when it loads this conftest, so a bare ``pytest`` (or
# ``uv run pytest``) collects the suite the same way ``python -m pytest`` does.
