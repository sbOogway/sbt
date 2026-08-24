"""Import gate: every module in the sbt package must parse and import.

No linting or CI exists in this repo, so a pure syntax slip (e.g. a bad
dedent inside a rarely-run CLI handler) used to ship silently until a
user hit the subcommand. This test walks every module under ``sbt/`` so
any unparseable or broken-import file fails the suite immediately.
"""

import importlib
import pathlib

import pytest

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1] / "sbt"


def sbt_modules():
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        rel = path.relative_to(PACKAGE_ROOT.parent)
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        yield ".".join(parts)


@pytest.mark.parametrize("module_name", list(sbt_modules()))
def test_module_imports(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


def test_registry_entries_import() -> None:
    from sbt.utils import get_strategy_class, get_strategy_names

    names = get_strategy_names()
    assert names, "strategy registry must not be empty"
    for name in names:
        strategy_cls, _config_cls = get_strategy_class(name)
        assert strategy_cls is not None
