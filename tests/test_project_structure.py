"""Keep the production package and executable layers separate."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PRODUCTION_IMPORTS = ("train_scripts", "scripts", "experiments", "ablations")


def imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def test_production_package_does_not_import_executables_or_experiments():
    violations = []
    for path in (ROOT / "miworld").rglob("*.py"):
        for module in imported_modules(path):
            if module.startswith(FORBIDDEN_PRODUCTION_IMPORTS):
                violations.append(f"{path.relative_to(ROOT)} imports {module}")
    assert not violations, violations


def test_models_do_not_depend_on_data_training_or_pipelines():
    forbidden = ("miworld.data", "miworld.training", "miworld.pipelines")
    violations = []
    for path in (ROOT / "miworld" / "models").rglob("*.py"):
        for module in imported_modules(path):
            if module.startswith(forbidden):
                violations.append(f"{path.relative_to(ROOT)} imports {module}")
    assert not violations, violations


def test_legacy_top_level_packages_are_gone():
    for name in ("models", "data_loaders", "trainers", "pipelines"):
        assert not (ROOT / name).exists()


def test_train_script_root_contains_canonical_miworld_entries():
    actual = {path.name for path in (ROOT / "train_scripts").glob("*.py")}
    canonical = {
        "train_m1.py",
        "train_m2.py",
        "train_m3_renderer.py",
        "train_m3_policy.py",
    }
    assert canonical <= actual
    assert "train_m4.py" not in actual
