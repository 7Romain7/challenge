"""T0/T1 - algorithm code must not touch the simulator's hidden state or the global RNG seed."""

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
FORBIDDEN_ATTRS = {"_sim", "_optimum", "reveal", "base_interdots", "contrast_model", "drift_matrix"}


def _files(pkg):
    return [p for p in (ROOT / pkg).rglob("*.py")]


def _violations(pkg, allowed_csd_imports):
    bad = []
    for path in _files(pkg):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "csd":
                names = {a.name for a in node.names}
                if not (names <= allowed_csd_imports and node.module == "csd"):
                    bad.append((path.name, node.lineno, f"import from {node.module}"))
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] == "csd":
                        bad.append((path.name, node.lineno, f"import {a.name}"))
            if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
                bad.append((path.name, node.lineno, f"attribute .{node.attr}"))
            if isinstance(node, ast.Attribute) and node.attr == "seed":
                # np.random.seed(...) / random.seed(...) would freeze the simulator noise
                if isinstance(node.value, ast.Attribute) and node.value.attr == "random":
                    bad.append((path.name, node.lineno, "np.random.seed"))
    return bad


def test_optimization_package_is_blind():
    assert _violations("optimization", set()) == []


def test_training_package_uses_public_api_only():
    assert _violations("training", {"new_experiment"}) == []
