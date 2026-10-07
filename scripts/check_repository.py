"""Validate repository sources without importing optional ML dependencies."""
from __future__ import annotations

import ast
from pathlib import Path
import sys
import tomllib

import yaml


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    python_files = sorted(set((root / "ultralytics").rglob("*.py")) | set(root.glob("*.py")) | set((root / "scripts").rglob("*.py")))
    yaml_files = sorted((root / "ultralytics/cfg").rglob("*.yaml"))
    errors = []
    for path in python_files:
        try:
            ast.parse(path.read_bytes(), filename=str(path.relative_to(root)))
        except (SyntaxError, UnicodeDecodeError) as error:
            errors.append(f"{path.relative_to(root)}: {error}")
    for path in yaml_files:
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
            if not isinstance(value, dict):
                raise ValueError("configuration must be a YAML mapping")
        except (yaml.YAMLError, UnicodeDecodeError, ValueError) as error:
            errors.append(f"{path.relative_to(root)}: {error}")
    try:
        with (root / "pyproject.toml").open("rb") as stream:
            tomllib.load(stream)
    except tomllib.TOMLDecodeError as error:
        errors.append(f"pyproject.toml: {error}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Validated {len(python_files)} Python sources, {len(yaml_files)} YAML configurations, and pyproject.toml.")
    print("Scope: syntax and configuration parsing; model training and inference require the project ML environment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
