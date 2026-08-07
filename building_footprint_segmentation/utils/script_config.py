"""Helpers so scripts can use an in-file CONFIG plus optional CLI overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def apply_cli_overrides(config: dict[str, Any], args: Any) -> dict[str, Any]:
    """
    Return a copy of CONFIG with any non-None argparse values applied.

    Argparse dest names should match CONFIG keys (use dest=... when needed).
    """
    settings = dict(config)
    for key in list(settings.keys()):
        if hasattr(args, key):
            value = getattr(args, key)
            if value is not None:
                settings[key] = value
    return settings


def create_next_run_dir(base_dir: Path | str, prefix: str) -> Path:
    """
    Create the next numbered run folder under ``base_dir``.

    Naming: ``{prefix}1``, ``{prefix}2``, ``{prefix}3``, ...

    Typical layout::

        runs/training/train1
        runs/inference/inference1
        runs/hyperparameter/hyperparameter1

    CONFIG ``output`` should point at the task folder
    (e.g. ``runs/training``).
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    index = 1
    while True:
        candidate = base / f"{prefix}{index}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        index += 1
