"""Small optional W&B helpers for optimization runs."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def init_wandb(
    *,
    project: str,
    name: str | None = None,
    config: dict[str, Any] | None = None,
    enabled: bool = True,
    mode: str | None = None,
):
    if not enabled:
        return None
    try:
        import wandb  # type: ignore
    except Exception as exc:
        print(f"[wandb] unavailable: {exc}")
        return None
    kwargs: dict[str, Any] = {
        "project": project,
        "name": name,
        "config": config or {},
    }
    run_mode = mode or os.environ.get("WANDB_MODE")
    if run_mode:
        kwargs["mode"] = run_mode
    try:
        return wandb.init(**kwargs)
    except Exception as exc:
        print(f"[wandb] init failed: {exc}")
        return None


def log(run: Any, data: dict[str, Any], step: int | None = None) -> None:
    if run is None:
        return
    try:
        run.log(data, step=step)
    except Exception as exc:
        print(f"[wandb] log failed: {exc}")


def artifact(run: Any, path: str | Path, *, name: str, type_: str) -> None:
    if run is None:
        return
    p = Path(path)
    if not p.exists():
        return
    try:
        import wandb  # type: ignore

        art = wandb.Artifact(name, type=type_)
        art.add_file(str(p))
        run.log_artifact(art)
    except Exception as exc:
        print(f"[wandb] artifact failed: {exc}")
