from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


class ExperimentTracker:
    def __init__(self, run: Any | None = None):
        self.run = run

    @classmethod
    def initialize(
        cls,
        config: dict[str, Any],
        stage: str,
    ) -> "ExperimentTracker":
        wandb_config = config.get("logging", {}).get("wandb", {})
        if not bool(wandb_config.get("enabled", False)):
            return cls()
        load_dotenv(Path.home() / ".env")
        mode = str(wandb_config.get("mode", "online"))
        if mode == "online" and not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError(
                "WANDB_API_KEY is missing. Add it to ~/.env or use W&B offline mode."
            )
        tracking_dir = Path(config["output_dir"]) / "wandb"
        tracking_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("WANDB_DIR", str(tracking_dir))
        os.environ.setdefault("WANDB_CACHE_DIR", str(tracking_dir / "cache"))
        os.environ.setdefault("WANDB_CONFIG_DIR", str(tracking_dir / "config"))
        import wandb

        initialization = {
            "project": str(wandb_config["project"]),
            "config": config,
            "mode": mode,
            "job_type": stage,
            "group": wandb_config.get("group"),
            "name": wandb_config.get("name"),
            "tags": list(wandb_config.get("tags", [])),
            "dir": str(tracking_dir),
        }
        if wandb_config.get("entity"):
            initialization["entity"] = wandb_config["entity"]
        run = wandb.init(**initialization)
        return cls(run)

    def log(self, values: dict[str, Any]) -> None:
        if self.run is not None:
            self.run.log(values)

    def log_figures(self, paths: list[Path], namespace: str) -> None:
        if self.run is None:
            return
        import wandb

        images = {
            f"figures/{namespace}/{path.stem}": wandb.Image(str(path))
            for path in paths
            if path.suffix.lower() == ".png"
        }
        if images:
            self.run.log(images)

    def update_summary(self, values: dict[str, Any]) -> None:
        if self.run is not None:
            self.run.summary.update(values)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
