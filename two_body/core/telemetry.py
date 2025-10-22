"""
Herramientas de registro y telemetria compartidas por todos los modulos.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config


def setup_logger(level: str = "INFO") -> logging.Logger:
    """Configura un logger estandar reutilizable en toda la aplicacion."""
    logger = logging.getLogger("master_two_body_opt")
    if logger.handlers:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s] %(levelname)s - %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


class MetricsLogger:
    """Registro ligero de metricas para pruebas iniciales."""

    def __init__(self) -> None:
        self.start_time = time.time()
        self.phases: Dict[str, float] = {}
        self.cache_hits_exact = 0
        self.cache_hits_approx = 0
        self.cache_misses_exact = 0
        self.cache_misses_approx = 0
        self.eval_count = 0
        self.epochs = 0
        self.reseeds = 0
        self.best_lambda_per_epoch: List[float] = []
        self.epoch_history: List[Dict[str, Any]] = []

    def mark_phase(self, name: str) -> None:
        self.phases[name] = time.time() - self.start_time

    def record_epoch(self, payload: Dict[str, Any]) -> None:
        """
        Registra informacion resumida de cada epoca para analisis posterior.
        """
        self.epoch_history.append(payload)

    def to_dict(self) -> Dict[str, Any]:
        total = max(time.time() - self.start_time, 1e-9)
        total_hits = self.cache_hits_exact + self.cache_hits_approx
        total_misses = self.cache_misses_exact + self.cache_misses_approx
        hit_rate = (
            total_hits / (total_hits + total_misses)
            if (total_hits + total_misses) > 0
            else 0.0
        )
        return {
            "wall_time_s": total,
            "phases": self.phases,
            "eval_count": self.eval_count,
            "epochs": self.epochs,
            "reseeds": self.reseeds,
            "cache": {
                "hits_exact": self.cache_hits_exact,
                "hits_approx": self.cache_hits_approx,
                "misses_exact": self.cache_misses_exact,
                "misses_approx": self.cache_misses_approx,
                "hit_rate": hit_rate,
            },
            "best_lambda_per_epoch": self.best_lambda_per_epoch,
            "epoch_history": self.epoch_history,
        }


class Reporter:
    """Gestiona archivos de salida en la carpeta de artefactos."""

    def __init__(self, artifacts_dir: str, logger: Optional[logging.Logger] = None) -> None:
        self.dir = Path(artifacts_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger("master_two_body_opt")

    def save_json(self, name: str, payload: Dict[str, Any]) -> Path:
        path = self.dir / name
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return path

    def save_config(self, cfg: Config) -> Path:
        return self.save_json("config.json", asdict(cfg))

    def save_metrics(self, metrics: MetricsLogger) -> Path:
        return self.save_json("metrics.json", metrics.to_dict())

    def save_results(self, results: Dict[str, Any]) -> Path:
        return self.save_json("results.json", results)

    def checkpoint(self, epoch: int, state: Dict[str, Any]) -> Path:
        return self.save_json(f"checkpoint_epoch_{epoch:03d}.json", state)

    def bootstrap(self, cfg: Config) -> None:
        self.save_config(cfg)
        if self.logger:
            self.logger.info("Saved config.json to artifacts directory")


if __name__ == "__main__":
    log = setup_logger()
    metrics = MetricsLogger()
    metrics.mark_phase("demo")
    rep = Reporter("artifacts", logger=log)
    rep.bootstrap(Config())
    rep.save_metrics(metrics)
    log.info("Telemetria registrada correctamente.")
