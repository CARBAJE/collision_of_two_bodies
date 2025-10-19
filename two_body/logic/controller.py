"""
Controlador continuo que coordina motor evolutivo, evaluaciones y logica de parametros.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from ..core.cache import HierarchicalCache
from ..core.config import Config
from ..core.telemetry import MetricsLogger
from .fitness import FitnessEvaluator
from .ga import StreamingGA
from .parameters import ParameterModifier


class ContinuousOptimizationController:
    def __init__(self, cfg: Config, logger: Optional[Any] = None) -> None:
        import logging

        self.cfg = cfg
        self.logger = logger or logging.getLogger("master_two_body_opt")
        self.metrics = MetricsLogger()
        self._modifier = ParameterModifier(cfg)

    def run(self) -> Dict[str, Any]:
        start = time.time()
        cache = HierarchicalCache(
            approx_max=self.cfg.cache_approx_max, exact_max=self.cfg.cache_exact_max
        )
        evaluator = FitnessEvaluator(cache, self.cfg, logger=self.logger)
        ga = StreamingGA(self.cfg, logger=self.logger)

        best: Optional[Tuple[float, float]] = None
        best_fit = -float("inf")

        stagnation_counter = 0
        radius = self.cfg.local_radius

        epoch = 0
        while True:
            now = time.time()
            if now - start > self.cfg.time_budget_s:
                break
            if self.metrics.eval_count >= self.cfg.eval_budget:
                break
            if epoch >= self.cfg.max_epochs:
                break

            X = ga.current_population()
            fits = evaluator.evaluate_batch(X, horizon="short")
            self.metrics.eval_count += len(X)

            if len(fits) > 0:
                idx = int(max(range(len(fits)), key=lambda i: fits[i]))
                epoch_best_fit = float(fits[idx])
                epoch_best = X[idx]
                self.metrics.best_lambda_per_epoch.append(-epoch_best_fit)
                if epoch_best_fit > best_fit:
                    best_fit = epoch_best_fit
                    best = epoch_best
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

            try:
                import numpy as _np

                ga._last_F = _np.array(fits, dtype=float).reshape(-1, 1)
            except Exception:
                ga._last_F = None  # type: ignore
            ga.step(generations=max(1, self.cfg.n_gen_step))

            if len(fits) > 0 and self.cfg.top_k_long > 0:
                k = min(self.cfg.top_k_long, len(fits))
                idxs = sorted(range(len(fits)), key=lambda i: fits[i], reverse=True)[:k]
                topX = [X[i] for i in idxs]
                long_fits = evaluator.evaluate_batch(topX, horizon="long")
                self.metrics.eval_count += len(topX)
                for xi, f_long in zip(topX, long_fits):
                    if f_long > best_fit:
                        best_fit = float(f_long)
                        best = xi

            if stagnation_counter >= self.cfg.stagnation_window:
                if self.logger:
                    self.logger.info("Stagnation detected; reseeding around best candidate.")
                radius = self._modifier.on_stagnation(ga, best, radius)
                stagnation_counter = 0

            epoch += 1
            self.metrics.epochs = epoch

        self.metrics.mark_phase("completed")
        result = {
            "status": "completed",
            "best": {
                "m1": float(best[0]) if best else None,
                "m2": float(best[1]) if best else None,
                "lambda": -float(best_fit) if best is not None else None,
                "fitness": float(best_fit) if best is not None else None,
            },
            "evals": self.metrics.eval_count,
            "epochs": epoch,
        }
        return result


if __name__ == "__main__":
    from ..core.config import Config
    from ..core.telemetry import setup_logger

    cfg = Config(max_epochs=1, time_budget_s=1.0, eval_budget=10)
    logger = setup_logger()
    controller = ContinuousOptimizationController(cfg, logger=logger)
    print("Resultado de smoke test:", controller.run())
