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

        best: Optional[Tuple[float, ...]] = None
        best_fit = -float("inf")

        stagnation_counter = 0
        radius = self.cfg.local_radius

        epoch = 0
        run_start = time.time()
        if self.logger:
            self.logger.info(
                "Starting optimization | pop=%d | dims=%d | time_budget=%.1fs | eval_budget=%d",
                self.cfg.pop_size,
                len(getattr(self.cfg, "mass_bounds", tuple())),
                self.cfg.time_budget_s,
                self.cfg.eval_budget,
            )
        while True:
            epoch_start = time.time()
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

            short_evals = len(X)
            long_evals = 0
            epoch_best_lambda_short: Optional[float] = None
            epoch_source = None
            epoch_candidate_short: Optional[Tuple[float, ...]] = None

            if len(fits) > 0:
                idx = int(max(range(len(fits)), key=lambda i: fits[i]))
                epoch_best_fit = float(fits[idx])
                epoch_best = X[idx]
                self.metrics.best_lambda_per_epoch.append(-epoch_best_fit)
                epoch_best_lambda_short = -epoch_best_fit
                epoch_candidate_short = tuple(epoch_best)
                if epoch_best_fit > best_fit:
                    best_fit = epoch_best_fit
                    best = epoch_best
                    stagnation_counter = 0
                    epoch_source = "short"
                    if self.logger:
                        self.logger.info(
                            "Epoch %d | new global best (short) λ≈ %.6f | masses=%s",
                            epoch,
                            -epoch_best_fit,
                            tuple(round(float(v), 6) for v in epoch_best),
                        )
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
                long_evals = len(topX)
                for xi, f_long in zip(topX, long_fits):
                    if f_long > best_fit:
                        best_fit = float(f_long)
                        best = xi
                        epoch_source = "long"
                        if self.logger:
                            self.logger.info(
                                "Epoch %d | new global best (long) λ≈ %.6f | masses=%s",
                                epoch,
                                -float(f_long),
                                tuple(round(float(v), 6) for v in xi),
                            )

            if stagnation_counter >= self.cfg.stagnation_window:
                if self.logger:
                    self.logger.info("Stagnation detected; reseeding around best candidate.")
                radius = self._modifier.on_stagnation(ga, best, radius)
                self.metrics.reseeds += 1
                stagnation_counter = 0

            epoch_duration = time.time() - epoch_start
            if self.logger:
                self.logger.info(
                    "Epoch %d complete | λ_short≈ %s | evals short/long=%d/%d | total evals=%d | radius=%.4f",
                    epoch,
                    f"{epoch_best_lambda_short:.6f}" if epoch_best_lambda_short is not None else "N/A",
                    short_evals,
                    long_evals,
                    self.metrics.eval_count,
                    radius,
                )
            self.metrics.record_epoch(
                {
                    "epoch": epoch,
                    "best_lambda_short": epoch_best_lambda_short,
                    "best_candidate_short": list(epoch_candidate_short) if epoch_candidate_short else None,
                    "best_lambda_global": -float(best_fit) if best is not None else None,
                    "best_candidate_global": list(best) if best else None,
                    "fitness_global": float(best_fit) if best is not None else None,
                    "evaluations": {
                        "short": short_evals,
                        "long": long_evals,
                        "total": self.metrics.eval_count,
                    },
                    "stagnation_counter": stagnation_counter,
                    "radius": radius,
                    "epoch_time_s": epoch_duration,
                    "source": epoch_source,
                    "timestamp": time.time(),
                }
            )

            epoch += 1
            self.metrics.epochs = epoch

        self.metrics.mark_phase("completed")
        final_lambda = -float(best_fit) if best is not None else None
        if self.logger:
            self.logger.info(
                "Optimization completed | epochs=%d | evals=%d | best λ≈ %s | wall=%.1fs",
                epoch,
                self.metrics.eval_count,
                f"{final_lambda:.6f}" if final_lambda is not None else "N/A",
                time.time() - run_start,
            )
        best_payload: Dict[str, Any] = {
            "masses": list(best) if best else None,
            "lambda": -float(best_fit) if best is not None else None,
            "fitness": float(best_fit) if best is not None else None,
        }
        if best:
            if len(best) > 0:
                best_payload["m1"] = float(best[0])
            if len(best) > 1:
                best_payload["m2"] = float(best[1])
            if len(best) > 2:
                best_payload["m3"] = float(best[2])

        result = {
            "status": "completed",
            "best": best_payload,
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
