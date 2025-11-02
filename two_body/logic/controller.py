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
        best_lambda_value: Optional[float] = None



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

            fits, short_details = evaluator.evaluate_batch(
                X, horizon="short", epoch=epoch, return_details=True
            )

            self.metrics.eval_count += len(X)



            short_evals = len(X)

            long_evals = 0

            epoch_best_lambda_short: Optional[float] = None
            epoch_best_fitness_short: Optional[float] = None
            epoch_best_lambda_global: Optional[float] = best_lambda_value
            epoch_best_fitness_global: Optional[float] = (
                float(best_fit) if best is not None and best_fit != -float("inf") else None
            )

            epoch_source = None

            epoch_candidate_short: Optional[Tuple[float, ...]] = None



            if len(fits) > 0:
                idx = int(max(range(len(fits)), key=lambda i: fits[i]))
                epoch_best_fit = float(fits[idx])
                epoch_best = X[idx]
                detail_short: Dict[str, Any] = short_details[idx] if idx < len(short_details) else {}
                lambda_short_val = detail_short.get("lambda") if isinstance(detail_short, dict) else None
                if lambda_short_val is None:
                    lambda_short_val = -epoch_best_fit
                penalty_short_val = (
                    detail_short.get("penalty") if isinstance(detail_short, dict) else None
                )
                epoch_best_lambda_short = float(lambda_short_val)
                epoch_best_fitness_short = epoch_best_fit
                epoch_candidate_short = tuple(epoch_best)
                if epoch_best_fit > best_fit:
                    best_fit = epoch_best_fit
                    best = epoch_best
                    best_lambda_value = float(lambda_short_val)
                    stagnation_counter = 0
                    epoch_source = "short"
                    epoch_best_lambda_global = best_lambda_value
                    epoch_best_fitness_global = best_fit
                    if self.logger:
                        penalty_suffix = (
                            f" | penalty={penalty_short_val:.6f}"
                            if penalty_short_val is not None
                            else ""
                        )
                        self.logger.info(
                            "Epoch %d | new global best (short) lambda=%.6f | fitness=%.6f%s | masses=%s",
                            epoch,
                            best_lambda_value,
                            best_fit,
                            penalty_suffix,
                            tuple(round(float(v), 6) for v in epoch_best),
                        )
                else:
                    stagnation_counter += 1



            try:

                import numpy as _np



                ga._last_F = _np.array(fits, dtype=float).reshape(-1, 1)

            except Exception:

                ga._last_F = None  # type: ignore

            ga.step(generations=max(1, self.cfg.n_gen_step), epoch=epoch)



            if len(fits) > 0 and self.cfg.top_k_long > 0:

                k = min(self.cfg.top_k_long, len(fits))

                idxs = sorted(range(len(fits)), key=lambda i: fits[i], reverse=True)[:k]

                topX = [X[i] for i in idxs]

                long_fits, long_details = evaluator.evaluate_batch(
                    topX, horizon="long", epoch=epoch, return_details=True
                )

                self.metrics.eval_count += len(topX)

                long_evals = len(topX)

                for xi, f_long, detail_long in zip(topX, long_fits, long_details):
                    if f_long > best_fit:
                        lambda_long_val = (
                            detail_long.get("lambda") if isinstance(detail_long, dict) else None
                        )
                        if lambda_long_val is None:
                            lambda_long_val = -float(f_long)
                        penalty_long_val = (
                            detail_long.get("penalty") if isinstance(detail_long, dict) else None
                        )
                        best_fit = float(f_long)
                        best = xi
                        best_lambda_value = float(lambda_long_val)
                        epoch_source = "long"
                        stagnation_counter = 0
                        epoch_best_lambda_global = best_lambda_value
                        epoch_best_fitness_global = best_fit
                        if self.logger:
                            penalty_suffix = (
                                f" | penalty={penalty_long_val:.6f}"
                                if penalty_long_val is not None
                                else ""
                            )
                            self.logger.info(
                                "Epoch %d | new global best (long) lambda=%.6f | fitness=%.6f%s | masses=%s",
                                epoch,
                                best_lambda_value,
                                best_fit,
                                penalty_suffix,
                                tuple(round(float(v), 6) for v in xi),
                            )



            if best is not None:
                epoch_best_lambda_global = (
                    float(best_lambda_value) if best_lambda_value is not None else -float(best_fit)
                )
                epoch_best_fitness_global = float(best_fit)



            if stagnation_counter >= self.cfg.stagnation_window:

                if self.logger:

                    self.logger.info("Stagnation detected; reseeding around best candidate.")

                radius = self._modifier.on_stagnation(ga, best, radius)

                self.metrics.reseeds += 1

                stagnation_counter = 0



            epoch_duration = time.time() - epoch_start
            epoch_short_lambda_str = (
                f"{epoch_best_lambda_short:.6f}" if epoch_best_lambda_short is not None else "N/A"
            )
            epoch_short_fit_str = (
                f"{epoch_best_fitness_short:.6f}" if epoch_best_fitness_short is not None else "N/A"
            )
            epoch_global_lambda_str = (
                f"{epoch_best_lambda_global:.6f}" if epoch_best_lambda_global is not None else "N/A"
            )
            epoch_global_fit_str = (
                f"{epoch_best_fitness_global:.6f}" if epoch_best_fitness_global is not None else "N/A"
            )
            if self.logger:
                self.logger.info(

                    "Epoch %d complete | lambda_short=%s | fitness_short=%s | lambda_best=%s | fitness_best=%s | evals short/long=%d/%d | total evals=%d | radius=%.4f",
                    epoch,

                    epoch_short_lambda_str,

                    epoch_short_fit_str,
                    epoch_global_lambda_str,

                    epoch_global_fit_str,
                    short_evals,

                    long_evals,
                    self.metrics.eval_count,
                    radius,

                )
            if epoch_best_lambda_global is not None:

                self.metrics.best_lambda_per_epoch.append(epoch_best_lambda_global)
            else:

                self.metrics.best_lambda_per_epoch.append(float("nan"))

            if epoch_best_fitness_global is not None:

                self.metrics.best_fitness_per_epoch.append(epoch_best_fitness_global)
            else:

                self.metrics.best_fitness_per_epoch.append(float("nan"))

            self.metrics.record_epoch(

                {

                    "epoch": epoch,

                    "best_lambda_short": epoch_best_lambda_short,

                    "best_fitness_short": epoch_best_fitness_short,

                    "best_candidate_short": list(epoch_candidate_short) if epoch_candidate_short else None,

                    "best_lambda_global": epoch_best_lambda_global,

                    "best_fitness_global": epoch_best_fitness_global,

                    "best_candidate_global": list(best) if best else None,

                    "fitness_short": epoch_best_fitness_short,

                    "fitness_global": epoch_best_fitness_global,

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

        if best is not None:

            final_lambda = (

                float(best_lambda_value) if best_lambda_value is not None else -float(best_fit)

            )

        else:

            final_lambda = None

        if self.logger:

            self.logger.info(

                "Optimization completed | epochs=%d | evals=%d | best lambda=%s | wall=%.1fs",

                epoch,

                self.metrics.eval_count,

                f"{final_lambda:.6f}" if final_lambda is not None else "N/A",

                time.time() - run_start,

            )

        best_payload: Dict[str, Any] = {

            "masses": list(best) if best else None,

            "lambda": final_lambda,

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

