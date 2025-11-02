"""
Evaluador de fitness para candidatos (m1, m2) basado en Lyapunov.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from ..core.cache import HierarchicalCache
from ..core.config import Config
from ..simulation.lyapunov import LyapunovEstimator
from ..simulation.rebound_adapter import ReboundSim
from ..perf_timings.timers import time_block


class FitnessEvaluator:
    def __init__(self, cache: HierarchicalCache, cfg: Config, logger: Optional[Any] = None) -> None:
        import logging

        self.cache = cache
        self.cfg = cfg
        self.logger = logger or logging.getLogger("master_two_body_opt")
        self._batch_counter = 0

    def evaluate_batch(
        self,
        candidates: List[Tuple[float, ...]],
        horizon: str = "short",
        *,
        epoch: int = -1,
        return_details: bool = False,
    ) -> Union[List[float], Tuple[List[float], List[Dict[str, Any]]]]:
        import numpy as _np

        if len(candidates) == 0:
            return ([], []) if return_details else []

        t_end = self.cfg.t_end_short if horizon == "short" else self.cfg.t_end_long
        dt = self.cfg.dt
        sim_builder = ReboundSim(G=self.cfg.G, integrator=self.cfg.integrator)
        estimator = LyapunovEstimator()
        bounds = self.cfg.mass_bounds
        periodicity_weight = float(getattr(self.cfg, "periodicity_weight", 0.0) or 0.0)
        try:
            base_r0 = tuple(
                tuple(float(coord) for coord in vec) for vec in self.cfg.r0  # type: ignore[arg-type]
            )
            base_v0 = tuple(
                tuple(float(coord) for coord in vec) for vec in self.cfg.v0  # type: ignore[arg-type]
            )
        except Exception as exc:
            raise ValueError(
                "Config.r0 y Config.v0 deben contener vectores iterables numericos de longitud 3."
            ) from exc
        for name, base_vecs in (("r0", base_r0), ("v0", base_v0)):
            for idx, vec in enumerate(base_vecs):
                if len(vec) != 3:
                    raise ValueError(f"Config.{name}[{idx}] debe contener exactamente 3 componentes.")

        batch_id = self._next_batch_id()
        results: List[float] = []
        details: List[Dict[str, Any]] = []
        with time_block(
            "batch_eval",
            epoch=epoch,
            batch_id=batch_id,
            extra={"size": len(candidates), "horizon": horizon},
        ):
            for individual_id, masses_raw in enumerate(candidates):
                masses_tuple = tuple(float(m) for m in masses_raw)
                key_exact = (
                    tuple(round(m, 12) for m in masses_tuple)
                    + (horizon, round(periodicity_weight, 12))
                )
                timing_meta = {
                    "epoch": epoch,
                    "batch_id": batch_id,
                    "individual_id": individual_id,
                }
                with time_block(
                    "fitness_eval",
                    epoch=epoch,
                    batch_id=batch_id,
                    individual_id=individual_id,
                    extra={"horizon": horizon},
                ):
                    cached_payload = self.cache.get_exact(key_exact)
                    lam_val: Optional[float] = None
                    penalty_val: Optional[float] = None
                    status = "cache" if cached_payload is not None else "ok"
                    if cached_payload is not None:
                        if isinstance(cached_payload, dict):
                            fit = float(cached_payload.get("fitness", -float("inf")))
                            raw_lambda = cached_payload.get("lambda")
                            lam_val = float(raw_lambda) if raw_lambda is not None else None
                            raw_penalty = cached_payload.get("penalty")
                            penalty_val = float(raw_penalty) if raw_penalty is not None else None
                            status = str(cached_payload.get("status", status))
                        else:
                            fit = float(cached_payload)
                    else:
                        try:
                            masses = masses_tuple
                            if len(masses) != len(bounds):
                                raise ValueError(
                                    f"Cantidad de masas ({len(masses)}) no coincide con mass_bounds ({len(bounds)})."
                                )
                            for idx, m in enumerate(masses):
                                lo, hi = bounds[idx]
                                if m < lo or m > hi:
                                    raise ValueError(f"Masa[{idx}]={m} fuera de [{lo}, {hi}].")
                            if len(base_r0) < len(masses) or len(base_v0) < len(masses):
                                raise ValueError(
                                    "Config.r0 y Config.v0 deben definir al menos tantas entradas como masas."
                                )
                            r0 = base_r0[: len(masses)]
                            v0 = base_v0[: len(masses)]
                            sim = sim_builder.setup_simulation(masses=masses, r0=r0, v0=v0)
                            initial_r: Optional[_np.ndarray] = None
                            initial_v: Optional[_np.ndarray] = None
                            if periodicity_weight != 0.0:
                                initial_coords = []
                                initial_vels = []
                                for idx, particle in enumerate(sim.particles):
                                    if idx >= len(masses):
                                        break
                                    initial_coords.append(
                                        [float(particle.x), float(particle.y), float(particle.z)]
                                    )
                                    initial_vels.append(
                                        [float(particle.vx), float(particle.vy), float(particle.vz)]
                                    )
                                if initial_coords:
                                    initial_r = _np.asarray(initial_coords, dtype=float)
                                if initial_vels:
                                    initial_v = _np.asarray(initial_vels, dtype=float)
                            ctx = {
                                "sim": sim,
                                "dt": dt,
                                "t_end": t_end,
                                "masses": masses,
                                "timing_meta": timing_meta,
                            }
                            if len(masses) > 0:
                                ctx["m1"] = masses[0]
                            if len(masses) > 1:
                                ctx["m2"] = masses[1]
                            if len(masses) > 2:
                                ctx["m3"] = masses[2]
                            ret = estimator.mLCE(ctx, window=t_end, timing_meta=timing_meta)
                            lam = float(ret.get("lambda", _np.inf))
                            lam_val = lam
                            if not _np.isfinite(lam):
                                status = "nonfinite_lambda"
                                self.logger.debug(
                                    "Lyapunov no finito para %s (horizon=%s): %s",
                                    masses_tuple,
                                    horizon,
                                    lam,
                                )
                                fit = -float("inf")
                            else:
                                penalty = 0.0
                                if (
                                    periodicity_weight != 0.0
                                    and initial_r is not None
                                    and initial_v is not None
                                ):
                                    try:
                                        rf_list: list[list[float]] = []
                                        vf_list: list[list[float]] = []
                                        for idx, particle in enumerate(sim.particles):
                                            if idx >= len(masses):
                                                break
                                            rf_list.append(
                                                [float(particle.x), float(particle.y), float(particle.z)]
                                            )
                                            vf_list.append(
                                                [float(particle.vx), float(particle.vy), float(particle.vz)]
                                            )
                                        rf_arr = _np.asarray(rf_list, dtype=float)
                                        vf_arr = _np.asarray(vf_list, dtype=float)
                                        if rf_arr.shape == initial_r.shape and vf_arr.shape == initial_v.shape:
                                            delta_r = _np.linalg.norm(rf_arr - initial_r, axis=1).sum()
                                            delta_v = _np.linalg.norm(vf_arr - initial_v, axis=1).sum()
                                            penalty = float(delta_r + delta_v)
                                        if not _np.isfinite(penalty):
                                            self.logger.debug(
                                                "Penalizacion no finita: delta_r=%.3e, delta_v=%.3e",
                                                delta_r,
                                                delta_v,
                                            )
                                            penalty = float("inf")
                                        else:
                                            self.logger.debug(
                                                "delta_r=%.6f, delta_v=%.6f, penalty=%.6f",
                                                delta_r,
                                                delta_v,
                                                penalty,
                                            )
                                            penalty = min(penalty, 1e6)
                                    except Exception:
                                        penalty = 0.0
                                penalty_val = penalty
                                fit = -lam - periodicity_weight * penalty
                        except Exception as exc:
                            status = "error"
                            self.logger.debug(
                                "Fallo evaluando candidato %s (horizon=%s): %s",
                                masses_tuple,
                                horizon,
                                exc,
                                exc_info=True,
                            )
                            fit = -float("inf")
                            lam_val = lam_val if lam_val is not None else None
                        cache_payload = {
                            "fitness": float(fit),
                            "lambda": float(lam_val) if lam_val is not None else None,
                            "penalty": float(penalty_val) if penalty_val is not None else None,
                            "status": status,
                        }
                        self.cache.set_exact(key_exact, cache_payload)

                results.append(float(fit))
                details.append(
                    {
                        "masses": masses_tuple,
                        "fitness": float(fit),
                        "lambda": float(lam_val) if lam_val is not None else None,
                        "penalty": float(penalty_val) if penalty_val is not None else None,
                        "cached": cached_payload is not None,
                        "status": status,
                        "horizon": horizon,
                    }
                )

        if return_details:
            return results, details
        return results

    def _next_batch_id(self) -> int:
        batch_id = self._batch_counter
        self._batch_counter += 1
        return batch_id


if __name__ == "__main__":
    from ..core.config import Config
    from ..core.cache import HierarchicalCache

    cfg = Config()
    cache = HierarchicalCache()
    evaluator = FitnessEvaluator(cache, cfg)
    print("Evaluacion ficticia:", evaluator.evaluate_batch([(1.0, 1.0)], horizon="short"))
