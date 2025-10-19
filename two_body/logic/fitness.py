"""
Evaluador de fitness para candidatos (m1, m2) basado en Lyapunov.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from ..core.cache import HierarchicalCache
from ..core.config import Config
from ..simulation.lyapunov import LyapunovEstimator
from ..simulation.rebound_adapter import ReboundSim


class FitnessEvaluator:
    def __init__(self, cache: HierarchicalCache, cfg: Config, logger: Optional[Any] = None) -> None:
        import logging

        self.cache = cache
        self.cfg = cfg
        self.logger = logger or logging.getLogger("master_two_body_opt")

    def evaluate_batch(self, candidates: List[Tuple[float, float]], horizon: str = "short") -> List[float]:
        import numpy as _np

        if len(candidates) == 0:
            return []

        t_end = self.cfg.t_end_short if horizon == "short" else self.cfg.t_end_long
        dt = self.cfg.dt
        sim_builder = ReboundSim(G=self.cfg.G, integrator=self.cfg.integrator)
        estimator = LyapunovEstimator()
        base_r0 = self.cfg.r0
        base_v0 = self.cfg.v0

        results: List[float] = []
        for (m1, m2) in candidates:
            key_exact = (round(float(m1), 12), round(float(m2), 12), horizon)
            cached = self.cache.get_exact(key_exact)
            if cached is not None:
                results.append(float(cached))
                continue

            try:
                masses = (float(m1), float(m2))
                if len(base_r0) < len(masses) or len(base_v0) < len(masses):
                    raise ValueError("Config.r0 y Config.v0 deben definir al menos tantas entradas como masas.")
                r0 = tuple(base_r0[i] for i in range(len(masses)))
                v0 = tuple(base_v0[i] for i in range(len(masses)))
                sim = sim_builder.setup_simulation(masses=masses, r0=r0, v0=v0)
                ctx = {
                    "sim": sim,
                    "dt": dt,
                    "t_end": t_end,
                    "masses": masses,
                    "m1": masses[0],
                    "m2": masses[1] if len(masses) > 1 else None,
                }
                ret = (
                    estimator.mLCE_short(ctx, window=t_end)
                    if horizon == "short"
                    else estimator.mLCE_long(ctx, window=t_end)
                )
                lam = float(ret.get("lambda", _np.inf))
                fit = -lam
            except Exception:
                fit = -float("inf")

            self.cache.set_exact(key_exact, fit)
            results.append(fit)

        return results


if __name__ == "__main__":
    from ..core.config import Config
    from ..core.cache import HierarchicalCache

    cfg = Config()
    cache = HierarchicalCache()
    evaluator = FitnessEvaluator(cache, cfg)
    print("Evaluacion ficticia:", evaluator.evaluate_batch([(1.0, 1.0)], horizon="short"))
