"""
Motor de optimizacion evolutiva basado en pymoo (modo streaming).
"""

from __future__ import annotations
import random
from typing import Any, List, Optional, Tuple

from ..core.config import Config


class StreamingGA:
    def __init__(self, cfg: Config, logger: Optional[Any] = None) -> None:
        import logging

        self.cfg = cfg
        self.logger = logger or logging.getLogger("master_two_body_opt")
        self._asked_pop = None
        self._asked_X: Optional[Any] = None
        self._last_F: Optional[Any] = None
        self._rng = random.Random(cfg.seed)
        self._reseed_center: Optional[Tuple[float, float]] = None
        self._reseed_radius: float = cfg.local_radius

        try:
            from pymoo.algorithms.soo.nonconvex.ga import GA
            from pymoo.core.problem import Problem

            class _TwoMassProblem(Problem):
                def __init__(self, lb1: float, ub1: float, lb2: float, ub2: float):
                    import numpy as _np

                    super().__init__(
                        n_var=2,
                        n_obj=1,
                        n_constr=0,
                        xl=_np.array([lb1, lb2], dtype=float),
                        xu=_np.array([ub1, ub2], dtype=float),
                    )

                def _evaluate(self, X, out, *args, **kwargs):
                    import numpy as _np

                    out["F"] = _np.zeros((len(X), 1), dtype=float)

            lo1, hi1 = cfg.m1_bounds
            lo2, hi2 = cfg.m2_bounds
            self._problem = _TwoMassProblem(lo1, hi1, lo2, hi2)
            self._algorithm = GA(pop_size=cfg.pop_size, eliminate_duplicates=True)
            self._algorithm.setup(self._problem, termination=None, seed=cfg.seed)
        except Exception as e:
            self._problem = None
            self._algorithm = None
            if self.logger:
                self.logger.warning(
                    "pymoo unavailable or setup failed: %s. Using internal sampler.", e
                )
            lo1, hi1 = cfg.m1_bounds
            lo2, hi2 = cfg.m2_bounds
            self._pop: List[Tuple[float, float]] = [
                (
                    self._rng.uniform(lo1, hi1),
                    self._rng.uniform(lo2, hi2),
                )
                for _ in range(cfg.pop_size)
            ]

    def step(self, generations: float = 1.0) -> None:
        if self._algorithm is None:
            if getattr(self, "_pop", None) is None:
                self._pop = self.current_population()
            lo1, hi1 = self.cfg.m1_bounds
            lo2, hi2 = self.cfg.m2_bounds
            center = self._reseed_center
            radius = self._reseed_radius
            new_pop: List[Tuple[float, float]] = []
            for _ in range(self.cfg.pop_size):
                if center is not None:
                    m1 = min(max(center[0] + self._rng.uniform(-radius, radius), lo1), hi1)
                    m2 = min(max(center[1] + self._rng.uniform(-radius, radius), lo2), hi2)
                else:
                    m1 = self._rng.uniform(lo1, hi1)
                    m2 = self._rng.uniform(lo2, hi2)
                new_pop.append((m1, m2))
            self._pop = new_pop
            self._asked_pop = None
            self._asked_X = None
            return

        if self._asked_pop is None or self._asked_X is None:
            _ = self.current_population()
        if self._last_F is None:
            if self.logger:
                self.logger.debug("No fitness to tell; skipping GA step.")
            return
        try:
            self._asked_pop.set("F", self._last_F)
            self._algorithm.tell(infills=self._asked_pop)
            for _ in range(max(1, int(generations))):
                self._algorithm.next()
        finally:
            self._asked_pop = None
            self._asked_X = None
            self._last_F = None

    def current_population(self) -> List[Tuple[float, float]]:
        import numpy as _np

        if self._algorithm is None:
            if getattr(self, "_pop", None) is None:
                lo1, hi1 = self.cfg.m1_bounds
                lo2, hi2 = self.cfg.m2_bounds
                center = self._reseed_center
                radius = self._reseed_radius
                self._pop = [
                    (
                        min(
                            max(
                                (center[0] if center else self._rng.uniform(lo1, hi1))
                                + (self._rng.uniform(-radius, radius) if center else 0.0),
                                lo1,
                            ),
                            hi1,
                        ),
                        min(
                            max(
                                (center[1] if center else self._rng.uniform(lo2, hi2))
                                + (self._rng.uniform(-radius, radius) if center else 0.0),
                                lo2,
                            ),
                            hi2,
                        ),
                    )
                    for _ in range(self.cfg.pop_size)
                ]
            return list(self._pop)

        if self._asked_pop is not None and self._asked_X is not None:
            X = self._asked_X
        else:
            pop = self._algorithm.ask()
            X = pop.get("X")
            if self._reseed_center is not None and self._reseed_radius > 0:
                lo1, hi1 = self.cfg.m1_bounds
                lo2, hi2 = self.cfg.m2_bounds
                c1, c2 = self._reseed_center
                rad = self._reseed_radius
                for i in range(len(X)):
                    X[i, 0] = float(_np.clip(c1 + (self._rng.uniform(-rad, rad)), lo1, hi1))
                    X[i, 1] = float(_np.clip(c2 + (self._rng.uniform(-rad, rad)), lo2, hi2))
            pop.set("X", X)
            self._asked_pop = pop
            self._asked_X = X

        return [(float(x[0]), float(x[1])) for x in X]

    def warm_start_around(self, x: Tuple[float, float], radius: float) -> None:
        self._reseed_center = (float(x[0]), float(x[1]))
        self._reseed_radius = max(1e-9, float(radius))

    def reseed_around(self, x: Tuple[float, float], radius: float) -> None:
        self._reseed_center = (float(x[0]), float(x[1]))
        self._reseed_radius = max(1e-9, float(radius))
        self._asked_pop = None
        self._asked_X = None

    def local_exploration(self, center: Tuple[float, float], radius: float) -> None:
        self.warm_start_around(center, radius)


if __name__ == "__main__":
    from ..core.config import Config

    ga = StreamingGA(Config())
    print("Poblacion inicial:", ga.current_population()[:3])
