"""
Motor de optimizacion evolutiva basado en pymoo (modo streaming).
"""

from __future__ import annotations
import random
from typing import Any, List, Optional, Sequence, Tuple

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
        self._reseed_center: Optional[Tuple[float, ...]] = None
        self._reseed_radius: float = cfg.local_radius
        self._bounds = [tuple(map(float, b)) for b in cfg.mass_bounds]
        self._dim = len(self._bounds)

        try:
            from pymoo.algorithms.soo.nonconvex.ga import GA
            from pymoo.core.problem import Problem
            from pymoo.operators.crossover.sbx import SBX
            from pymoo.operators.mutation.pm import PolynomialMutation
            from pymoo.operators.selection.tournament import TournamentSelection

            class _MassProblem(Problem):
                def __init__(self, lows: Sequence[float], highs: Sequence[float]):
                    import numpy as _np

                    super().__init__(
                        n_var=len(lows),
                        n_obj=1,
                        n_constr=0,
                        xl=_np.array(lows, dtype=float),
                        xu=_np.array(highs, dtype=float),
                    )

                def _evaluate(self, X, out, *args, **kwargs):
                    import numpy as _np

                    out["F"] = _np.zeros((len(X), 1), dtype=float)

            lows = [b[0] for b in self._bounds]
            highs = [b[1] for b in self._bounds]
            self._problem = _MassProblem(lows, highs)
            crossover = SBX(prob=max(0.0, min(1.0, float(cfg.crossover))), eta=15.0)
            mut_prob = float(cfg.mutation)
            if mut_prob <= 0.0:
                mut_prob = 1.0 / self._dim
            mutation = PolynomialMutation(prob=min(1.0, mut_prob), eta=20.0)
            def _tournament(pop, P, **_kwargs):
                import numpy as _np

                F = pop.get("F")
                winners: list[int] = []
                for row in _np.atleast_2d(P):
                    best_idx = int(row[0])
                    best_fit = float(F[best_idx, 0])
                    for cand in row[1:]:
                        cand_idx = int(cand)
                        cand_fit = float(F[cand_idx, 0])
                        if cand_fit > best_fit:
                            best_idx = cand_idx
                            best_fit = cand_fit
                    winners.append(best_idx)
                return _np.array(winners, dtype=int)

            selection = TournamentSelection(func_comp=_tournament)

            self._algorithm = GA(
                pop_size=cfg.pop_size,
                eliminate_duplicates=bool(cfg.elitism > 0),
                crossover=crossover,
                mutation=mutation,
                selection=selection if cfg.selection == "tournament" else None,
            )
            if cfg.selection != "tournament" and self.logger:
                self.logger.warning(
                    "Selection '%s' not supported explicitly; pymoo default will be used.",
                    cfg.selection,
                )
            self._algorithm.setup(self._problem, termination=None, seed=cfg.seed)
        except Exception as e:
            self._problem = None
            self._algorithm = None
            if self.logger:
                self.logger.warning(
                    "pymoo unavailable or setup failed: %s. Using internal sampler.", e
                )
            self._pop = self._init_random_population()

    def _init_random_population(self) -> List[Tuple[float, ...]]:
        return [
            tuple(self._rng.uniform(*self._bounds[d]) for d in range(self._dim))
            for _ in range(self.cfg.pop_size)
        ]

    def _sample_offspring(self, bounds: List[Tuple[float, float]]) -> List[float]:
        child = [self._rng.uniform(*bounds[idx]) for idx in range(self._dim)]
        population = getattr(self, "_pop", [])
        cross_prob = max(0.0, min(1.0, float(self.cfg.crossover)))
        if population and len(population) >= 2 and self._rng.random() < cross_prob:
            p1, p2 = self._rng.sample(population, 2)
            child = [
                self._rng.uniform(min(p1[j], p2[j]), max(p1[j], p2[j]))
                for j in range(self._dim)
            ]
        mut_prob = float(self.cfg.mutation)
        if mut_prob <= 0.0:
            mut_prob = 1.0 / self._dim
        if self._rng.random() < min(1.0, mut_prob):
            gene_idx = self._rng.randrange(self._dim)
            span = bounds[gene_idx][1] - bounds[gene_idx][0]
            perturb = self._rng.uniform(-0.5, 0.5) * span * 0.1
            child[gene_idx] = min(
                max(child[gene_idx] + perturb, bounds[gene_idx][0]),
                bounds[gene_idx][1],
            )
        return child

    def step(self, generations: float = 1.0) -> None:
        if self._algorithm is None:
            if getattr(self, "_pop", None) is None:
                self._pop = self._init_random_population()
            bounds = self._bounds
            center = self._reseed_center
            radius = self._reseed_radius
            elites: List[Tuple[float, ...]] = []
            if (
                self.cfg.elitism > 0
                and self._last_F is not None
                and getattr(self, "_pop", None) is not None
            ):
                try:
                    scores = [float(val[0]) for val in list(self._last_F)]
                except Exception:
                    scores = []
                if scores:
                    ranked = sorted(
                        range(len(self._pop)),
                        key=lambda i: scores[i],
                        reverse=True,
                    )
                    for idx in ranked[: min(self.cfg.elitism, len(self._pop))]:
                        elites.append(self._pop[idx])
            new_pop: List[Tuple[float, ...]] = list(elites)
            while len(new_pop) < self.cfg.pop_size:
                if center is not None:
                    genes = [
                        min(
                            max(
                                center[idx] + self._rng.uniform(-radius, radius),
                                bounds[idx][0],
                            ),
                            bounds[idx][1],
                        )
                        for idx in range(self._dim)
                    ]
                else:
                    genes = self._sample_offspring(bounds)
                new_pop.append(tuple(genes))
            self._pop = new_pop
            self._asked_pop = None
            self._asked_X = None
            self._last_F = None
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

    def current_population(self) -> List[Tuple[float, ...]]:
        import numpy as _np

        if self._algorithm is None:
            if getattr(self, "_pop", None) is None:
                bounds = self._bounds
                center = self._reseed_center
                radius = self._reseed_radius
                self._pop = [
                    tuple(
                        min(
                            max(
                                (center[idx] if center else self._rng.uniform(*bounds[idx]))
                                + (self._rng.uniform(-radius, radius) if center else 0.0),
                                bounds[idx][0],
                            ),
                            bounds[idx][1],
                        )
                        for idx in range(self._dim)
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
                bounds = self._bounds
                center = self._reseed_center
                rad = self._reseed_radius
                for i in range(len(X)):
                    for j in range(self._dim):
                        X[i, j] = float(
                            _np.clip(
                                center[j] + self._rng.uniform(-rad, rad),
                                bounds[j][0],
                                bounds[j][1],
                            )
                        )
            pop.set("X", X)
            self._asked_pop = pop
            self._asked_X = X

        return [tuple(float(x[j]) for j in range(self._dim)) for x in X]

    def warm_start_around(self, x: Tuple[float, ...], radius: float) -> None:
        self._reseed_center = tuple(float(xi) for xi in x)
        self._reseed_radius = max(1e-9, float(radius))

    def reseed_around(self, x: Tuple[float, ...], radius: float) -> None:
        self._reseed_center = tuple(float(xi) for xi in x)
        self._reseed_radius = max(1e-9, float(radius))
        self._asked_pop = None
        self._asked_X = None

    def local_exploration(self, center: Tuple[float, ...], radius: float) -> None:
        self.warm_start_around(center, radius)


if __name__ == "__main__":
    from ..core.config import Config

    ga = StreamingGA(Config())
    print("Poblacion inicial:", ga.current_population()[:3])
