"""
Configuraciones base y utilidades de seeding para el optimizador de dos cuerpos.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import numpy as np  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - dependencia opcional
    np = None  # type: ignore


@dataclass
class Config:
    """
    Parametros de simulacion, busqueda evolutiva y recursos de ejecucion.

    Se mantiene identico al esquema previo para preservar compatibilidad con
    scripts existentes.
    """

    # Simulacion
    t_end_short: float = 100.0
    t_end_long: float = 1000.0
    dt: float = 0.5
    integrator: str = "whfast"
    periodicity_weight: float = 0.0
    r0: Tuple[Tuple[float, float, float], ...] = (
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    v0: Tuple[Tuple[float, float, float], ...] = (
        (0.0, -0.5, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, 0.0),
    )

    # Parametros fisicos
    mass_bounds: Tuple[Tuple[float, float], ...] = (
        (0.05, 10.0),
        (0.05, 10.0),
    )
    mass_distribution: str = "uniform"  # "uniform" | "beta"
    mass_beta_alpha: float = 1.0
    mass_beta_beta: float = 1.0
    G: float = 1.0
    # Estado inicial estandar (legacy) para compatibilidad con scripts antiguos
    x0: Tuple[float, ...] = (-1.0, 0.0, 0.0, -0.5, 1.0, 0.0, 0.0, 0.5)

    # GA
    pop_size: int = 64
    n_gen_step: int = 3
    crossover: float = 0.9
    mutation: float = 0.2
    selection: str = "tournament"
    elitism: int = 1
    seed: int = 42

    # Optimizacion continua
    max_epochs: int = 50
    top_k_long: int = 8
    stagnation_window: int = 5
    stagnation_tol: float = 1e-3
    local_radius: float = 0.1
    radius_decay: float = 0.9
    time_budget_s: float = 1800.0
    eval_budget: int = 10000

    # Backend
    use_gpu: str = "auto"  # "auto" | "true" | "false"
    batch_size: int = 64
    cache_exact_max: int = 2000
    cache_approx_max: int = 5000

    # I/O
    artifacts_dir: str = "artifacts"
    save_plots: bool = False
    headless: bool = True


def set_global_seeds(seed: int) -> None:
    """Inicializa generadores pseudoaleatorios reproducibles."""
    random.seed(seed)
    if np is not None:
        try:
            np.random.seed(seed)  # type: ignore[call-arg]
        except Exception:
            pass


if __name__ == "__main__":
    cfg = Config()
    set_global_seeds(cfg.seed)
    print("Config de prueba:", cfg)
