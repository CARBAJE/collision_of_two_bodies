#!/usr/bin/env python3
"""
Master optimizer para la estabilidad de dos cuerpos (arquitectura modular).

El antiguo script monolitico se dividio en paquetes que reflejan la capa de
presentacion, simulacion y logica mostradas en el diagrama del usuario:

* Capa de presentacion (`two_body.presentation`):
  - `ui.py`: placeholder de interfaz Qt.
  - `visualization.py`: visualizacion rapida con Matplotlib.
* Capa de simulacion (`two_body.simulation`):
  - `rebound_adapter.py`: integracion con REBOUND.
  - `lyapunov.py`: calculo del exponente de Lyapunov.
  - `dynamics.py`: dinamica basica CPU/GPU.
* Capa logica (`two_body.logic`):
  - `ga.py`: motor de optimizacion (pymoo / muestreo interno).
  - `fitness.py`: evaluacion de candidatos.
  - `parameters.py`: modificacion de parametros del sistema.
  - `controller.py`: orquestador continuo.
* Nucleo (`two_body.core`):
  - Configuracion, telemetria y caches.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from two_body.core.config import Config, set_global_seeds
from two_body.core.telemetry import Reporter, setup_logger
from two_body.logic.controller import ContinuousOptimizationController
from two_body.presentation.visualization import Visualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-body GA optimizer (modular skeleton)")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pop-size", type=int, default=64)
    parser.add_argument("--n-gen-step", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--top-k-long", type=int, default=8)

    parser.add_argument("--t-end-short", type=float, default=100.0)
    parser.add_argument("--t-end-long", type=float, default=1000.0)
    parser.add_argument("--dt", type=float, default=0.5)

    parser.add_argument("--time-budget-s", type=float, default=1800.0)
    parser.add_argument("--eval-budget", type=int, default=10000)

    parser.add_argument("--use-gpu", type=str, default="auto", choices=["auto", "true", "false"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cache-approx-max", type=int, default=5000)
    parser.add_argument("--cache-exact-max", type=int, default=2000)

    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--log-level", type=str, default="INFO")

    parser.add_argument(
        "--install-hint",
        action="store_true",
        help="Imprime sugerencias de instalacion para dependencias opcionales y termina.",
    )
    parser.add_argument(
        "--run-optimization",
        action="store_true",
        help="Ejecuta un ciclo del controlador continuo (puede requerir dependencias pesadas).",
    )
    parser.add_argument(
        "--preview-visualization",
        action="store_true",
        help="Genera una visualizacion rapida de prueba si Matplotlib esta disponible.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        t_end_short=args.t_end_short,
        t_end_long=args.t_end_long,
        dt=args.dt,
        integrator="whfast",
        preset_orbita="circular_cm",
        pop_size=args.pop_size,
        n_gen_step=args.n_gen_step,
        max_epochs=args.max_epochs,
        top_k_long=args.top_k_long,
        time_budget_s=args.time_budget_s,
        eval_budget=args.eval_budget,
        use_gpu=args.use_gpu,
        batch_size=args.batch_size,
        cache_exact_max=args.cache_exact_max,
        cache_approx_max=args.cache_approx_max,
        artifacts_dir=args.artifacts_dir,
        save_plots=args.save_plots,
        headless=args.headless,
        seed=args.seed,
    )


def print_install_hint() -> None:
    pkgs = [
        "numpy",
        "matplotlib",
        "rebound",
        "pymoo",
        "joblib",
        "psutil",
        "cupy-cuda11x (opcional, GPU)",
        "numba (opcional)",
    ]
    lines = [
        "Dependencias opcionales no requeridas en el esqueleto:",
        " - " + ", ".join(pkgs[:4]),
        " - " + ", ".join(pkgs[4:]),
        "Instalacion base:",
        "   pip install numpy matplotlib rebound pymoo joblib psutil",
        "Extras GPU:",
        "   pip install cupy-cuda12x numba  # seleccione la version de CUDA adecuada",
    ]
    print("\n".join(lines))


def maybe_run_controller(cfg: Config, logger: Any) -> None:
    controller = ContinuousOptimizationController(cfg, logger=logger)
    try:
        result = controller.run()
        reporter = Reporter(cfg.artifacts_dir, logger=logger)
        reporter.save_results(result)
        logger.info("Optimizacion finalizada. Resultado: %s", result["best"])
    except Exception as exc:
        logger.error(
            "Fallo al ejecutar la optimizacion. Verifique dependencias opcionales: %s",
            exc,
        )


def maybe_preview_visualization(cfg: Config) -> None:
    viz = Visualizer(headless=cfg.headless)
    viz.quick_view([])


def main() -> None:
    args = parse_args()
    if args.install_hint:
        print_install_hint()
        return

    logger = setup_logger(args.log_level)
    cfg = build_config(args)
    set_global_seeds(cfg.seed)

    reporter = Reporter(cfg.artifacts_dir, logger=logger)
    reporter.bootstrap(cfg)

    logger.info("Arquitectura modular inicializada. Config guardada en %s", os.path.abspath(cfg.artifacts_dir))
    logger.info("Capa presentacion -> two_body.presentation.*")
    logger.info("Capa simulacion -> two_body.simulation.*")
    logger.info("Capa logica -> two_body.logic.*")

    if args.run_optimization:
        maybe_run_controller(cfg, logger)
    if args.preview_visualization:
        maybe_preview_visualization(cfg)


if __name__ == "__main__":
    main()
