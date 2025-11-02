#!/usr/bin/env python
"""
Demo de referencia para validar la visualizacion y la dinamica Sol-Tierra.

El script reutiliza los componentes del paquete `two_body` para:
  1. Construir una simulacion con datos astronomicos simplificados (UA, anos, masas solares)
  2. Integrar la orbita circular de la Tierra alrededor del Sol
  3. Mostrar la traza 2D/3D existentes en el proyecto
  4. Registrar metricas clave que ayudan a diagnosticar el comportamiento (radio orbital,
     variacion de energia, periodos aproximados, etc.)
"""

from __future__ import annotations

import sys
import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

#
# Ajustar PYTHONPATH automáticamente cuando se ejecuta el script via ruta absoluta.
#
_THIS_FILE = Path(__file__).resolve()
_PACKAGE_ROOT = _THIS_FILE.parent.parent  # .../two_body
_PROJECT_PARENT = _PACKAGE_ROOT.parent

if str(_PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_PARENT))

from two_body import Config, set_global_seeds  # noqa: E402
from two_body.presentation.triDTry import Visualizer as Visualizer3D  # noqa: E402
from two_body.presentation.visualization import Visualizer as PlanarVisualizer  # noqa: E402
from two_body.simulation.rebound_adapter import ReboundSim  # noqa: E402


LOGGER_NAME = "two_body.demo_tierra"


@dataclass
class DemoOptions:
    """Parametros de linea de comandos para el demo."""

    duration_years: float = 1.0
    time_step: float = 0.002
    headless: bool = False
    move_to_com: bool = False
    disable_3d: bool = False
    disable_2d: bool = False
    log_level: str = "INFO"


def parse_args() -> DemoOptions:
    parser = argparse.ArgumentParser(description="Simulacion de la orbita Tierra-Sol")
    parser.add_argument(
        "--duration-years",
        type=float,
        default=1.0,
        help="Tiempo total de integracion en anos (default: 1.0).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.002,
        help="Paso temporal en anos (default: 0.002).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Cerrar figuras automaticamente (util en pipelines).",
    )
    parser.add_argument(
        "--move-to-com",
        action="store_true",
        help="Reposicionar el sistema en el centro de masa (por defecto el Sol queda fijo).",
    )
    parser.add_argument(
        "--no-3d",
        action="store_true",
        help="Desactivar la animacion 3D.",
    )
    parser.add_argument(
        "--no-2d",
        action="store_true",
        help="Desactivar la visualizacion en 2D.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Nivel de logging (DEBUG, INFO, ...).",
    )
    args = parser.parse_args()
    return DemoOptions(
        duration_years=args.duration_years,
        time_step=args.dt,
        headless=args.headless,
        move_to_com=args.move_to_com,
        disable_3d=args.no_3d,
        disable_2d=args.no_2d,
        log_level=args.log_level.upper(),
    )


def setup_logging(level: str) -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)5s | %(name)s | %(message)s",
        level=getattr(logging, level, logging.INFO),
    )
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level, logging.INFO))
    return logger


def build_config(opts: DemoOptions) -> Config:
    """
    Crea una configuracion consistente en unidades astronomicas:
        - distancias en UA
        - tiempos en anos
        - masas en masas solares
    """
    earth_mass = 3.0e-6  # ~ M_earth / M_sun
    cfg = Config(
        t_end_short=opts.duration_years,
        t_end_long=opts.duration_years,
        dt=opts.time_step,
        integrator="whfast",
        seed=12345,
        headless=opts.headless,
        G=4 * np.pi**2,
        r0=(
            (0.0, 0.0, 0.0),  # Sol
            (1.0, 0.0, 0.0),  # Tierra
        ),
        v0=(
            (0.0, 0.0, 0.0),
            (0.0, 2 * np.pi, 0.0),  # 1 orbita por ano
        ),
        mass_bounds=(
            (1.0, 1.0),
            (earth_mass, earth_mass),
        ),
        pop_size=1,
        n_gen_step=1,
        mutation=0.0,
        crossover=0.0,
        elitism=1,
        max_epochs=1,
        top_k_long=1,
        stagnation_window=1,
        stagnation_tol=1e-6,
        local_radius=0.0,
        radius_decay=1.0,
        time_budget_s=60.0,
        eval_budget=4,
        periodicity_weight=0.0,
        artifacts_dir="artifacts/demo_tierra",
        save_plots=False,
    )
    return cfg


def summarize_trajectory(
    logger: logging.Logger,
    traj: np.ndarray,
    masses: Sequence[float],
    cfg: Config,
) -> None:
    """
    Calcula metricas utiles para diagnosticar el estado orbital.
    """
    positions = traj[:, :, :3]
    velocities = traj[:, :, 3:6]
    masses_arr = np.asarray(masses)

    barycenter = np.average(positions, axis=1, weights=masses_arr)
    rel_positions = positions - barycenter[:, np.newaxis, :]
    bary_velocity = np.average(velocities, axis=1, weights=masses_arr)
    rel_velocities = velocities - bary_velocity[:, np.newaxis, :]

    logger.info("Resumen de simulacion")
    logger.info("  pasos=%d | dt=%.6f anos | duracion total=%.3f anos", len(traj), cfg.dt, cfg.t_end_long)
    logger.info("  masas=%s (M_sun) | G=%.6f", masses, cfg.G)
    logger.info("  centro de masa: desplazamiento maximo = %.3e UA", np.linalg.norm(barycenter, axis=1).max())

    for idx in range(len(masses_arr)):
        radius = np.linalg.norm(rel_positions[:, idx, :2], axis=1)
        speed = np.linalg.norm(rel_velocities[:, idx, :2], axis=1)
        logger.info(
            "  cuerpo %d -> radio[min, max]=[%.4f, %.4f] UA | radio sigma=%.4e | velocidad media=%.4f UA/ano",
            idx,
            radius.min(),
            radius.max(),
            radius.std(),
            speed.mean(),
        )

    total_energy = compute_total_energy(rel_positions, rel_velocities, masses_arr, cfg.G)
    logger.info(
        "  energia total (media)=%.6e | variacion relativa=%.3e",
        total_energy.mean(),
        (total_energy.max() - total_energy.min()) / abs(total_energy.mean()),
    )

    approx_period = estimate_orbital_period(rel_positions[:, 1, :2], cfg.dt)
    if approx_period is not None:
        logger.info("  periodo orbital estimado para la Tierra ~= %.6f anos", approx_period)
        logger.info("  error relativo vs 1 ano ~= %.3e", abs(approx_period - 1.0))
    else:
        logger.warning("  no se pudo estimar periodo orbital (trayectoria no cerrada dentro del intervalo)")


def compute_total_energy(
    rel_positions: np.ndarray,
    rel_velocities: np.ndarray,
    masses: np.ndarray,
    G: float,
) -> np.ndarray:
    """
    Energia total del sistema (cinetica + potencial) para cada instante.
    """
    kinetic = 0.5 * masses[np.newaxis, :] * np.sum(rel_velocities**2, axis=2)
    total_energy = np.sum(kinetic, axis=1)

    n_bodies = rel_positions.shape[1]
    for i in range(n_bodies):
        for j in range(i + 1, n_bodies):
            dist = np.linalg.norm(rel_positions[:, i, :] - rel_positions[:, j, :], axis=1)
            total_energy -= G * masses[i] * masses[j] / dist
    return total_energy


def estimate_orbital_period(track_xy: np.ndarray, dt: float) -> float | None:
    """
    Periodo aproximado con base en el avance angular acumulado.

    Se calcula el angulo polar para cada muestra, se "desenvuelve" (unwrap) y se
    busca el primer instante en que la variacion alcanza +- 2*pi dependiendo del
    sentido de giro. Si no se completa una vuelta en el intervalo, se retorna None.
    """
    if track_xy.shape[0] < 2:
        return None

    x = track_xy[:, 0]
    y = track_xy[:, 1]
    angles = np.unwrap(np.arctan2(y, x))
    delta = angles - angles[0]

    avg_rate = np.mean(np.diff(angles))
    if np.isclose(avg_rate, 0.0):
        return None
    target = 2 * np.pi if avg_rate > 0 else -2 * np.pi

    crossing_idx = np.where(delta >= target if target > 0 else delta <= target)[0]
    if crossing_idx.size == 0:
        return None

    idx = crossing_idx[0]
    if idx == 0:
        return idx * dt

    prev = idx - 1
    delta_prev = delta[prev]
    delta_curr = delta[idx]
    if np.isclose(delta_curr, delta_prev):
        return idx * dt

    fraction = (target - delta_prev) / (delta_curr - delta_prev)
    return (prev + fraction) * dt


def main() -> None:
    opts = parse_args()
    logger = setup_logging(opts.log_level)
    cfg = build_config(opts)
    cfg.headless = opts.headless

    set_global_seeds(cfg.seed)
    masses = tuple(bounds[0] for bounds in cfg.mass_bounds)

    logger.info(
        "Iniciando demo con duration_years=%.2f, dt=%.4f, headless=%s, move_to_com=%s",
        cfg.t_end_long,
        cfg.dt,
        cfg.headless,
        opts.move_to_com,
    )

    sim_builder = ReboundSim(G=cfg.G, integrator=cfg.integrator)
    sim = sim_builder.setup_simulation(
        masses=masses,
        r0=cfg.r0,
        v0=cfg.v0,
        move_to_com=opts.move_to_com,
    )
    trajectory = sim_builder.integrate(sim, t_end=cfg.t_end_long, dt=cfg.dt)
    summarize_trajectory(logger, trajectory, masses, cfg)

    xyz_tracks = [trajectory[:, i, :3] for i in range(trajectory.shape[1])]

    if not opts.disable_2d:
        logger.info("Mostrando visualizacion 2D...")
        viz2d = PlanarVisualizer(headless=cfg.headless)
        viz2d.quick_view(xyz_tracks)

    if not opts.disable_3d:
        logger.info("Generando animacion 3D...")
        viz3d = Visualizer3D(headless=cfg.headless, cfg=cfg, sim_builder=sim_builder)
        viz3d.animate_3d(
            trajectories=xyz_tracks,
            interval_ms=40,
            title="Orbita Sol-Tierra (UA, anos)",
            total_frames=len(xyz_tracks[0]),
        )

    logger.info("Demo finalizado.")


if __name__ == "__main__":
    main()
