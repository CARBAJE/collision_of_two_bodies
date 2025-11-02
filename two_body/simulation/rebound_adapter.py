"""
Adaptador ligero sobre REBOUND para crear y propagar simulaciones.
"""

from __future__ import annotations

from typing import Any, Iterable, Tuple

from ..perf_timings.timers import time_block


class ReboundSim:
    def __init__(self, G: float = 1.0, integrator: str = "whfast") -> None:
        self.G = G
        self.integrator = integrator

    def setup_simulation(
        self,
        masses: Iterable[float],
        r0: Iterable[Iterable[float]],
        v0: Iterable[Iterable[float]],
        move_to_com: bool = True,
    ) -> Any:
        try:
            import rebound  # type: ignore
        except Exception as e:
            raise ImportError(
                "rebound is required for ReboundSim. Use --install-hint for help."
            ) from e

        mass_tuple = tuple(float(m) for m in masses)
        n_bodies = len(mass_tuple)
        if n_bodies < 2 or n_bodies > 3:
            raise ValueError("ReboundSim soporta entre 2 y 3 cuerpos en la configuración base.")

        def _normalize(vecs: Iterable[Iterable[float]], name: str) -> Tuple[Tuple[float, float, float], ...]:
            out: list[Tuple[float, float, float]] = []
            for idx, vec in enumerate(vecs):
                try:
                    x, y, z = vec
                except Exception as exc:
                    raise ValueError(f"{name}[{idx}] debe proporcionar tres componentes (x, y, z).") from exc
                out.append((float(x), float(y), float(z)))
            if len(out) != n_bodies:
                raise ValueError(f"{name} debe contener exactamente {n_bodies} vectores.")
            return tuple(out)

        pos_tuple = _normalize(r0, "r0")
        vel_tuple = _normalize(v0, "v0")

        sim = rebound.Simulation()
        sim.G = self.G
        sim.integrator = self.integrator

        for idx in range(n_bodies):
            sim.add(
                m=mass_tuple[idx],
                x=pos_tuple[idx][0],
                y=pos_tuple[idx][1],
                z=pos_tuple[idx][2],
                vx=vel_tuple[idx][0],
                vy=vel_tuple[idx][1],
                vz=vel_tuple[idx][2],
            )
        if move_to_com:
            sim.move_to_com()
        return sim

    def integrate(self, sim: Any, t_end: float, dt: float) -> Any:
        """Integra el sistema y devuelve la trayectoria como arreglo numpy [pasos, cuerpos, 6]."""
        import numpy as _np

        if not hasattr(sim, "particles"):
            raise ValueError("Se esperaba una instancia de rebound.Simulation.")

        sim.dt = dt
        steps = max(1, int(_np.floor(t_end / dt)))
        n_bodies = len(sim.particles)
        traj = _np.zeros((steps, n_bodies, 6), dtype=float)
        base_t = sim.t if hasattr(sim, "t") else 0.0
        for i in range(steps):
            t_target = base_t + (i + 1) * dt
            with time_block(
                "simulation_step",
                batch_id=i,
                extra={"dt": dt, "t_target": t_target, "n_bodies": n_bodies},
            ):
                sim.integrate(t_target)
                for j, p in enumerate(sim.particles):
                    traj[i, j, 0] = float(p.x)
                    traj[i, j, 1] = float(p.y)
                    traj[i, j, 2] = float(p.z)
                    traj[i, j, 3] = float(p.vx)
                    traj[i, j, 4] = float(p.vy)
                    traj[i, j, 5] = float(p.vz)
        return traj


if __name__ == "__main__":
    sim = ReboundSim()
    try:
        masses = (1.0, 1.0)
        r0 = ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        v0 = ((0.0, -0.5, 0.0), (0.0, 0.5, 0.0))
        ctx = sim.setup_simulation(masses, r0, v0)
        traj = sim.integrate(ctx, 10.0, 0.5)
        print("Trayectoria generada con shape", traj.shape)
    except ImportError as err:
        print("REBOUND no instalado:", err)
