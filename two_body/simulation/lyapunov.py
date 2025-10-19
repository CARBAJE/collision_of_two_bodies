"""
Calculo (placeholder) del exponente de Lyapunov maximo.
"""
from __future__ import annotations
import math
from typing import Any, Dict, Iterable, Optional, Tuple
from .rebound_adapter import ReboundSim

class LyapunovEstimator:
    def mLCE(
        self,
        trajectory: Any = None,
        x0: Optional[Iterable[float]] = None,
        window: float = 100.0,
    ) -> Dict[str, Any]:
        try:
            lam, series, meta = self._mlce_with_context(trajectory, window)
            return {"lambda": lam, "series": series, "meta": meta}
        except Exception as e:
            return {"lambda": float("inf"), "series": None, "meta": {"impl": "error", "err": str(e)}}

    def _mlce_with_context(self, ctx: Any, window: float) -> Tuple[float, Optional[Any], Dict[str, Any]]:
        import numpy as _np

        try:
            import rebound  # noqa: F401
        except Exception:
            raise ValueError("REBOUND no esta instalado")

        if ctx is None:
            raise ValueError("Se requiere una simulacion o contexto discreto para estimar Lyapunov")

        if isinstance(ctx, dict) and "sim" in ctx:
            sim = ctx["sim"]
            dt = float(ctx.get("dt", 0.5))
            t_end = float(ctx.get("t_end", window))
            masses_ctx = ctx.get("masses")
        else:
            sim = ctx
            dt = 0.5
            t_end = window
            masses_ctx = None

        if masses_ctx is not None:
            masses = tuple(float(m) for m in masses_ctx)
        else:
            masses, _, _ = self._extract_rebound_state(sim)

        if dt <= 0.0:
            raise ValueError("dt debe ser positivo para el Lyapunov discreto")

        steps = max(1, math.ceil(t_end / dt)) if t_end is not None else 1

        lam, info = self._mlce_rebound_variational(sim, dt=dt, steps=steps)
        meta = {"steps": steps, "dt": dt, "n_bodies": len(masses), "masses": masses}
        if info:
            meta.update(info)
        meta.setdefault("impl", "rebound_megno")
        return lam, None, meta

    @staticmethod
    def _extract_rebound_state(
        sim: Any,
    ) -> Tuple[Tuple[float, ...], Tuple[Tuple[float, float, float], ...], Tuple[Tuple[float, float, float], ...]]:
        if not hasattr(sim, "particles"):
            raise ValueError("La simulacion proporcionada no contiene particulas.")

        masses: list[float] = []
        positions: list[Tuple[float, float, float]] = []
        velocities: list[Tuple[float, float, float]] = []

        for p in sim.particles:
            masses.append(float(getattr(p, "m", 0.0)))
            positions.append((float(p.x), float(p.y), float(p.z)))
            velocities.append((float(p.vx), float(p.vy), float(p.vz)))

        return tuple(masses), tuple(positions), tuple(velocities)

    def _mlce_rebound_variational(self, sim: Any, dt: float, steps: int) -> Tuple[float, Dict[str, Any]]:
        if not hasattr(sim, "particles"):
            raise RuntimeError("Invalid REBOUND Simulation provided")
        try:
            steps = max(1, int(steps))
            sim.dt = dt
            base_t = getattr(sim, "t", 0.0)

            sim.init_megno()
            for i in range(steps):
                sim.integrate(base_t + (i + 1) * dt)

            megno_val: Optional[float]
            megno_val = None
            try:
                megno_val = float(sim.calculate_megno())
            except AttributeError:
                megno_val = None

            lam_val = float(sim.lyapunov())
            info: Dict[str, Any] = {"impl": "rebound_megno"}
            if megno_val is not None:
                info["megno"] = megno_val
            return lam_val, info
        except AttributeError as exc:
            raise RuntimeError("REBOUND MEGNO API no disponible") from exc
        except Exception as e:
            raise RuntimeError(f"REBOUND MEGNO path failed: {e}")

if __name__ == "__main__":
    est = LyapunovEstimator()
    try:
        masses = (1.0, 1.0)
        r0 = ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        v0 = ((0.0, -0.5, 0.0), (0.0, 0.5, 0.0))
        sim = ReboundSim().setup_simulation(masses, r0, v0)
        print(est.mLCE({"sim": sim, "dt": 0.5, "t_end": 5.0, "masses": masses}))
    except ImportError as err:
        print("No es posible calcular Lyapunov sin REBOUND:", err)
