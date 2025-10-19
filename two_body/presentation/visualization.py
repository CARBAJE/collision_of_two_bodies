"""
Modulo de visualizacion con Matplotlib (placeholder sin dependencias fuertes).
"""

from __future__ import annotations

from typing import Any, Iterable, Optional


class Visualizer:
    def __init__(self, headless: bool = True) -> None:
        self.headless = headless

    def quick_view(
        self,
        trajectories: Optional[Iterable[Any]] = None,
        title: str = "Visualizacion de trayectoria",
    ) -> None:
        """
        Renderiza una vista rapida siempre que Matplotlib este disponible.
        """
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception as exc:
            print("Matplotlib no disponible, se omite la visualizacion:", exc)
            return

        plt.figure(figsize=(6, 6))
        if trajectories:
            for idx, traj in enumerate(trajectories):
                if traj is None:
                    continue
                xs = traj[:, 0]
                ys = traj[:, 1]
                plt.plot(xs, ys, label=f"Cuerpo {idx+1}")
        else:
            plt.text(0.5, 0.5, "Sin datos", ha="center", va="center")
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        if self.headless:
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    import numpy as np

    dummy = np.column_stack((np.cos(np.linspace(0, 2 * np.pi, 100)), np.sin(np.linspace(0, 2 * np.pi, 100))))
    viz = Visualizer(headless=False)
    viz.quick_view([dummy])
