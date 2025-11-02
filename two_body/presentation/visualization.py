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
            import numpy as np  # type: ignore
        except Exception as exc:
            print("Matplotlib no disponible, se omite la visualizacion:", exc)
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=False, sharey=False)
        planes = [
            ("Plano XY", (0, 1), ("X", "Y")),
            ("Plano XZ", (0, 2), ("X", "Z")),
            ("Plano YZ", (1, 2), ("Y", "Z")),
        ]

        trajectories_list = list(trajectories or [])
        valid_count = sum(1 for traj in trajectories_list if traj is not None)
        colors = plt.cm.get_cmap("tab10", max(valid_count, 1))(np.arange(max(valid_count, 1)))

        for ax, (plane_title, _, (xlabel, ylabel)) in zip(axes, planes):
            ax.set_title(plane_title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

        plotted_planes = [False, False, False]
        color_idx = 0
        for idx, traj in enumerate(trajectories_list):
            if traj is None:
                continue
            arr = np.asarray(traj, dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 2:
                continue
            color = colors[min(color_idx, len(colors) - 1)]
            label = f"Cuerpo {idx + 1}"
            color_idx += 1

            for plane_idx, (_, (i, j), _) in enumerate(planes):
                if arr.shape[1] <= max(i, j):
                    continue
                axes[plane_idx].plot(arr[:, i], arr[:, j], label=label, color=color)
                plotted_planes[plane_idx] = True

        if not any(plotted_planes):
            for ax in axes:
                ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax.transAxes)
        else:
            for ax, has_data in zip(axes, plotted_planes):
                if has_data:
                    ax.legend(loc="best")
                    break

        fig.suptitle(title)
        plt.tight_layout()
        if self.headless:
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    import numpy as np

    dummy = np.column_stack((np.cos(np.linspace(0, 2 * np.pi, 100)), np.sin(np.linspace(0, 2 * np.pi, 100))))
    viz = Visualizer(headless=False)
    viz.quick_view([dummy])
