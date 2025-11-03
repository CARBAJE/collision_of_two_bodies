from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

from .visualization import Visualizer as _PlanarVisualizer
from ..core.config import Config
from ..simulation.rebound_adapter import ReboundSim

try:  # Dependencias opcionales para la animacion 3D
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from matplotlib import colors as mcolors
    import numpy as np

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover - entorno sin GUI
    MATPLOTLIB_AVAILABLE = False

if TYPE_CHECKING:  # pragma: no cover - solo para type checkers
    import numpy as np
    from matplotlib.animation import FuncAnimation
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    Trajectory3D = NDArray[np.floating[Any]]
else:  # Evitamos referencias inexistentes en tiempo de ejecucion
    FuncAnimation = Any  # type: ignore[misc,assignment]
    Figure = Any
    Trajectory3D = Any


@dataclass(slots=True)
class MassComparisonData:
    original_xyz: "np.ndarray"
    optimized_xyz: "np.ndarray"
    dt_value: float
    num_frames: int
    steps: "np.ndarray"
    body_labels: tuple[str, ...]
    centers: "np.ndarray"
    max_range: float
    n_bodies: int


class Visualizer(_PlanarVisualizer):
    """
    Extiende el visualizador 2D base con soporte para animacion de trayectorias 3D.
    """

    def __init__(
        self,
        headless: bool = True,
        cfg: Config | None = None,
        sim_builder: ReboundSim | None = None,
    ) -> None:
        super().__init__(headless=headless)
        self.cfg = cfg or Config()
        self._sim_builder = sim_builder or ReboundSim(G=self.cfg.G, integrator=self.cfg.integrator)

    def animate_3d(
        self,
        trajectories: Sequence[Trajectory3D],
        interval_ms: int = 50,
        title: str = "Simulacion de orbita en 3D",
        total_frames: int = 300,
    ) -> FuncAnimation | None:
        """
        Crea y muestra (o cierra en modo headless) una animacion 3D de las trayectorias.
        """

        if not trajectories:
            raise ValueError("Se requiere al menos una trayectoria para animar.")

        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib/NumPy no disponibles; se omite la visualizacion 3D.")
            return None

        arrays = [np.asarray(traj, dtype=float) for traj in trajectories]
        min_len = min(traj.shape[0] for traj in arrays)
        if total_frames <= 0 or min_len == 0:
            raise ValueError("Las trayectorias no contienen muestras suficientes.")

        num_frames = min(total_frames, min_len)
        trimmed = [traj[:num_frames, :3] for traj in arrays]
        all_data = np.concatenate(trimmed, axis=0)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(title)
        ax.set_xlabel("Eje X")
        ax.set_ylabel("Eje Y")
        ax.set_zlabel("Eje Z")
        ax.grid(True)

        ranges = all_data.ptp(axis=0)
        max_range = ranges.max() / 2.0 if ranges.size else 1.0
        centers = all_data.mean(axis=0)
        ax.set_xlim(centers[0] - max_range, centers[0] + max_range)
        ax.set_ylim(centers[1] - max_range, centers[1] + max_range)
        ax.set_zlim(centers[2] - max_range, centers[2] + max_range)

        for idx, traj in enumerate(trimmed, start=1):
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                traj[:, 2],
                linestyle="--",
                alpha=0.35,
                label=f"Orbita {idx}",
            )

        points = [
            ax.plot(
                [traj[0, 0]],
                [traj[0, 1]],
                [traj[0, 2]],
                marker="o",
                markersize=5,
                label=f"Cuerpo {idx}",
            )[0]
            for idx, traj in enumerate(trimmed, start=1)
        ]

        def update_frame(frame_index: int) -> list[Any]:
            for point, traj in zip(points, trimmed):
                x, y, z = traj[frame_index, :3]
                point.set_data([x], [y])
                point.set_3d_properties([z])

            ax.set_title(f"{title}\nTiempo: {frame_index}/{num_frames}")
            return points

        ani = animation.FuncAnimation(
            fig,
            update_frame,
            frames=num_frames,
            interval=interval_ms,
            blit=False,
            repeat=True,
        )

        if self.headless:
            plt.close(fig)
        else:
            plt.show()

        return ani

    def plot_mass_distance_evolution(
        self,
        original_masses: Sequence[float] | None = None,
        optimized_masses: Sequence[float] | None = None,
        *,
        original_tracks: Sequence[Trajectory3D] | None = None,
        optimized_tracks: Sequence[Trajectory3D] | None = None,
        comparison_data: MassComparisonData | None = None,
        body_labels: Sequence[str] | None = None,
        title: str = "Crecimiento de la perturbacion",
        ylabel: str = "Distancia [UA]",
        t_end: float | None = None,
        dt: float | None = None,
        total_frames: int | None = None,
        r0: Sequence[Sequence[float]] | None = None,
        v0: Sequence[Sequence[float]] | None = None,
    ) -> Figure | None:
        """
        Grafica la evolucion estatica (no animada) de ``||R_2 - R||`` para cada cuerpo.
        Puede reutilizar los datos generados por :meth:`plot_mass_comparison` mediante
        el parametro ``comparison_data`` o recalcularlos a partir de las masas.
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib/NumPy no disponibles; se omite la grafica de distancias.")
            return None

        if comparison_data is None:
            if (original_masses is None or optimized_masses is None) and (
                original_tracks is None or optimized_tracks is None
            ):
                raise ValueError(
                    "Proporciona comparison_data o bien masas o trayectorias para ambos casos."
                )
            try:
                data = self._prepare_mass_comparison_data(
                    original_masses,
                    optimized_masses,
                    body_labels=body_labels,
                    original_tracks=original_tracks,
                    optimized_tracks=optimized_tracks,
                    t_end=t_end,
                    dt=dt,
                    total_frames=total_frames,
                    r0=r0,
                    v0=v0,
                )
            except ImportError as exc:
                print(f"{exc}")
                return None
        else:
            data = comparison_data

        distances = np.linalg.norm(data.optimized_xyz - data.original_xyz, axis=2)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title(title)
        ax.set_xlabel("Paso (k)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_yscale("log")

        colors = plt.cm.get_cmap("tab10", data.n_bodies)(np.arange(data.n_bodies))

        plotted_any = False
        for idx in range(data.n_bodies):
            series = distances[:, idx]
            finite_mask = np.isfinite(series) & (series > 0)
            if not np.any(finite_mask):
                continue
            sanitized = np.full_like(series, np.nan, dtype=float)
            sanitized[finite_mask] = series[finite_mask]
            ax.plot(data.steps, sanitized, color=colors[idx], label=data.body_labels[idx])
            plotted_any = True

        if data.num_frames > 1:
            ax.set_xlim(int(data.steps[0]), int(data.steps[-1]))
        if plotted_any:
            ax.legend(loc="best")
        else:
            ax.text(
                0.5,
                0.5,
                "Sin diferencias finitas entre trayectorias",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        fig.tight_layout()

        if self.headless:
            plt.close(fig)
        else:
            plt.show()

        return fig

    def plot_lambda_evolution(
        self,
        lambda_history: Sequence[float] | None = None,
        *,
        epoch_history: Sequence[dict[str, Any]] | None = None,
        title: str = "Evolucion de lambda (Lyapunov)",
        ylabel: str = "lambda",
        show_moving_average: bool = True,
        moving_average_window: int = 3,
        annotate_best: bool = True,
    ) -> Figure | None:
        """
        Grafica la evolucion del mejor valor de lambda por epoca y (opcionalmente) el mejor global acumulado.
        Puede alimentarse con metrics.best_lambda_per_epoch o directamente con metrics.epoch_history.
        """

        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib/NumPy no disponibles; se omite la grafica de lambda.")
            return None

        if lambda_history is None and epoch_history is None:
            raise ValueError("Se requiere lambda_history o epoch_history para graficar.")

        def _safe_get(item: Any, key: str) -> Any:
            if isinstance(item, dict):
                return item.get(key)
            return getattr(item, key, None)

        def _coerce_scalar(value: Any) -> float:
            if value is None:
                return float("nan")
            try:
                return float(value)
            except (TypeError, ValueError):
                return float("nan")

        def _coerce_series(data: Sequence[Any], name: str) -> "np.ndarray":
            series = [_coerce_scalar(val) for val in data]
            arr = np.asarray(series, dtype=float)
            if arr.size == 0:
                raise ValueError(f"{name} no contiene elementos.")
            return arr

        epochs: list[int] | None = None
        short_series: "np.ndarray | None" = None
        global_series: "np.ndarray | None" = None

        if epoch_history is not None:
            epochs = []
            short_vals = []
            global_vals = []
            for idx, entry in enumerate(epoch_history):
                payload = entry or {}
                epoch_val = _safe_get(payload, "epoch")
                try:
                    epoch_idx = int(epoch_val) if epoch_val is not None else idx
                except (TypeError, ValueError):
                    epoch_idx = idx
                epochs.append(epoch_idx)
                short_vals.append(_coerce_scalar(_safe_get(payload, "best_lambda_short")))
                global_vals.append(_coerce_scalar(_safe_get(payload, "best_lambda_global")))
            if epochs:
                short_series = np.asarray(short_vals, dtype=float)
                global_series = np.asarray(global_vals, dtype=float)
                if np.isnan(global_series).all():
                    global_series = None

        if lambda_history is not None:
            custom_series = _coerce_series(lambda_history, "lambda_history")
            if epochs is None:
                epochs = list(range(1, custom_series.size + 1))
            elif custom_series.size != len(epochs):
                raise ValueError(
                    "lambda_history y epoch_history deben tener la misma longitud para superponerse."
                )
            if short_series is None:
                short_series = custom_series
            else:
                short_series = np.where(np.isnan(short_series), custom_series, short_series)
            if global_series is None:
                global_series = custom_series
            else:
                global_series = np.where(np.isnan(global_series), custom_series, global_series)

        if global_series is not None and np.isnan(global_series).all():
            global_series = None

        if epochs is None or short_series is None:
            raise ValueError("No fue posible construir una serie de datos para graficar lambda.")

        epoch_arr = np.asarray(epochs, dtype=float)
        short_arr = np.asarray(short_series, dtype=float)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(
            epoch_arr,
            short_arr,
            label="lambda mejor por epoca",
            color="#1f77b4",
            marker="o",
            linewidth=1.5,
        )

        raw_global_arr: "np.ndarray | None" = None
        if global_series is not None:
            arr = np.asarray(global_series, dtype=float)
            if arr.size > 0:
                raw_global_arr = arr

        def _build_cumulative_series(
            primary: "np.ndarray",
            secondary: "np.ndarray | None",
        ) -> "np.ndarray | None":
            total = primary.size
            if total == 0:
                return None
            result = np.full(primary.shape, np.nan, dtype=float)
            running_best: float | None = None
            for idx in range(total):
                candidates: list[float] = []
                primary_val = primary[idx]
                if np.isfinite(primary_val):
                    candidates.append(float(primary_val))
                if secondary is not None and secondary.shape == primary.shape:
                    secondary_val = secondary[idx]
                    if np.isfinite(secondary_val):
                        candidates.append(float(secondary_val))

                if candidates:
                    current_best = min(candidates)
                    if running_best is not None:
                        current_best = min(current_best, running_best)
                    result[idx] = current_best
                    running_best = current_best
                elif running_best is not None:
                    result[idx] = running_best
            if np.isnan(result).all():
                return None
            return result

        global_arr = _build_cumulative_series(short_arr, raw_global_arr)
        if global_arr is None and raw_global_arr is not None:
            global_arr = _build_cumulative_series(raw_global_arr, None)

        if global_arr is not None:
            ax.plot(
                epoch_arr,
                global_arr,
                label="lambda global acumulado",
                color="#2ca02c",
                linestyle="--",
                linewidth=1.3,
            )

        if show_moving_average and moving_average_window > 1:
            def _moving_average(values: "np.ndarray", window: int) -> tuple["np.ndarray", "np.ndarray"]:
                if values.size < window:
                    return np.array([]), np.array([])
                mask = ~np.isnan(values)
                if mask.sum() < window:
                    return np.array([]), np.array([])
                kernel = np.ones(window, dtype=float)
                sums = np.convolve(np.nan_to_num(values, nan=0.0), kernel, mode="valid")
                counts = np.convolve(mask.astype(float), kernel, mode="valid")
                valid = counts > 0
                moving = np.full_like(sums, np.nan)
                moving[valid] = sums[valid] / counts[valid]
                x_ma = epoch_arr[window - 1 :]
                return x_ma, moving

            x_ma, ma_vals = _moving_average(short_arr, moving_average_window)
            if x_ma.size > 0:
                ax.plot(
                    x_ma,
                    ma_vals,
                    label=f"Media movil ({moving_average_window})",
                    color="#9467bd",
                    linewidth=2.0,
                )

        if annotate_best:
            best_idx: int | None = None
            best_value: float | None = None

            def _consider_series(series: "np.ndarray | None") -> None:
                nonlocal best_idx, best_value
                if series is None:
                    return
                finite_mask = np.isfinite(series)
                if not np.any(finite_mask):
                    return
                finite_indices = np.flatnonzero(finite_mask)
                finite_values = series[finite_mask]
                local_idx = int(np.argmin(finite_values))
                candidate_idx = int(finite_indices[local_idx])
                candidate_value = float(finite_values[local_idx])

                if best_value is None:
                    best_idx = candidate_idx
                    best_value = candidate_value
                    return

                if candidate_value < best_value - 1e-12:
                    best_idx = candidate_idx
                    best_value = candidate_value
                    return

                if np.isclose(candidate_value, best_value, rtol=1e-9, atol=1e-12):
                    if best_idx is None or candidate_idx < best_idx:
                        best_idx = candidate_idx
                        best_value = candidate_value

            _consider_series(global_arr)
            _consider_series(short_arr)

            if best_idx is not None and best_value is not None and np.isfinite(best_value):
                best_epoch = epoch_arr[best_idx]
                ax.scatter(
                    [best_epoch],
                    [best_value],
                    color="#d62728",
                    zorder=5,
                    label="Mejor lambda",
                )
                display_epoch = int(best_epoch) if np.isclose(best_epoch, round(best_epoch)) else best_epoch
                ax.annotate(
                    f"lambda={best_value:.4g}\nepoca={display_epoch}",
                    xy=(best_epoch, best_value),
                    xytext=(5, -25),
                    textcoords="offset points",
                    arrowprops={"arrowstyle": "->", "color": "#d62728"},
                    fontsize=9,
                )

        ax.set_title(title)
        ax.set_xlabel("Epoca")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        plt.tight_layout()

        if self.headless:
            plt.close(fig)
        else:
            plt.show()

        return fig

    def plot_fitness_evolution(
        self,
        fitness_history: Sequence[float] | None = None,
        *,
        epoch_history: Sequence[dict[str, Any]] | None = None,
        title: str = "Evolucion del fitness",
        ylabel: str = "fitness",
        show_moving_average: bool = True,
        moving_average_window: int = 3,
        annotate_best: bool = True,
    ) -> Figure | None:
        """Grafica los valores de fitness por epoca y el mejor global acumulado."""

        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib/NumPy no disponibles; se omite la grafica de fitness.")
            return None

        if fitness_history is None and epoch_history is None:
            raise ValueError("Se requiere fitness_history o epoch_history para graficar.")

        def _safe_get(item: Any, key: str) -> Any:
            if isinstance(item, dict):
                return item.get(key)
            return getattr(item, key, None)

        def _coerce_scalar(value: Any) -> float:
            if value is None:
                return float("nan")
            try:
                return float(value)
            except (TypeError, ValueError):
                return float("nan")

        def _coerce_series(data: Sequence[Any], name: str) -> "np.ndarray":
            series = [_coerce_scalar(val) for val in data]
            arr = np.asarray(series, dtype=float)
            if arr.size == 0:
                raise ValueError(f"{name} no contiene elementos.")
            return arr

        epochs: list[int] | None = None
        short_series: "np.ndarray | None" = None
        global_series: "np.ndarray | None" = None

        if epoch_history is not None:
            epochs = []
            short_vals = []
            global_vals = []
            for idx, entry in enumerate(epoch_history):
                payload = entry or {}
                epoch_val = _safe_get(payload, "epoch")
                try:
                    epoch_idx = int(epoch_val) if epoch_val is not None else idx
                except (TypeError, ValueError):
                    epoch_idx = idx
                epochs.append(epoch_idx)
                short_value = _safe_get(payload, "best_fitness_short")
                if short_value is None:
                    short_value = _safe_get(payload, "fitness_short")
                short_vals.append(_coerce_scalar(short_value))
                global_value = _safe_get(payload, "best_fitness_global")
                if global_value is None:
                    global_value = _safe_get(payload, "fitness_global")
                global_vals.append(_coerce_scalar(global_value))
            if epochs:
                short_series = np.asarray(short_vals, dtype=float)
                global_series = np.asarray(global_vals, dtype=float)
                if np.isnan(global_series).all():
                    global_series = None

        if fitness_history is not None:
            custom_series = _coerce_series(fitness_history, "fitness_history")
            if epochs is None:
                epochs = list(range(1, custom_series.size + 1))
            elif custom_series.size != len(epochs):
                raise ValueError(
                    "fitness_history y epoch_history deben tener la misma longitud para superponerse."
                )
            if short_series is None:
                short_series = custom_series
            else:
                short_series = np.where(np.isnan(short_series), custom_series, short_series)
            if global_series is None:
                global_series = custom_series
            else:
                global_series = np.where(np.isnan(global_series), custom_series, global_series)

        if global_series is not None and np.isnan(global_series).all():
            global_series = None

        if epochs is None or short_series is None:
            raise ValueError("No fue posible construir una serie de datos para graficar fitness.")

        epoch_arr = np.asarray(epochs, dtype=float)
        short_arr = np.asarray(short_series, dtype=float)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(
            epoch_arr,
            short_arr,
            label="fitness mejor por epoca",
            color="#ff7f0e",
            marker="o",
            linewidth=1.5,
        )

        global_arr: "np.ndarray | None" = None
        if global_series is not None:
            global_arr = np.asarray(global_series, dtype=float)
            if global_arr.size > 0:
                valid_mask = ~np.isnan(global_arr)
                if np.any(valid_mask):
                    global_arr = global_arr.copy()
                    global_arr[valid_mask] = np.maximum.accumulate(global_arr[valid_mask])
            ax.plot(
                epoch_arr,
                global_arr,
                label="fitness global acumulado",
                color="#d62728",
                linestyle="--",
                linewidth=1.3,
            )

        if show_moving_average and moving_average_window > 1:

            def _moving_average(values: "np.ndarray", window: int) -> tuple["np.ndarray", "np.ndarray"]:
                if values.size < window:
                    return np.array([]), np.array([])
                mask = ~np.isnan(values)
                if mask.sum() < window:
                    return np.array([]), np.array([])
                kernel = np.ones(window, dtype=float)
                sums = np.convolve(np.nan_to_num(values, nan=0.0), kernel, mode="valid")
                counts = np.convolve(mask.astype(float), kernel, mode="valid")
                valid = counts > 0
                moving = np.full_like(sums, np.nan)
                moving[valid] = sums[valid] / counts[valid]
                x_ma = epoch_arr[window - 1 :]
                return x_ma, moving

            x_ma, ma_vals = _moving_average(short_arr, moving_average_window)
            if x_ma.size > 0:
                ax.plot(
                    x_ma,
                    ma_vals,
                    label=f"Media movil ({moving_average_window})",
                    color="#9467bd",
                    linewidth=2.0,
                )

        if annotate_best:
            reference = (
                global_arr
                if global_arr is not None and not np.isnan(global_arr).all()
                else short_arr
            )
            if reference.size > 0 and not np.isnan(reference).all():
                best_idx = int(np.nanargmax(reference))
                best_epoch = epoch_arr[best_idx]
                best_value = reference[best_idx]
                ax.scatter(
                    [best_epoch],
                    [best_value],
                    color="#2ca02c",
                    zorder=5,
                    label="Mejor fitness",
                )
                ax.annotate(
                    f"fitness={best_value:.4g}\nepoca={int(best_epoch)}",
                    xy=(best_epoch, best_value),
                    xytext=(5, 12),
                    textcoords="offset points",
                    arrowprops={"arrowstyle": "->", "color": "#2ca02c"},
                    fontsize=9,
                )

        ax.set_title(title)
        ax.set_xlabel("Epoca")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        plt.tight_layout()

        if self.headless:
            plt.close(fig)
        else:
            plt.show()

        return fig

    def _prepare_mass_comparison_data(
        self,
        original_masses: Sequence[float] | None = None,
        optimized_masses: Sequence[float] | None = None,
        body_labels: Sequence[str] | None = None,
        *,
        original_tracks: Sequence[Trajectory3D] | None = None,
        optimized_tracks: Sequence[Trajectory3D] | None = None,
        t_end: float | None = None,
        dt: float | None = None,
        total_frames: int | None = None,
        r0: Sequence[Sequence[float]] | None = None,
        v0: Sequence[Sequence[float]] | None = None,
    ) -> MassComparisonData:

        def _stack_tracks(tracks: Sequence[Trajectory3D], name: str) -> tuple["np.ndarray", int]:
            if not tracks:
                raise ValueError(f"{name} debe contener al menos una trayectoria.")
            arrays = [np.asarray(track, dtype=float) for track in tracks]
            lengths = [arr.shape[0] for arr in arrays]
            min_len = min(lengths)
            if min_len < 2:
                raise ValueError(f"{name} requiere al menos 2 muestras temporales.")
            trimmed = []
            for idx, arr in enumerate(arrays):
                if arr.shape[-1] < 3:
                    raise ValueError(f"{name}[{idx}] debe contener componentes X, Y, Z.")
                trimmed.append(arr[:min_len, :3])
            stacked = np.stack(trimmed, axis=1)
            return stacked, min_len

        if original_tracks is not None or optimized_tracks is not None:
            if original_tracks is None or optimized_tracks is None:
                raise ValueError(
                    "Debe proporcionar original_tracks y optimized_tracks o ninguno de ellos."
                )
            original_xyz, len_orig = _stack_tracks(original_tracks, "original_tracks")
            optimized_xyz, len_opt = _stack_tracks(optimized_tracks, "optimized_tracks")
            num_frames = min(len_orig, len_opt)
            original_xyz = original_xyz[:num_frames]
            optimized_xyz = optimized_xyz[:num_frames]
            if original_xyz.shape[1] != optimized_xyz.shape[1]:
                raise ValueError(
                    "El numero de cuerpos en original_tracks y optimized_tracks debe coincidir."
                )
            n_bodies = original_xyz.shape[1]
            if n_bodies == 0:
                raise ValueError("Se requiere al menos un cuerpo para comparar.")
            dt_value = float(dt if dt is not None else 1.0)
            labels_tuple: tuple[str, ...]
            if body_labels is None:
                labels_tuple = tuple(f"Cuerpo {idx + 1}" for idx in range(n_bodies))
            else:
                if len(body_labels) != n_bodies:
                    raise ValueError("Numero de etiquetas distinto al de cuerpos.")
                labels_tuple = tuple(str(label) for label in body_labels)
        else:
            if original_masses is None or optimized_masses is None:
                raise ValueError(
                    "Proporciona masas o trayectorias para preparar la comparacion de masas."
                )
            original = np.asarray(original_masses, dtype=float).ravel()
            optimized = np.asarray(optimized_masses, dtype=float).ravel()
            if original.shape != optimized.shape:
                raise ValueError("Las listas de masas deben tener la misma longitud.")

            n_bodies = original.size
            if n_bodies == 0:
                raise ValueError("Se requiere al menos un cuerpo para comparar.")
            if n_bodies > 3:
                raise ValueError("La animacion 3D soporta hasta 3 cuerpos con el backend actual.")

            if body_labels is None:
                labels_tuple = tuple(f"Cuerpo {idx + 1}" for idx in range(n_bodies))
            else:
                if len(body_labels) != n_bodies:
                    raise ValueError("Numero de etiquetas distinto al de cuerpos.")
                labels_tuple = tuple(str(label) for label in body_labels)

            dt_value = float(dt if dt is not None else getattr(self.cfg, "dt", 0.5))
            if dt_value <= 0:
                raise ValueError("El paso de integracion (dt) debe ser positivo.")
            base_horizon = float(
                t_end if t_end is not None else getattr(self.cfg, "t_end_short", 100.0)
            )
            if base_horizon <= 0:
                raise ValueError("El horizonte de simulacion debe ser positivo.")

            max_steps = max(1, int(np.floor(base_horizon / dt_value)))
            desired_frames = total_frames if total_frames is not None else max_steps
            if desired_frames <= 0:
                raise ValueError("total_frames debe ser positivo.")
            num_frames = min(max_steps, desired_frames)
            if num_frames < 2:
                raise ValueError("Se requieren al menos 2 cuadros para una animacion significativa.")
            integration_horizon = num_frames * dt_value

            def _slice_vectors(
                vectors: Sequence[Sequence[float]] | None,
                fallback: Sequence[Sequence[float]],
                name: str,
            ) -> tuple[tuple[float, float, float], ...]:
                base_vectors = vectors if vectors is not None else fallback
                if len(base_vectors) < n_bodies:
                    raise ValueError(f"{name} debe contener al menos {n_bodies} vectores 3D.")
                sliced: list[tuple[float, float, float]] = []
                for idx in range(n_bodies):
                    vec = base_vectors[idx]
                    if len(vec) < 3:
                        raise ValueError(f"{name}[{idx}] debe proporcionar tres componentes.")
                    sliced.append((float(vec[0]), float(vec[1]), float(vec[2])))
                return tuple(sliced)

            r0_slice = _slice_vectors(r0, self.cfg.r0, "r0")
            v0_slice = _slice_vectors(v0, self.cfg.v0, "v0")

            sim_builder = self._sim_builder

            def _simulate_case(masses_vec: np.ndarray) -> np.ndarray:
                mass_tuple = tuple(float(m) for m in masses_vec.tolist())
                sim = sim_builder.setup_simulation(masses=mass_tuple, r0=r0_slice, v0=v0_slice)
                traj = sim_builder.integrate(sim, t_end=integration_horizon, dt=dt_value)
                if traj.shape[1] != n_bodies:
                    raise RuntimeError("La simulacion devolvio un numero inesperado de cuerpos.")
                return traj[:num_frames, :, :3]

            try:
                original_xyz = _simulate_case(original)
                optimized_xyz = _simulate_case(optimized)
            except ImportError as exc:
                raise ImportError(
                    f"Dependencia faltante para la simulacion 3D ({exc}); se omite la comparacion."
                ) from exc

        combined_points = np.concatenate(
            (original_xyz.reshape(-1, 3), optimized_xyz.reshape(-1, 3)),
            axis=0,
        )
        finite_mask = np.all(np.isfinite(combined_points), axis=1)
        if not np.any(finite_mask):
            raise ValueError(
                "Las trayectorias simuladas contienen solo valores no finitos; "
                "no se puede construir la animacion."
            )
        finite_points = combined_points[finite_mask]
        ranges = finite_points.ptp(axis=0)
        if not np.all(np.isfinite(ranges)):
            ranges = np.where(np.isfinite(ranges), ranges, 0.0)
        raw_max = ranges.max() if ranges.size else 0.0
        if not np.isfinite(raw_max):
            raw_max = 0.0
        max_range = float(max(raw_max / 2.0, 1e-6))
        centers = finite_points.mean(axis=0)
        if not np.all(np.isfinite(centers)):
            centers = np.where(np.isfinite(centers), centers, 0.0)
        centers = centers.astype(float, copy=False)

        steps = np.arange(num_frames, dtype=int)

        return MassComparisonData(
            original_xyz=original_xyz,
            optimized_xyz=optimized_xyz,
            dt_value=dt_value,
            num_frames=num_frames,
            steps=steps,
            body_labels=labels_tuple,
            centers=centers,
            max_range=max_range,
            n_bodies=n_bodies,
        )

    def plot_mass_comparison(
        self,
        original_masses: Sequence[float] | None = None,
        optimized_masses: Sequence[float] | None = None,
        body_labels: Sequence[str] | None = None,
        title: str = "Comparativa de masas",
        *,
        original_tracks: Sequence[Trajectory3D] | None = None,
        optimized_tracks: Sequence[Trajectory3D] | None = None,
        t_end: float | None = None,
        dt: float | None = None,
        interval_ms: int = 50,
        trail_length: int = 50,
        total_frames: int | None = None,
        r0: Sequence[Sequence[float]] | None = None,
        v0: Sequence[Sequence[float]] | None = None,
    ) -> FuncAnimation | None:
        """
        Genera una figura con tres subplots 3D animados:
        1) Trayectorias simuladas con las masas originales.
        2) Trayectorias con las masas optimizadas.
        3) Ambas simulaciones superpuestas para comparar su dinamica.

        Se puede suministrar bien las masas (se recalcula la simulacion) o bien las
        trayectorias ya integradas mediante ``original_tracks`` y ``optimized_tracks``.
        El objeto ``FuncAnimation`` resultante incluye los datos en ``mass_comparison_data``
        para reutilizarlos en otras graficas.
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib/NumPy no disponibles; se omite la comparacion de masas.")
            return None

        try:
            data = self._prepare_mass_comparison_data(
                original_masses,
                optimized_masses,
                body_labels=body_labels,
                original_tracks=original_tracks,
                optimized_tracks=optimized_tracks,
                t_end=t_end,
                dt=dt,
                total_frames=total_frames,
                r0=r0,
                v0=v0,
            )
        except ImportError as exc:
            print(f"{exc}")
            return None

        fig = plt.figure(figsize=(18, 6))
        axes = [
            fig.add_subplot(1, 3, 1, projection="3d"),
            fig.add_subplot(1, 3, 2, projection="3d"),
            fig.add_subplot(1, 3, 3, projection="3d"),
        ]
        subtitles = [
            "Simulacion con masas originales",
            "Simulacion con masas optimizadas",
            "Comparacion superpuesta",
        ]
        colors = plt.cm.get_cmap("tab10", data.n_bodies)(np.arange(data.n_bodies))

        def _blend_color(base_color: Any, target: str, weight: float) -> tuple[float, float, float]:
            base_rgb = np.array(mcolors.to_rgb(base_color), dtype=float)
            target_rgb = np.array(mcolors.to_rgb(target), dtype=float)
            mixed = (1.0 - weight) * base_rgb + weight * target_rgb
            return tuple(np.clip(mixed, 0.0, 1.0))

        overlay_original_colors = [
            _blend_color(color, "white", 0.35) for color in colors
        ]
        overlay_optimized_colors = [
            _blend_color(color, "black", 0.25) for color in colors
        ]

        def _configure_axis(ax: Any, subtitle: str) -> None:
            ax.set_title(subtitle)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.grid(True, alpha=0.2)
            ax.set_xlim(data.centers[0] - data.max_range, data.centers[0] + data.max_range)
            ax.set_ylim(data.centers[1] - data.max_range, data.centers[1] + data.max_range)
            ax.set_zlim(data.centers[2] - data.max_range, data.centers[2] + data.max_range)
            if hasattr(ax, "set_box_aspect"):
                ax.set_box_aspect([1, 1, 1])

        for ax, subtitle in zip(axes, subtitles):
            _configure_axis(ax, subtitle)

        for body_idx in range(data.n_bodies):
            axes[0].plot(
                data.original_xyz[:, body_idx, 0],
                data.original_xyz[:, body_idx, 1],
                data.original_xyz[:, body_idx, 2],
                color=colors[body_idx],
                linestyle="--",
                alpha=0.6,
            )
            axes[1].plot(
                data.optimized_xyz[:, body_idx, 0],
                data.optimized_xyz[:, body_idx, 1],
                data.optimized_xyz[:, body_idx, 2],
                color=colors[body_idx],
                linestyle="-",
                alpha=0.8,
            )
            axes[2].plot(
                data.original_xyz[:, body_idx, 0],
                data.original_xyz[:, body_idx, 1],
                data.original_xyz[:, body_idx, 2],
                color=overlay_original_colors[body_idx],
                linestyle=(0, (4, 2)),
                linewidth=1.6,
                alpha=0.85,
            )
            axes[2].plot(
                data.optimized_xyz[:, body_idx, 0],
                data.optimized_xyz[:, body_idx, 1],
                data.optimized_xyz[:, body_idx, 2],
                color=overlay_optimized_colors[body_idx],
                linestyle="-",
                linewidth=2.1,
                alpha=0.95,
            )

        orig_points = [
            axes[0].plot(
                [data.original_xyz[0, idx, 0]],
                [data.original_xyz[0, idx, 1]],
                [data.original_xyz[0, idx, 2]],
                marker="o",
                markersize=5,
                color=colors[idx],
            )[0]
            for idx in range(data.n_bodies)
        ]
        opt_points = [
            axes[1].plot(
                [data.optimized_xyz[0, idx, 0]],
                [data.optimized_xyz[0, idx, 1]],
                [data.optimized_xyz[0, idx, 2]],
                marker="o",
                markersize=5,
                color=colors[idx],
            )[0]
            for idx in range(data.n_bodies)
        ]
        overlay_orig_points = [
            axes[2].plot(
                [data.original_xyz[0, idx, 0]],
                [data.original_xyz[0, idx, 1]],
                [data.original_xyz[0, idx, 2]],
                marker="o",
                markersize=5,
                color=overlay_original_colors[idx],
                markerfacecolor="none",
                markeredgecolor=overlay_original_colors[idx],
                markeredgewidth=1.4,
                alpha=0.85,
            )[0]
            for idx in range(data.n_bodies)
        ]
        overlay_opt_points = [
            axes[2].plot(
                [data.optimized_xyz[0, idx, 0]],
                [data.optimized_xyz[0, idx, 1]],
                [data.optimized_xyz[0, idx, 2]],
                marker="o",
                markersize=5,
                color=overlay_optimized_colors[idx],
                markerfacecolor=overlay_optimized_colors[idx],
                markeredgecolor=overlay_optimized_colors[idx],
                markeredgewidth=1.2,
                alpha=1.0,
            )[0]
            for idx in range(data.n_bodies)
        ]

        _ = trail_length  # Reservado por compatibilidad; no se utiliza en esta version.

        def _update(frame_idx: int):
            time_value = frame_idx * data.dt_value
            for body_idx in range(data.n_bodies):
                x_o, y_o, z_o = data.original_xyz[frame_idx, body_idx]
                orig_points[body_idx].set_data([x_o], [y_o])
                orig_points[body_idx].set_3d_properties([z_o])
                overlay_orig_points[body_idx].set_data([x_o], [y_o])
                overlay_orig_points[body_idx].set_3d_properties([z_o])

                x_opt, y_opt, z_opt = data.optimized_xyz[frame_idx, body_idx]
                opt_points[body_idx].set_data([x_opt], [y_opt])
                opt_points[body_idx].set_3d_properties([z_opt])
                overlay_opt_points[body_idx].set_data([x_opt], [y_opt])
                overlay_opt_points[body_idx].set_3d_properties([z_opt])

            axes[0].set_title(f"{subtitles[0]}\n t={time_value:.2f}")
            axes[1].set_title(f"{subtitles[1]}\n t={time_value:.2f}")
            axes[2].set_title(f"{subtitles[2]}\n t={time_value:.2f}")
            return (
                orig_points
                + opt_points
                + overlay_orig_points
                + overlay_opt_points
            )

        fig.suptitle(title)

        body_handles = [
            Patch(facecolor=colors[idx], edgecolor="none", label=data.body_labels[idx])
            for idx in range(data.n_bodies)
        ]
        body_legend = axes[2].legend(
            handles=body_handles,
            title="Cuerpos",
            loc="upper left",
            frameon=False,
        )
        axes[2].add_artist(body_legend)
        sample_orig_color = overlay_original_colors[0] if overlay_original_colors else (0.6, 0.6, 0.6)
        sample_opt_color = overlay_optimized_colors[0] if overlay_optimized_colors else (0.3, 0.3, 0.3)
        scenario_handles = [
            Line2D(
                [0],
                [0],
                color=sample_orig_color,
                linestyle=(0, (4, 2)),
                linewidth=1.8,
                marker="o",
                markerfacecolor="none",
                markeredgecolor=sample_orig_color,
                markeredgewidth=1.2,
                label="Original",
            ),
            Line2D(
                [0],
                [0],
                color=sample_opt_color,
                linestyle="-",
                linewidth=2.2,
                marker="o",
                markerfacecolor=sample_opt_color,
                markeredgecolor=sample_opt_color,
                markeredgewidth=1.0,
                label="Optimizado",
            ),
        ]
        axes[2].legend(
            handles=scenario_handles,
            title="Caso",
            loc="upper right",
            frameon=False,
        )

        plt.tight_layout(rect=(0, 0, 1, 0.92))

        ani = animation.FuncAnimation(
            fig,
            _update,
            frames=data.num_frames,
            interval=interval_ms,
            blit=False,
            repeat=True,
        )

        setattr(ani, "mass_comparison_data", data)

        if self.headless:
            plt.close(fig)
        else:
            plt.show()

        return ani


if __name__ == "__main__":
    if MATPLOTLIB_AVAILABLE:
        steps = 500
        t = np.linspace(0, 10 * np.pi, steps)
        path = np.column_stack((np.cos(t), np.sin(t), t * 0.1))

        viz = Visualizer(headless=False)
        viz.animate_3d(
            trajectories=[path],
            interval_ms=50,
            title="Prueba de helice 3D",
        )
    else:
        print("Instala NumPy y Matplotlib para ejecutar el ejemplo de animacion 3D.")
