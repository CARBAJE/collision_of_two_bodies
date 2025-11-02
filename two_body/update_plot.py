from pathlib import Path

path = Path('presentation/triDTry.py')
text = path.read_text(encoding='utf-8')
start = text.index('    def plot_lambda_evolution')
end = text.index('    def plot_mass_comparison')
new_block = """    def plot_lambda_evolution(
        self,
        lambda_history: Sequence[float] | None = None,
        *,
        epoch_history: Sequence[dict[str, Any]] | None = None,
        title: str = 'Evolucion de lambda (Lyapunov)',
        ylabel: str = 'lambda',
        show_moving_average: bool = True,
        moving_average_window: int = 3,
        annotate_best: bool = True,
    ) -> Figure | None:
        \"\"\"Grafica los valores de lambda por epoca y el mejor global acumulado.\"\"\"

        if not MATPLOTLIB_AVAILABLE:
            print('Matplotlib/NumPy no disponibles; se omite la grafica de lambda.')
            return None

        if lambda_history is None and epoch_history is None:
            raise ValueError('Se requiere lambda_history o epoch_history para graficar.')

        def _safe_get(item: Any, key: str) -> Any:
            if isinstance(item, dict):
                return item.get(key)
            return getattr(item, key, None)

        def _coerce_scalar(value: Any) -> float:
            if value is None:
                return float('nan')
            try:
                return float(value)
            except (TypeError, ValueError):
                return float('nan')

        def _coerce_series(data: Sequence[Any], name: str) -> 'np.ndarray':
            series = [_coerce_scalar(val) for val in data]
            arr = np.asarray(series, dtype=float)
            if arr.size == 0:
                raise ValueError(f"{name} no contiene elementos.")
            return arr

        epochs: list[int] | None = None
        short_series: 'np.ndarray | None' = None
        global_series: 'np.ndarray | None' = None

        if epoch_history is not None:
            epochs = []
            short_vals = []
            global_vals = []
            for idx, entry in enumerate(epoch_history):
                payload = entry or {}
                epoch_val = _safe_get(payload, 'epoch')
                try:
                    epoch_idx = int(epoch_val) if epoch_val is not None else idx
                except (TypeError, ValueError):
                    epoch_idx = idx
                epochs.append(epoch_idx)
                short_vals.append(_coerce_scalar(_safe_get(payload, 'best_lambda_short')))
                global_vals.append(_coerce_scalar(_safe_get(payload, 'best_lambda_global')))
            if epochs:
                short_series = np.asarray(short_vals, dtype=float)
                global_series = np.asarray(global_vals, dtype=float)
                if np.isnan(global_series).all():
                    global_series = None

        if lambda_history is not None:
            custom_series = _coerce_series(lambda_history, 'lambda_history')
            if epochs is None:
                epochs = list(range(1, custom_series.size + 1))
            elif custom_series.size != len(epochs):
                raise ValueError(
                    'lambda_history y epoch_history deben tener la misma longitud para superponerse.'
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
            raise ValueError('No fue posible construir una serie de datos para graficar lambda.')

        epoch_arr = np.asarray(epochs, dtype=float)
        short_arr = np.asarray(short_series, dtype=float)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(
            epoch_arr,
            short_arr,
            label='lambda mejor por epoca',
            color='#1f77b4',
            marker='o',
            linewidth=1.5,
        )

        global_arr: 'np.ndarray | None' = None
        if global_series is not None:
            global_arr = np.asarray(global_series, dtype=float)
            ax.plot(
                epoch_arr,
                global_arr,
                label='lambda global acumulado',
                color='#2ca02c',
                linestyle='--',
                linewidth=1.3,
            )

        if show_moving_average and moving_average_window > 1:

            def _moving_average(values: 'np.ndarray', window: int) -> tuple['np.ndarray', 'np.ndarray']:
                if values.size < window:
                    return np.array([]), np.array([])
                mask = ~np.isnan(values)
                if mask.sum() < window:
                    return np.array([]), np.array([])
                kernel = np.ones(window, dtype=float)
                sums = np.convolve(np.nan_to_num(values, nan=0.0), kernel, mode='valid')
                counts = np.convolve(mask.astype(float), kernel, mode='valid')
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
                    label=f'Media movil ({moving_average_window})',
                    color='#9467bd',
                    linewidth=2.0,
                )

        if annotate_best:
            reference = (
                global_arr
                if global_arr is not None and not np.isnan(global_arr).all()
                else short_arr
            )
            if reference.size > 0 and not np.isnan(reference).all():
                best_idx = int(np.nanargmin(reference))
                best_epoch = epoch_arr[best_idx]
                best_value = reference[best_idx]
                ax.scatter(
                    [best_epoch],
                    [best_value],
                    color='#d62728',
                    zorder=5,
                    label='Mejor lambda',
                )
                ax.annotate(
                    f'lambda={best_value:.4g}\nepoca={int(best_epoch)}',
                    xy=(best_epoch, best_value),
                    xytext=(5, -25),
                    textcoords='offset points',
                    arrowprops={'arrowstyle': '->', 'color': '#d62728'},
                    fontsize=9,
                )

        ax.set_title(title)
        ax.set_xlabel('Epoca')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc='best')
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
        title: str = 'Evolucion del fitness',
        ylabel: str = 'fitness',
        show_moving_average: bool = True,
        moving_average_window: int = 3,
        annotate_best: bool = True,
    ) -> Figure | None:
        \"\"\"Grafica los valores de fitness por epoca y el mejor global acumulado.\"\"\"

        if not MATPLOTLIB_AVAILABLE:
            print('Matplotlib/NumPy no disponibles; se omite la grafica de fitness.')
            return None

        if fitness_history is None and epoch_history is None:
            raise ValueError('Se requiere fitness_history o epoch_history para graficar.')

        def _safe_get(item: Any, key: str) -> Any:
            if isinstance(item, dict):
                return item.get(key)
            return getattr(item, key, None)

        def _coerce_scalar(value: Any) -> float:
            if value is None:
                return float('nan')
            try:
                return float(value)
            except (TypeError, ValueError):
                return float('nan')

        def _coerce_series(data: Sequence[Any], name: str) -> 'np.ndarray':
            series = [_coerce_scalar(val) for val in data]
            arr = np.asarray(series, dtype=float)
            if arr.size == 0:
                raise ValueError(f"{name} no contiene elementos.")
            return arr

        epochs: list[int] | None = None
        short_series: 'np.ndarray | None' = None
        global_series: 'np.ndarray | None' = None

        if epoch_history is not None:
            epochs = []
            short_vals = []
            global_vals = []
            for idx, entry in enumerate(epoch_history):
                payload = entry or {}
                epoch_val = _safe_get(payload, 'epoch')
                try:
                    epoch_idx = int(epoch_val) if epoch_val is not None else idx
                except (TypeError, ValueError):
                    epoch_idx = idx
                epochs.append(epoch_idx)
                short_value = _safe_get(payload, 'best_fitness_short')
                if short_value is None:
                    short_value = _safe_get(payload, 'fitness_short')
                short_vals.append(_coerce_scalar(short_value))
                global_value = _safe_get(payload, 'best_fitness_global')
                if global_value is None:
                    global_value = _safe_get(payload, 'fitness_global')
                global_vals.append(_coerce_scalar(global_value))
            if epochs:
                short_series = np.asarray(short_vals, dtype=float)
                global_series = np.asarray(global_vals, dtype=float)
                if np.isnan(global_series).all():
                    global_series = None

        if fitness_history is not None:
            custom_series = _coerce_series(fitness_history, 'fitness_history')
            if epochs is None:
                epochs = list(range(1, custom_series.size + 1))
            elif custom_series.size != len(epochs):
                raise ValueError(
                    'fitness_history y epoch_history deben tener la misma longitud para superponerse.'
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
            raise ValueError('No fue posible construir una serie de datos para graficar fitness.')

        epoch_arr = np.asarray(epochs, dtype=float)
        short_arr = np.asarray(short_series, dtype=float)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(
            epoch_arr,
            short_arr,
            label='fitness mejor por epoca',
            color='#ff7f0e',
            marker='o',
            linewidth=1.5,
        )

        global_arr: 'np.ndarray | None' = None
        if global_series is not None:
            global_arr = np.asarray(global_series, dtype=float)
            ax.plot(
                epoch_arr,
                global_arr,
                label='fitness global acumulado',
                color='#d62728',
                linestyle='--',
                linewidth=1.3,
            )

        if show_moving_average and moving_average_window > 1:

            def _moving_average(values: 'np.ndarray', window: int) -> tuple['np.ndarray', 'np.ndarray']:
                if values size < window:
                    return np.array([]), np.array([])
                mask = ~np.isnan(values)
                if mask.sum() < window:
                    return np.array([]), np.array([])
                kernel = np.ones(window, dtype=float)
                sums = np.convolve(np.nan_to_num(values, nan=0.0), kernel, mode='valid')
                counts = np.convolve(mask.astype(float), kernel, mode='valid')
                valid = counts > 0
                moving = np.full_like(sums, np.nan)
                moving[valid] = sums[valid] / counts[valid]
                x_ma = epoch_arr[window - 1 :]
                return x_ma, moving

            x_ma, ma_vals = _moving_average(short_arr, moving_average_window)
            if x_ma size > 0:
                ax.plot(
                    x_ma,
                    ma_vals,
                    label=f'Media movil ({moving_average_window})',
                    color='#9467bd',
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
                    color='#2ca02c',
                    zorder=5,
                    label='Mejor fitness',
                )
                ax.annotate(
                    f'fitness={best_value:.4g}\nepoca={int(best_epoch)}',
                    xy=(best_epoch, best_value),
                    xytext=(5, 12),
                    textcoords='offset points',
                    arrowprops={'arrowstyle': '->', 'color': '#2ca02c'},
                    fontsize=9,
                )

        ax.set_title(title)
        ax.set_xlabel('Epoca')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc='best')
        plt.tight_layout()

        if self.headless:
            plt.close(fig)
        else:
            plt.show()

        return fig

"""
path.write_text(text[:start] + new_block + text[end:], encoding='utf-8')
