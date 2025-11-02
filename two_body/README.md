## Instrumentacion de desempeno

El proyecto dispone de un sistema ligero para medir secciones criticas (simulacion, Lyapunov, GA, seleccion, crossover, mutacion y evaluaciones de fitness).

- `PERF_TIMINGS_ENABLED=1` (default) habilita los medidores. Usa `0/false/off` para desactivarlos sin tocar el codigo.
- Los registros se escriben en `data/timings/timings_{run_id}_{YYYYMMDD_HHMMSS}.csv` (y un `.jsonl` espejo si `PERF_TIMINGS_JSONL=1`).
- Cada fila contiene `run_id,epoch,batch_id,individual_id,section,start_ns,end_ns,duration_us,extra`. Los `extras` son JSON compacto con metadatos como dt, horizon, etc.
- El buffer interno evita flush constantes; se fuerza con `TimingLogger.flush()` y cierra limpio al terminar el proceso.

## Visualizacion

1. Ejecuta cualquier flujo del GA/Simulacion para generar el CSV.
2. Corre el script:

   ```bash
   python scripts/plot_timings.py --run-id <uuid> --epoch 3 --batch-id 0 --top-n 15 --sections simulation_step,lyapunov_compute
   ```

   Si omites `--run-id`, se usa el CSV mas reciente en `data/timings/`.

3. El script genera cuatro PNG en `reports/` (timelines por individuo, por batch, simulacion y una grafica de pastel por seccion). Todos los ejes estan en microsegundos relativos al inicio del subconjunto graficado.

## Pruebas

Ejecuta el smoke test para validar el logger y los decoradores:

```bash
python -m pytest tests/test_timings_smoke.py
```

Esto crea un CSV temporal, verifica que existan filas con `duration_us >= 1` y cierra el logger global al final.
