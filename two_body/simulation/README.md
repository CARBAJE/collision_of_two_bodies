# Modulo de Simulacion

Este modulo agrupa los componentes necesarios para construir una simulacion gravitatoria basada en REBOUND y calcular exponentes de Lyapunov mediante la API MEGNO del propio motor.

## Contenido

- `rebound_adapter.py`
- `lyapunov.py`
- `__init__.py`

---

## `rebound_adapter.py` — Adaptador a REBOUND

- La clase `ReboundSim` encapsula la creacion de una instancia `rebound.Simulation`.
- Valida y normaliza masas, posiciones y velocidades (tres componentes por cuerpo).
- Permite integrar la simulacion y devolver una trayectoria discreta con forma `[pasos, cuerpos, 6]`.
- Se puede reutilizar para cualquier flujo que requiera inicializar REBOUND con una API homogenea.

### Diagrama de flujo (pendiente)

```mermaid
%% TODO: agregar diagrama definitivo para rebound_adapter.py
```

---

## `lyapunov.py` — Estimador de Lyapunov

- La clase `LyapunovEstimator` calcula el maximo exponente de Lyapunov utilizando directamente la ruta MEGNO de REBOUND.
- Flujo general:
  1. Recibe una simulacion o un diccionario con `{sim, dt, t_end, masses}`.
  2. Comprueba que `dt > 0` y determina el numero de pasos.
  3. Configura `sim.dt`, llama `sim.init_megno()` e integra paso a paso.
  4. Obtiene `lambda = sim.lyapunov()` y, si existe, `megno = sim.calculate_megno()`.
  5. Devuelve el resultado junto con un diccionario de metadatos (`impl`, `steps`, `dt`, `masses`, `megno`).

### Diagrama de flujo

```mermaid
flowchart TD
    START([Inicio Lyapunov]):::start
    INPUT[Entrada de trayectoria/contexto<br/>y ventana]:::input
    EXTRACT[Obtener sim, dt, t_end<br/>y masas (si existen)]:::process
    CHECK_DT{dt > 0?}:::decision
    ERR_DT[Error: dt debe ser positivo]:::error
    STEPS[steps = max(1, ceil(t_end / dt))]:::process
    PREP[sim.dt = dt;<br/>t0 = sim.t o 0]:::process
    MEGNO[sim.init_megno()]:::process
    LOOP{{Para i = 1 .. steps}}:::loop
    INTEGRATE[sim.integrate(t0 + i * dt)]:::process
    CALC_ME{`calculate_megno()` disponible?}:::decision
    MEGNO_VAL[megno = sim.calculate_megno()]:::process
    LYAP[lambda = sim.lyapunov()]:::result
    META[meta = {impl: rebound_megno,<br/>steps, dt, masses, megno?}]:::process
    RETURN[Retornar lambda y meta]:::finish
    FAIL[Error: API MEGNO no disponible]:::error

    START --> INPUT --> EXTRACT --> CHECK_DT
    CHECK_DT -- No --> ERR_DT --> RETURN
    CHECK_DT -- Si --> STEPS --> PREP --> MEGNO --> LOOP
    MEGNO -. AttributeError .-> FAIL --> RETURN
    LOOP --> INTEGRATE --> LOOP
    LOOP -->|Fin loop| CALC_ME
    CALC_ME -- Si --> MEGNO_VAL --> LYAP
    CALC_ME -- No --> LYAP
    LYAP --> META --> RETURN

    classDef start fill:#4CAF50,stroke:#2E7D32,color:#fff,stroke-width:3px;
    classDef input fill:#2196F3,stroke:#1565C0,color:#fff,stroke-width:2px;
    classDef process fill:#00BCD4,stroke:#00838F,color:#fff,stroke-width:2px;
    classDef decision fill:#FF9800,stroke:#E65100,color:#fff,stroke-width:2px;
    classDef loop fill:#9C27B0,stroke:#6A1B9A,color:#fff,stroke-width:2px;
    classDef result fill:#8BC34A,stroke:#558B2F,color:#fff,stroke-width:2px;
    classDef finish fill:#F06292,stroke:#AD1457,color:#fff,stroke-width:3px;
    classDef error fill:#F44336,stroke:#C62828,color:#fff,stroke-width:2px;
```

---

## `__init__.py` — Punto de entrada del paquete

- Reexporta `ReboundSim` y `LyapunovEstimator` para ofrecer una API sencilla: `from simulation import ReboundSim, LyapunovEstimator`.
- Incluye un ejemplo minimo al ejecutarse como script, mostrando los componentes disponibles.

---

## Interrelaciones clave

- `LyapunovEstimator` consume simulaciones REBOUND y produce estimaciones de Lyapunov usando MEGNO.
- `ReboundSim` construye los contextos REBOUND que se pueden pasar directamente al estimador o a otros scripts.
- `__init__.py` expone ambos componentes de forma unificada.
