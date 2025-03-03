import rebound
import numpy as np
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from scipy.optimize import curve_fit

#############################################
# VARIABLES GLOBALES
#############################################

# Parámetros generales de la simulación
GLOBAL_INTEGRATOR = 'ias15'
GLOBAL_DT = 0.01
GLOBAL_T_MAX = 10.0      # Tiempo máximo de integración (años)
GLOBAL_N_STEPS = 100     # Número de pasos de integración

# Límites para las variables libres (para cada cuerpo libre: [x, y, z, vx, vy, vz])
# Ejemplo para 3 cuerpos libres (cada uno con 6 variables):
GLOBAL_LOWER_BOUND = np.array([0.5, -0.5, -0.5, -10.0, 5.0, -1.0])  # Límites por cuerpo
GLOBAL_UPPER_BOUND = np.array([1.5,  0.5,  0.5,  10.0, 10.0,  1.0])

# En ModularOptimizationProblem, se aplican automáticamente con np.tile.

# Configuración de los cuerpos (cada diccionario incluye masa, radio y si es central)
GLOBAL_BODIES_CONFIG = [
    {"mass": 1.0, "is_central": True,  "radius": 0.005},  # Estrella central
    {"mass": 1e-3, "is_central": False, "radius": 0.001},  # Planeta 1
    {"mass": 1e-3, "is_central": False, "radius": 0.001},  # Planeta 2
    {"mass": 1e-4, "is_central": False, "radius": 0.0005}  # Asteroide
]

# Restricciones globales de estabilidad (definidas como funciones)
def constraint_e_positive(sim_results):
    # Restricción: e_final >= 0 para todos los cuerpos --> max(-e_final) <= 0
    e_final = sim_results["e_arr"][-1, :]  # Excentricidades finales (todos los cuerpos)
    return -np.min(e_final)  # Asegura que la mínima excentricidad sea >= 0

def constraint_e_less_than_one(sim_results):
    """
    Restricción: e_final < 1 para todos los cuerpos.
    """
    e_final = sim_results["e_arr"][-1, :]  # Excentricidades finales de todos los cuerpos
    return np.max(e_final) - 1  # Máxima excentricidad debe ser < 1

def constraint_a_variation(sim_results):
    # Restricción: min(a) >= 0.75 * max(a) para cada cuerpo individualmente
    a_arr = sim_results["a_arr"]  # Forma: (pasos_tiempo, cuerpos)
    violations = []
    for j in range(a_arr.shape[1]):  # Iterar sobre cada cuerpo
        a_min = np.min(a_arr[:, j])
        a_max = np.max(a_arr[:, j])
        violations.append(0.75 * a_max - a_min)
    return np.max(violations)  # La peor violación

def constraint_lyapunov(sim_runner, opt_vars):
    # Restricción: λ < 1e-9 (escalar)
    res_nom = sim_runner.run_simulation(opt_vars, perturb=False)
    res_pert = sim_runner.run_simulation(opt_vars, perturb=True)
    lam = sim_runner.calculate_lyapunov(res_nom["times"], res_nom["a_arr"], res_nom["e_arr"],
                                          res_pert["a_arr"], res_pert["e_arr"])
    return lam - 1e-9 if not np.isnan(lam) else 1e6

# Lista global de funciones de restricción
GLOBAL_CONSTRAINTS = [
    constraint_e_positive,
    constraint_e_less_than_one,
    constraint_a_variation,
    constraint_lyapunov
]

#############################################
# CONFIGURACIÓN MODULAR DE LA SIMULACIÓN
#############################################

class SimulationConfig:
    def __init__(self, integrator=GLOBAL_INTEGRATOR, dt=GLOBAL_DT,
                 t_max=GLOBAL_T_MAX, N_steps=GLOBAL_N_STEPS, bodies_config=None):
        self.integrator = integrator
        self.dt = dt
        self.t_max = t_max
        self.N_steps = N_steps
        self.bodies_config = bodies_config if bodies_config is not None else GLOBAL_BODIES_CONFIG

#############################################
# EJECUTOR DE SIMULACIÓN (RUNNER)
#############################################

class SimulationRunner:
    def __init__(self, config: SimulationConfig):
        self.config = config

    def initialize_simulation(self, opt_vars, perturb=False, delta=1e-6):
        """
        Inicializa la simulación usando la configuración.
        opt_vars es un array con las condiciones iniciales para cada cuerpo libre,
        en el orden: [x, y, z, vx, vy, vz] (para cada cuerpo).
        Si perturb=True se añade una pequeña perturbación (por ejemplo, en vx).
        Se incluye el radio (r) según se especifique en bodies_config.
        """
        sim = rebound.Simulation()
        sim.units = ('AU', 'yr', 'Msun')
        free_body_index = 0
        for body in self.config.bodies_config:
            r_val = body.get("radius", 0.0)
            if body.get("is_central", False):
                sim.add(m=body["mass"], r=r_val)
            else:
                if opt_vars is not None:
                    idx = free_body_index * 6
                    x = opt_vars[idx + 0]
                    y = opt_vars[idx + 1]
                    z = opt_vars[idx + 2]
                    vx = opt_vars[idx + 3]
                    vy = opt_vars[idx + 4]
                    vz = opt_vars[idx + 5]
                    if perturb:
                        vx += delta
                    sim.add(m=body["mass"], x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, r=r_val)
                    free_body_index += 1
                else:
                    sim.add(m=body["mass"], r=r_val)
        sim.integrator = self.config.integrator
        sim.dt = self.config.dt
        return sim

    def run_simulation(self, opt_vars, perturb=False, delta=1e-6):
        """
        Corre la simulación y retorna parámetros orbitales para múltiples cuerpos.
        """
        sim = self.initialize_simulation(opt_vars, perturb=perturb, delta=delta)
        times = np.linspace(0, self.config.t_max, self.config.N_steps)

        # Contar cuerpos no centrales
        n_non_central = sum(1 for body in self.config.bodies_config if not body.get("is_central", False))

        # Inicializar arrays para cada cuerpo no central
        a_arr = np.zeros((self.config.N_steps, n_non_central))  # Forma: (pasos_tiempo, cuerpos)
        e_arr = np.zeros_like(a_arr)
        inc_arr = np.zeros_like(a_arr)

        primary = sim.particles[0]  # Cuerpo central (índice 0)

        for i, t in enumerate(times):
            sim.integrate(t)
            for j in range(n_non_central):  # Índices de cuerpos no centrales: 1, 2, ...
                secondary = sim.particles[j + 1]
                orb = rebound.Orbit(simulation=sim, particle=secondary, primary=primary)
                a_arr[i, j] = orb.a
                e_arr[i, j] = orb.e
                inc_arr[i, j] = orb.inc

        return {"times": times, "a_arr": a_arr, "e_arr": e_arr, "inc_arr": inc_arr}

    def calculate_lyapunov(self, times, a_nom, e_nom, a_pert, e_pert):
        """
        Calcula el exponente de Lyapunov λ a partir de la divergencia entre la órbita nominal y la perturbada.
        """
        delta_orbit = np.sqrt((a_nom - a_pert)**2 + (e_nom - e_pert)**2)
        if np.any(delta_orbit > 1e-3):
            cutoff = np.argmax(delta_orbit > 1e-3)
            valid = (delta_orbit > 0) & (delta_orbit < 1e-3) & (times < times[cutoff])
        else:
            valid = (delta_orbit > 0) & (delta_orbit < 1e-3)
        lam = np.nan
        if np.sum(valid) > 10:
            def model(t, lam):
                return delta_orbit[valid][0] * np.exp(np.clip(lam * t, -100, 100))
            try:
                params, _ = curve_fit(model, times[valid], delta_orbit[valid], maxfev=5000)
                lam = params[0]
            except Exception:
                lam = np.nan
        return lam

#############################################
# FUNCIONES OBJETIVO (MODULARES)
#############################################

def objective_variation_a(sim_results):
    """
    Objetivo: Minimizar la suma de variaciones de 'a' para todos los cuerpos.
    """
    variations = np.max(sim_results["a_arr"], axis=0) - np.min(sim_results["a_arr"], axis=0)
    return np.sum(variations)  # Suma de variaciones de todos los cuerpos

def objective_lyapunov(sim_runner, opt_vars):
    """
    Objetivo: minimizar el exponente de Lyapunov.
    Se corre la simulación nominal y perturbada para calcular λ.
    """
    res_nom = sim_runner.run_simulation(opt_vars, perturb=False)
    res_pert = sim_runner.run_simulation(opt_vars, perturb=True)
    lam = sim_runner.calculate_lyapunov(res_nom["times"], res_nom["a_arr"], res_nom["e_arr"],
                                          res_pert["a_arr"], res_pert["e_arr"])
    return lam if not np.isnan(lam) else 1e6

#############################################
# PROBLEMA DE OPTIMIZACIÓN MODULAR CON PYMOO
#############################################

class ModularOptimizationProblem(Problem):
    def __init__(self, sim_runner: SimulationRunner, n_free_bodies=1,
                 objective_funcs=None, constraint_funcs=None):
        """
        Parámetros:
         - sim_runner: Instancia de SimulationRunner.
         - n_free_bodies: Número de cuerpos libres (cada uno con 6 variables).
         - objective_funcs: Lista de funciones objetivo.
         - constraint_funcs: Lista de funciones de restricción.
        """
        self.sim_runner = sim_runner
        self.objective_funcs = objective_funcs
        self.constraint_funcs = constraint_funcs

        n_var = n_free_bodies * 6  # 6 variables por cada cuerpo libre.
        n_obj = len(objective_funcs)
        n_ieq_constr = len(constraint_funcs)
        # Usar los límites globales
        xl = np.tile(GLOBAL_LOWER_BOUND, n_free_bodies)
        xu = np.tile(GLOBAL_UPPER_BOUND, n_free_bodies)
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_ieq_constr, n_eq_constr=0, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        n_samples = X.shape[0]
        F = np.zeros((n_samples, self.n_obj))
        G = np.zeros((n_samples, self.n_ieq_constr))

        for i in range(n_samples):
            opt_vars = X[i]
            # Ejecutar la simulación nominal.
            sim_results = self.sim_runner.run_simulation(opt_vars, perturb=False)
            # Evaluar las funciones objetivo.
            for j, func in enumerate(self.objective_funcs):
                if func.__name__ == "objective_lyapunov":
                    F[i, j] = func(self.sim_runner, opt_vars)
                else:
                    F[i, j] = func(sim_results)
            # Evaluar las restricciones.
            for k, func in enumerate(self.constraint_funcs):
                if func.__name__ == "constraint_lyapunov":
                    G[i, k] = func(self.sim_runner, opt_vars)
                else:
                    G[i, k] = func(sim_results)

        out["F"] = F
        out["G"] = G

#############################################
# EJECUCIÓN DE LA OPTIMIZACIÓN
#############################################

if __name__ == "__main__":
    # Crear la configuración de la simulación usando variables globales.
    config = SimulationConfig()
    sim_runner = SimulationRunner(config)

    # Lista de funciones objetivo (buscando estabilidad, sin target específico)
    objectives = [
        objective_variation_a,
        objective_lyapunov  # Esta función requiere el runner y opt_vars
    ]

    # Usar la lista global de restricciones
    constraints = GLOBAL_CONSTRAINTS

    # Definir el problema de optimización.
    problem = ModularOptimizationProblem(
        sim_runner,
        n_free_bodies=3,  # 3 cuerpos no centrales en GLOBAL_BODIES_CONFIG
        objective_funcs=objectives,
        constraint_funcs=constraints
    )

    # Seleccionar NSGA-II como algoritmo multiobjetivo
    algorithm = NSGA2(pop_size=50)

    # Ejecutar la optimización (ajustar el criterio de terminación según sea necesario)
    res = minimize(problem, algorithm, termination=('n_gen', 100), seed=1, verbose=True)

    # Mostrar resultados
    print("Soluciones no dominadas (condiciones iniciales):")
    print(res.X)
    print("\nValores de los objetivos (variación en a y λ):")
    print(res.F)
    print("\nRestricciones (deben ser <= 0):")
    print(res.G)