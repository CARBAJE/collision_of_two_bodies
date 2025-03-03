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
GLOBAL_DT = 0.001
GLOBAL_T_MAX = 1.0      # Tiempo máximo de integración (años)
GLOBAL_N_STEPS = 100    # Número de pasos de integración

# Límites para las variables libres (por cuerpo)
GLOBAL_LOWER_BOUND = np.array([0.5, -0.1, -0.1, -0.5, 5.0, -0.1])
GLOBAL_UPPER_BOUND = np.array([1.5,  0.1,  0.1,  0.5, 7.0,  0.1])
# Configuración de los cuerpos
GLOBAL_BODIES_CONFIG = [
    {"mass": 1.0, "is_central": True,  "radius": 0.005},  # Estrella
    {"mass": 1e-3, "is_central": False, "radius": 0.001}   # 1 planeta
]

#############################################
# FUNCIONES DE RESTRICCIÓN
#############################################

def constraint_e_positive(sim_results):
    """Excentricidad final >= 0 para todos los cuerpos"""
    e_final = sim_results["e_arr"][-1, :]
    return -np.min(e_final)

def constraint_e_less_than_one(sim_results):
    """Excentricidad final < 1 para todos los cuerpos"""
    e_final = sim_results["e_arr"][-1, :]
    return np.max(e_final) - 1.0

def constraint_stable_orbit(sim_results):
    """Variación máxima del semieje mayor < 10%"""
    if sim_results is None:
        return 1e6  # Penalización alta si la simulación falla

    a_arr = sim_results["a_arr"]
    epsilon = 1e-9  # Evitar división por cero
    max_variation = np.max(np.abs(a_arr - np.mean(a_arr, axis=0)), axis=0)
    return np.max(max_variation / (np.mean(a_arr, axis=0) + epsilon)) - 0.1

GLOBAL_CONSTRAINTS = [
    constraint_e_positive,
    constraint_e_less_than_one,
    constraint_stable_orbit
]

#############################################
# CONFIGURACIÓN DE SIMULACIÓN
#############################################

class SimulationConfig:
    def __init__(self, integrator=GLOBAL_INTEGRATOR, dt=GLOBAL_DT,
                 t_max=GLOBAL_T_MAX, N_steps=GLOBAL_N_STEPS, bodies_config=None):
        self.integrator = integrator
        self.dt = dt
        self.t_max = t_max
        self.N_steps = N_steps
        self.bodies_config = bodies_config or GLOBAL_BODIES_CONFIG

#############################################
# EJECUTOR DE SIMULACIÓN
#############################################

class SimulationRunner:
    def __init__(self, config: SimulationConfig):
        self.config = config

    def initialize_simulation(self, opt_vars, perturb=False, delta=1e-6):
        """Inicializa la simulación con condiciones iniciales"""
        sim = rebound.Simulation()
        sim.units = ('AU', 'yr', 'Msun')
        sim.integrator = self.config.integrator
        sim.dt = self.config.dt

        # Añadir cuerpos
        free_idx = 0
        for body in self.config.bodies_config:
            if body["is_central"]:
                sim.add(m=body["mass"], r=body.get("radius", 0))
            else:
                idx = free_idx * 6
                x, y, z = opt_vars[idx:idx+3]
                vx, vy, vz = opt_vars[idx+3:idx+6]

                if perturb:
                    vx += delta

                sim.add(
                    m=body["mass"],
                    x=x, y=y, z=z,
                    vx=vx, vy=vy, vz=vz,
                    r=body.get("radius", 0)
                )
                free_idx += 1
        return sim

    def run_simulation(self, opt_vars, perturb=False):
        """Ejecuta la simulación y devuelve resultados"""
        try:
            sim = self.initialize_simulation(opt_vars, perturb)
        except rebound.ReboundError:
            return None

        times = np.linspace(0, self.config.t_max, self.config.N_steps)

        # Almacenar resultados
        n_bodies = sum(not b["is_central"] for b in self.config.bodies_config)
        a_arr = np.zeros((len(times), n_bodies))
        e_arr = np.zeros_like(a_arr)

        primary = sim.particles[0]

        try:
            for i, t in enumerate(times):
                sim.integrate(t, exact_finish_time=0)
                for j in range(n_bodies):
                    # CORRECCIÓN AQUÍ: Usar keyword arguments
                    orb = rebound.Orbit(
                        simulation=sim,
                        particle=sim.particles[j+1],
                        primary=primary
                    )
                    a_arr[i, j] = orb.a
                    e_arr[i, j] = orb.e
        except rebound.Collision:
            return None

        return {"times": times, "a_arr": a_arr, "e_arr": e_arr}

    def calculate_lyapunov(self, res_nom, res_pert):
        """Calcula el exponente de Lyapunov"""
        if res_nom is None or res_pert is None:
            return np.nan

        delta = np.sqrt(
            (res_nom["a_arr"] - res_pert["a_arr"])**2 +
            (res_nom["e_arr"] - res_pert["e_arr"])**2
        )

        try:
            valid = delta > 1e-12
            if np.sum(valid) < 10:
                return np.nan

            log_delta = np.log(delta[valid])
            t = res_nom["times"][valid]
            coeffs = np.polyfit(t, log_delta, 1)
            return coeffs[0]

        except Exception as e:
            print(f"Error calculando Lyapunov: {str(e)}")
            return np.nan

#############################################
# FUNCIONES OBJETIVO
#############################################

def objective_stability(sim_results):
    """Minimizar la variación del semieje mayor"""
    if sim_results is None or sim_results["a_arr"] is None:
        return 1e6  # Penalización por simulación fallida
    return np.mean(np.ptp(sim_results["a_arr"], axis=0))

def objective_lyapunov(sim_runner, opt_vars):
    """Minimizar el exponente de Lyapunov"""
    res_nom = sim_runner.run_simulation(opt_vars)
    res_pert = sim_runner.run_simulation(opt_vars, perturb=True)
    lam = sim_runner.calculate_lyapunov(res_nom, res_pert)
    return lam if not np.isnan(lam) else 1e6

#############################################
# PROBLEMA DE OPTIMIZACIÓN
#############################################

class OrbitalOptimizationProblem(Problem):
    def __init__(self, sim_runner, n_free_bodies=1):
        self.runner = sim_runner
        n_var = n_free_bodies * 6
        n_obj = 2
        n_ieq_constr = len(GLOBAL_CONSTRAINTS)

        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_ieq_constr=n_ieq_constr,
            xl=np.tile(GLOBAL_LOWER_BOUND, n_free_bodies),
            xu=np.tile(GLOBAL_UPPER_BOUND, n_free_bodies)
        )

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.full((X.shape[0], self.n_obj), 1e6)  # Inicializar con valores altos
        G = np.full((X.shape[0], self.n_ieq_constr), 1e6)

        for i in range(X.shape[0]):
            res = self.runner.run_simulation(X[i])

            if res is not None and res["a_arr"] is not None:
                # Cálculo de objetivos y restricciones
                F[i, 0] = objective_stability(res)
                F[i, 1] = objective_lyapunov(self.runner, X[i])

                for j, constr in enumerate(GLOBAL_CONSTRAINTS):
                    G[i, j] = constr(res)

        out["F"] = F
        out["G"] = G

#############################################
# EJECUCIÓN PRINCIPAL
#############################################

if __name__ == "__main__":
    config = SimulationConfig(
        t_max=0.5,
        N_steps=50,
        bodies_config=GLOBAL_BODIES_CONFIG
    )

    runner = SimulationRunner(config)
    problem = OrbitalOptimizationProblem(runner, n_free_bodies=1)

    algorithm = NSGA2(pop_size=30)
    res = minimize(
        problem,
        algorithm,
        ("n_gen", 20),
        seed=42,
        verbose=True
    )

    print("\nMejores soluciones encontradas:")
    for i, (x, f) in enumerate(zip(res.X, res.F)):
        print(f"Sol {i+1}:")
        print(f" - Parámetros: {x}")
        print(f" - Objetivos: {f}\n")