#!/usr/bin/env python3
"""
Sistema de Optimización Gravitacional PARALELO con Exponentes de Lyapunov
========================================================================

Versión optimizada con paralelización y métodos de aceleración:
1. Paralelización con multiprocessing
2. Evaluación por lotes con joblib
3. Optimización de PyMOO con evaluación vectorizada
4. Cache de resultados
5. Aproximaciones rápidas para cribado inicial

Autor: Versión optimizada del sistema gravitacional
Bibliotecas requeridas: rebound, pymoo, lyapynov, mpmath, numpy, matplotlib, joblib
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import rebound
import lyapynov
import mpmath
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population

# Para paralelización
import multiprocessing as mp
from joblib import Parallel, delayed
from functools import lru_cache
import time
import warnings
warnings.filterwarnings('ignore')

# Configurar mpmath para alta precisión
mpmath.mp.dps = 30  # Reducido para mejor rendimiento

class FastGravitationalSystem:
    """Versión optimizada del sistema gravitacional"""

    def __init__(self, G=1.0, dt=0.01):
        self.G = G
        self.dt = dt

    def setup_simulation(self, m1, m2, perturbation=False, perturb_factor=1e-12):
        """Configuración rápida de simulación"""
        sim = rebound.Simulation()
        sim.G = self.G
        sim.dt = self.dt
        sim.integrator = "whfast"  # Integrador más rápido para sistemas estables

        # Configuración inicial optimizada
        sim.add(m=float(m1), x=-1.0, y=0.0, z=0.0, vx=0.0, vy=-0.5, vz=0.0)
        sim.add(m=float(m2), x=1.0, y=0.0, z=0.0, vx=0.0, vy=0.5, vz=0.0)
        sim.move_to_com()

        if perturbation:
            sim.particles[1].x *= (1 + perturb_factor)
            sim.particles[1].y *= (1 + perturb_factor)

        return sim

    def quick_stability_check(self, m1, m2, t_max=10.0):
        """Verificación rápida de estabilidad básica"""
        try:
            sim = self.setup_simulation(m1, m2)
            sim.integrate(t_max)

            # Verificar si los cuerpos siguen existiendo
            if len(sim.particles) < 2:
                return False, 1e6

            # Verificar distancia razonable
            p1, p2 = sim.particles[0], sim.particles[1]
            distance = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

            if distance > 50 or distance < 0.1:  # Muy lejos o muy cerca
                return False, 1e6

            return True, distance

        except:
            return False, 1e6

class ParallelLyapunovCalculator:
    """Calculadora paralela de exponentes de Lyapunov"""

    def __init__(self):
        self.grav_system = FastGravitationalSystem()

    @lru_cache(maxsize=1000)
    def cached_lyapunov(self, m1, m2, t_max):
        """Cálculo de Lyapunov con cache"""
        return self._calculate_lyapunov_core(float(m1), float(m2), float(t_max))

    def _calculate_lyapunov_core(self, m1, m2, t_max=50.0):
        """Cálculo optimizado del exponente de Lyapunov"""
        try:
            # Verificación rápida primero
            stable, distance = self.grav_system.quick_stability_check(m1, m2, t_max/5)
            if not stable:
                return 1e6

            # Si pasa la verificación rápida, calcular Lyapunov completo
            m1_mp = mpmath.mpf(str(m1))
            m2_mp = mpmath.mpf(str(m2))

            # Estado inicial optimizado
            x0 = np.array([-1.0, 0.0, 0.0, -0.5, 1.0, 0.0, 0.0, 0.5])
            t0 = 0.0
            dt = 0.02  # Paso temporal más grande para mayor velocidad

            def f(x, t):
                return self._gravitational_dynamics_fast(x, t, m1_mp, m2_mp)

            def jac(x, t):
                return self._jacobian_fast(x, t, m1_mp, m2_mp)

            system = lyapynov.ContinuousDS(x0, t0, f, jac, dt)

            # Parámetros optimizados para velocidad
            n_forward = 50   # Reducido para mayor velocidad
            n_compute = int(t_max / dt / 2)  # Reducido

            max_lyap = lyapynov.mLCE(system, n_forward, n_compute, keep=False)
            return float(max_lyap)

        except Exception as e:
            return 1e6

    def _gravitational_dynamics_fast(self, state, t, m1, m2, G=1.0):
        """Versión optimizada de las dinámicas gravitacionales"""
        x1, y1, vx1, vy1, x2, y2, vx2, vy2 = state

        dx = x2 - x1
        dy = y2 - y1
        r_sq = dx*dx + dy*dy

        if r_sq < 1e-20:
            return np.zeros(8)

        r = mpmath.sqrt(r_sq)
        r3 = r_sq * r

        fx = G * dx / r3
        fy = G * dy / r3

        return np.array([
            vx1, vy1, float(m2 * fx), float(m2 * fy),
            vx2, vy2, float(-m1 * fx), float(-m1 * fy)
        ])

    def _jacobian_fast(self, state, t, m1, m2, G=1.0):
        """Jacobiano optimizado"""
        x1, y1, vx1, vy1, x2, y2, vx2, vy2 = state

        dx = x2 - x1
        dy = y2 - y1
        r_sq = dx*dx + dy*dy

        if r_sq < 1e-20:
            return np.eye(8)

        r = mpmath.sqrt(r_sq)
        r3 = r_sq * r
        r5 = r_sq * r3

        # Derivadas optimizadas
        dfx_dx1 = float(G * (3*dx*dx/r5 - 1/r3))
        dfx_dy1 = float(G * 3*dx*dy/r5)
        dfy_dy1 = float(G * (3*dy*dy/r5 - 1/r3))

        J = np.zeros((8, 8))

        # Velocidades
        J[0, 2] = J[1, 3] = J[4, 6] = J[5, 7] = 1.0

        # Aceleraciones (matriz simétrica optimizada)
        J[2, 0] = m2 * dfx_dx1
        J[2, 1] = m2 * dfx_dy1
        J[2, 4] = -J[2, 0]
        J[2, 5] = -J[2, 1]

        J[3, 0] = m2 * dfx_dy1
        J[3, 1] = m2 * dfy_dy1
        J[3, 4] = -J[3, 0]
        J[3, 5] = -J[3, 1]

        J[6, 0] = -m1 * dfx_dx1
        J[6, 1] = -m1 * dfx_dy1
        J[6, 4] = -J[6, 0]
        J[6, 5] = -J[6, 1]

        J[7, 0] = -m1 * dfx_dy1
        J[7, 1] = -m1 * dfy_dy1
        J[7, 4] = -J[7, 0]
        J[7, 5] = -J[7, 1]

        return J

def evaluate_individual(individual, t_eval=30.0):
    """Función para evaluar un individuo (para paralelización)"""
    m1, m2 = individual

    if m1 <= 0 or m2 <= 0:
        return 1e6

    calc = ParallelLyapunovCalculator()
    lyap_exp = calc.cached_lyapunov(m1, m2, t_eval)

    if not np.isfinite(lyap_exp) or lyap_exp > 100:
        return 1e6

    return abs(lyap_exp)

class ParallelStabilityOptimizationProblem(Problem):
    """Problema de optimización con evaluación paralela"""

    def __init__(self, mass_bounds, t_eval=30.0, n_jobs=-1):
        n_var = 2
        n_obj = 1
        n_constr = 0

        xl = np.array([mass_bounds[0][0], mass_bounds[1][0]])
        xu = np.array([mass_bounds[0][1], mass_bounds[1][1]])

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)

        self.t_eval = t_eval
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()

        print(f"Configurando evaluación paralela con {self.n_jobs} procesos")

    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluación paralela de la población"""
        start_time = time.time()

        # Evaluación paralela usando joblib
        fitness_values = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(evaluate_individual)(individual, self.t_eval)
            for individual in X
        )

        eval_time = time.time() - start_time
        print(f"Evaluación de {len(X)} individuos completada en {eval_time:.2f}s")

        out["F"] = np.array(fitness_values).reshape(-1, 1)

class MultiAlgorithmOptimizer:
    """Optimizador que usa múltiples algoritmos en paralelo"""

    def __init__(self, problem, pop_size=50):
        self.problem = problem
        self.pop_size = pop_size

    def optimize_with_algorithm(self, alg_name, algorithm, generations=20):
        """Ejecuta optimización con un algoritmo específico"""
        print(f"\n--- Ejecutando {alg_name} ---")

        termination = get_termination("n_gen", generations)

        try:
            res = minimize(
                self.problem,
                algorithm,
                termination,
                seed=np.random.randint(1000),
                verbose=False
            )

            if res.X is not None:
                return {
                    'algorithm': alg_name,
                    'best_x': res.X,
                    'best_f': res.F[0],
                    'success': True
                }
        except Exception as e:
            print(f"Error en {alg_name}: {e}")

        return {
            'algorithm': alg_name,
            'best_x': None,
            'best_f': 1e6,
            'success': False
        }

    def parallel_multi_algorithm_search(self, generations=20):
        """Ejecuta múltiples algoritmos en paralelo"""

        algorithms = [
            ("Genetic Algorithm", GA(pop_size=self.pop_size, eliminate_duplicates=True))
            #,
            #("Particle Swarm", PSO(pop_size=self.pop_size)),
            #("Differential Evolution", DE(pop_size=self.pop_size))
        ]

        print("Iniciando búsqueda multi-algoritmo paralela...")

        # Ejecutar algoritmos en paralelo
        results = Parallel(n_jobs=min(3, mp.cpu_count()), backend='threading')(
            delayed(self.optimize_with_algorithm)(name, alg, generations)
            for name, alg in algorithms
        )

        # Encontrar el mejor resultado
        best_result = min(results, key=lambda x: x['best_f'])

        print("\n" + "="*60)
        print("RESULTADOS DE OPTIMIZACIÓN MULTI-ALGORITMO")
        print("="*60)

        for result in results:
            status = "✓" if result['success'] else "✗"
            print(f"{status} {result['algorithm']}: f = {result['best_f']:.6f}")

        return best_result, results

def fast_visualization(best_masses, t_max=20.0):
    """Visualización optimizada"""
    m1, m2 = best_masses
    grav_system = FastGravitationalSystem()

    # Simulación más eficiente
    sim = grav_system.setup_simulation(m1, m2)

    n_outputs = 1000  # Reducido para mayor velocidad
    times = np.linspace(0, t_max, n_outputs)
    positions = np.zeros((n_outputs, 4))

    try:
        for i, t in enumerate(times):
            sim.integrate(t)
            if len(sim.particles) < 2:
                break
            positions[i, 0] = sim.particles[0].x
            positions[i, 1] = sim.particles[0].y
            positions[i, 2] = sim.particles[1].x
            positions[i, 3] = sim.particles[1].y
    except:
        pass

    # Visualización rápida
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.7, label=f'Cuerpo 1 (m={m1:.4f})')
    plt.plot(positions[:, 2], positions[:, 3], 'r-', alpha=0.7, label=f'Cuerpo 2 (m={m2:.4f})')
    plt.title("Órbitas Estables")
    plt.axis('equal')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    distances = np.sqrt((positions[:, 2] - positions[:, 0])**2 +
                       (positions[:, 3] - positions[:, 1])**2)
    plt.plot(times[:len(distances)], distances, 'g-')
    plt.title("Distancia entre Cuerpos")
    plt.xlabel("Tiempo")
    plt.ylabel("Distancia")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\n=== ÓRBITA ESTABLE ===")
    print(f"Masas: m1={m1:.6f}, m2={m2:.6f}")
    print(f"Relación: m1/m2={m1/m2:.6f}")
    print(f"Distancia promedio: {np.mean(distances):.6f}")

def main():
    """Función principal optimizada"""
    print("=== OPTIMIZADOR GRAVITACIONAL PARALELO ===")
    print("Usando evaluación paralela y múltiples algoritmos\n")

    # Configuración
    mass_bounds = [(0.05, 10.0), (0.05, 10.0)]
    t_eval = 20.0  # Tiempo reducido para mayor velocidad
    n_jobs = mp.cpu_count()

    print(f"CPUs disponibles: {n_jobs}")
    print(f"Tiempo de evaluación: {t_eval}s por individuo")
    print(f"Rangos de masa: {mass_bounds}")

    start_time = time.time()

    # Crear problema paralelo
    pc = .90    #Probabilidad de cruza
    nc = 2  #amplitud cruza 
    nm = 20 #amplitud mutacion
    pm = 0.03   # Probabilidad de mutar
    tam_pob = 50  # Tamaño de la población
    generations = 20  # Número de generaciones

    def generar_poblacion():
        return [[random.uniform(mass_bounds[i][0], mass_bounds[i][1]) for i in range(len(mass_bounds[0]))] for _ in range(tam_pob)]

    def seleccion_padres(poblacion):
        padres = []
        for _ in range(2):
            torneo = random.sample(poblacion, 3)
            aptitudes = aptitud(torneo)
            ganador = torneo[np.argmin(aptitudes)]
            padres.append(ganador)
        return padres

    def aptitud( X):
        """Evaluación paralela de la población"""
        start_time = time.time()

        # Evaluación paralela usando joblib
        fitness_values = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(evaluate_individual)(individual, t_eval)
            for individual in X
        )

        eval_time = time.time() - start_time
        print(f"Evaluación de {len(X)} individuos completada en {eval_time:.2f}s")

        return np.array(fitness_values)
    def cruza(padre_1, padre_2):

        if random.random() <= pc:
            u = random.random()
            h1,h2 = [], []
            for i in range(len(padre_1)):
                b = 1 + 2 * min(padre_1[i] - mass_bounds[i][0], mass_bounds[i][1] - padre_2[i]) / max(abs(padre_2[i] - padre_1[i]), 1e-9)
                a = 2 - pow(abs(b),-(nc+1))
                if u <= 1/a:
                    b = pow(u*a, 1/(nc+1))
                else:
                    b = pow(1/(2-u*a),1/(nc+1))
                h1.append(.5*(padre_1[i]+padre_2[i]-b*abs(padre_2[i]-padre_1[i])))
                h2.append(.5*(padre_1[i]+padre_2[i]+b*abs(padre_2[i]-padre_1[i])))
        else:
            h1,h2 = padre_1[:],padre_2[:]
            
        return h1,h2

    def mutation(individuo):
        for i in range(len(individuo)):
            if random.random() <= pm:
                r = 0.4
                d = min(individuo[i]-mass_bounds[i][0],mass_bounds[i][1]-individuo[i])/ (mass_bounds[i][0]-mass_bounds[i][1])
                if r <= 0.5:
                    d = pow(2*r+(1-2*r)*pow(1-d,(nm+1)), 1/(nm+1)) - 1
                else:
                    d = 1 - pow(2*(1-r)+2*(r-0.5)*pow(1-d,(nm+1)), 1/(nm+1))
                individuo[i]=individuo[i]+d*(mass_bounds[i][0]-mass_bounds[i][1])
            
        return individuo

    poblacion = generar_poblacion()
    mejores = []

    for gen in range(generations):
        hijos = []
        
        for _ in range(tam_pob // 2):
            padres = seleccion_padres(poblacion)
            h1, h2 = cruza(padres[0], padres[1])
            hijos.append(mutation(h1))
            hijos.append(mutation(h2))
        poblacion = hijos  # Reemplazo generacional
        aptitudes = aptitud(poblacion)
        mejor = poblacion[np.argmin(aptitudes)]
        mejores.append(mejor)
        #print(f"Generación {gen + 1}: Mejor solución: {mejor} con aptitud {aptitud(mejor)}")
    apt_mej = aptitud(mejores)
    best_result = mejores[np.argmin(apt_mej)]
    print(f"Mejor solución: {best_result} con aptitud {apt_mej[np.argmin(apt_mej)]}")


    # Optimizador multi-algoritmo
    #optimizer = MultiAlgorithmOptimizer(problem, pop_size=30)  # Población reducida


    # Ejecutar optimización paralela
    #best_result, all_results = optimizer.parallel_multi_algorithm_search(generations=15)

    total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print("OPTIMIZACIÓN COMPLETADA")
    print(f"Tiempo total: {total_time:.2f} segundos")
    print(f"{'='*60}")

    if best_result:
        best_masses = best_result
        best_fitness = apt_mej[np.argmin(apt_mej)]

        print(f"✓ Mejores masas: m1={best_masses[0]:.8f}, m2={best_masses[1]:.8f}")
        print(f"✓ Fitness: {best_fitness:.8f}")
        print(f"✓ Relación de masas: {best_masses[0]/best_masses[1]:.6f}")

        # Evaluación final detallada
        calc = ParallelLyapunovCalculator()
        final_lyap = calc._calculate_lyapunov_core(
            best_masses[0], best_masses[1], t_max=50.0
        )
        print(f"✓ Exponente de Lyapunov: {final_lyap:.10f}")

        if final_lyap < 0:
            print("✓ Sistema ESTABLE")
        elif final_lyap < 0.01:
            print("✓ Sistema CUASI-ESTABLE")
        else:
            print("⚠ Sistema potencialmente INESTABLE")

        print("\nGenerando visualización...")
        fast_visualization(best_masses, t_max=30.0)

    else:
        print("✗ No se encontró solución óptima")
        print("Considera aumentar generaciones o ajustar parámetros")

if __name__ == "__main__":
    main()