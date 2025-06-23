#!/usr/bin/env python3
"""
Sistema de Optimización Gravitacional con Exponentes de Lyapunov
================================================================

Este script implementa un sistema que utiliza algoritmos genéticos para encontrar
configuraciones de masa que resulten en órbitas gravitacionales estables,
medidas por el Exponente de Lyapunov.

Autor: Basado en el análisis del proyecto gravitacional
Bibliotecas requeridas: rebound, pymoo, lyapynov, mpmath, numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import rebound
import lyapynov
import mpmath
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import warnings
warnings.filterwarnings('ignore')

# Configurar mpmath para alta precisión
mpmath.mp.dps = 50  # 50 dígitos decimales de precisión

class GravitationalSystem:
    """Clase para manejar simulaciones gravitacionales con REBOUND"""

    def __init__(self, G=1.0, dt=0.01):
        self.G = G
        self.dt = dt

    def setup_simulation(self, m1, m2, perturbation=False, perturb_factor=1e-12):
        """
        Configura una simulación de REBOUND con dos cuerpos

        Args:
            m1, m2: Masas de los cuerpos
            perturbation: Si aplicar perturbación para calcular Lyapunov
            perturb_factor: Factor de perturbación
        """
        sim = rebound.Simulation()
        sim.G = self.G
        sim.dt = self.dt

        # Configuración inicial: órbita circular
        # Cuerpo 1: a la izquierda con velocidad hacia abajo
        sim.add(m=float(m1), x=-1.0, y=0.0, z=0.0, vx=0.0, vy=-0.5, vz=0.0)

        # Cuerpo 2: a la derecha con velocidad hacia arriba
        sim.add(m=float(m2), x=1.0, y=0.0, z=0.0, vx=0.0, vy=0.5, vz=0.0)

        # Mover al centro de masas
        sim.move_to_com()

        # Aplicar perturbación si se requiere
        if perturbation:
            sim.particles[1].x *= (1 + perturb_factor)
            sim.particles[1].y *= (1 + perturb_factor)

        return sim

    def run_simulation(self, sim, t_max, n_outputs=1000):
        """
        Ejecuta una simulación y devuelve las trayectorias

        Args:
            sim: Simulación de REBOUND
            t_max: Tiempo máximo de simulación
            n_outputs: Número de puntos de salida

        Returns:
            times, positions: Arrays con tiempos y posiciones
        """
        times = np.linspace(0, t_max, n_outputs)
        positions = np.zeros((n_outputs, 4))  # x1, y1, x2, y2

        try:
            for i, t in enumerate(times):
                sim.integrate(t)
                if len(sim.particles) < 2:
                    # Si un cuerpo escapa, marcar como inestable
                    return times[:i], positions[:i]

                positions[i, 0] = sim.particles[0].x
                positions[i, 1] = sim.particles[0].y
                positions[i, 2] = sim.particles[1].x
                positions[i, 3] = sim.particles[1].y

        except Exception as e:
            print(f"Error en simulación: {e}")
            return times[:i] if 'i' in locals() else times[:1], positions[:i] if 'i' in locals() else positions[:1]

        return times, positions

class LyapunovCalculator:
    """Calculadora de exponentes de Lyapunov usando la biblioteca lyapynov y mpmath"""

    def __init__(self, grav_system):
        self.grav_system = grav_system

    def gravitational_dynamics(self, state, t, m1, m2, G=1.0):
        """
        Define las ecuaciones diferenciales del sistema gravitacional

        Args:
            state: [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
            t: tiempo
            m1, m2: masas
            G: constante gravitacional
        """
        x1, y1, vx1, vy1, x2, y2, vx2, vy2 = state

        # Distancia entre cuerpos
        dx = x2 - x1
        dy = y2 - y1
        r = mpmath.sqrt(dx**2 + dy**2)

        if r < 1e-10:  # Evitar división por cero
            return np.zeros(8)

        # Fuerzas gravitacionales
        r3 = r**3
        fx = G * dx / r3
        fy = G * dy / r3

        # Ecuaciones de movimiento
        dstate = np.zeros(8)
        dstate[0] = vx1                    # dx1/dt = vx1
        dstate[1] = vy1                    # dy1/dt = vy1
        dstate[2] = float(m2 * fx)         # dvx1/dt = m2*fx/m1 (asumiendo m1=1)
        dstate[3] = float(m2 * fy)         # dvy1/dt = m2*fy/m1
        dstate[4] = vx2                    # dx2/dt = vx2
        dstate[5] = vy2                    # dy2/dt = vy2
        dstate[6] = float(-m1 * fx)        # dvx2/dt = -m1*fx/m2 (asumiendo m2=1)
        dstate[7] = float(-m1 * fy)        # dvy2/dt = -m1*fy/m2

        return dstate

    def jacobian(self, state, t, m1, m2, G=1.0):
        """
        Calcula el Jacobiano del sistema gravitacional
        """
        x1, y1, vx1, vy1, x2, y2, vx2, vy2 = state

        dx = x2 - x1
        dy = y2 - y1
        r = mpmath.sqrt(dx**2 + dy**2)

        if r < 1e-10:
            return np.eye(8)

        r3 = r**3
        r5 = r**5

        # Derivadas parciales de las fuerzas
        dfx_dx1 = float(G * (3*dx**2/r5 - 1/r3))
        dfx_dy1 = float(G * 3*dx*dy/r5)
        dfx_dx2 = float(-G * (3*dx**2/r5 - 1/r3))
        dfx_dy2 = float(-G * 3*dx*dy/r5)

        dfy_dx1 = float(G * 3*dx*dy/r5)
        dfy_dy1 = float(G * (3*dy**2/r5 - 1/r3))
        dfy_dx2 = float(-G * 3*dx*dy/r5)
        dfy_dy2 = float(-G * (3*dy**2/r5 - 1/r3))

        # Construir el Jacobiano
        J = np.zeros((8, 8))

        # Derivadas de las velocidades
        J[0, 2] = 1.0  # dx1/dvx1
        J[1, 3] = 1.0  # dy1/dvy1
        J[4, 6] = 1.0  # dx2/dvx2
        J[5, 7] = 1.0  # dy2/dvy2

        # Derivadas de las aceleraciones
        J[2, 0] = m2 * dfx_dx1
        J[2, 1] = m2 * dfx_dy1
        J[2, 4] = m2 * dfx_dx2
        J[2, 5] = m2 * dfx_dy2

        J[3, 0] = m2 * dfy_dx1
        J[3, 1] = m2 * dfy_dy1
        J[3, 4] = m2 * dfy_dx2
        J[3, 5] = m2 * dfy_dy2

        J[6, 0] = -m1 * dfx_dx1
        J[6, 1] = -m1 * dfx_dy1
        J[6, 4] = -m1 * dfx_dx2
        J[6, 5] = -m1 * dfx_dy2

        J[7, 0] = -m1 * dfy_dx1
        J[7, 1] = -m1 * dfy_dy1
        J[7, 4] = -m1 * dfy_dx2
        J[7, 5] = -m1 * dfy_dy2

        return J

    def calculate_lyapunov_exponent(self, m1, m2, t_max=100.0, dt=0.01):
        """
        Calcula el exponente de Lyapunov máximo usando la biblioteca lyapynov

        Args:
            m1, m2: Masas de los cuerpos
            t_max: Tiempo de evaluación
            dt: Paso temporal

        Returns:
            max_lyapunov: Exponente de Lyapunov máximo
        """
        try:
            # Convertir a alta precisión
            m1 = mpmath.mpf(str(m1))
            m2 = mpmath.mpf(str(m2))

            # Estado inicial: [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
            x0 = np.array([-1.0, 0.0, 0.0, -0.5, 1.0, 0.0, 0.0, 0.5])
            t0 = 0.0

            # Crear el sistema dinámico
            def f(x, t):
                return self.gravitational_dynamics(x, t, m1, m2)

            def jac(x, t):
                return self.jacobian(x, t, m1, m2)

            system = lyapynov.ContinuousDS(x0, t0, f, jac, dt)

            # Calcular el exponente de Lyapunov máximo
            n_forward = 100   # Pasos para estabilizar
            n_compute = int(t_max / dt)  # Pasos para computar

            max_lyap = lyapynov.mLCE(system, n_forward, n_compute, keep=False)

            return float(max_lyap)

        except Exception as e:
            print(f"Error calculando Lyapunov para masas ({m1}, {m2}): {e}")
            return 1e6  # Valor de penalización

class StabilityOptimizationProblem(Problem):
    """Problema de optimización para encontrar masas que minimicen el exponente de Lyapunov"""

    def __init__(self, mass_bounds, t_eval=50.0):
        """
        Args:
            mass_bounds: [(min_m1, max_m1), (min_m2, max_m2)]
            t_eval: Tiempo de evaluación para cada simulación
        """
        n_var = 2  # Dos variables: m1, m2
        n_obj = 1  # Un objetivo: minimizar el exponente de Lyapunov
        n_constr = 0  # Sin restricciones explícitas

        xl = np.array([mass_bounds[0][0], mass_bounds[1][0]])  # Límites inferiores
        xu = np.array([mass_bounds[0][1], mass_bounds[1][1]])  # Límites superiores

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)

        self.grav_system = GravitationalSystem()
        self.lyap_calculator = LyapunovCalculator(self.grav_system)
        self.t_eval = t_eval

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evalúa la población de individuos

        Args:
            X: Array de individuos (población) de forma (n_pop, 2)
            out: Diccionario para almacenar los resultados
        """
        fitness_values = []

        for individual in X:
            m1, m2 = individual

            # Verificar que las masas sean positivas
            if m1 <= 0 or m2 <= 0:
                fitness_values.append(1e6)
                continue

            # Calcular el exponente de Lyapunov
            lyap_exp = self.lyap_calculator.calculate_lyapunov_exponent(
                m1, m2, t_max=self.t_eval
            )

            # Si el exponente es muy grande o no finito, penalizar
            if not np.isfinite(lyap_exp) or lyap_exp > 100:
                fitness_values.append(1e6)
            else:
                # Queremos minimizar el exponente (valores negativos son mejores)
                fitness_values.append(abs(lyap_exp))

        out["F"] = np.array(fitness_values).reshape(-1, 1)

def visualize_stable_orbit(best_masses, t_max=20.0, n_outputs=2000):
    """
    Visualiza la órbita estable encontrada

    Args:
        best_masses: Tupla (m1, m2) de las mejores masas
        t_max: Tiempo máximo de simulación para visualización
        n_outputs: Número de puntos para la visualización
    """
    m1, m2 = best_masses
    grav_system = GravitationalSystem()

    # Configurar y ejecutar simulación final
    sim = grav_system.setup_simulation(m1, m2)
    times, positions = grav_system.run_simulation(sim, t_max, n_outputs)

    if len(positions) == 0:
        print("Error: No se pudo generar la trayectoria.")
        return

    # Extraer posiciones
    x1, y1 = positions[:, 0], positions[:, 1]
    x2, y2 = positions[:, 2], positions[:, 3]

    # Crear la visualización
    plt.figure(figsize=(12, 10))

    # Subplot 1: Trayectorias completas
    plt.subplot(2, 2, 1)
    plt.plot(x1, y1, 'b-', label=f'Cuerpo 1 (m={m1:.4f})', alpha=0.7)
    plt.plot(x2, y2, 'r-', label=f'Cuerpo 2 (m={m2:.4f})', alpha=0.7)
    plt.scatter(x1[0], y1[0], c='blue', s=100, marker='o', label='Inicio 1')
    plt.scatter(x2[0], y2[0], c='red', s=100, marker='o', label='Inicio 2')
    plt.title("Órbitas Estables Completas")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Subplot 2: Zoom central
    plt.subplot(2, 2, 2)
    plt.plot(x1, y1, 'b-', alpha=0.7)
    plt.plot(x2, y2, 'r-', alpha=0.7)
    plt.title("Vista Central")
    plt.xlabel("X")
    plt.ylabel("Y")
    center_range = max(np.std(x1), np.std(y1), np.std(x2), np.std(y2)) * 2
    plt.xlim(-center_range, center_range)
    plt.ylim(-center_range, center_range)
    plt.grid(True, alpha=0.3)

    # Subplot 3: Distancia entre cuerpos vs tiempo
    plt.subplot(2, 2, 3)
    distances = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    plt.plot(times[:len(distances)], distances, 'g-', linewidth=2)
    plt.title("Distancia entre Cuerpos")
    plt.xlabel("Tiempo")
    plt.ylabel("Distancia")
    plt.grid(True, alpha=0.3)

    # Subplot 4: Energía del sistema (aproximada)
    plt.subplot(2, 2, 4)
    # Calcular energía cinética y potencial aproximadas
    if len(positions) > 1:
        dt = times[1] - times[0] if len(times) > 1 else 0.01
        vx1 = np.gradient(x1, dt)
        vy1 = np.gradient(y1, dt)
        vx2 = np.gradient(x2, dt)
        vy2 = np.gradient(y2, dt)

        kinetic = 0.5 * (m1 * (vx1**2 + vy1**2) + m2 * (vx2**2 + vy2**2))
        potential = -m1 * m2 / distances
        total_energy = kinetic + potential

        plt.plot(times[:len(total_energy)], total_energy, 'purple', linewidth=2)
        plt.title("Energía Total del Sistema")
        plt.xlabel("Tiempo")
        plt.ylabel("Energía")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Información adicional
    print(f"\n=== INFORMACIÓN DE LA ÓRBITA ESTABLE ===")
    print(f"Masas óptimas: m1 = {m1:.6f}, m2 = {m2:.6f}")
    print(f"Relación de masas: m1/m2 = {m1/m2:.6f}")
    print(f"Tiempo simulado: {t_max} unidades")
    print(f"Distancia promedio: {np.mean(distances):.6f}")
    print(f"Distancia mín/máx: {np.min(distances):.6f} / {np.max(distances):.6f}")

def main():
    """Función principal del programa"""
    print("=== OPTIMIZADOR DE ÓRBITAS GRAVITACIONALES ===")
    print("Buscando configuraciones de masa para órbitas estables...")
    print("Utilizando exponentes de Lyapunov con precisión extendida\n")

    # Configuración del problema
    mass_bounds = [(0.05, 10.0), (0.05, 10.0)]  # Rangos para m1 y m2
    t_eval = 30.0  # Tiempo de evaluación por individuo

    # Crear el problema de optimización
    problem = StabilityOptimizationProblem(mass_bounds, t_eval)

    # Configurar el algoritmo genético
    algorithm = GA(
        pop_size=50,              # Tamaño de población
        eliminate_duplicates=True  # Eliminar duplicados
    )

    # Criterio de terminación
    termination = get_termination("n_gen", 30)  # 30 generaciones

    # Ejecutar optimización
    print("Iniciando proceso de optimización...")
    print(f"Población: {algorithm.pop_size}")
    print(f"Generaciones máximas: 30")
    print(f"Tiempo de evaluación por individuo: {t_eval}s")
    print(f"Rangos de masa: m1 ∈ {mass_bounds[0]}, m2 ∈ {mass_bounds[1]}")
    print("-" * 60)

    try:
        res = minimize(
            problem,
            algorithm,
            termination,
            seed=42,
            verbose=True
        )

        # Mostrar resultados
        print("\n" + "="*60)
        print("OPTIMIZACIÓN COMPLETADA")
        print("="*60)

        if res.X is not None:
            best_masses = res.X
            best_fitness = res.F[0]

            print(f"✓ Mejores masas encontradas:")
            print(f"  m1 = {best_masses[0]:.8f}")
            print(f"  m2 = {best_masses[1]:.8f}")
            print(f"✓ Mejor fitness (|Exponente de Lyapunov|): {best_fitness:.8f}")
            print(f"✓ Relación de masas (m1/m2): {best_masses[0]/best_masses[1]:.6f}")

            # Calcular el exponente de Lyapunov exacto para las mejores masas
            lyap_calc = LyapunovCalculator(GravitationalSystem())
            exact_lyap = lyap_calc.calculate_lyapunov_exponent(
                best_masses[0], best_masses[1], t_max=100.0
            )
            print(f"✓ Exponente de Lyapunov exacto: {exact_lyap:.10f}")

            if exact_lyap < 0:
                print("✓ Sistema ESTABLE (exponente negativo)")
            elif exact_lyap < 0.01:
                print("✓ Sistema CUASI-ESTABLE (exponente muy pequeño)")
            else:
                print("⚠ Sistema potencialmente INESTABLE")

            print("\nGenerando visualización de la órbita estable...")
            visualize_stable_orbit(best_masses, t_max=50.0)

        else:
            print("✗ No se encontró una solución óptima.")
            print("Considera ajustar los parámetros del algoritmo.")

    except Exception as e:
        print(f"✗ Error durante la optimización: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()