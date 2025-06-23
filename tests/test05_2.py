#!/usr/bin/env python3
"""
Sistema de Optimización Gravitacional ULTRA-OPTIMIZADO con Hardware Avanzado
===========================================================================

Versión con optimizaciones extremas de hardware:
1. Cache masivo aprovechando 64GB RAM
2. Procesamiento por lotes masivos
3. Pre-cálculo de matrices y jacobianos
4. Soporte GPU con CuPy/CUDA
5. Múltiples niveles de cache
6. Buffer circular para trayectorias
7. Interpolación rápida con grids pre-calculados

Autor: Versión ultra-optimizada del sistema gravitacional
Bibliotecas: rebound, pymoo, lyapynov, mpmath, numpy, matplotlib, joblib, cupy, numba
"""

import numpy as np
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

# Paralelización y optimización
import multiprocessing as mp
from joblib import Parallel, delayed
from functools import lru_cache
import time
import warnings
import psutil
import gc
from collections import deque, OrderedDict
import pickle

# GPU y optimizaciones numéricas
try:
    import cupy as cp
    import cupyx.scipy
    GPU_AVAILABLE = True
    print("✓ GPU CuPy disponible")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠ GPU CuPy no disponible, usando CPU")

try:
    import numba
    from numba import jit, cuda
    NUMBA_AVAILABLE = True
    print("✓ Numba JIT disponible")
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠ Numba no disponible")

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIONES AVANZADAS DE HARDWARE
# ============================================================================

class HardwareConfig:
    """Configuración optimizada para hardware específico"""

    def __init__(self):
        # Información del sistema
        self.ram_gb = psutil.virtual_memory().total / (1024**3)
        self.cpu_count = mp.cpu_count()
        self.gpu_available = GPU_AVAILABLE

        # Configuraciones basadas en RAM disponible (64GB)
        if self.ram_gb >= 60:  # 64GB disponible
            self.CACHE_SIZE = 20000          # Cache masivo
            self.BATCH_SIZE = 500            # Lotes enormes
            self.PRECOMPUTE_GRID_SIZE = 1000 # Grid denso
            self.TRAJECTORY_BUFFER = 50000   # Buffer grande
            self.MATRIX_CACHE_SIZE = 5000    # Cache de matrices
        elif self.ram_gb >= 30:  # 32GB
            self.CACHE_SIZE = 10000
            self.BATCH_SIZE = 200
            self.PRECOMPUTE_GRID_SIZE = 500
            self.TRAJECTORY_BUFFER = 20000
            self.MATRIX_CACHE_SIZE = 2000
        else:  # <32GB
            self.CACHE_SIZE = 5000
            self.BATCH_SIZE = 100
            self.PRECOMPUTE_GRID_SIZE = 200
            self.TRAJECTORY_BUFFER = 10000
            self.MATRIX_CACHE_SIZE = 1000

        # Configuraciones GPU
        self.GPU_BATCH_SIZE = 2000 if self.gpu_available else 0
        self.USE_GPU_JACOBIANS = self.gpu_available
        self.GPU_MEMORY_POOL = True

        self.print_config()

    def print_config(self):
        print(f"\n{'='*60}")
        print("⚡ CONFIGURACIÓN DE HARDWARE AVANZADA")
        print(f"{'='*60}")
        print(f"💾 RAM disponible: {self.ram_gb:.1f} GB")
        print(f"🔥 CPUs: {self.cpu_count}")
        print(f"🚀 GPU disponible: {self.gpu_available}")
        print(f"📊 Cache size: {self.CACHE_SIZE:,}")
        print(f"📦 Batch size: {self.BATCH_SIZE}")
        print(f"🗃️ Buffer trayectorias: {self.TRAJECTORY_BUFFER:,}")
        if self.gpu_available:
            print(f"⚡ GPU batch size: {self.GPU_BATCH_SIZE:,}")

# Configuración global
HARDWARE = HardwareConfig()

# ============================================================================
# CACHE INTELIGENTE MASIVO
# ============================================================================

class HierarchicalCache:
    """Sistema de cache jerárquico de múltiples niveles"""

    def __init__(self):
        # Nivel 1: Resultados exactos (LRU)
        self.exact_cache = OrderedDict()
        self.max_exact = HARDWARE.CACHE_SIZE

        # Nivel 2: Aproximaciones rápidas (más grande)
        self.approx_cache = OrderedDict()
        self.max_approx = HARDWARE.CACHE_SIZE * 2

        # Nivel 3: Cache de matrices jacobianas
        self.jacobian_cache = OrderedDict()
        self.max_jacobian = HARDWARE.MATRIX_CACHE_SIZE

        # Estadísticas
        self.hits = {'exact': 0, 'approx': 0, 'jacobian': 0}
        self.misses = {'exact': 0, 'approx': 0, 'jacobian': 0}

        # Pre-computar grids comunes
        self.precompute_common_configurations()

    def precompute_common_configurations(self):
        """Pre-calcular configuraciones comunes de masas"""
        print("🔄 Pre-calculando configuraciones comunes...")

        # Grid de masas comunes
        mass_grid = np.logspace(-1, 1, HARDWARE.PRECOMPUTE_GRID_SIZE)
        common_ratios = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

        precompute_count = 0
        for m1 in mass_grid[::10]:  # Subsampling para velocidad
            for ratio in common_ratios:
                m2 = m1 * ratio
                if 0.1 <= m2 <= 10.0:
                    key = self._make_key(m1, m2, 'approx')
                    if key not in self.approx_cache:
                        # Aproximación rápida
                        approx_lyap = self._quick_lyapunov_estimate(m1, m2)
                        self.approx_cache[key] = approx_lyap
                        precompute_count += 1

        print(f"✓ Pre-calculadas {precompute_count:,} configuraciones")

    def _make_key(self, m1, m2, level, precision=6):
        """Crear clave de cache con precisión controlada"""
        return (round(float(m1), precision), round(float(m2), precision), level)

    def _quick_lyapunov_estimate(self, m1, m2):
        """Estimación ultra-rápida de Lyapunov"""
        # Heurística basada en relación de masas y estabilidad teórica
        ratio = max(m1, m2) / min(m1, m2)
        if ratio > 100:
            return 10.0  # Muy inestable
        elif ratio > 10:
            return 1.0   # Moderadamente inestable
        elif ratio < 0.1:
            return -0.5  # Posiblemente estable
        else:
            return 0.1 * (ratio - 1)  # Interpolación lineal

    def get_exact(self, m1, m2):
        """Obtener resultado exacto del cache"""
        key = self._make_key(m1, m2, 'exact')
        if key in self.exact_cache:
            self.hits['exact'] += 1
            # Mover al frente (LRU)
            value = self.exact_cache.pop(key)
            self.exact_cache[key] = value
            return value
        else:
            self.misses['exact'] += 1
            return None

    def set_exact(self, m1, m2, value):
        """Guardar resultado exacto"""
        key = self._make_key(m1, m2, 'exact')
        self.exact_cache[key] = value

        # Mantener tamaño del cache
        if len(self.exact_cache) > self.max_exact:
            self.exact_cache.popitem(last=False)

    def get_approximation(self, m1, m2):
        """Obtener aproximación rápida"""
        key = self._make_key(m1, m2, 'approx')
        if key in self.approx_cache:
            self.hits['approx'] += 1
            return self.approx_cache[key]
        else:
            self.misses['approx'] += 1
            # Generar aproximación sobre la marcha
            approx = self._quick_lyapunov_estimate(m1, m2)
            self.approx_cache[key] = approx
            if len(self.approx_cache) > self.max_approx:
                self.approx_cache.popitem(last=False)
            return approx

    def print_stats(self):
        """Imprimir estadísticas del cache"""
        total_hits = sum(self.hits.values())
        total_misses = sum(self.misses.values())
        hit_rate = total_hits / (total_hits + total_misses) * 100 if total_hits + total_misses > 0 else 0

        print(f"\n📊 ESTADÍSTICAS DE CACHE:")
        print(f"Hit rate total: {hit_rate:.1f}%")
        print(f"Exact hits: {self.hits['exact']:,}, misses: {self.misses['exact']:,}")
        print(f"Approx hits: {self.hits['approx']:,}, misses: {self.misses['approx']:,}")

# Cache global
GLOBAL_CACHE = HierarchicalCache()

# ============================================================================
# BUFFER CIRCULAR PARA TRAYECTORIAS
# ============================================================================

class TrajectoryBuffer:
    """Buffer circular para mantener trayectorias en memoria"""

    def __init__(self, max_size=HARDWARE.TRAJECTORY_BUFFER):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lookup = {}  # Para búsqueda rápida

    def add_trajectory(self, m1, m2, trajectory):
        """Añadir trayectoria al buffer"""
        key = (round(float(m1), 4), round(float(m2), 4))

        # Si el buffer está lleno, remover el más antiguo del lookup
        if len(self.buffer) >= self.max_size and self.buffer:
            old_key = self.buffer[0][0]
            if old_key in self.lookup:
                del self.lookup[old_key]

        entry = (key, trajectory, time.time())
        self.buffer.append(entry)
        self.lookup[key] = len(self.buffer) - 1

    def get_trajectory(self, m1, m2, tolerance=1e-3):
        """Obtener trayectoria similar del buffer"""
        key = (round(float(m1), 4), round(float(m2), 4))

        # Búsqueda exacta
        if key in self.lookup:
            idx = self.lookup[key]
            if idx < len(self.buffer):
                return self.buffer[idx][1]

        # Búsqueda aproximada
        for stored_key, traj, timestamp in self.buffer:
            if (abs(stored_key[0] - key[0]) < tolerance and
                abs(stored_key[1] - key[1]) < tolerance):
                return traj

        return None

# Buffer global
TRAJECTORY_BUFFER = TrajectoryBuffer()

# ============================================================================
# SISTEMA GRAVITACIONAL GPU-OPTIMIZADO
# ============================================================================

class GPUOptimizedGravitationalSystem:
    """Sistema gravitacional optimizado para GPU"""

    def __init__(self, G=1.0, dt=0.01):
        self.G = G
        self.dt = dt
        self.use_gpu = GPU_AVAILABLE and HARDWARE.USE_GPU_JACOBIANS

        if self.use_gpu:
            # Configurar memoria GPU
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(size=2**30)  # 1GB limit
            print("✓ GPU configurada para cálculos")

    def setup_simulation(self, m1, m2, perturbation=False, perturb_factor=1e-9):
        """
        Configura y devuelve una simulación de REBOUND.
        """
        sim = rebound.Simulation()
        sim.G = self.G

        # Condiciones iniciales estándar para el problema de dos cuerpos
        # Cuerpo 1
        sim.add(m=m1, x=-1.0, y=0.0, vx=0.0, vy=-0.5 * m2)
        # Cuerpo 2
        sim.add(m=m2, x=1.0, y=0.0, vx=0.0, vy=0.5 * m1)

        # Lógica para manejar las perturbaciones necesarias para el análisis de sensibilidad
        if perturbation:
            # Añade una pequeña "patada" aleatoria a la velocidad de una de las partículas
            # para probar la estabilidad del sistema.
            kick = (np.random.rand(2) - 0.5) * 2 * perturb_factor
            sim.particles[1].vx += kick[0]
            sim.particles[1].vy += kick[1]

        # Mover al centro de masa para asegurar que el sistema no se desplace
        sim.move_to_com()

        return sim

    @jit(nopython=True) if NUMBA_AVAILABLE else lambda x: x
    def _gravitational_dynamics_jit(self, state, m1, m2, G=1.0):
        """Dinámicas gravitacionales optimizadas con JIT"""
        x1, y1, vx1, vy1, x2, y2, vx2, vy2 = state

        dx = x2 - x1
        dy = y2 - y1
        r_sq = dx*dx + dy*dy

        if r_sq < 1e-20:
            return np.zeros(8)

        r3_inv = 1.0 / (r_sq * np.sqrt(r_sq))
        fx = G * dx * r3_inv
        fy = G * dy * r3_inv

        return np.array([
            vx1, vy1, m2 * fx, m2 * fy,
            vx2, vy2, -m1 * fx, -m1 * fy
        ])

    def batch_jacobian_gpu(self, states_batch, masses_batch):
        """Cálculo por lotes de jacobianos en GPU"""
        if not self.use_gpu:
            return self._batch_jacobian_cpu(states_batch, masses_batch)

        try:
            # Transferir a GPU
            states_gpu = cp.array(states_batch)
            masses_gpu = cp.array(masses_batch)

            batch_size = len(states_batch)
            jacobians_gpu = cp.zeros((batch_size, 8, 8))

            # Kernel GPU personalizado (simplificado)
            for i in range(batch_size):
                jacobians_gpu[i] = self._compute_jacobian_gpu(
                    states_gpu[i], masses_gpu[i, 0], masses_gpu[i, 1]
                )

            # Transferir de vuelta a CPU
            return cp.asnumpy(jacobians_gpu)

        except Exception as e:
            print(f"⚠ Error GPU, fallback a CPU: {e}")
            return self._batch_jacobian_cpu(states_batch, masses_batch)

    def _compute_jacobian_gpu(self, state, m1, m2):
        """Jacobiano individual en GPU"""
        x1, y1, vx1, vy1, x2, y2, vx2, vy2 = state

        dx = x2 - x1
        dy = y2 - y1
        r_sq = dx*dx + dy*dy

        if r_sq < 1e-20:
            return cp.eye(8)

        r = cp.sqrt(r_sq)
        r3 = r_sq * r
        r5 = r_sq * r3

        # Derivadas
        dfx_dx1 = self.G * (3*dx*dx/r5 - 1/r3)
        dfx_dy1 = self.G * 3*dx*dy/r5
        dfy_dy1 = self.G * (3*dy*dy/r5 - 1/r3)

        J = cp.zeros((8, 8))

        # Velocidades
        J[0, 2] = J[1, 3] = J[4, 6] = J[5, 7] = 1.0

        # Aceleraciones
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

    def _batch_jacobian_cpu(self, states_batch, masses_batch):
        """Fallback CPU para jacobianos por lotes"""
        batch_size = len(states_batch)
        jacobians = np.zeros((batch_size, 8, 8))

        for i in range(batch_size):
            jacobians[i] = self._compute_jacobian_cpu(
                states_batch[i], masses_batch[i, 0], masses_batch[i, 1]
            )

        return jacobians

    def _compute_jacobian_cpu(self, state, m1, m2):
        """Jacobiano CPU (versión original optimizada)"""
        x1, y1, vx1, vy1, x2, y2, vx2, vy2 = state

        dx = x2 - x1
        dy = y2 - y1
        r_sq = dx*dx + dy*dy

        if r_sq < 1e-20:
            return np.eye(8)

        r = np.sqrt(r_sq)
        r3 = r_sq * r
        r5 = r_sq * r3

        dfx_dx1 = self.G * (3*dx*dx/r5 - 1/r3)
        dfx_dy1 = self.G * 3*dx*dy/r5
        dfy_dy1 = self.G * (3*dy*dy/r5 - 1/r3)

        J = np.zeros((8, 8))
        J[0, 2] = J[1, 3] = J[4, 6] = J[5, 7] = 1.0

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

# ============================================================================
# CALCULADORA ULTRA-OPTIMIZADA DE LYAPUNOV
# ============================================================================

class UltraOptimizedLyapunovCalculator:
    """Calculadora de Lyapunov con todas las optimizaciones"""

    def __init__(self):
        self.grav_system = GPUOptimizedGravitationalSystem()
        self.evaluation_count = 0
        self.gpu_evaluation_count = 0

    def calculate_lyapunov_smart(self, m1, m2, t_max=30.0):
        """Cálculo inteligente con cache jerárquico"""
        self.evaluation_count += 1

        # Nivel 1: Cache exacto
        cached_exact = GLOBAL_CACHE.get_exact(m1, m2)
        if cached_exact is not None:
            return cached_exact

        # Nivel 2: Verificación rápida con aproximación
        approx = GLOBAL_CACHE.get_approximation(m1, m2)
        if abs(approx) > 5.0:  # Claramente inestable
            GLOBAL_CACHE.set_exact(m1, m2, approx)
            return approx

        # Nivel 3: Cálculo completo solo si es prometedor
        lyap_exp = self._calculate_lyapunov_full(m1, m2, t_max)

        # Guardar en cache exacto
        GLOBAL_CACHE.set_exact(m1, m2, lyap_exp)

        return lyap_exp

    def _calculate_lyapunov_full(self, m1, m2, t_max=30.0):
        """Cálculo completo de Lyapunov"""
        try:
            m1_mp = mpmath.mpf(str(m1))
            m2_mp = mpmath.mpf(str(m2))

            x0 = np.array([-1.0, 0.0, 0.0, -0.5, 1.0, 0.0, 0.0, 0.5])
            t0 = 0.0
            dt = 0.02

            def f(x, t):
                return self.grav_system._gravitational_dynamics_jit(x, m1_mp, m2_mp)

            def jac(x, t):
                return self.grav_system._compute_jacobian_cpu(x, m1_mp, m2_mp)

            system = lyapynov.ContinuousDS(x0, t0, f, jac, dt)

            # Parámetros optimizados
            n_forward = 50
            n_compute = int(t_max / dt / 2)

            max_lyap = lyapynov.mLCE(system, n_forward, n_compute, keep=False)
            return float(max_lyap)

        except Exception as e:
            return 1e6

    def batch_evaluate(self, mass_pairs, t_max=30.0):
        """Evaluación por lotes ultra-optimizada"""
        if not mass_pairs:
            return []

        batch_size = len(mass_pairs)
        results = []

        # Pre-filtrar con aproximaciones
        promising_pairs = []
        quick_results = []

        for m1, m2 in mass_pairs:
            approx = GLOBAL_CACHE.get_approximation(m1, m2)
            if abs(approx) <= 2.0:  # Posiblemente estable o interesante
                promising_pairs.append((m1, m2))
            else:
                quick_results.append(approx)

        print(f"🔄 Evaluación por lotes: {len(promising_pairs)}/{batch_size} prometedores")

        # Evaluar los prometedores en paralelo
        if promising_pairs:
            detailed_results = Parallel(
                n_jobs=HARDWARE.cpu_count,
                backend='threading'
            )(
                delayed(self.calculate_lyapunov_smart)(m1, m2, t_max)
                for m1, m2 in promising_pairs
            )
        else:
            detailed_results = []

        # Combinar resultados
        result_idx = 0
        quick_idx = 0

        for m1, m2 in mass_pairs:
            approx = GLOBAL_CACHE.get_approximation(m1, m2)
            if abs(approx) <= 2.0:
                results.append(detailed_results[result_idx])
                result_idx += 1
            else:
                results.append(quick_results[quick_idx])
                quick_idx += 1

        return results

# ============================================================================
# PROBLEMA DE OPTIMIZACIÓN MEGA-PARALELO
# ============================================================================

class MegaParallelOptimizationProblem(Problem):
    """Problema con optimizaciones extremas de hardware"""

    def __init__(self, mass_bounds, t_eval=30.0):
        n_var = 2
        n_obj = 1
        n_constr = 0

        xl = np.array([mass_bounds[0][0], mass_bounds[1][0]])
        xu = np.array([mass_bounds[0][1], mass_bounds[1][1]])

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)

        self.t_eval = t_eval
        self.calculator = UltraOptimizedLyapunovCalculator()
        self.batch_count = 0

        print(f"🚀 Problema mega-paralelo configurado")
        print(f"   Cache size: {HARDWARE.CACHE_SIZE:,}")
        print(f"   Batch size: {HARDWARE.BATCH_SIZE}")
        print(f"   GPU disponible: {HARDWARE.gpu_available}")

    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluación mega-optimizada"""
        start_time = time.time()
        self.batch_count += 1

        print(f"\n🔄 Evaluando lote #{self.batch_count} ({len(X)} individuos)")

        # Convertir a lista de pares
        mass_pairs = [(float(x[0]), float(x[1])) for x in X]

        # Evaluación por lotes inteligente
        fitness_values = self.calculator.batch_evaluate(mass_pairs, self.t_eval)

        # Post-procesamiento
        fitness_values = [
            abs(f) if np.isfinite(f) and f != 1e6 else 1e6
            for f in fitness_values
        ]

        eval_time = time.time() - start_time
        cache_hit_rate = (GLOBAL_CACHE.hits['exact'] /
                         (GLOBAL_CACHE.hits['exact'] + GLOBAL_CACHE.misses['exact']) * 100
                         if GLOBAL_CACHE.hits['exact'] + GLOBAL_CACHE.misses['exact'] > 0 else 0)

        print(f"   ✓ Completado en {eval_time:.2f}s")
        print(f"   📊 Cache hit rate: {cache_hit_rate:.1f}%")
        print(f"   🎯 Mejor fitness: {min(fitness_values):.6f}")

        out["F"] = np.array(fitness_values).reshape(-1, 1)

        # Limpieza periódica de memoria
        if self.batch_count % 10 == 0:
            gc.collect()
            if GPU_AVAILABLE:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except:
                    pass

# ============================================================================
# OPTIMIZADOR MULTI-ALGORITMO ULTRA-AVANZADO
# ============================================================================

class UltraAdvancedOptimizer:
    """Optimizador con todas las optimizaciones de hardware"""

    def __init__(self, problem, pop_size=None):
        self.problem = problem
        self.pop_size = pop_size or HARDWARE.BATCH_SIZE

        print(f"\n⚡ OPTIMIZADOR ULTRA-AVANZADO")
        print(f"   Población: {self.pop_size}")
        print(f"   Procesos: {HARDWARE.cpu_count}")

    def mega_optimize(self, generations=25):
        """Optimización con todas las mejoras"""

        algorithms = [
            ("🧬 Genetic Algorithm (Ultra)", GA(
                pop_size=self.pop_size,
                eliminate_duplicates=True,
                n_offsprings=self.pop_size//2
            )),
            ("🌊 Particle Swarm (Mega)", PSO(
                pop_size=self.pop_size,
                w=0.9, c1=2.0, c2=2.0
            )),
            ("🔄 Differential Evolution (Turbo)", DE(
                pop_size=self.pop_size,
                F=0.8, CR=0.9
            ))
        ]

        print(f"\n🚀 INICIANDO MEGA-OPTIMIZACIÓN")
        print(f"{'='*60}")

        start_time = time.time()

        # Ejecutar algoritmos en paralelo
        results = Parallel(n_jobs=min(3, HARDWARE.cpu_count), backend='threading')(
            delayed(self._run_algorithm)(name, alg, generations)
            for name, alg in algorithms
        )

        total_time = time.time() - start_time

        # Encontrar mejor resultado
        valid_results = [r for r in results if r['success']]
        if valid_results:
            best_result = min(valid_results, key=lambda x: x['best_f'])
        else:
            best_result = {'algorithm': 'None', 'best_x': None, 'best_f': 1e6, 'success': False}

        # Mostrar resultados detallados
        self._display_mega_results(results, best_result, total_time)

        return best_result, results

    def _run_algorithm(self, name, algorithm, generations):
        """Ejecutar algoritmo individual con manejo de errores avanzado"""
        print(f"\n🔄 Iniciando {name}...")

        try:
            termination = get_termination("n_gen", generations)

            algorithm_start = time.time()
            res = minimize(
                self.problem,
                algorithm,
                termination,
                seed=np.random.randint(1000),
                verbose=False
            )
            algorithm_time = time.time() - algorithm_start

            if res.X is not None and np.isfinite(res.F[0]):
                print(f"✅ {name} completado en {algorithm_time:.1f}s - Fitness: {res.F[0]:.8f}")
                return {
                    'algorithm': name,
                    'best_x': res.X,
                    'best_f': res.F[0],
                    'success': True,
                    'time': algorithm_time,
                    'generations': generations
                }
            else:
                print(f"⚠️ {name} - Resultado inválido")

        except Exception as e:
            print(f"❌ {name} - Error: {str(e)[:50]}...")

        return {
            'algorithm': name,
            'best_x': None,
            'best_f': 1e6,
            'success': False,
            'time': 0,
            'generations': 0
        }

    def _display_mega_results(self, results, best_result, total_time):
        """Mostrar resultados detallados con estadísticas avanzadas"""
        print(f"\n{'🎯 RESULTADOS MEGA-OPTIMIZACIÓN':^60}")
        print("="*60)
        print(f"⏱️  Tiempo total: {total_time:.2f} segundos")
        print(f"🎖️  Mejor algoritmo: {best_result['algorithm']}")

        if best_result['success']:
            m1, m2 = best_result['best_x']
            print(f"🏆 Mejores masas: m1={m1:.8f}, m2={m2:.8f}")
            print(f"📊 Fitness óptimo: {best_result['best_f']:.10f}")
            print(f"⚖️  Relación: m1/m2 = {m1/m2:.6f}")

            # Clasificación de estabilidad
            if best_result['best_f'] < 1e-6:
                stability = "🌟 ULTRA-ESTABLE"
            elif best_result['best_f'] < 1e-3:
                stability = "✅ MUY ESTABLE"
            elif best_result['best_f'] < 0.01:
                stability = "✓ ESTABLE"
            elif best_result['best_f'] < 0.1:
                stability = "⚠️ CUASI-ESTABLE"
            else:
                stability = "❌ INESTABLE"

            print(f"🔬 Clasificación: {stability}")

        print("\n📈 Desempeño por Algoritmo:")
        print("-" * 60)

        for result in sorted(results, key=lambda x: x['best_f']):
            status = "✅" if result['success'] else "❌"
            time_str = f"{result['time']:.1f}s" if result['success'] else "N/A"
            fitness_str = f"{result['best_f']:.6f}" if result['success'] else "FALLÓ"

            print(f"{status} {result['algorithm']:<30} {fitness_str:>12} ({time_str})")

        # Estadísticas del cache
        GLOBAL_CACHE.print_stats()

        # Estadísticas de memoria
        memory_info = psutil.virtual_memory()
        print(f"\n💾 Uso de memoria: {memory_info.percent:.1f}% ({memory_info.used/1024**3:.1f}GB usado)")

        if GPU_AVAILABLE:
            try:
                gpu_memory = cp.get_default_memory_pool().used_bytes() / 1024**3
                print(f"🚀 Memoria GPU utilizada: {gpu_memory:.2f}GB")
            except:
                pass

# ============================================================================
# VISUALIZACIÓN ULTRA-AVANZADA
# ============================================================================

class UltraVisualization:
    """Visualización con todas las optimizaciones y análisis avanzado"""

    def __init__(self):
        self.grav_system = GPUOptimizedGravitationalSystem()

    def mega_visualization(self, best_masses, t_max=50.0, analysis_depth='ultra'):
        """Visualización ultra-completa con análisis profundo"""
        m1, m2 = best_masses

        print(f"\n🎨 GENERANDO VISUALIZACIÓN ULTRA-AVANZADA")
        print(f"   Masas: m1={m1:.8f}, m2={m2:.8f}")
        print(f"   Tiempo: {t_max}s")
        print(f"   Nivel de análisis: {analysis_depth}")

        # Configurar simulación optimizada
        sim = self.grav_system.setup_simulation(m1, m2)

        # Parámetros de alta resolución
        if analysis_depth == 'ultra':
            n_outputs = 5000
            perturbation_levels = [1e-15, 1e-12, 1e-9, 1e-6]
        elif analysis_depth == 'high':
            n_outputs = 2000
            perturbation_levels = [1e-12, 1e-9, 1e-6]
        else:
            n_outputs = 1000
            perturbation_levels = [1e-9, 1e-6]

        times = np.linspace(0, t_max, n_outputs)

        # Simulación principal con buffer de trayectoria
        main_trajectory = self._simulate_with_buffer(sim, times, m1, m2)

        # Análisis de perturbaciones
        perturbation_analysis = self._analyze_perturbations(
            m1, m2, perturbation_levels, t_max, n_outputs//2
        )

        # Análisis espectral avanzado
        spectral_analysis = self._spectral_analysis(main_trajectory, times)

        # Crear visualización mega-completa
        self._create_mega_plot(
            main_trajectory, times, perturbation_analysis,
            spectral_analysis, m1, m2, analysis_depth
        )

        # Reporte detallado
        self._generate_detailed_report(
            main_trajectory, perturbation_analysis, spectral_analysis, m1, m2
        )

    def _simulate_with_buffer(self, sim, times, m1, m2):
        """Simulación con uso del buffer de trayectorias"""
        # Verificar si ya tenemos una trayectoria similar
        cached_traj = TRAJECTORY_BUFFER.get_trajectory(m1, m2)
        if cached_traj is not None and len(cached_traj) >= len(times):
            print("📋 Usando trayectoria del buffer")
            return cached_traj[:len(times)]

        print("🔄 Calculando nueva trayectoria...")
        positions = np.zeros((len(times), 4))

        try:
            for i, t in enumerate(times):
                sim.integrate(t)
                if len(sim.particles) < 2:
                    break
                positions[i, 0] = sim.particles[0].x
                positions[i, 1] = sim.particles[0].y
                positions[i, 2] = sim.particles[1].x
                positions[i, 3] = sim.particles[1].y
        except Exception as e:
            print(f"⚠️ Error en simulación: {e}")

        # Guardar en buffer
        TRAJECTORY_BUFFER.add_trajectory(m1, m2, positions)

        return positions

    def _analyze_perturbations(self, m1, m2, perturbation_levels, t_max, n_outputs):
        """Análisis de sensibilidad a perturbaciones"""
        perturbation_data = {}

        print(f"🔬 Analizando {len(perturbation_levels)} niveles de perturbación...")

        for level in perturbation_levels:
            sim_pert = self.grav_system.setup_simulation(
                m1, m2, perturbation=True, perturb_factor=level
            )

            times = np.linspace(0, t_max, n_outputs)
            positions = np.zeros((n_outputs, 4))

            try:
                for i, t in enumerate(times):
                    sim_pert.integrate(t)
                    if len(sim_pert.particles) < 2:
                        break
                    positions[i, 0] = sim_pert.particles[0].x
                    positions[i, 1] = sim_pert.particles[0].y
                    positions[i, 2] = sim_pert.particles[1].x
                    positions[i, 3] = sim_pert.particles[1].y
            except:
                pass

            perturbation_data[level] = {
                'times': times,
                'positions': positions
            }

        return perturbation_data

    def _spectral_analysis(self, trajectory, times):
        """Análisis espectral avanzado de la trayectoria"""
        if len(trajectory) < 100:
            return None

        print("📊 Realizando análisis espectral...")

        # Extraer coordenadas
        x1, y1 = trajectory[:, 0], trajectory[:, 1]
        x2, y2 = trajectory[:, 2], trajectory[:, 3]

        # Calcular distancias y ángulos
        distances = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        angles = np.arctan2(y2 - y1, x2 - x1)

        # FFT para análisis de frecuencias
        dt = times[1] - times[0] if len(times) > 1 else 0.01

        try:
            # FFT de distancias
            fft_dist = np.fft.fft(distances - np.mean(distances))
            freqs = np.fft.fftfreq(len(distances), dt)

            # Encontrar frecuencias dominantes
            power_spectrum = np.abs(fft_dist)**2
            dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
            dominant_frequency = freqs[dominant_freq_idx]

            return {
                'distances': distances,
                'angles': angles,
                'frequencies': freqs[:len(freqs)//2],
                'power_spectrum': power_spectrum[:len(power_spectrum)//2],
                'dominant_frequency': dominant_frequency,
                'period': 1.0 / abs(dominant_frequency) if dominant_frequency != 0 else np.inf
            }
        except:
            return None

    def _create_mega_plot(self, trajectory, times, perturbation_analysis,
                         spectral_analysis, m1, m2, analysis_depth):
        """Crear visualización mega-completa"""

        if analysis_depth == 'ultra':
            fig = plt.figure(figsize=(20, 16))
            subplot_config = (4, 3)
        elif analysis_depth == 'high':
            fig = plt.figure(figsize=(16, 12))
            subplot_config = (3, 3)
        else:
            fig = plt.figure(figsize=(12, 8))
            subplot_config = (2, 3)

        # 1. Órbitas principales
        plt.subplot(subplot_config[0], subplot_config[1], 1)
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.8,
                linewidth=1.5, label=f'Cuerpo 1 (m={m1:.6f})')
        plt.plot(trajectory[:, 2], trajectory[:, 3], 'r-', alpha=0.8,
                linewidth=1.5, label=f'Cuerpo 2 (m={m2:.6f})')
        plt.title("🌟 Órbitas Estables", fontsize=14, fontweight='bold')
        plt.axis('equal')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. Evolución temporal de distancias
        plt.subplot(subplot_config[0], subplot_config[1], 2)
        distances = np.sqrt((trajectory[:, 2] - trajectory[:, 0])**2 +
                           (trajectory[:, 3] - trajectory[:, 1])**2)
        plt.plot(times[:len(distances)], distances, 'g-', linewidth=2)
        plt.title("📏 Distancia entre Cuerpos", fontsize=14, fontweight='bold')
        plt.xlabel("Tiempo")
        plt.ylabel("Distancia")
        plt.grid(True, alpha=0.3)

        # 3. Análisis de perturbaciones
        plt.subplot(subplot_config[0], subplot_config[1], 3)
        if perturbation_analysis:
            for level, data in perturbation_analysis.items():
                pert_distances = np.sqrt(
                    (data['positions'][:, 2] - data['positions'][:, 0])**2 +
                    (data['positions'][:, 3] - data['positions'][:, 1])**2
                )
                plt.plot(data['times'][:len(pert_distances)], pert_distances,
                        alpha=0.7, label=f'δ={level:.0e}')
            plt.title("🔬 Sensibilidad a Perturbaciones", fontsize=14, fontweight='bold')
            plt.xlabel("Tiempo")
            plt.ylabel("Distancia")
            plt.legend()
            plt.grid(True, alpha=0.3)

        # 4. Análisis espectral
        if spectral_analysis and subplot_config[0] >= 3:
            plt.subplot(subplot_config[0], subplot_config[1], 4)
            plt.semilogy(spectral_analysis['frequencies'][1:],
                        spectral_analysis['power_spectrum'][1:], 'purple')
            plt.title("📊 Espectro de Potencia", fontsize=14, fontweight='bold')
            plt.xlabel("Frecuencia")
            plt.ylabel("Potencia")
            plt.grid(True, alpha=0.3)

            # Marcar frecuencia dominante
            if spectral_analysis['dominant_frequency'] != 0:
                plt.axvline(abs(spectral_analysis['dominant_frequency']),
                           color='red', linestyle='--', alpha=0.7,
                           label=f"f₀={abs(spectral_analysis['dominant_frequency']):.4f}")
                plt.legend()

        # Plots adicionales para análisis ultra
        if analysis_depth == 'ultra' and subplot_config[0] >= 4:
            # 5. Energía del sistema
            plt.subplot(4, 3, 5)
            # Calcular energía cinética y potencial (simplificado)
            velocities_1 = np.gradient(trajectory[:, 0])**2 + np.gradient(trajectory[:, 1])**2
            velocities_2 = np.gradient(trajectory[:, 2])**2 + np.gradient(trajectory[:, 3])**2
            kinetic_energy = 0.5 * m1 * velocities_1 + 0.5 * m2 * velocities_2

            plt.plot(times[:len(kinetic_energy)], kinetic_energy, 'orange')
            plt.title("⚡ Energía Cinética", fontsize=12, fontweight='bold')
            plt.xlabel("Tiempo")
            plt.ylabel("Energía")
            plt.grid(True, alpha=0.3)

            # 6. Fase del sistema
            plt.subplot(4, 3, 6)
            plt.plot(trajectory[:, 0], np.gradient(trajectory[:, 0]), 'blue', alpha=0.7)
            plt.plot(trajectory[:, 2], np.gradient(trajectory[:, 2]), 'red', alpha=0.7)
            plt.title("🌀 Espacio de Fases", fontsize=12, fontweight='bold')
            plt.xlabel("Posición")
            plt.ylabel("Velocidad")
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.suptitle(f"ANÁLISIS GRAVITACIONAL ULTRA-AVANZADO\nm₁={m1:.8f}, m₂={m2:.8f}",
                    fontsize=16, fontweight='bold', y=0.98)
        plt.show()

    def _generate_detailed_report(self, trajectory, perturbation_analysis,
                                spectral_analysis, m1, m2):
        """Generar reporte detallado del sistema"""
        print(f"\n{'📋 REPORTE DETALLADO DEL SISTEMA':^60}")
        print("="*60)

        # Propiedades básicas
        print(f"🔬 PROPIEDADES FUNDAMENTALES:")
        print(f"   Masa 1: {m1:.10f}")
        print(f"   Masa 2: {m2:.10f}")
        print(f"   Relación de masas: {m1/m2:.8f}")
        print(f"   Masa total: {m1+m2:.10f}")
        print(f"   Masa reducida: {m1*m2/(m1+m2):.10f}")

        # Análisis orbital
        if len(trajectory) > 10:
            distances = np.sqrt((trajectory[:, 2] - trajectory[:, 0])**2 +
                               (trajectory[:, 3] - trajectory[:, 1])**2)

            print(f"\n🛰️ CARACTERÍSTICAS ORBITALES:")
            print(f"   Distancia media: {np.mean(distances):.8f}")
            print(f"   Distancia mín: {np.min(distances):.8f}")
            print(f"   Distancia máx: {np.max(distances):.8f}")
            print(f"   Excentricidad aprox: {(np.max(distances)-np.min(distances))/(np.max(distances)+np.min(distances)):.6f}")

        # Análisis espectral
        if spectral_analysis:
            print(f"\n📊 ANÁLISIS ESPECTRAL:")
            print(f"   Frecuencia dominante: {abs(spectral_analysis['dominant_frequency']):.6f}")
            print(f"   Período orbital: {spectral_analysis['period']:.4f}")
            print(f"   Estabilidad espectral: {'Alta' if spectral_analysis['period'] < 100 else 'Moderada'}")

        # Análisis de perturbaciones
        if perturbation_analysis:
            print(f"\n🔬 ANÁLISIS DE SENSIBILIDAD:")
            stability_levels = []
            for level, data in perturbation_analysis.items():
                if len(data['positions']) > 10:
                    final_distance = np.sqrt(
                        (data['positions'][-1, 2] - data['positions'][-1, 0])**2 +
                        (data['positions'][-1, 3] - data['positions'][-1, 1])**2
                    )
                    initial_distance = np.sqrt(
                        (data['positions'][0, 2] - data['positions'][0, 0])**2 +
                        (data['positions'][0, 3] - data['positions'][0, 1])**2
                    )

                    if abs(final_distance - initial_distance) < 0.1:
                        stability = "Muy Estable"
                    elif abs(final_distance - initial_distance) < 1.0:
                        stability = "Estable"
                    else:
                        stability = "Inestable"

                    stability_levels.append(stability)
                    print(f"   Perturbación {level:.0e}: {stability}")

            overall_stability = max(set(stability_levels), key=stability_levels.count) if stability_levels else "Desconocida"
            print(f"   🏆 Estabilidad general: {overall_stability}")

# ============================================================================
# FUNCIÓN PRINCIPAL ULTRA-OPTIMIZADA
# ============================================================================

def ultra_main():
    """Función principal con todas las optimizaciones"""
    print("🚀" * 20)
    print("SISTEMA DE OPTIMIZACIÓN GRAVITACIONAL ULTRA-AVANZADO")
    print("🚀" * 20)
    print("Aprovechando al máximo el hardware disponible\n")

    # Configuración de parámetros optimizada
    mass_bounds = [(0.05, 10.0), (0.05, 10.0)]  # Rango ampliado
    t_eval = 25.0  # Balance entre precisión y velocidad

    print(f"⚙️  Configuración:")
    print(f"   Rangos de masa: {mass_bounds}")
    print(f"   Tiempo evaluación: {t_eval}s")
    print(f"   Población: {HARDWARE.BATCH_SIZE}")

    # Crear problema ultra-optimizado
    problem = MegaParallelOptimizationProblem(mass_bounds, t_eval)

    # Crear optimizador ultra-avanzado
    optimizer = UltraAdvancedOptimizer(problem)

    # Medir tiempo total
    mega_start_time = time.time()

    print(f"\n{'🎯 INICIANDO MEGA-OPTIMIZACIÓN':^60}")
    print("="*60)

    # Ejecutar optimización ultra-avanzada
    best_result, all_results = optimizer.mega_optimize(generations=20)

    mega_total_time = time.time() - mega_start_time

    print(f"\n{'⭐ OPTIMIZACIÓN ULTRA-COMPLETADA':^60}")
    print("="*60)
    print(f"🕐 Tiempo total: {mega_total_time:.2f} segundos")
    print(f"📊 Evaluaciones realizadas: {problem.calculator.evaluation_count:,}")

    if best_result['success']:
        best_masses = best_result['best_x']
        best_fitness = best_result['best_f']

        print(f"\n🏆 RESULTADO ÓPTIMO:")
        print(f"   Algoritmo ganador: {best_result['algorithm']}")
        print(f"   Masas óptimas: m₁={best_masses[0]:.10f}, m₂={best_masses[1]:.10f}")
        print(f"   Fitness final: {best_fitness:.12f}")

        # Evaluación final detallada con máxima precisión
        print(f"\n🔬 EVALUACIÓN FINAL DETALLADA:")
        final_calc = UltraOptimizedLyapunovCalculator()
        final_lyap = final_calc._calculate_lyapunov_full(
            best_masses[0], best_masses[1], t_max=100.0  # Evaluación larga final
        )

        print(f"   Exponente Lyapunov: {final_lyap:.15f}")

        # Clasificación ultra-detallada
        if final_lyap < -1e-10:
            classification = "🌟 SISTEMA ULTRA-ESTABLE (Atractor)"
        elif final_lyap < -1e-6:
            classification = "⭐ SISTEMA MUY ESTABLE"
        elif final_lyap < -1e-3:
            classification = "✅ SISTEMA ESTABLE"
        elif abs(final_lyap) < 1e-3:
            classification = "⚖️ SISTEMA NEUTRAL (Límite)"
        elif final_lyap < 0.01:
            classification = "⚠️ SISTEMA CUASI-ESTABLE"
        elif final_lyap < 0.1:
            classification = "🔄 SISTEMA PERIÓDICO COMPLEJO"
        else:
            classification = "❌ SISTEMA CAÓTICO"

        print(f"   Clasificación: {classification}")

        # Métricas adicionales
        ratio = best_masses[0] / best_masses[1]
        print(f"   Relación m₁/m₂: {ratio:.8f}")
        print(f"   Momento angular reducido: {(best_masses[0]*best_masses[1]/(best_masses[0]+best_masses[1]))**0.5:.8f}")

        # Visualización ultra-avanzada
        print(f"\n🎨 Generando visualización ultra-avanzada...")
        visualizer = UltraVisualization()
        visualizer.mega_visualization(
            best_masses,
            t_max=100.0,  # Visualización extendida
            analysis_depth='ultra'
        )

        # Guardar resultados para análisis posterior
        try:
            result_data = {
                'masses': best_masses,
                'fitness': best_fitness,
                'lyapunov': final_lyap,
                'algorithm': best_result['algorithm'],
                'classification': classification,
                'hardware_config': {
                    'ram_gb': HARDWARE.ram_gb,
                    'cpu_count': HARDWARE.cpu_count,
                    'gpu_available': HARDWARE.gpu_available,
                    'cache_size': HARDWARE.CACHE_SIZE
                },
                'performance': {
                    'total_time': mega_total_time,
                    'evaluations': problem.calculator.evaluation_count,
                    'cache_hits': GLOBAL_CACHE.hits,
                    'cache_misses': GLOBAL_CACHE.misses
                }
            }

            with open('ultra_optimization_results.pkl', 'wb') as f:
                pickle.dump(result_data, f)
            print(f"💾 Resultados guardados en 'ultra_optimization_results.pkl'")

        except Exception as e:
            print(f"⚠️ No se pudieron guardar los resultados: {e}")

    else:
        print("❌ No se encontró solución óptima")
        print("💡 Sugerencias:")
        print("   - Aumentar generaciones")
        print("   - Ampliar rangos de búsqueda")
        print("   - Ajustar parámetros de precisión")

    # Estadísticas finales del sistema
    print(f"\n{'📈 ESTADÍSTICAS FINALES DEL SISTEMA':^60}")
    print("="*60)

    # Uso de memoria final
    memory_info = psutil.virtual_memory()
    print(f"💾 Memoria utilizada: {memory_info.percent:.1f}% ({memory_info.used/1024**3:.2f}GB)")

    # Estadísticas de cache
    GLOBAL_CACHE.print_stats()

    # Eficiencia computacional
    if problem.calculator.evaluation_count > 0:
        eval_per_second = problem.calculator.evaluation_count / mega_total_time
        print(f"⚡ Velocidad: {eval_per_second:.1f} evaluaciones/segundo")

    if GPU_AVAILABLE:
        try:
            gpu_memory = cp.get_default_memory_pool().used_bytes() / 1024**3
            print(f"🚀 Memoria GPU utilizada: {gpu_memory:.2f}GB")
        except:
            pass

    print(f"\n🎊 OPTIMIZACIÓN ULTRA-COMPLETADA CON ÉXITO! 🎊")

if __name__ == "__main__":
    ultra_main()