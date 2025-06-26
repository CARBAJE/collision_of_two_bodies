import numpy as np
from libs.functions import f_riesgos, g2_finanzas, h1_finanzas
# ---------------------------
# Parámetros del algoritmo
# ---------------------------
POP_SIZE = 50           # Número de individuos en la población
NUM_GENERATIONS = 20     # Número de generaciones
NUM_RUNS = 1             # Número de ejecuciones completas (ciclos)

# Parámetros de la función de Langermann
a = np.array([3, 5, 2, 1, 7])
b = np.array([5, 2, 1, 4, 9])
c = np.array([1, 2, 5, 2, 3])

# Parámetros del torneo
TOURNAMENT_SIZE = 7  # Número de individuos participantes en cada torneo

# Parámetros del cruzamiento SBX
CROSSOVER_PROB = 0.9  # Probabilidad de aplicar cruzamiento
ETA_C = 0.1            # Índice de distribución para SBX

# Limites superiores e inferiores
LB = np.array([0.5, 0.5])
UB = np.array([10, 10])

# Parámetros de la mutación polinomial
MUTATION_PROB = 1 / 20  # Probabilidad de mutar cada gen
ETA_MUT = 0.1                         # Índice de distribución para mutación polinomial

# Ruta a la imagen a contrastar
# IMG_PATH = get_image_paths('imgs')

best_solutions_list = [] 
all_runs_history = []  # Para graficar luego

LAMBDA = 1*10e1

FUNCTIONS = {
    "Problem 3": {
        "func": f_riesgos,
        "g": None,
        "h": None,
        "lb": LB,
        "ub": UB,
        "name": "Problema 1",
        "num_runs": NUM_RUNS
    }
}
