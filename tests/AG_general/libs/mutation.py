import numpy as np

def polynomial_mutation(child, lower_bound, upper_bound, mutation_prob, eta_mut):
    """Aplica mutación polinomial a un hijo."""
    mutant = child.copy()
    for i in range(len(child)):
        if np.random.rand() < mutation_prob:
            r = np.random.rand()
            diff = upper_bound[i] - lower_bound[i]
            if r < 0.5:
                delta = (2*r)**(1/(eta_mut+1)) - 1
            else:
                delta = 1 - (2*(1-r))**(1/(eta_mut+1))
            mutant[i] = child[i] + delta * diff
            mutant[i] = np.clip(mutant[i], lower_bound[i], upper_bound[i])
    return mutant


def polynomial_mutation_with_boundaries(child, lower_bound, upper_bound,
                                        mutation_prob, eta_mut,
                                        use_global_r=False, global_r=None):
    """
    Aplica mutación polinomial (con límites) a un vector 'child'.
    Puede usar un único 'r' global para todas las variables (si use_global_r=True)
    o generar un 'r' distinto para cada variable.
    
    Args:
      - child : array-like
          Cromosoma (vector de decisión) a mutar.
      - lower_bound, upper_bound : array-like
          Límites inferiores y superiores para cada variable.
      - mutation_prob : float
          Probabilidad de mutación (en [0,1]) para cada variable.
      - eta_mut : float
          Índice de distribución para la mutación.
      - use_global_r : bool
          Si True, se utiliza un único valor 'r' para todas las variables.
      - global_r : float, opcional
          Valor de 'r' global a usar; si no se proporciona, se genera uno.
    
    Retutrns:
      - mutant : np.ndarray
          Nuevo vector mutado (manteniendo la dimensión de 'child').
    """
    mutant = np.array(child, copy=True, dtype=float)
    num_vars = len(child)
    
    # Si se desea usar un 'r' global y no se ha proporcionado, se genera uno una sola vez.
    if use_global_r:
        if global_r is None:
            global_r = np.random.rand()
    
    for i in range(num_vars):
        # Decidir si mutar esta variable
        if np.random.rand() < mutation_prob:
            x = mutant[i]
            xl = lower_bound[i]
            xu = upper_bound[i]
            
            # Evitar división por cero si los límites son casi iguales
            if abs(xu - xl) < 1e-14:
                continue

            # d = distancia normalizada al límite más cercano
            d = min(xu - x, x - xl) / (xu - xl)
            
            # Elegir r: global o individual para cada variable
            if use_global_r:
                r = global_r
            else:
                r = np.random.rand()

            nm = eta_mut + 1.0

            # Calcular delta_q según el valor de r
            if r < 0.5:
                bl = 2.0 * r + (1.0 - 2.0 * r) * ((1.0 - d) ** nm)
                delta_q = (bl ** (1.0 / nm)) - 1.0
            else:
                bl = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * ((1.0 - d) ** nm)
                delta_q = 1.0 - (bl ** (1.0 / nm))
            
            # Calcular la nueva posición y asegurarse que esté dentro de los límites
            y = x + delta_q * (xu - xl)
            mutant[i] = np.clip(y, xl, xu)
    
    return mutant