import numpy as np

def sbx_crossover(parent1, parent2, lower_bound, upper_bound, eta, crossover_prob):
    """Realiza el cruzamiento SBX para dos padres y devuelve dos hijos."""
    child1 = np.empty_like(parent1)
    child2 = np.empty_like(parent2)
    
    if np.random.rand() <= crossover_prob:
        for i in range(len(parent1)):
            u = np.random.rand()
            if u <= 0.5:
                beta = (2*u)**(1/(eta+1))
            else:
                beta = (1/(2*(1-u)))**(1/(eta+1))
            
            # Genera los dos hijos
            child1[i] = 0.5*((1+beta)*parent1[i] + (1-beta)*parent2[i])
            child2[i] = 0.5*((1-beta)*parent1[i] + (1+beta)*parent2[i])
            
            # Asegurar que los hijos estén dentro de los límites
            child1[i] = np.clip(child1[i], lower_bound[i], upper_bound[i])
            child2[i] = np.clip(child2[i], lower_bound[i], upper_bound[i])
    else:
        child1 = parent1.copy()
        child2 = parent2.copy()
    
    return child1, child2

def sbx_crossover_with_boundaries(parent1, parent2, lower_bound, upper_bound,
                                  eta, crossover_prob, use_global_u=False, global_u=None):
    """
    Realiza el cruzamiento SBX con límites, usando fórmulas que ajustan beta en función
    de la cercanía a las fronteras. Permite usar un único 'u' global para todos los individuos 
    de la generación o, de forma estándar, un 'u' distinto por cada gen.
    
    Args:
      - parent1, parent2: arrays con los padres.
      - lower_bound, upper_bound: arrays con los límites inferiores y superiores.
      - eta: índice de distribución para SBX.
      - crossover_prob: probabilidad de aplicar el cruce.
      - use_global_u: si es True se utilizará el mismo valor de 'u' para todas las variables.
      - global_u: valor de 'u' que se aplicará globalmente (si se proporciona).
      
    Returns:
      - child1, child2: arrays con los hijos resultantes.
    """
    parent1 = np.asarray(parent1)
    parent2 = np.asarray(parent2)
    child1 = np.empty_like(parent1)
    child2 = np.empty_like(parent2)
    
    # Si no se realiza el crossover, retornamos copias de los padres.
    if np.random.rand() > crossover_prob:
        return parent1.copy(), parent2.copy()
    
    # Si se quiere usar un 'u' global y no se ha pasado, se genera uno.
    if use_global_u:
        if global_u is None:
            global_u = np.random.rand()
    
    for i in range(len(parent1)):
        x1 = parent1[i]
        x2 = parent2[i]
        lb = lower_bound[i]
        ub = upper_bound[i]
        
        # Aseguramos que x1 sea menor o igual que x2
        if x1 > x2:
            x1, x2 = x2, x1
        
        dist = x2 - x1
        if dist < 1e-14:
            child1[i] = x1
            child2[i] = x2
            continue
        
        # Calcular la mínima distancia a las fronteras
        min_val = min(x1 - lb, ub - x2)
        if min_val < 0:
            min_val = 0
        
        beta = 1.0 + (2.0 * min_val / dist)
        alpha = 2.0 - beta**(-(eta+1))
        
        # Si se usa u global, se usa el mismo valor para cada variable
        if use_global_u:
            u = global_u
        else:
            u = np.random.rand()
        
        if u <= (1.0 / alpha):
            betaq = (alpha * u)**(1.0/(eta+1))
        else:
            betaq = (1.0 / (2.0 - alpha*u))**(1.0/(eta+1))
        
        # Calcular los hijos
        c1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))
        c2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))
        
        # Ajustar a los límites
        child1[i] = np.clip(c1, lb, ub)
        child2[i] = np.clip(c2, lb, ub)
    
    return child1, child2
