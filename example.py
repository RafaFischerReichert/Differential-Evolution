from scipy.optimize import differential_evolution
import numpy as np


def func(p):
    x, y = p
    r = np.sqrt(x**2 + y**2)
    return np.sqrt(r)


bounds = [[-4, 4], [-4, 4]]

# execute differential evolution search
result = differential_evolution(func, bounds)

print(result)
