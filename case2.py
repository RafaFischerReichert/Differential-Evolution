import numpy as np
from scipy.optimize import differential_evolution

popsize = 13
power_target = 1800

Pmin = [0, 0, 0, 60, 60, 60, 60, 60, 60, 40, 40, 55, 55]
Pmax = [680, 360, 360, 200, 200, 200, 200, 200, 200, 120, 120, 120, 120]
a = [
    0.00028,
    0.00056,
    0.00056,
    0.00324,
    0.00324,
    0.00324,
    0.00324,
    0.00324,
    0.00324,
    0.00284,
    0.00284,
    0.00284,
    0.00284,
]
b = [8.10, 8.10, 8.10, 7.74, 7.74, 7.74, 7.74, 7.74, 7.74, 8.60, 8.60, 8.60, 8.60]
c = [550, 309, 307, 240, 240, 240, 240, 240, 240, 126, 126, 126, 126]
e = [300, 200, 150, 150, 150, 150, 150, 150, 150, 100, 100, 100, 100]
f = [
    0.035,
    0.042,
    0.042,
    0.063,
    0.063,
    0.063,
    0.063,
    0.063,
    0.063,
    0.084,
    0.084,
    0.084,
    0.084,
]


def objective(x):
    sum = 0
    for i in range(popsize):
        sum += (
            a[i] * x[i]**2
            + b[i] * x[i]
            + c[i]
            + np.abs(e[i] * np.sin(f[i] * (Pmin[i] - x[i])))
        )
    return np.abs(sum - power_target)


bounds = []
for i in range(popsize):
    bounds.append((Pmin[i], Pmax[i]))

# print(bounds)
for i in range(5):
    results = differential_evolution(
        objective,
        bounds,
        popsize=popsize,
        polish=False,
        mutation=0.91,
        recombination=0.7,
        tol=0.0005,
        strategy="currenttobest1exp",
    )
    print("resultado: {}, iteracoes: {}".format(results.fun, results.nit))