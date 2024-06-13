import numpy as np
from ant_colony_algorithm import AntColonyOptimizationValue

def rosebrock(x):
    return np.power(x, 2).sum() + 1

name = 'power_sum'
num_dimensions = 2
bounds = [(-10, 10)] * num_dimensions
num_ants = 20
num_iterations = 100
alpha = 1
beta = 2
rho = 0.5
q = 1

aco = AntColonyOptimizationValue(name, num_ants, num_iterations, alpha, beta, rho, q, num_dimensions, bounds)
best_position, best_fitness = aco.run(rosebrock)
print(aco.best_position)
print(aco.best_fitness)