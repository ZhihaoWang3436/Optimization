from ant_colony_algorithm import AntColonyOptimizationValue

def rosebrock(x):
    return sum(100 * (x[1:] - x[:-1]**2) ** 2 + (1 - x[:-1]) ** 2)

name = 'rosebrock'
num_dimensions = 2
bounds = [(-5, 5)] * num_dimensions
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