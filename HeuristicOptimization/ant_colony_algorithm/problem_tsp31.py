import numpy as np
import pandas as pd
from ant_colony_algorithm  import AntColonyOptimizationTSP

filename = '..\\data\\citys_data.csv'

data = pd.read_csv(filename)[['横坐标', '纵坐标']].values
num_cities = len(data)
distance_matrix = np.zeros((num_cities, num_cities))
for i in range(num_cities-1):
    for j in range(i+1, num_cities):
        distance_matrix[i, j] = distance_matrix[j, i] = np.sqrt(
            np.power(data[i] - data[j], 2).sum()
        )

name = 'tsp31'
num_ants = 50
alpha = 1
beta = 5
rho = 0.1
q = 1
num_iterations = 200
aco = AntColonyOptimizationTSP(name, num_ants, num_iterations, alpha, beta, rho, q)
aco.run(distance_matrix)
print(aco.best_tour_length)
print(aco.best_tour)