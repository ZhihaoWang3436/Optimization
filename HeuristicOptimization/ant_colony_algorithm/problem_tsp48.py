import csv
import gzip
import numpy as np
from ant_colony_algorithm import AntColonyOptimizationTSP

filename = '..\\data\\att48.tsp.gz'

data = []
with gzip.open(filename, 'rt') as file:
    reader = csv.reader(file)
    for row in reader:
        data.append([t for t in row][0])

data = np.array([d.split() for d in data[6:54]]).astype(int)[:, 1:]

num_cities = len(data)
distance_matrix = np.zeros((num_cities, num_cities))
for i in range(num_cities-1):
    for j in range(i+1, num_cities):
        distance_matrix[i, j] = distance_matrix[j, i] = np.sqrt(
            np.power(data[i] - data[j], 2).sum() / 10
        )

name = 'tsp48'
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