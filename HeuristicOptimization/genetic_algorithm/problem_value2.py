import numpy as np
from genetic_algorithm import GeneticAlgorithmRI

def obj_func(x):
    return np.power(x, 2).sum()

name = 'problem2'
maxormins = -1
dim = 5
lb = [-100] * dim
ub = [100] * dim
lbin = [1] * dim
ubin = [1] * dim
decimal = 1

NIND = 100
MAXGEN = 1000
GGAP = 0.9
Pc = 0.9
Pm = 0.1


ga = GeneticAlgorithmRI(name, maxormins, dim, lb, ub, lbin, ubin, decimal)
ga.run(obj_func, NIND, MAXGEN, GGAP, Pc, Pm)
print(ga.result)
print(ga.x)