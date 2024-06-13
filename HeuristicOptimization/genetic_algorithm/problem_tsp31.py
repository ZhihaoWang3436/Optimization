import numpy as np
import pandas as pd
from genetic_algorithm import GeneticAlgorithmP

filename = '..\\data\\citys_data.csv'

data = pd.read_csv(filename)[['横坐标', '纵坐标']].values

def distance(x, data):
    dst = 0
    n = len(x)
    x = x.astype(int)
    for i in range(n):
        if i < n - 1:
            c1 = data[x[i], :]
            c2 = data[x[i+1], :]
            dst += np.power(np.power((c2 - c1), 2).sum(), 1/2)
        else:
            c1 = data[x[i], :]
            c2 = data[x[0], :]
            dst += np.power(np.power((c2 - c1), 2).sum(), 1/2)
    return dst

name = 'problem4'
maxormins = -1
n = len(data)
dim = n
lb = 0
ub = n-1
lbin = 1
ubin = 1

NIND = 30
MAXGEN = 10000
GGAP = 0.9
Pc = 0.8
Pm = 0.5

kwargs = {'data': data}

ga = GeneticAlgorithmP(name, maxormins, dim, lb, ub, lbin, ubin)
ga.run(distance, NIND, MAXGEN, GGAP, Pc, Pm, **kwargs)
print(ga.result)
print(ga.x)