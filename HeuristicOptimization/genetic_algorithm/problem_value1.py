import numpy as np
from genetic_algorithm import GeneticAlgorithmRI

def constrain(x):
    c = min(np.power((x - 5), 2).sum() - 100, 0)
    return c

def obj_func(x):
    return np.power((x - np.array([10, 20])), 3).sum() - constrain(x) * 1e3

name = 'problem1'
maxormins = -1
dim = 2
lb = [13, 0]
ub = [100, 100]
lbin = [1] * dim
ubin = [1] * dim
decimal = 3

NIND = 500
MAXGEN = 500
GGAP = 0.9
Pc = 0.9
Pm = 0.1

# class Problem(GeneticAlgorithmRI):
#
#     def Repair(self, Chrom, *args):
#         repaired_Chrom = np.clip(Chrom, self.lower_bound, self.upper_bound)
#         judge = (np.apply_along_axis(args[0], axis=1, arr=repaired_Chrom) < 0)
#         repaired_Chrom[judge, 0] = 15
#         repaired_Chrom[judge, 1] = 0
#         return repaired_Chrom
#
# args = (constrain,)

ga = GeneticAlgorithmRI(name, maxormins, dim, lb, ub, lbin, ubin, decimal)
ga.run(obj_func, NIND, MAXGEN, GGAP, Pc, Pm)
print(ga.result)
print(ga.x)