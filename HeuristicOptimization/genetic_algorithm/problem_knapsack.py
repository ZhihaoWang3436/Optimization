import numpy as np
from genetic_algorithm import GeneticAlgorithmBG

def constrain(x, w, cap):
    weight = np.inner(x, w)
    return max(weight - cap, 0)

def obj_func(x, p):
    return np.inner(x, p)

w = np.array([80, 82, 85, 70, 72, 70, 82, 75, 78, 45,
              49, 76, 45, 35, 94, 49, 76, 79, 84, 74,
              76, 63, 35, 26, 52, 12, 56, 78, 16, 52,
              16, 42, 18, 46, 39, 80, 41, 41, 16, 35,
              70, 72, 70, 66, 50, 55, 25, 50, 55, 40])
p = np.array([200, 208, 198, 192, 180, 180, 168, 176, 182, 168,
              187, 138, 184, 154, 168, 175, 198, 184, 158, 148,
              174, 135, 126, 156, 123, 145, 164, 145, 134, 164,
              134, 174, 102, 149, 134, 156, 172, 164, 101, 154,
              192, 180, 180, 165, 162, 160, 158, 155, 130, 125])
cap = 1000


name = 'problem3'
maxormins = 1
dim = len(w)
lb = [0] * dim
ub = [1] * dim
lbin = [1] * dim
ubin = [1] * dim
decimal = 0

NIND = 500
MAXGEN = 500
GGAP = 0.9
Pc = 0.9
Pm = 0.1

def repair_individual(Individual, w, p, cap):
    flag = constrain(Individual, w, cap)
    if flag > 0:
        pack_item = np.where(Individual == 1)[0]
        w_pack = w[pack_item]
        total_w = w_pack.sum()
        p_pack = p[pack_item]
        ratio_pack = p_pack / w_pack
        rps_index = ratio_pack.argsort()
        for i in rps_index:
            total_w -= w_pack[i]
            Individual[pack_item[i]] = 0
            if total_w <= cap:
                break
        unpack_item = np.where(Individual == 0)[0]
        w_unpack = w[unpack_item]
        p_unpack = p[unpack_item]
        ratio_unpack = p_unpack / w_unpack
        rups_index = ratio_unpack.argsort()[::-1]
        for j in rups_index:
            total_w += w_unpack[j]
            if total_w > cap:
                break
            Individual[unpack_item[j]] = 1
    return Individual

class ProblemKnapsack(GeneticAlgorithmBG):

    def Repair(self, Chrom, *args):
        repaired_Chrom = np.clip(Chrom, self.lower_bound, self.upper_bound)
        repaired_Chrom = np.apply_along_axis(repair_individual, axis=1, arr=repaired_Chrom,
                                             w=args[0], p=args[1], cap=args[2])
        return repaired_Chrom

kwargs = {'p': p}
args = (w, p, cap)

ga = ProblemKnapsack(name, maxormins, dim, lb, ub, lbin, ubin, decimal)
ga.run(obj_func, NIND, MAXGEN, GGAP, Pc, Pm, *args, **kwargs)
print(ga.result)
print(ga.x)
print(np.where(ga.x == 1)[0] + 1)