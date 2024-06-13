import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['Microsoft YaHei']

class BaseGeneticAlgorithm:
    def __init__(self, name, maxormin, dim):
        self.name = name
        self.maxormin = maxormin
        self.dim = dim
    def Encode(self, Value):
        raise NotImplementedError('未定义编码函数!')
    def Decode(self, Individual):
        raise NotImplementedError('未定义解码函数!')
    def Repair(self, Chrom, *args):
        raise NotImplementedError('未定义整理函数!')
    def aimFunc(self, Chrom, func, **kwargs):
        raise NotImplementedError('未定义目标函数!')
    def InitPop(self, NIND, *args):
        raise NotImplementedError('未定义初始化种群函数!')
    def Select(self, Chrom, FitnV, NSel):
        raise NotImplementedError('未定义选择函数!')
    def Crossover(self, SelCh, Pc, NSel):
        raise NotImplementedError('未定义交叉函数!')
    def Mutate(self, SelCh, Pm, NSel):
        raise NotImplementedError('未定义变异函数!')
    def Reins(self, Chrom, SelCh, FitnV, NIND, NSel):
        raise NotImplementedError('未定义组合函数!')
    def run(self, func, NIND, MAXGEN, GGAP, Pc, Pm, *args, **kwargs):
        raise NotImplementedError('未定义主函数!')
    def trace_plot(self, MAXGEN, BestObj):
        plt.figure()
        plt.plot(np.arange(1, MAXGEN+1), BestObj)
        plt.xlabel('迭代次数')
        plt.ylabel('目标函数值(物品总价值)')
        plt.title(self.name)
        plt.savefig(f'rub\\{self.name}迭代过程.png')
        plt.show()


class GeneticAlgorithmBG(BaseGeneticAlgorithm):

    def __init__(self, name, maxormin, dim, lb, ub, lbin, ubin, decimal):
        super().__init__(name, maxormin, dim)
        self.decimal_ratio = 1 / (10 ** decimal)
        self.lower_bound = np.array([lb[i] + (1 - lbin[i]) * self.decimal_ratio for i in range(dim)])
        self.upper_bound = np.array([ub[i] - (1 - ubin[i]) * self.decimal_ratio for i in range(dim)])
        self.dna_info = self.DnaSize().astype(int)
        self.dna_size = self.dna_info.sum()

    def DnaSize(self):
        dna_info = np.ceil(np.log2((self.upper_bound - self.lower_bound) / self.decimal_ratio + 1))
        return dna_info

    def Encode(self, Value):
        result = []
        for i in range(self.dim):
            part = np.array(list(bin(int((Value[i] - self.lower_bound[i]) / self.decimal_ratio))[2:]), dtype=int)
            result.append(part)
        return np.hstack(result)

    def Decode(self, Individual):
        Value = np.zeros(shape=(self.dim,))
        left = 0
        for i in range(self.dim):
            right = left + self.dna_info[i]
            part = Individual[left:right]
            two_array = np.power(2, np.arange(self.dna_info[i]))[::-1]
            Value[i] = self.lower_bound[i] + np.inner(part, two_array) * self.decimal_ratio
            left = right
        return Value

    def Repair(self, Chrom, *args):
        Values = np.apply_along_axis(self.Decode, axis=1, arr=Chrom)
        repaired_Values = np.clip(Values, self.lower_bound, self.upper_bound)
        repaired_Chrom = np.apply_along_axis(self.Encode, axis=1, arr=repaired_Values)
        return repaired_Chrom

    def aimFunc(self, Chrom, func, **kwargs):
        Values = np.apply_along_axis(self.Decode, axis=1, arr=Chrom)
        Obj = np.apply_along_axis(func, axis=1, arr=Values, **kwargs)
        return Obj

    def InitPop(self, NIND, *args):
        Chrom = np.random.choice([0, 1], size=(NIND, self.dna_size))
        Chrom = self.Repair(Chrom,  *args)
        return Chrom

    def Select(self, Chrom, FitnV, NSel):
        adjusted_FitnV = FitnV - FitnV.min() + self.decimal_ratio
        total_FitnV = adjusted_FitnV.sum()
        select_p = adjusted_FitnV / total_FitnV
        select_index = np.zeros(shape=(NSel,))
        c = np.cumsum(select_p)
        for i in range(NSel):
            r = np.random.rand()
            index = np.where(c >= r)[0][0]
            select_index[i] = index
        return Chrom[select_index.astype(int), :]

    def Crossover(self, SelCh, Pc, NSel):
        for i in range(0, NSel - 1, 2):
            if np.random.rand() <= Pc:
                R = np.sort(np.random.choice(range(self.dna_size), size=2, replace=True))
                left, right = R[0], R[1]
                temp = SelCh[i, left:right + 1].copy()  # 需注意这里需要复制
                SelCh[i, left:right + 1] = SelCh[i + 1, left:right + 1]
                SelCh[i + 1, left:right + 1] = temp
        return SelCh

    def Mutate(self, SelCh, Pm, NSel):
        for i in range(NSel):
            if np.random.rand() <= Pm:
                R = np.sort(np.random.choice(range(self.dna_size), size=2, replace=True))
                left, right = R[0], R[1]
                SelCh[i, left:right+1] = SelCh[i, left:right+1][::-1]
        return SelCh

    def Reins(self, Chrom, SelCh, FitnV, NIND, NSel):
        indices = FitnV.argsort(axis=0)[::-1]
        Chrom = np.vstack([Chrom[indices[:NIND-NSel]], SelCh])
        return Chrom

    def trace_plot(self, MAXGEN, BestObj):
        plt.figure()
        plt.plot(np.arange(1, MAXGEN+1), BestObj)
        plt.xlabel('迭代次数')
        plt.ylabel('目标函数值(物品总价值)')
        plt.title(self.name)
        plt.savefig(f'rub\\{self.name}迭代过程.png')
        plt.show()

    def run(self, func, NIND, MAXGEN, GGAP, Pc, Pm, *args, **kwargs):
        Chrom = self.InitPop(NIND, *args)
        Obj = self.aimFunc(Chrom, func, **kwargs)
        bestIndividual = Chrom[0, :]
        bestObj = Obj[0]
        BestObj = np.zeros(shape=(MAXGEN,))
        NSel = int(NIND * GGAP)
        gen = 0
        while gen < MAXGEN:
            FitnV = Obj * self.maxormin
            SelCh = self.Select(Chrom, FitnV, NSel)
            SelCh = self.Crossover(SelCh, Pc, NSel)
            SelCh = self.Mutate(SelCh, Pm, NSel)
            Chrom = self.Reins(Chrom, SelCh, FitnV, NIND, NSel)
            Chrom = self.Repair(Chrom, *args)
            Obj = self.aimFunc(Chrom, func, **kwargs)
            cur_bestIndex = np.argmax(Obj)
            cur_bestObj = Obj[cur_bestIndex]
            cur_bestIndividual = Chrom[cur_bestIndex, :]
            if (cur_bestObj - bestObj) * self.maxormin >= 0:
                bestObj = cur_bestObj
                bestIndividual = cur_bestIndividual
            BestObj[gen] = bestObj
            gen += 1
            print(f'第{gen}次迭代的全局最优解如下:', bestObj)

        self.x = self.Decode(bestIndividual)
        self.result = bestObj
        self.trace_plot(MAXGEN, BestObj)


class GeneticAlgorithmRI(BaseGeneticAlgorithm):

    def __init__(self, name, maxormin, dim, lb, ub, lbin, ubin, decimal=0):
        super().__init__(name, maxormin, dim)
        self.decimal_ratio = 1 / (10 ** decimal)
        self.lower_bound = np.array([lb[i] + (1 - lbin[i]) * self.decimal_ratio for i in range(dim)])
        self.upper_bound = np.array([ub[i] - (1 - ubin[i]) * self.decimal_ratio for i in range(dim)])

    def Repair(self, Chrom, *args):
        repaired_Chrom = np.clip(Chrom, self.lower_bound, self.upper_bound)
        return repaired_Chrom

    def aimFunc(self, Chrom, func, **kwargs):
        Obj = np.apply_along_axis(func, axis=1, arr=Chrom, **kwargs)
        return Obj

    def InitPop(self, NIND, *args):
        Chrom = np.zeros(shape=(NIND, self.dim))
        for j in range(self.dim):
            Chrom[:, j] = np.random.choice(
                np.arange(start=self.lower_bound[j],
                          stop=self.upper_bound[j] + self.decimal_ratio / 10,
                          step=self.decimal_ratio),
                size=(NIND,)
            )
        Chrom = self.Repair(Chrom, *args)
        return Chrom

    def Select(self, Chrom, FitnV, NSel):
        adjusted_FitnV = FitnV - FitnV.min() + self.decimal_ratio
        total_FitnV = adjusted_FitnV.sum()
        select_p = adjusted_FitnV / total_FitnV
        select_index = np.zeros(shape=(NSel,))
        c = np.cumsum(select_p)
        for i in range(NSel):
            r = np.random.rand()
            index = np.where(c >= r)[0][0]
            select_index[i] = index
        return Chrom[select_index.astype(int), :]

    def Crossover(self, SelCh, Pc, NSel):
        Parents = SelCh.copy()
        for i in range(NSel):
            if np.random.rand() <= Pc:
                parent1, parent2 = Parents[np.random.choice(range(NSel), size=2, replace=False), :]
                beta = np.random.rand()
                child = ((1 + beta) * parent1 + (1 - beta) * parent2) / 2
            else:
                child = Parents[np.random.choice(range(NSel))]
            SelCh[i, :] = child
        return SelCh

    def Mutate(self, SelCh, Pm, NSel):
        for i in range(NSel):
            if np.random.rand() <= Pm:
                SelCh[i] += np.random.normal(0, self.decimal_ratio*2, size=self.dim)
        return SelCh

    def Reins(self, Chrom, SelCh, FitnV, NIND, NSel):
        indices = FitnV.argsort(axis=0)[::-1]
        Chrom = np.vstack([Chrom[indices[:NIND-NSel]], SelCh])
        return Chrom

    def run(self, func, NIND, MAXGEN, GGAP, Pc, Pm, *args, **kwargs):
        Chrom = self.InitPop(NIND, *args)
        Obj = self.aimFunc(Chrom, func, **kwargs)
        bestIndividual = Chrom[0, :]
        bestObj = Obj[0]
        BestObj = np.zeros(shape=(MAXGEN,))
        NSel = int(NIND * GGAP)
        gen = 0
        while gen < MAXGEN:
            FitnV = Obj * self.maxormin
            SelCh = self.Select(Chrom, FitnV, NSel)
            SelCh = self.Crossover(SelCh, Pc, NSel)
            SelCh = self.Mutate(SelCh, Pm, NSel)
            Chrom = self.Reins(Chrom, SelCh, FitnV, NIND, NSel)
            Chrom = self.Repair(Chrom, *args)
            Obj = self.aimFunc(Chrom, func, **kwargs)
            cur_bestIndex = np.argmax(Obj)
            cur_bestObj = Obj[cur_bestIndex]
            cur_bestIndividual = Chrom[cur_bestIndex, :]
            if (cur_bestObj - bestObj) * self.maxormin >= 0:
                bestObj = cur_bestObj
                bestIndividual = cur_bestIndividual
            BestObj[gen] = bestObj
            gen += 1
            print(f'第{gen}次迭代的全局最优解如下:', bestObj)

        self.x = bestIndividual
        self.result = bestObj
        self.trace_plot(MAXGEN, BestObj)


class GeneticAlgorithmP(BaseGeneticAlgorithm):

    def __init__(self, name, maxormin, dim, lb, ub, lbin, ubin):
        super().__init__(name, maxormin, dim)
        self.lower_bound = lb + (1 - lbin)
        self.upper_bound = ub - (1 - ubin)

    def aimFunc(self, Chrom, func, **kwargs):
        Obj = np.apply_along_axis(func, axis=1, arr=Chrom, **kwargs)
        return Obj

    def InitPop(self, NIND, *args):
        Chrom = np.zeros(shape=(NIND, self.dim))
        for i in range(NIND):
            Chrom[i, :] = np.random.permutation(range(self.lower_bound, self.upper_bound+1, 1))
        return Chrom

    def Select(self, Chrom, FitnV, NSel):
        adjusted_FitnV = FitnV - FitnV.min() + 1
        total_FitnV = adjusted_FitnV.sum()
        select_p = adjusted_FitnV / total_FitnV
        select_index = np.zeros(shape=(NSel,))
        c = np.cumsum(select_p)
        for i in range(NSel):
            r = np.random.rand()
            index = np.where(c >= r)[0][0]
            select_index[i] = index
        return Chrom[select_index.astype(int), :]

    def _map_genes(self, child, parent, left, right):
        for i in range(self.dim):
            if not left <= i <= right:
                gene = parent[i]
                while gene in child[left:right+1]:
                    gene = parent[np.where(child==gene)[0]]
                child[i] = gene
        return child

    def Crossover(self, SelCh, Pc, NSel):
        for i in range(0, NSel - 1, 2):
            if np.random.rand() <= Pc:
                R = np.sort(np.random.choice(range(self.dim), size=2, replace=True))
                left, right = R[0], R[1]
                child1, child2 = np.array([np.nan] * self.dim), np.array([np.nan] * self.dim)
                child1[left:right+1] = SelCh[i+1, left:right+1].copy()
                child2[left:right+1] = SelCh[i, left:right+1].copy()
                SelCh[i, :] = self._map_genes(child1, SelCh[i+1, :], left, right)
                SelCh[i+1, :] = self._map_genes(child2, SelCh[i, :], left, right)
        return SelCh

    def Mutate(self, SelCh, Pm, NSel):
        for i in range(NSel):
            if np.random.rand() <= Pm:
                R = np.sort(np.random.choice(range(self.dim), size=2, replace=True))
                left, right = R[0], R[1]
                SelCh[i, left], SelCh[i, right] = SelCh[i, right], SelCh[i, left]
        return SelCh

    def Reins(self, Chrom, SelCh, FitnV, NIND, NSel):
        indices = FitnV.argsort(axis=0)[::-1]
        Chrom = np.vstack([Chrom[indices[:NIND - NSel]], SelCh])
        return Chrom

    def run(self, func, NIND, MAXGEN, GGAP, Pc, Pm, *args, **kwargs):
        Chrom = self.InitPop(NIND, *args)
        Obj = self.aimFunc(Chrom, func, **kwargs)
        bestIndividual = Chrom[0, :]
        bestObj = Obj[0]
        BestObj = np.zeros(shape=(MAXGEN,))
        NSel = int(NIND * GGAP)
        gen = 0
        while gen < MAXGEN:
            FitnV = Obj * self.maxormin
            SelCh = self.Select(Chrom, FitnV, NSel)
            SelCh = self.Crossover(SelCh, Pc, NSel)
            SelCh = self.Mutate(SelCh, Pm, NSel)
            Chrom = self.Reins(Chrom, SelCh, FitnV, NIND, NSel)
            Obj = self.aimFunc(Chrom, func, **kwargs)
            cur_bestIndex = np.argmax(Obj)
            cur_bestObj = Obj[cur_bestIndex]
            cur_bestIndividual = Chrom[cur_bestIndex, :]
            if (cur_bestObj - bestObj) * self.maxormin >= 0:
                bestObj = cur_bestObj
                bestIndividual = cur_bestIndividual
            BestObj[gen] = bestObj
            gen += 1
            print(f'第{gen}次迭代的全局最优解如下:', bestObj)

        self.x = bestIndividual
        self.result = bestObj
        self.trace_plot(MAXGEN, BestObj)