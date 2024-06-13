import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['font.family'] = ['Microsoft YaHei']

# 判断函数:判断个体是否满足约束(容量限制)
def judge_individual(Individual: np.array, w: np.array, cap: int):
    total_w = w[Individual == 1].sum()
    return total_w <= cap

# 约束处理函数
def repair_individual(Individual: np.array, w: np.array, p: np.array, cap: int):
    flag = judge_individual(Individual, w, cap)
    if not flag:
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

def encode(n: int, w: np.array, p: np.array, cap: int):
    Individual = np.random.choice([0, 1], size=n)
    Individual = repair_individual(Individual, w, p, cap)
    return Individual

def InitPop(NIND: int, n: int, w: np.array, p: np.array, cap: int):
    Chrom = np.zeros(shape=(NIND, n))
    for i in range(NIND):
        Chrom[i, :] = encode(n, w, p, cap)
    return Chrom

def Individual_P_W(Individual: np.array, p: np.array, w:np.array):
    pack_item = (Individual == 1)
    sumP = p[pack_item].sum()
    sumW = w[pack_item].sum()
    return [sumP, sumW]

def Obj_Fun(Chrom: np.array, p: np.array, w: np.array):
    Obj = np.apply_along_axis(Individual_P_W, axis=1, arr=Chrom, p=p, w=w)[:, 0:1]
    return Obj

def Select(Chrom: np.array, FitnV: np.array, GGAP: float):
    NIND = Chrom.shape[0]
    Nsel = int(NIND * GGAP)
    total_FitnV = FitnV.sum()
    select_p = FitnV / total_FitnV
    select_index = np.zeros(shape=(Nsel,))
    c = np.cumsum(select_p)
    for i in range(Nsel):
        r = np.random.rand()
        index = np.where(c >= r)[0][0]
        select_index[i] = index
    return Chrom[select_index.astype(np.int), :]

def Crossover(SelCh: np.array, Pc: float):
    NSel, n = SelCh.shape
    for i in range(0, NSel, 2):
        if np.random.rand() <= Pc:
            # cross_pos = np.random.randint(1, n+1)
            # cross_Selch1 = SelCh[i, :]
            # cross_Selch2 = SelCh[i+1, :]
            # cross_part1 = cross_Selch1[:cross_pos]
            # cross_part2 = cross_Selch2[:cross_pos]
            # cross_Selch1[:cross_pos] = cross_part2
            # cross_Selch2[:cross_pos] = cross_part1
            # SelCh[i, :] = cross_Selch1
            # SelCh[i+1, :] = cross_Selch2
            cross_pos = np.random.randint(1, n + 1)
            temp = SelCh[i, :cross_pos].copy()  # 需注意这里需要复制
            SelCh[i, :cross_pos] = SelCh[i + 1, :cross_pos]
            SelCh[i + 1, :cross_pos] = temp
    return SelCh

def Mutatee(SelCh: np.array, Pm: float):
    NSel, n = SelCh.shape
    for i in range(NSel):
        if np.random.rand() < Pm:
            R = np.random.permutation(n)[:2]
            left, right = R.min(), R.max()
            mutate_Selch = SelCh[i, :]
            mutate_part = mutate_Selch[left:right+1][::-1]
            mutate_Selch[left:right+1] = mutate_part
            SelCh[i, :] = mutate_Selch
    return SelCh

def Reins(Chrom: np.array, SelCh: np.array, Obj: np.array):
    NIND = Chrom.shape[0]
    NSel = SelCh.shape[0]
    index = Obj.flatten().argsort(axis=0)[::-1]
    Chrom = np.vstack([Chrom[index[:NIND-NSel]], SelCh])
    return Chrom

def main():
    w = np.array([80, 82, 85, 70, 72, 70, 82, 75, 78, 45, 49, 76, 45, 35, 94, 49, 76, 79, 84, 74, 76, 63, 35, 26, 52,
                  12, 56, 78, 16, 52, 16, 42, 18, 46, 39, 80, 41, 41, 16, 35, 70, 72, 70, 66, 50, 55, 25, 50, 55, 40])
    p = np.array([200, 208, 198, 192, 180, 180, 168, 176, 182, 168, 187, 138, 184, 154, 168, 175, 198, 184, 158, 148,
                  174, 135, 126, 156, 123, 145, 164, 145, 134, 164, 134, 174, 102, 149, 134, 156, 172, 164, 101, 154,
                  192, 180, 180, 165, 162, 160, 158, 155, 130, 125])
    cap = 1000
    n = len(p)
    NIND = 500
    MAXGEN = 500
    Pc = 0.9
    Pm = 0.08
    GGAP = 0.9
    Chrom = InitPop(NIND, n, w, p, cap)
    gen = 0
    bestIndividual = Chrom[0, :]
    bestObj = Individual_P_W(bestIndividual, p, w)[0]
    BestObj = np.zeros(shape=(MAXGEN,))
    Obj = Obj_Fun(Chrom, p, w)
    while gen < MAXGEN:
        FitnV = Obj
        SelCh = Select(Chrom, FitnV, GGAP)
        SelCh = Crossover(SelCh, Pc)
        SelCh = Mutatee(SelCh, Pm)
        Chrom = Reins(Chrom, SelCh, Obj)
        Chrom = np.apply_along_axis(repair_individual, axis=1, arr=Chrom, w=w, p=p, cap=cap)
        Obj = Obj_Fun(Chrom, p, w)
        cur_bestIndex = np.argmax(Obj)
        cur_bestObj = Obj[cur_bestIndex, 0]
        cur_bestIndividual = Chrom[cur_bestIndex, :]
        if cur_bestObj >= bestObj:
            bestObj = cur_bestObj
            bestIndividual = cur_bestIndividual
        BestObj[gen] = bestObj
        gen += 1
        print(f'第{gen}次迭代的全局最优解如下:', bestObj)

    plt.figure()
    plt.plot(np.arange(1,501), BestObj)
    plt.xlabel('迭代次数')
    plt.ylabel('目标函数值(物品总价值)')
    plt.savefig('rub\\迭代过程.png')
    plt.show()

    bestP, bestW = Individual_P_W(bestIndividual, p, w)
    return bestIndividual, bestP, bestW


bestIndividual, bestP, bestW = main()
print('---------------------------------')
print('最好的物品选取: ', np.where(bestIndividual == 1)[0]+1)
print(f'最佳价值 {bestP} 元, 最佳重量 {bestW} kg.')