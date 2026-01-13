import math
import random
import time

import networkx as nx
from numpy.random import random_integers
import matplotlib.pyplot as plt
from SphericalSoftRandomGeomtricGraph import RandomGenerator
import numpy as np

import numpy as np
from scipy.special import gamma


def d0_from_avg(N, avg, beta=1.0, eta=2.0, d=2):
    """
    通用反解公式计算 d0 (无限空间近似)

    参数：
    N     : 节点数
    avg   : 期望平均度
    beta  : 最大连边概率 (0<beta<=1)
    eta   : 衰减指数
    d     : 空间维度 (默认二维)

    返回：
    d0    : 连接尺度参数
    """
    rho = N / 1.0  # 假设单位体积（单位正方形或单位立方体）
    # d维球面面积
    if d == 1:
        S_d = 2
    elif d == 2:
        S_d = 2 * np.pi
    elif d == 3:
        S_d = 4 * np.pi
    else:
        # 高维空间通用公式
        S_d = 2 * np.pi ** (d / 2) / gamma(d / 2)

    # 反解 d0
    d0 = (avg * eta / (rho * beta * S_d * gamma(d / eta))) ** (1 / d)
    return d0



def plot_rayleigh_function():
    d = np.linspace(0, 1, 500)

    # ------------------------
    # 参数设置
    # ------------------------
    # 左图：不同 beta，固定 d0
    d0_fixed = 1.0
    beta_values = [0.2, 0.5, 0.8, 1.0]

    # 右图：不同 d0，固定 beta
    beta_fixed = 1
    d0_values = [0.05,0.1,0.4,0.5,2]

    # ------------------------
    # 绘图
    # ------------------------
    plt.figure(figsize=(12, 5))

    # 左图：不同 beta
    plt.subplot(1, 2, 1)



    N=1000
    for avg in [2,5,10,100,1000]:
        d0 = d0_from_avg(N, avg)
        f = 1 * np.exp(-(d / d0) ** 2)
        plt.plot(d, f, label=f'avg={avg}')
    plt.xlabel('d')
    plt.ylabel('f(d)')
    plt.legend()
    plt.grid(True)

    # 右图：不同 d0
    plt.subplot(1, 2, 2)
    for d0 in d0_values:
        f = beta_fixed * np.exp(-(d / d0) ** 2)
        plt.plot(d, f, label=f'd0={d0}')
    plt.xlabel('d')
    plt.ylabel('f(d)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()




def R2SRGG_waman(N, avg, eta, rg, beta=1, Coorx=None, Coory=None, SaveNetworkPath=None):
    """
    Program generates Soft Random Geometric Graph on a 2d unit square
    Connection probability function is f(d) = \beta e^{- \left(\frac{d}{d_0}\right)^{\eta}}
    :param N: number of nodes
    :param avg: d_0
    :param beta: beta (controlling max probability)
    :param rg:  random seed
    :param Coorx: coordinates of x
    :param Coory: coordinates of y
    :param SaveNetworkPath:
    :return:
    """
    # calculate parameters
    assert avg>0
    assert N > 1

    s = []
    t = []

    # Assign coordinates
    if Coorx is not None and Coory is not None:
        xx = Coorx
        yy = Coory
    else:
        xx = []
        yy = []
        for i in range(N):
            xx.append(rg.ran1())
            yy.append(rg.ran1())

    d0 = d0_from_avg(N, avg, beta, eta)
    # print(d0)

    # make connections
    for i in range(N):
        for j in range(i + 1, N):
            dist = math.sqrt((xx[i] - xx[j]) ** 2 + (yy[i] - yy[j]) ** 2)
            assert dist > 0

            try:
                prob = beta * np.exp(-(dist / d0) ** 2)
            except:
                prob = 0
            if rg.ran1() < prob:
                s.append(i)
                t.append(j)


    if SaveNetworkPath is not None:
        with open(SaveNetworkPath, "w") as file:
            for nodei, nodej in zip(s, t):
                file.write(f"{nodei}\t{nodej}\n")
    # Create graph and remove self-loops
    G = nx.Graph()
    G.add_edges_from(zip(s, t))
    if G.number_of_nodes()<N:
        ExpectedNodeList = [i for i in range(0, N)]
        Nodelist = list(G.nodes)
        difference = [item for item in ExpectedNodeList if item not in Nodelist]
        G.add_nodes_from(difference)
    return G, xx, yy


if __name__ == '__main__':
    rg = RandomGenerator(-12)
    for i in range(random.randint(1, 1000)):
        rg.ran1()

    G,xx,yy = R2SRGG_waman(1000,100,2,rg)
    real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
    print("real ED:", real_avg)

