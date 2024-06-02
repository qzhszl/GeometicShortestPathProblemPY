# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 31-5-2024
"""
import math
import random
import networkx as nx
from SphericalSoftRandomGeomtricGraph import RandomGenerator


def R2SRGG(N, avg, beta, rg, Coorx=None, Coory=None, SaveNetworkPath=None):
    """
    Program generates Soft Random Geometric Graph on a 2d unit square
    Connection probability function is (1 + (a*d)^{b})^{-1}
    :param N: number of nodes
    :param avg: <k>
    :param beta: beta (controlling clustering)
    :param rg:  random seed
    :param Coorx: coordinates of x
    :param Coory: coordinates of y
    :param SaveNetworkPath:
    :return:
    """
    # calculate parameters
    assert beta > 2
    assert avg > 0
    assert N > 1

    R = 2.0  # manually tuned value
    alpha = (2 * N / avg * R * R) * (math.pi / (math.sin(2 * math.pi / beta) * beta))
    alpha = math.sqrt(alpha)
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

    # make connections
    for i in range(N):
        for j in range(i + 1, N):
            dist = math.sqrt((xx[i] - xx[j]) ** 2 + (yy[i] - yy[j]) ** 2)
            assert dist > 0
            prob = 1 / (1 + math.exp(beta * math.log(alpha * dist)))
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



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    rg = RandomGenerator(-12)  # Seed initialization
    # for _ in range(random.randint(0, 100)):
    #     rg.ran1()
    R2SRGG(100, 5, 3.5, rg)
