# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 1-5-2024
"""
import math
import random
import pandas as pd
import numpy as np
import networkx as nx

from R2RGG import RandomGeometricGraph
from SphericalSoftRandomGeomtricGraph import RandomGenerator


def degree_vs_radius(N,avg):
    """
    % INPUT: expected degree ED;expected clustering coefficent CC
    :return: radius
    """
    # Generata a random number generator for ran1 and warm up a bit
    rg = RandomGenerator(-12)
    for i in range(random.randint(1,100)):
        rg.ran1()

    # Loop through ED and CC combinations to find optimal beta

    flag = False

    radiusmin = 0.0001  # Minimum bound for radius
    radius = math.sqrt(avg/((N-1)*math.pi))
    radiusmax = radius+0.01  # Maximum bound for radius

    while not flag:
        G,_,_= RandomGeometricGraph(N,avg,rg,radius)
        real_degree = G.number_of_edges() * 2 / G.number_of_nodes()
        # print("current <k>:", real_degree)
        if abs(avg - real_degree) < 0.1:
            flag = True
            print("current <k>:", real_degree)
            print("Optimal radius:", radius)
        elif real_degree - avg > 0:
            radiusmax = radius
            radius -= 0.5 * (radius - radiusmin)
        else:
            radiusmin = radius
            radius += 0.5 * (radiusmax - radius)
    return radius


def compute_degree_vs_radius_diffN():
    """
    % INPUT: expected degree ED; Node number N
    :return: radius
    """
    ED = [3, 5, 7, 10, 15, 20]  # Expected degrees
    N = [100,500,1000,10000]  # Clustering coefficients
    betavec = np.zeros((len(ED), len(N)))  # Matrix to store optimal radius values

    # Generata a random number generator for ran1 and warm up a bit
    rg = RandomGenerator(-12)
    for i in range(random.randint(1,100)):
        rg.ran1()

    # Loop through ED and N combinations to find optimal radius
    for i, ED_i in enumerate(ED):
        print("E_d:",ED_i)
        for j, N_j in enumerate(N):
            print("N:", N_j)
            betavec[i][j] = degree_vs_radius(N_j, ED_i)
    print(betavec)
    return 0

def read_radius(N,avg):
    m1 = [[0.1069628,  0.0462457,  0.0321674,  0.00977254],
        [0.13522968, 0.05772546, 0.04116419, 0.01261629],
    [0.15877253,0.06932267,0.04847711,0.0149278],
    [0.18930023, 0.08236836, 0.05769719, 0.01799838],
    [0.22960928,0.10313087,0.07163341,0.02204737],
    [0.26358436, 0.11857592, 0.08275807, 0.02554509]]
    df = pd.DataFrame(m1, columns=[100,500,1000,10000],index=[3, 5, 7, 10, 15, 20])
    radius = df.loc[avg,N]
    return radius


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Verification
    # N = 10000
    # avg =20

    # r = degree_vs_radius(200, 5)
    # print(r)

    # rg = RandomGenerator(-12)
    # for i in range(random.randint(1, 100)):
    #     rg.ran1()
    #
    # for _ in range(10):
    #     G, _, _ = RandomGeometricGraph(N, avg, rg, r)
    #     print("degree:", G.number_of_edges() * 2 / G.number_of_nodes())

    # m1=[]
    # m1.append([1,2])
    # m1.append([4, 3])
    #
    # df = pd.DataFrame(m1, columns=[100,10000],index=[5,20])
    # print(df)
    # print(df.loc[5,100])

    compute_degree_vs_radius_diffN()



