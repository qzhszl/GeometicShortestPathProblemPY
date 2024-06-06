# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 1-5-2024
"""
import math
import random
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
    radiusmax = 1  # Maximum bound for radius
    radiusmin = 0.0001  # Minimum bound for radius
    radius = math.sqrt(avg/((N-1)*math.pi))

    while not flag:
        G,_,_= RandomGeometricGraph(N,avg,rg,radius)
        real_degree = G.number_of_edges() * 2 / G.number_of_nodes()
        print("current <k>:", real_degree)
        if abs(avg - real_degree) < 0.1:
            flag = True
            print("Optimal radius:", radius)
        elif real_degree - avg > 0:
            radiusmax = radius
            radius -= 0.5 * (radius - radiusmin)
        else:
            radiusmin = radius
            radius += 0.5 * (radiusmax - radius)
    return radius


# def degree_vs_radius():
#     """
#     % INPUT: expected degree ED;expected clustering coefficent CC
#     :return: beta
#     """
#     N = 10000  # Number of nodes
#     ED = [3, 5, 7, 10, 15, 20, 50, 100]  # Expected degrees
#     CC = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45]  # Clustering coefficients
#     betavec = np.zeros((len(ED), len(CC)))  # Matrix to store optimal beta values
#
#     # Generata a random number generator for ran1 and warm up a bit
#     rg = RandomGenerator(-12)
#     for i in range(random.randint(1,100)):
#         rg.ran1()
#
#     # Loop through ED and CC combinations to find optimal beta
#     for i, ED_i in enumerate(ED):
#         print("E_d:",ED_i)
#         for j, CC_i in enumerate(CC):
#             print("CC:", CC_i)
#             flag = False
#             betamax = 200  # Maximum bound for beta
#             betamin = 1.2  # Minimum bound for beta
#             beta = 5  # Starting beta
#
#             while not flag:
#                 G,_,_= SphericalSoftRGG(N, ED_i, beta, rg)
#                 cc = nx.average_clustering(G)
#                 print("current cc:", cc)
#                 if abs(cc - CC_i) < 0.01:
#                     flag = True
#                     betavec[i][j] = beta
#                     print("Optimal beta:", beta)
#                 elif cc - CC_i > 0:
#                     betamax = beta
#                     beta -= 0.5 * (beta - betamin)
#                 else:
#                     betamin = beta
#                     beta += 0.5 * (betamax - beta)
#     return betavec

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Verification
    N = 10000
    avg =5

    r = degree_vs_radius(N, avg)
    print(r)

    rg = RandomGenerator(-12)
    for i in range(random.randint(1, 100)):
        rg.ran1()

    for _ in range(10):
        G, _, _ = RandomGeometricGraph(N, avg, rg, r)
        print("degree:", G.number_of_edges() * 2 / G.number_of_nodes())

