# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 1-5-2024
"""

import random
import numpy as np
import networkx as nx
from SphericalSoftRandomGeomtricGraph import RandomGenerator, SphericalSoftRGGwithGivenNode,SphericalSoftRGG

def DegreeVsClu():
    """
    % INPUT: expected degree ED;expected clustering coefficent CC
    :return: beta
    """
    N = 10000  # Number of nodes
    ED = [3, 5, 7, 10, 15, 20, 50, 100]  # Expected degrees
    CC = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45]  # Clustering coefficients
    betavec = np.zeros((len(ED), len(CC)))  # Matrix to store optimal beta values

    # Generata a random number generator for ran1 and warm up a bit
    rg = RandomGenerator(-12)
    for i in range(random.randint(1,100)):
        rg.ran1()

    # Loop through ED and CC combinations to find optimal beta
    for i, ED_i in enumerate(ED):
        print("E_d:",ED_i)
        for j, CC_i in enumerate(CC):
            print("CC:", CC_i)
            flag = False
            betamax = 200  # Maximum bound for beta
            betamin = 1.2  # Minimum bound for beta
            beta = 5  # Starting beta

            while not flag:
                G,_,_= SphericalSoftRGG(N, ED_i, beta, rg)
                cc = nx.average_clustering(G)
                print("current cc:", cc)
                if abs(cc - CC_i) < 0.01:
                    flag = True
                    betavec[i][j] = beta
                    print("Optimal beta:", beta)
                elif cc - CC_i > 0:
                    betamax = beta
                    beta -= 0.5 * (beta - betamin)
                else:
                    betamin = beta
                    beta += 0.5 * (betamax - beta)
    return betavec



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(DegreeVsClu())
    # Verification
    # N = 10000
    # beta = 7
    # avg =
    # rg = RandomGenerator(-12)
    # for i in range(random.randint(1, 100)):
    #     rg.ran1()
    #
    # for _ in range(10):
    #     G, _, _ = SphericalSoftRGG(N, avg, beta, rg)
    #     cc = nx.average_clustering(G)
    #     avg_clustering = np.mean(list(cc))
    #     print("Average clustering coefficient:", avg_clustering)

