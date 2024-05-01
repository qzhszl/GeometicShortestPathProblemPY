# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 2024/4/30
"""
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from SphericalSoftRandomGeomtricGraph import RandomGenerator, SphericalSoftRGGwithGivenNode


# Function to find nodes that frequently appear in the shortest paths
def FindNearlySPNodes(G, nodei, nodej, RelevanceSimTimes=1000):
    N = G.number_of_nodes()  # Total number of nodes in the graph
    Noderelevance = np.zeros(N)  # Initialize node relevance

    # Simulate the removal of random edges and calculate shortest paths
    for Simutime in range(RelevanceSimTimes):
        # print(Simutime)
        ShuffleTable = np.random.rand(G.number_of_edges())  # Random numbers for shuffle
        H = G.copy()  # Create a copy of the graph
        edges_to_remove = [e for e, shuffle_value in zip(G.edges, ShuffleTable) if shuffle_value < 0.5]
        H.remove_edges_from(edges_to_remove)  # Remove edges with shuffle value < 0.5
        # time3 = time.time()

        # Find all shortest paths between nodei and nodej
        try:
            # timespecial = time.time()
            # shortest_paths = nx.all_shortest_paths(H, nodei, nodej)
            # print("timeallsp",time.time()-timespecial)
            # print("pathlength", sum(1 for _ in shortest_paths))
            shortest_paths = nx.all_shortest_paths(H, nodei, nodej)
            PNodeList = set()  # Use a set to keep unique nodes
            count =0
            for path in shortest_paths:
                PNodeList.update(path)
                count+=1
                if count>1000000:
                    PNodeList = set()
                    break
            # print("pathlength", len(path))
            # print("pathnum",count)
        except nx.NetworkXNoPath:
            PNodeList = set()  # If there's no path, continue with an empty set
        # time31 = time.time()
        # print("timeallsp0",time31-time3)
        # Remove the starting and ending nodes from the list
        PNodeList.discard(nodei)
        PNodeList.discard(nodej)

        # Increment relevance count for nodes appearing in the path
        for node in PNodeList:
            Noderelevance[node] += 1

    # Normalize relevance values by the number of simulations
    Noderelevance /= RelevanceSimTimes

    # Find nodes with relevance greater than 0.05
    NearlySPNodelist = [i for i, relevance in enumerate(Noderelevance) if relevance > 0.05]

    return NearlySPNodelist, Noderelevance


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    rg = RandomGenerator(-12)  # Seed initialization
    N = 10000
    avg = 5
    beta = 5
    tic=time.time()
    print(tic)
    G, Coortheta, Coorphi = SphericalSoftRGGwithGivenNode(N, avg, beta, rg, math.pi / 2, 0, math.pi / 2, 1)
    toc1 = time.time()
    print(time.time()-tic)

    nodei = N-1
    nodej = N-2

    NSP,NSPrelevance = FindNearlySPNodes(G, nodei, nodej,RelevanceSimTimes=1000)
    print(len(NSP))

    print("time:", time.time()-toc1)
    plt.hist(NSPrelevance)
    plt.show()
