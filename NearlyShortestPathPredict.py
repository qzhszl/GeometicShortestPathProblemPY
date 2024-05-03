# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 2024/4/30
"""
import itertools
import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from SphericalSoftRandomGeomtricGraph import RandomGenerator, SphericalSoftRGGwithGivenNode, SphericalSoftRGG, \
    dist_to_geodesic_S2
from sklearn.metrics import precision_recall_curve, auc
import sys

# Function to find nodes that frequently appear in the shortest paths
def FindNearlySPNodes(G, nodei, nodej, RelevanceSimTimes=1000):
    N = G.number_of_nodes()  # Total number of nodes in the graph
    Noderelevance = np.zeros(N)  # Initialize node relevance

    # Simulate the removal of random edges and calculate shortest paths
    for Simutime in range(RelevanceSimTimes):
        print("NSP Simutime:",Simutime)
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


def TestFindNSPnodes():
    # Demo 1, use a and
    rg = RandomGenerator(-12)  # Seed initialization
    N = 10000
    avg = 5
    beta = 5
    tic = time.time()
    print(tic)
    G, Coortheta, Coorphi = SphericalSoftRGGwithGivenNode(N, avg, beta, rg, math.pi / 2, 0, math.pi / 2, 1)
    toc1 = time.time()
    print(time.time() - tic)
    nodei = N - 1
    nodej = N - 2
    NSP, NSPrelevance = FindNearlySPNodes(G, nodei, nodej, RelevanceSimTimes=1000)
    print(len(NSP))
    print("time:", time.time() - toc1)
    plt.hist(NSPrelevance)
    plt.show()


def GeodiscPRAUC(Edindex,betaindex,ExternalSimutime):
    """
    :param ED: average degree
    :param beta: parameter to control the clustering coefficient
    :return: ED and beta
    """
    N = 10000
    ED_list = [5, 7, 10, 15, 20, 50, 100]  # Expected degrees
    ED = ED_list[Edindex]
    print("ED:",ED)
    CC_list = [0.2, 0.25, 0.3, 0.35, 0.4]  # Clustering coefficients
    CC = CC_list[betaindex]
    beta_list = [[3.575,3.69375,4.05,4.7625,5.57128906],
                [3.21875,3.575,4.05,4.525,5.38085938],
                [3.21875,3.575,4.05,4.525,5.38085938],
                [3.21875,3.575,4.05,4.525,5.19042969],
                [3.21875,3.575,4.05,4.525,5.38085938],
                [3.1,3.575,4.05,4.525,5.19042969],
                [3.1,3.45625,3.93125,4.525,5.19042969]]
    beta = beta_list[Edindex][betaindex]
    print("beta:", beta)
    PRAUC_nodepair = [] # save the PRAUC for each node pair, we selected 1000 node pair intotal
    rg = RandomGenerator(-12)
    for i in range(random.randint(0,100)):
        rg.ran1()

    G, CoorTheta, CoorPhi = SphericalSoftRGG(N, ED, beta, rg)
    while abs(nx.average_clustering(G)-CC)>0.1:
        G, CoorTheta, CoorPhi = SphericalSoftRGG(N, ED, beta, rg)

    print("We have a network now!")
    nodepair_num = 10
    # Random select nodepair_num nodes in the largest connected component
    components = list(nx.connected_components(G))
    largest_component = max(components, key=len)
    nodes = list(largest_component)
    unique_pairs = set(tuple(sorted(pair)) for pair in itertools.combinations(nodes, 2))
    possible_num_nodepair = len(unique_pairs)
    if possible_num_nodepair > nodepair_num:
        random_pairs = random.sample(sorted(unique_pairs), nodepair_num)
    else:
        random_pairs = random.sample(sorted(unique_pairs), possible_num_nodepair)
    count = 0
    for nodepair in random_pairs:
        count = count + 1
        print(count)
        nodei = nodepair[0]
        nodej = nodepair[1]

        thetaSource = CoorTheta[nodei]
        phiSource = CoorPhi[nodei]
        thetaEnd = CoorTheta[nodej]
        phiEnd = CoorPhi[nodej]

        # Find nearly shortest path nodes
        NearlySPNodelist, Noderelevance = FindNearlySPNodes(G, nodei, nodej)

        # Create label array
        Label_med = np.zeros(N)
        Label_med[NearlySPNodelist] = 1  # True cases
        distance_med = np.zeros(N)

        # Calculate distances to geodesic
        for NodeC in range(0, N):
            if NodeC not in [nodei, nodej]:
                thetaMed = CoorTheta[NodeC]
                phiMed = CoorPhi[NodeC]
                dist = dist_to_geodesic_S2(thetaMed, phiMed, thetaSource, phiSource, thetaEnd, phiEnd)
                distance_med[NodeC] = dist

        # Remove source and target nodes from consideration
        Label_med = np.delete(Label_med, [nodei, nodej])
        distance_med = np.delete(distance_med, [nodei, nodej])
        distance_score = [1/x for x in distance_med]
        # Calculate precision-recall curve and AUC
        precisions, recalls, _ = precision_recall_curve(Label_med, distance_score)
        AUCWithoutNornodeij = auc(recalls, precisions)

        # Store AUC values
        PRAUC_nodepair.append(AUCWithoutNornodeij)

    # Calculate means and standard deviations of AUC
    # AUCWithoutnorMean = np.mean(PRAUC_nodepair[~np.isnan(PRAUC_nodepair)])
    # AUCWithoutnorStd = np.std(PRAUC_nodepair[~np.isnan(PRAUC_nodepair)])

    PRAUCName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\AUCED{EDn}Beta{betan}PYSimu{ST}.txt".format(EDn=ED, betan=beta,ST=ExternalSimutime)
    np.savetxt(PRAUCName,PRAUC_nodepair)

    print("Mean AUC Without Normalization:", np.mean(PRAUC_nodepair))
    print("Standard Deviation of AUC Without Normalization:", np.std(PRAUC_nodepair))


# def PlotPRAUC():


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # TestFindNSPnodes()
    # ED = sys.argv[1]
    # beta = sys.argv[2]
    # ExternalSimutime = sys.argv[3]
    # GeodiscPRAUC(int(ED),int(beta),int(ExternalSimutime))

    GeodiscPRAUC(0, 0,1)




