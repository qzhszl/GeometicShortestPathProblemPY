# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 29-5-2024
Generate the graph, remove links, blur node coordinates:  x = x + E(A), y = y + E(A),
where E ~ Unif(0,A), A is noise amplitude. Or do it in a more “kosher” way, uniformly place it within a 2D circle of radius A.

For the node pair ij:
	a) find shortest path nodes using distance to geodesic (with blurred node coordinates).
	b) find shortest path nodes by reconstructing the graph.

Use the same parameter combinations as before.
Vary noise magnitude A, see what happens to predictions.

"""
import itertools
import sys
import time

import numpy as np
import networkx as nx
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score

from NearlyShortestPathPredict import FindNearlySPNodes, FindNearlySPNodesRemoveSpecficLink
from SphericalSoftRandomGeomtricGraph import RandomGenerator, SphericalRandomGeometricGraph, distS2, \
    SphericalRGGwithGivenNode, dist_to_geodesic_S2, SphericalSoftRGGwithGivenNode, SphericalSoftRGG
from main import find_nonzero_indices


def add_uniform_random_noise_to_coordinates(lst, noise_amplitude, coor_type="theta"):
    angle = [x + random.uniform(-noise_amplitude, noise_amplitude) for x in lst]
    if coor_type=="theta":
        angle = [2 * math.pi-x if x > math.pi else x for x in angle]
        angle = [-x if x < 0 else x for x in angle]
    else:
        angle = [x - 2 * math.pi if x > 2 * math.pi else x for x in angle]
        angle = [2 * math.pi+x if x < 0 else x for x in angle]

    # angle  = [x for ]
    return angle


def PredictGeodistanceVsRGG_withnoise(Edindex, betaindex, ExternalSimutime):
    """
    :param Edindex: average degree
    :param betaindex: parameter to control the clustering coefficient
    :return: PRAUC control and test simu for diff ED and beta
    Only four data point, beta can reach 100
    """
    N = 10000
    ED_list = [5, 20]  # Expected degrees
    ED = ED_list[Edindex]
    print("ED:", ED)

    beta_list = [4, 100]
    beta = beta_list[betaindex]
    print("beta:", beta)
    Precision_RGG_nodepair = []  # save the precision_RGG for each node pair, we selected 100 node pair in total
    Recall_RGG_nodepair = []  # we selected 100 node pair in total
    Precision_Geodis_nodepair = []
    Recall_Geodis_nodepair = []
    NSPnum_nodepair = []  # save the Number of nearly shortest path for each node pair
    geodistance_between_nodepair = []  # save the geodeisc length between each node pair

    random.seed(ExternalSimutime)
    rg = RandomGenerator(-12)
    for i in range(random.randint(0, 100)):
        rg.ran1()

    G, CoorTheta, CoorPhi = SphericalSoftRGG(N, ED, beta, rg)

    print("We have a network now!")
    FileNetworkName = "D:\\data\\geometric shortest path problem\\SphericalRGG\\PRAUC\\NetworkED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    nx.write_edgelist(G, FileNetworkName)
    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\SphericalRGG\\PRAUC\\CoorED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    with open(FileNetworkCoorName, "w") as file:
        for data1, data2 in zip(CoorTheta, CoorPhi):
            file.write(f"{data1}\t{data2}\n")

    nodepair_num = 5
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
    components = []
    largest_component = []
    nodes = []
    unique_pairs = []
    unique_pairs = []
    filename_selecetednodepair = "D:\\data\\geometric shortest path problem\\SphericalRGG\\PRAUC\\SelecetedNodepairED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(filename_selecetednodepair, random_pairs, fmt="%i")

    for nodepair in random_pairs:
        count = count + 1
        print("Simunodepair:", count)
        nodei = nodepair[0]
        nodej = nodepair[1]

        thetaSource = CoorTheta[nodei]
        phiSource = CoorPhi[nodei]
        thetaEnd = CoorTheta[nodej]
        phiEnd = CoorPhi[nodej]
        geodistance_between_nodepair.append(distS2(thetaSource, phiSource, thetaEnd, phiEnd))

        tic = time.time()
        # Find nearly shortest path nodes
        NearlySPNodelist, _ = FindNearlySPNodes(G, nodei, nodej)
        NSPnum_nodepair.append(len(NearlySPNodelist))
        toc  = time.time()-tic
        print("NSP finding time:", toc)

        Geodistance = {}
        for NodeC in range(N):
            if NodeC in [nodei, nodej]:
                Geodistance[NodeC] = 0
            else:
                thetaMed = CoorTheta[NodeC]
                phiMed = CoorPhi[NodeC]
                dist = dist_to_geodesic_S2(thetaMed, phiMed, thetaSource, phiSource, thetaEnd, phiEnd)
                Geodistance[NodeC] = dist

        # Create label array
        Label_med = np.zeros(N)
        Label_med[NearlySPNodelist] = 1  # True cases
        distance_med = np.zeros(N)

        # Generate an RGG with the coordinates and predict it
        NSPNodeList_RGG = NSPnodes_inRGG_with_coordinates(N, ED, rg, CoorTheta, CoorPhi, nodei, nodej)
        Predicted_truecase_num = len(NSPNodeList_RGG)
        toc2 = time.time() - toc
        print("RGG generate time:", toc2)

        PredictNSPNodeList_RGG = np.zeros(N)
        PredictNSPNodeList_RGG[NSPNodeList_RGG] = 1  # True cases

        precision_RGG = precision_score(Label_med, PredictNSPNodeList_RGG)
        recall_RGG = recall_score(Label_med, PredictNSPNodeList_RGG)

        # Store precision and recall values
        Precision_RGG_nodepair.append(precision_RGG)
        Recall_RGG_nodepair.append(recall_RGG)

        # Calculate precision-recall curve and AUC for control group
        # Predict nsp nodes use distance, where top Predicted_truecase_num nodes will be regarded as predicted nsp according to distance form the geodesic
        Geodistance = sorted(Geodistance.items(), key=lambda kv: (kv[1], kv[0]))
        Geodistance = Geodistance[:Predicted_truecase_num + 2]
        Top100closednode = [t[0] for t in Geodistance]
        Top100closednode = [n for n in Top100closednode if n not in [nodei, nodej]]
        NSPNodeList_Geo = np.zeros(N)
        NSPNodeList_Geo[Top100closednode] = 1  # True cases
        precision_Geo = precision_score(Label_med, NSPNodeList_Geo)
        recall_Geo = recall_score(Label_med, NSPNodeList_Geo)

        # Store precision and recall values
        Precision_Geodis_nodepair.append(precision_Geo)
        Recall_Geodis_nodepair.append(recall_Geo)

    # Calculate means and standard deviations of AUC
    # AUCWithoutnorMean = np.mean(PRAUC_nodepair[~np.isnan(PRAUC_nodepair)])
    # AUCWithoutnorStd = np.std(PRAUC_nodepair[~np.isnan(PRAUC_nodepair)])

    precision_RGG_Name = "D:\\data\\geometric shortest path problem\\SphericalRGG\\PRAUC\\PrecisionRGGED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(precision_RGG_Name, Precision_RGG_nodepair)

    recall_RGG_Name = "D:\\data\\geometric shortest path problem\\SphericalRGG\\PRAUC\\RecallRGGED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(recall_RGG_Name, Recall_RGG_nodepair)

    precision_Geodis_Name = "D:\\data\\geometric shortest path problem\\SphericalRGG\\PRAUC\\PrecisionGeodisED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(precision_Geodis_Name, Precision_Geodis_nodepair)

    recall_Geodis_Name = "D:\\data\\geometric shortest path problem\\SphericalRGG\\PRAUC\\RecallGeodisED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(recall_Geodis_Name, Recall_Geodis_nodepair)

    NSPnum_nodepairName = "D:\\data\\geometric shortest path problem\\SphericalRGG\\PRAUC\\NSPNumED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(NSPnum_nodepairName, NSPnum_nodepair, fmt="%i")

    geodistance_between_nodepair_Name = "D:\\data\\geometric shortest path problem\\SphericalRGG\\PRAUC\\GeodistanceBetweenTwoNodesED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(geodistance_between_nodepair_Name, geodistance_between_nodepair)

    print("Mean Pre RGG:", np.mean(Precision_RGG_nodepair))
    print("Mean Recall RGG:", np.mean(Recall_RGG_nodepair))
    print("Mean Pre Geodistance:", np.mean(Precision_Geodis_nodepair))
    print("Mean Recall Geodistance:", np.mean(Recall_Geodis_nodepair))
    # print("Standard Deviation of AUC Without Normalization:", np.std(PRAUC_nodepair))



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 示例列表
    my_list = [1, 2, 3, 4]

    upper_bound = 5

    # 对列表的每个元素添加均匀分布的随机数
    new_list = add_uniform_random_noise_to_coordinates(my_list, upper_bound, coor_type="phi")
    print(new_list)
