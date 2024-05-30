# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 17-5-2024
"""
import itertools
import math
import random
import time

# import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from NearlyShortestPathPredict import FindNearlySPNodes, FindNearlySPNodesRemoveSpecficLink
from SphericalSoftRandomGeomtricGraph import RandomGenerator, SphericalSoftRGGwithGivenNode, SphericalSoftRGG, \
    dist_to_geodesic_S2, distS2, loadNodeSSRGG
from sklearn.metrics import precision_recall_curve, auc
import sys
# import seaborn as sns
# import pandas as pd

# from main import find_nonnan_indices


def nodeNSPfrequency(N, avg, beta, rg, Coortheta, Coorphi, nodei, nodej):
    """
        Given nodes of the SRGG.
        For i = 1 to 100 independent realizations:
    	Reconstruct G_i using original connection probabilities. Find shortest path nodes SPi.
        Characterize each node by the frequency it belong to the shortest path.
        Use this frequency to computer AUPRC.
        :return: Frequency to predict NSP
        """
    NodeFrequency = np.zeros(N)  # Initialize node frequency
    for i in range(100):
        print("simu time:", i)
        tic = time.time()
        H, angle1, angle2 = SphericalSoftRGG(N, avg, beta, rg, Coortheta=Coortheta, Coorphi=Coorphi)
        FreNodeList, _ = FindNearlySPNodesRemoveSpecficLink(H, nodei, nodej, Linkremoveratio=0.5)
        for node in FreNodeList:
            NodeFrequency[node] += 1
        print(time.time() - tic)
    return NodeFrequency

def nodeSPfrequency(N, avg, beta, rg, Coortheta, Coorphi, nodei, nodej):
    """
        Given nodes of the SRGG.
        For i = 1 to 100 independent realizations:
    	Reconstruct G_i using original connection probabilities. Find shortest path nodes SPi.
        Characterize each node by the frequency it belong to the shortest path.
        Use this frequency to computer AUPRC.
        :return: Frequency to predict NSP
        """
    NodeFrequency = np.zeros(N)  # Initialize node frequency
    for i in range(100):
        print("fresimu time:", i)
        tic = time.time()
        H, angle1, angle2 = SphericalSoftRGG(N, avg, beta, rg, Coortheta=Coortheta, Coorphi=Coorphi)

        try:
            shortest_paths = nx.all_shortest_paths(H, nodei, nodej)
            PNodeList = set()  # Use a set to keep unique nodes
            count = 0
            for path in shortest_paths:
                PNodeList.update(path)
                count += 1
                if count > 1000000:
                    PNodeList = set()
                    break
            print("pathlength", len(path))
            print("pathnum",count)
        except nx.NetworkXNoPath:
            PNodeList = set()  # If there's no path, continue with an empty set
        # time31 = time.time()
        # print("timeallsp0",time31-time3)
        # Remove the starting and ending nodes from the list
        PNodeList.discard(nodei)
        PNodeList.discard(nodej)

        for node in PNodeList:
            NodeFrequency[node] += 1
        print("time",time.time() - tic)
    return NodeFrequency


def frequency_controlgroup_PRAUC(Edindex, betaindex, ExternalSimutime):
    """
        :param Edindex: average degree
        :param betaindex: parameter to control the clustering coefficient
        :return: PRAUC control and test simu for diff ED and beta
        """
    N = 10000
    ED_list = [5, 7, 10, 15, 20, 50, 100]  # Expected degrees
    ED = ED_list[Edindex]
    print("ED:", ED)
    CC_list = [0.2, 0.25, 0.3, 0.35, 0.4]  # Clustering coefficients
    CC = CC_list[betaindex]
    beta_list = [[3.2781, 3.69375, 4.05, 4.7625, 5.57128906],
                 [3.21875, 3.575, 4.05, 4.525, 5.38085938],
                 [3.21875, 3.575, 4.05, 4.525, 5.38085938],
                 [3.21875, 3.575, 4.05, 4.525, 5.19042969],
                 [3.21875, 3.575, 4.05, 4.525, 5.38085938],
                 [3.1, 3.575, 4.05, 4.525, 5.19042969],
                 [3.1, 3.45625, 3.93125, 4.525, 5.19042969]]
    beta = beta_list[Edindex][betaindex]
    print("beta:", beta)
    PRAUC_nodepair = []  # save the PRAUC for each node pair, we selected 100 node pair in total
    PRAUC_fre_controlgroup_nodepair = []  # save the PRAUC for each node pair, we selected 100 node pair in total
    NSPnum_nodepair = []  # save the Number of nearly shortest path for each node pair
    geodistance_between_nodepair = []  # save the geodeisc length between each node pair

    random.seed(ExternalSimutime)
    rg = RandomGenerator(-12)
    for i in range(random.randint(0, 100)):
        rg.ran1()

    G, CoorTheta, CoorPhi = SphericalSoftRGG(N, ED, beta, rg)
    while abs(nx.average_clustering(G) - CC) > 0.1:
        G, CoorTheta, CoorPhi = SphericalSoftRGG(N, ED, beta, rg)
    print("We have a network now!")
    FileNetworkName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\compare\\NetworkED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    nx.write_edgelist(G, FileNetworkName)
    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\compare\\CoorED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    with open(FileNetworkCoorName, "w") as file:
        for data1, data2 in zip(CoorTheta, CoorPhi):
            file.write(f"{data1}\t{data2}\n")

    nodepair_num = 2
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
    filename_selecetednodepair = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\compare\\SelecetedNodepairED{EDn}Beta{betan}PYSimu{ST}.txt".format(
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

        # tic = time.time()
        # Find nearly shortest path nodes
        NearlySPNodelist, Noderelevance = FindNearlySPNodes(G, nodei, nodej)
        NSPnum_nodepair.append(len(NearlySPNodelist))
        # toc  = time.time()-tic
        # print("NSP finding time:", toc)

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
        distance_score = [1 / x for x in distance_med]
        # Calculate precision-recall curve and AUC
        precisions, recalls, _ = precision_recall_curve(Label_med, distance_score)
        AUCWithoutNornodeij = auc(recalls, precisions)

        # Store AUC values
        PRAUC_nodepair.append(AUCWithoutNornodeij)

        # Calculate precision-recall curve and AUC for control group
        node_fre = nodeNSPfrequency(N, ED, beta, rg, CoorTheta, CoorPhi, nodei, nodej)
        node_fre = np.delete(node_fre, [nodei, nodej])
        precisionsfre, recallsfre, _ = precision_recall_curve(Label_med, node_fre)
        AUCfrenodeij = auc(recallsfre, precisionsfre)
        PRAUC_fre_controlgroup_nodepair.append(AUCfrenodeij)

    # Calculate means and standard deviations of AUC
    # AUCWithoutnorMean = np.mean(PRAUC_nodepair[~np.isnan(PRAUC_nodepair)])
    # AUCWithoutnorStd = np.std(PRAUC_nodepair[~np.isnan(PRAUC_nodepair)])

    PRAUCName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\compare\\AUCED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(PRAUCName, PRAUC_nodepair)

    FrePRAUCName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\compare\\ControlFreAUCED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(FrePRAUCName, PRAUC_fre_controlgroup_nodepair)

    NSPnum_nodepairName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\compare\\NSPNumED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(NSPnum_nodepairName, NSPnum_nodepair, fmt="%i")

    geodistance_between_nodepair_Name = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\compare\\GeodistanceBetweenTwoNodesED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(geodistance_between_nodepair_Name, geodistance_between_nodepair)

    print("Mean AUC Without Normalization:", np.mean(PRAUC_nodepair))
    print("Mean AUC CONTROL:", np.mean(PRAUC_fre_controlgroup_nodepair))
    # print("Standard Deviation of AUC Without Normalization:", np.std(PRAUC_nodepair))


def frequency_controlgroup_PRAUC_highbeta(Edindex, betaindex, ExternalSimutime):
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
    print("ExternalSimutime",ExternalSimutime)
    PRAUC_nodepair = []  # save the PRAUC for each node pair, we selected 100 node pair in total
    PRAUC_fre_controlgroup_nodepair = []  # save the PRAUC for each node pair, we selected 100 node pair in total
    NSPnum_nodepair = []  # save the Number of nearly shortest path for each node pair
    geodistance_between_nodepair = []  # save the geodeisc length between each node pair

    random.seed(ExternalSimutime)
    rg = RandomGenerator(-12)
    for i in range(random.randint(0, 100)):
        rg.ran1()

    G, CoorTheta, CoorPhi = SphericalSoftRGG(N, ED, beta, rg)

    print("We have a network now!")
    FileNetworkName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\compare\\NetworkED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    nx.write_edgelist(G, FileNetworkName)
    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\compare\\CoorED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    with open(FileNetworkCoorName, "w") as file:
        for data1, data2 in zip(CoorTheta, CoorPhi):
            file.write(f"{data1}\t{data2}\n")

    nodepair_num = 2
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
    filename_selecetednodepair = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\compare\\SelecetedNodepairED{EDn}Beta{betan}PYSimu{ST}.txt".format(
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

        # tic = time.time()
        # Find nearly shortest path nodes
        NearlySPNodelist, Noderelevance = FindNearlySPNodes(G, nodei, nodej)
        NSPnum_nodepair.append(len(NearlySPNodelist))
        # toc  = time.time()-tic
        # print("NSP finding time:", toc)

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
        distance_score = [1 / x for x in distance_med]
        # Calculate precision-recall curve and AUC
        precisions, recalls, _ = precision_recall_curve(Label_med, distance_score)
        AUCWithoutNornodeij = auc(recalls, precisions)

        # Store AUC values
        PRAUC_nodepair.append(AUCWithoutNornodeij)

        # Calculate precision-recall curve and AUC for control group
        node_fre = nodeNSPfrequency(N, ED, beta, rg, CoorTheta, CoorPhi, nodei, nodej)
        node_fre = np.delete(node_fre, [nodei, nodej])
        precisionsfre, recallsfre, _ = precision_recall_curve(Label_med, node_fre)
        AUCfrenodeij = auc(recallsfre, precisionsfre)
        PRAUC_fre_controlgroup_nodepair.append(AUCfrenodeij)

    # Calculate means and standard deviations of AUC
    # AUCWithoutnorMean = np.mean(PRAUC_nodepair[~np.isnan(PRAUC_nodepair)])
    # AUCWithoutnorStd = np.std(PRAUC_nodepair[~np.isnan(PRAUC_nodepair)])

    PRAUCName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\compare\\AUCED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(PRAUCName, PRAUC_nodepair)

    FrePRAUCName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\compare\\ControlFreAUCED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(FrePRAUCName, PRAUC_fre_controlgroup_nodepair)

    NSPnum_nodepairName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\compare\\NSPNumED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(NSPnum_nodepairName, NSPnum_nodepair, fmt="%i")

    geodistance_between_nodepair_Name = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\compare\\GeodistanceBetweenTwoNodesED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(geodistance_between_nodepair_Name, geodistance_between_nodepair)

    print("Mean AUC Without Normalization:", np.mean(PRAUC_nodepair))
    print("Mean AUC CONTROL:", np.mean(PRAUC_fre_controlgroup_nodepair))
    # print("Standard Deviation of AUC Without Normalization:", np.std(PRAUC_nodepair))


def frequency_controlgroup_PRAUC_bySP(Edindex, betaindex, ExternalSimutime):
    """
     We use shortest path rather than nearly shortest path
    :param Edindex: average degree
    :param betaindex: parameter to control the clustering coefficient
    :return: PRAUC control and test simu for diff ED and beta
    """
    N = 10000
    ED_list = [5, 20]  # Expected degrees
    ED = ED_list[Edindex]
    print("ED:", ED)
    # CC_list = [0.2, 0.25, 0.3, 0.35, 0.4]  # Clustering coefficients
    # CC = CC_list[betaindex]
    # beta_list = [[3.2781, 3.69375, 4.05, 4.7625, 5.57128906],
    #              [3.21875, 3.575, 4.05, 4.525, 5.38085938],
    #              [3.21875, 3.575, 4.05, 4.525, 5.38085938],
    #              [3.21875, 3.575, 4.05, 4.525, 5.19042969],
    #              [3.21875, 3.575, 4.05, 4.525, 5.38085938],
    #              [3.1, 3.575, 4.05, 4.525, 5.19042969],
    #              [3.1, 3.45625, 3.93125, 4.525, 5.19042969]]
    # beta = beta_list[Edindex][betaindex]
    beta_list = [4, 100]
    beta = beta_list[betaindex]
    print("beta:", beta)
    PRAUC_nodepair = []  # save the PRAUC for each node pair, we selected 100 node pair in total
    PRAUC_fre_controlgroup_nodepair = []  # save the PRAUC for each node pair, we selected 100 node pair in total
    SPnum_nodepair = []  # save the Number of nearly shortest path for each node pair
    geodistance_between_nodepair = []  # save the geodeisc length between each node pair

    random.seed(ExternalSimutime)
    rg = RandomGenerator(-12)
    for i in range(random.randint(0, 100)):
        rg.ran1()

    G, CoorTheta, CoorPhi = SphericalSoftRGG(N, ED, beta, rg)
    # while abs(nx.average_clustering(G) - CC) > 0.1:
    #     G, CoorTheta, CoorPhi = SphericalSoftRGG(N, ED, beta, rg)
    print("We have a network now!")
    FileNetworkName = "D:\\data\\geometric shortest path problem\\SSRGG\\ShortestPathasActualCase\\NetworkED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    nx.write_edgelist(G, FileNetworkName)
    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\SSRGG\\ShortestPathasActualCase\\CoorED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    with open(FileNetworkCoorName, "w") as file:
        for data1, data2 in zip(CoorTheta, CoorPhi):
            file.write(f"{data1}\t{data2}\n")

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
    components = []
    largest_component = []
    nodes = []
    unique_pairs = []
    unique_pairs = []
    filename_selecetednodepair = "D:\\data\\geometric shortest path problem\\SSRGG\\ShortestPathasActualCase\\SelecetedNodepairED{EDn}Beta{betan}PYSimu{ST}.txt".format(
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

        # tic = time.time()
        # Find  shortest path nodes
        try:
            shortest_paths = nx.all_shortest_paths(G, nodei, nodej)
            PNodeList = set()  # Use a set to keep unique nodes
            spcount = 0
            for path in shortest_paths:
                PNodeList.update(path)
                spcount += 1
                if spcount > 1000000:
                    PNodeList = set()
                    break
            print("pathlength", len(path))
            print("pathnum",spcount)
        except nx.NetworkXNoPath:
            PNodeList = set()  # If there's no path, continue with an empty set
        # time31 = time.time()
        # print("timeallsp0",time31-time3)
        # Remove the starting and ending nodes from the list
        PNodeList.discard(nodei)
        PNodeList.discard(nodej)



        SPnum_nodepair.append(len(PNodeList))
        # toc  = time.time()-tic
        # print("NSP finding time:", toc)

        # Create label array
        Label_med = np.zeros(N)
        for i in PNodeList:
            Label_med[i] = 1  # True cases
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
        distance_score = [1 / x for x in distance_med]
        # Calculate precision-recall curve and AUC
        precisions, recalls, _ = precision_recall_curve(Label_med, distance_score)
        AUCWithoutNornodeij = auc(recalls, precisions)

        # Store AUC values
        PRAUC_nodepair.append(AUCWithoutNornodeij)

        # Calculate precision-recall curve and AUC for control group
        node_fre = nodeSPfrequency(N, ED, beta, rg, CoorTheta, CoorPhi, nodei, nodej)
        node_fre = np.delete(node_fre, [nodei, nodej])
        precisionsfre, recallsfre, _ = precision_recall_curve(Label_med, node_fre)
        AUCfrenodeij = auc(recallsfre, precisionsfre)
        PRAUC_fre_controlgroup_nodepair.append(AUCfrenodeij)

    # Calculate means and standard deviations of AUC
    # AUCWithoutnorMean = np.mean(PRAUC_nodepair[~np.isnan(PRAUC_nodepair)])
    # AUCWithoutnorStd = np.std(PRAUC_nodepair[~np.isnan(PRAUC_nodepair)])

    PRAUCName = "D:\\data\\geometric shortest path problem\\SSRGG\\ShortestPathasActualCase\\AUCED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(PRAUCName, PRAUC_nodepair)

    FrePRAUCName = "D:\\data\\geometric shortest path problem\\SSRGG\\ShortestPathasActualCase\\ControlFreAUCED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(FrePRAUCName, PRAUC_fre_controlgroup_nodepair)

    NSPnum_nodepairName = "D:\\data\\geometric shortest path problem\\SSRGG\\ShortestPathasActualCase\\NSPNumED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(NSPnum_nodepairName, SPnum_nodepair, fmt="%i")

    geodistance_between_nodepair_Name = "D:\\data\\geometric shortest path problem\\SSRGG\\ShortestPathasActualCase\\GeodistanceBetweenTwoNodesED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(geodistance_between_nodepair_Name, geodistance_between_nodepair)

    print("Mean AUC Without Normalization:", np.mean(PRAUC_nodepair))
    print("Mean AUC CONTROL:", np.mean(PRAUC_fre_controlgroup_nodepair))
    # print("Standard Deviation of AUC Without Normalization:", np.std(PRAUC_nodepair))


def plot_frequency_controlgroup_PRAUC_bySP():

    PRAUC_matrix = np.zeros((2, 2))
    PRAUC_std_matrix = np.zeros((2, 2))
    PRAUC_fre_matrix = np.zeros((2, 2))
    PRAUC_fre_std_matrix = np.zeros((2, 2))

    for EDindex in [0, 1]:
        ED_list = [5, 20]  # Expected degrees
        ED = ED_list[EDindex]
        print("ED:", ED)

        for betaindex in [0, 1]:
            beta_list = [4,100]
            beta = beta_list[betaindex]
            print(beta)
            PRAUC_list = []
            PRAUC_fre_list = []
            for ExternalSimutime in range(10):
                PRAUCName = "D:\\data\\geometric shortest path problem\\SSRGG\\ShortestPathasActualCase\\AUCED{EDn}Beta{betan}PYSimu{ST}.txt".format(
                    EDn=ED, betan=beta, ST=ExternalSimutime)
                PRAUC_list_10times = np.loadtxt(PRAUCName)
                PRAUC_list.extend(PRAUC_list_10times)

                FrePRAUCName = "D:\\data\\geometric shortest path problem\\SSRGG\\ShortestPathasActualCase\\ControlFreAUCED{EDn}Beta{betan}PYSimu{ST}.txt".format(
                    EDn=ED, betan=beta, ST=ExternalSimutime)
                PRAUC_fre_list_10times = np.loadtxt(FrePRAUCName)
                PRAUC_fre_list.extend(PRAUC_fre_list_10times)

            nonzero_indices_geo = find_nonnan_indices(PRAUC_list)
            PRAUC_list = [PRAUC_list[x] for x in nonzero_indices_geo]

            mean_PRAUC = np.mean(PRAUC_list)
            PRAUC_matrix[EDindex][betaindex] = mean_PRAUC
            PRAUC_std_matrix[EDindex][betaindex] = np.std(PRAUC_list)
            print(mean_PRAUC)

            nonzero_indices_geo = find_nonnan_indices(PRAUC_fre_list)
            PRAUC_fre_list = [PRAUC_fre_list[x] for x in nonzero_indices_geo]
            mean_fre_PRAUC = np.mean(PRAUC_fre_list)
            PRAUC_fre_matrix[EDindex][betaindex] = mean_fre_PRAUC
            PRAUC_fre_std_matrix[EDindex][betaindex] = np.std(PRAUC_list)
            print(mean_fre_PRAUC)


    plt.figure()
    df = pd.DataFrame(PRAUC_matrix,
                      index=[5,20],  # DataFrame的行标签设置为大写字母
                      columns=[4,100])  # 设置DataFrame的列标签
    sns.heatmap(data=df, vmin=0,vmax=1, annot=True, fmt=".2f", cbar=True,
                cbar_kws={'label': 'AUPRC'})
    plt.title("Distance of node from the geodesic")
    plt.xlabel("beta")
    plt.ylabel("average degree")
    plt.savefig(
        "D:\\data\\geometric shortest path problem\\SSRGG\\ShortestPathasActualCase\\GeoDisPRAUCHeatmapNSP0_1LinkRemove.pdf",
        format='pdf', bbox_inches='tight', dpi=600)
    plt.close()

    plt.figure()
    df = pd.DataFrame(PRAUC_fre_matrix,
                      index=[5, 20],  # DataFrame的行标签设置为大写字母
                      columns=[4,100])  # 设置DataFrame的列标签
    sns.heatmap(data=df, vmin=0, vmax=1, annot=True, fmt=".2f", cbar=True,
                cbar_kws={'label': 'AUPRC'})
    plt.title("Frequency")
    plt.xlabel("beta")
    plt.ylabel("average degree")
    plt.savefig(
        "D:\\data\\geometric shortest path problem\\SSRGG\\ShortestPathasActualCase\\FreGeoDisPRAUCHeatmapNSP0_1LinkRemove.pdf",
        format='pdf', bbox_inches='tight', dpi=600)
    plt.close()


def frequency_controlgroup_PRAUC_givennodepair():
    """
    For a given node pair
    :return:
    """
    # Input data parameters
    N = 10000
    avg = 5
    beta = 3.69375
    rg = RandomGenerator(-12)
    # for _ in range(random.randint(0,100)):
    #     rg.ran1()

    # Network and coordinates
    G, Coortheta, Coorphi = SphericalSoftRGGwithGivenNode(N, avg, beta, rg, math.pi / 4, 0, 3 * math.pi / 8, 0)
    print("LinkNum:", G.number_of_edges())
    print("AveDegree:", G.number_of_edges() * 2 / G.number_of_nodes())
    print("ClusteringCoefficient:", nx.average_clustering(G))

    FileNetworkName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\ControlGroupnetwork.txt"
    nx.write_edgelist(G, FileNetworkName)
    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\ControlGroupnetworkCoor.txt"
    with open(FileNetworkCoorName, "w") as file:
        for data1, data2 in zip(Coortheta, Coorphi):
            file.write(f"{data1}\t{data2}\n")

    # test form here
    # nodepair_num = 1
    # # Random select nodepair_num nodes in the largest connected component
    # components = list(nx.connected_components(G))
    # largest_component = max(components, key=len)
    # nodes = list(largest_component)
    # unique_pairs = set(tuple(sorted(pair)) for pair in itertools.combinations(nodes, 2))
    # possible_num_nodepair = len(unique_pairs)
    # if possible_num_nodepair > nodepair_num:
    #     random_pairs = random.sample(sorted(unique_pairs), nodepair_num)
    # else:
    #     random_pairs = random.sample(sorted(unique_pairs), possible_num_nodepair)
    count = 0
    # for nodepair in random_pairs:
    #     count = count + 1
    #     print(count,"Simu")
    #     nodei = nodepair[0]
    #     nodej = nodepair[1]
    #     print("nodei",nodei)
    #     print(nodej)
    components = []
    largest_component = []
    nodes = []
    unique_pairs = []
    unique_pairs = []
    # test end here

    nodei = N - 2
    nodej = N - 1

    print("Node Geo distance", distS2(Coortheta[nodei], Coorphi[nodei], Coortheta[nodej], Coorphi[nodej]))
    # All shortest paths
    AllSP = nx.all_shortest_paths(G, nodei, nodej)
    AllSPlist = list(AllSP)
    print("SPnum", len(AllSPlist))
    print("SPlength", len(AllSPlist[0]))

    FileASPName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\ControlGroupASP.txt"
    np.savetxt(FileASPName, AllSPlist, fmt="%i")

    # All shortest path node
    AllSPNode = set()
    for path in AllSPlist:
        AllSPNode.update(path)
    AllSPNode.discard(nodei)
    AllSPNode.discard(nodej)
    FileASPNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\ControlGroupASPNode.txt"
    np.savetxt(FileASPNodeName, list(AllSPNode), fmt="%i")
    tic = time.time()
    # Nearly shortest path node
    NSPNode, relevance = FindNearlySPNodes(G, nodei, nodej)
    print("time for finding NSP", time.time() - tic)
    print("NSP num", len(NSPNode))
    FileNSPNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\ControlGroupNSPNode.txt"
    np.savetxt(FileNSPNodeName, NSPNode, fmt="%i")
    FileNodeRelevanceName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\ControlGroupRelevance.txt"
    np.savetxt(FileNodeRelevanceName, relevance, fmt="%.3f")

    # Geodesic
    thetaSource = Coortheta[nodei]
    phiSource = Coorphi[nodei]
    thetaEnd = Coortheta[nodej]
    phiEnd = Coorphi[nodej]

    Geodistance = {}
    for NodeC in range(N):
        if NodeC in [nodei, nodej]:
            Geodistance[NodeC] = 0
        else:
            thetaMed = Coortheta[NodeC]
            phiMed = Coorphi[NodeC]
            dist = dist_to_geodesic_S2(thetaMed, phiMed, thetaSource, phiSource, thetaEnd, phiEnd)
            Geodistance[NodeC] = dist
    FileGeodistanceName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\ControlGroupGeoDistance.txt"
    np.savetxt(FileGeodistanceName, list(Geodistance.values()), fmt="%.8f")
    Geodistance = sorted(Geodistance.items(), key=lambda kv: (kv[1], kv[0]))
    Geodistance = Geodistance[:102]
    Top100closednode = [t[0] for t in Geodistance]
    Top100closednode = [n for n in Top100closednode if n not in [nodei, nodej]]

    FileTop100closedNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\ControlGroupTop100closedNode.txt"
    np.savetxt(FileTop100closedNodeName, Top100closednode, fmt="%i")

    # Create label array
    Label_med = np.zeros(N)
    Label_med[NSPNode] = 1  # True cases
    distance_med = np.zeros(N)

    # Calculate distances to geodesic
    for NodeC in range(0, N):
        if NodeC not in [nodei, nodej]:
            thetaMed = Coortheta[NodeC]
            phiMed = Coorphi[NodeC]
            dist = dist_to_geodesic_S2(thetaMed, phiMed, thetaSource, phiSource, thetaEnd, phiEnd)
            distance_med[NodeC] = dist

    # Remove source and target nodes from consideration
    Label_med = np.delete(Label_med, [nodei, nodej])
    distance_med = np.delete(distance_med, [nodei, nodej])
    distance_score = [1 / x for x in distance_med]
    # Calculate precision-recall curve and AUC
    precisions, recalls, _ = precision_recall_curve(Label_med, distance_score)
    AUCWithoutNornodeij = auc(recalls, precisions)
    print("PRAUC", AUCWithoutNornodeij)

    # Calculate precision-recall curve and AUC for control group
    node_fre = nodeNSPfrequency(N, avg, beta, rg, Coortheta, Coorphi, nodei, nodej)
    FileNodefreName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\ControlGroupNodefre.txt"
    np.savetxt(FileNodefreName, node_fre)

    node_fre = np.delete(node_fre, [nodei, nodej])
    precisionsfre, recallsfre, _ = precision_recall_curve(Label_med, node_fre)
    AUCfrenodeij = auc(recallsfre, precisionsfre)
    print(AUCfrenodeij)


def frequency_controlgroup_PRAUC_givennodepair_diffgeolength(theta_A, phi_A, theta_B, phi_B, ExternalSimutime):
    """
    For a given node pair. This function is designed for the cluster
    :param theta_A:
    :param phi_A:
    :param theta_B:
    :param phi_B:
    :param ExternalSimutime:
    :return:
    """
    # Input data parameters
    N = 10000
    avg = 5
    beta = 4
    # random.seed(ExternalSimutime)
    rg = RandomGenerator(-12)
    for _ in range(random.randint(0, 100)):
        rg.ran1()

    # Network and coordinates
    G, Coortheta, Coorphi = SphericalSoftRGGwithGivenNode(N, avg, beta, rg, theta_A, phi_A, theta_B, phi_B)
    print("LinkNum:", G.number_of_edges())
    print("AveDegree:", G.number_of_edges() * 2 / G.number_of_nodes())
    print("ClusteringCoefficient:", nx.average_clustering(G))
    geo_length = distS2(theta_A, phi_A, theta_B, phi_B)
    print("Geo Length:", geo_length / math.pi)

    FileNetworkName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\Geolength\\ControlGroupnetworkGeolen{le}Simu{ST}beta{b}.txt".format(
        le=geo_length, ST=ExternalSimutime, b=beta)
    nx.write_edgelist(G, FileNetworkName)
    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\Geolength\\ControlGroupnetworkCoorGeolen{le}Simu{ST}beta{b}.txt".format(
        le=geo_length, ST=ExternalSimutime, b=beta)
    with open(FileNetworkCoorName, "w") as file:
        for data1, data2 in zip(Coortheta, Coorphi):
            file.write(f"{data1}\t{data2}\n")

    nodei = N - 2
    nodej = N - 1

    print("Node Geo distance", distS2(Coortheta[nodei], Coorphi[nodei], Coortheta[nodej], Coorphi[nodej]))
    # All shortest paths
    AllSP = nx.all_shortest_paths(G, nodei, nodej)
    AllSPlist = list(AllSP)
    print("SPnum", len(AllSPlist))
    print("SPlength", len(AllSPlist[0]))

    FileASPName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\Geolength\\ControlGroupASPGeolen{le}Simu{ST}beta{b}.txt".format(
        le=geo_length, ST=ExternalSimutime, b=beta)
    np.savetxt(FileASPName, AllSPlist, fmt="%i")

    # All shortest path node
    AllSPNode = set()
    for path in AllSPlist:
        AllSPNode.update(path)
    AllSPNode.discard(nodei)
    AllSPNode.discard(nodej)
    FileASPNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\Geolength\\ControlGroupASPNodeGeolen{le}Simu{ST}beta{b}.txt".format(
        le=geo_length, ST=ExternalSimutime, b=beta)
    np.savetxt(FileASPNodeName, list(AllSPNode), fmt="%i")
    tic = time.time()
    # Nearly shortest path node
    NSPNode, relevance = FindNearlySPNodes(G, nodei, nodej)
    print("time for finding NSP", time.time() - tic)
    print("NSP num", len(NSPNode))
    FileNSPNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\Geolength\\ControlGroupNSPNodeGeolen{le}Simu{ST}beta{b}.txt".format(
        le=geo_length, ST=ExternalSimutime, b=beta)
    np.savetxt(FileNSPNodeName, NSPNode, fmt="%i")
    FileNodeRelevanceName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\Geolength\\ControlGroupRelevanceGeolen{le}Simu{ST}beta{b}.txt".format(
        le=geo_length, ST=ExternalSimutime, b=beta)
    np.savetxt(FileNodeRelevanceName, relevance, fmt="%.3f")

    # Geodesic
    thetaSource = Coortheta[nodei]
    phiSource = Coorphi[nodei]
    thetaEnd = Coortheta[nodej]
    phiEnd = Coorphi[nodej]

    Geodistance = {}
    for NodeC in range(N):
        if NodeC in [nodei, nodej]:
            Geodistance[NodeC] = 0
        else:
            thetaMed = Coortheta[NodeC]
            phiMed = Coorphi[NodeC]
            dist = dist_to_geodesic_S2(thetaMed, phiMed, thetaSource, phiSource, thetaEnd, phiEnd)
            Geodistance[NodeC] = dist
    FileGeodistanceName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\Geolength\\ControlGroupGeoDistanceGeolen{le}Simu{ST}beta{b}.txt".format(
        le=geo_length, ST=ExternalSimutime, b=beta)
    np.savetxt(FileGeodistanceName, list(Geodistance.values()), fmt="%.8f")
    Geodistance = sorted(Geodistance.items(), key=lambda kv: (kv[1], kv[0]))
    Geodistance = Geodistance[:102]
    Top100closednode = [t[0] for t in Geodistance]
    Top100closednode = [n for n in Top100closednode if n not in [nodei, nodej]]

    FileTop100closedNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\Geolength\\ControlGroupTop100closedNodeGeolen{le}Simu{ST}beta{b}.txt".format(
        le=geo_length, ST=ExternalSimutime, b=beta)
    np.savetxt(FileTop100closedNodeName, Top100closednode, fmt="%i")

    # Create label array
    Label_med = np.zeros(N)
    Label_med[NSPNode] = 1  # True cases
    distance_med = np.zeros(N)

    # Calculate distances to geodesic
    for NodeC in range(0, N):
        if NodeC not in [nodei, nodej]:
            thetaMed = Coortheta[NodeC]
            phiMed = Coorphi[NodeC]
            dist = dist_to_geodesic_S2(thetaMed, phiMed, thetaSource, phiSource, thetaEnd, phiEnd)
            distance_med[NodeC] = dist

    # Remove source and target nodes from consideration
    Label_med = np.delete(Label_med, [nodei, nodej])
    distance_med = np.delete(distance_med, [nodei, nodej])
    distance_score = [1 / x for x in distance_med]
    # Calculate precision-recall curve and AUC
    precisions, recalls, _ = precision_recall_curve(Label_med, distance_score)
    AUCWithoutNornodeij = auc(recalls, precisions)
    print("PRAUC", AUCWithoutNornodeij)

    # Calculate precision-recall curve and AUC for control group
    node_fre = nodeNSPfrequency(N, avg, beta, rg, Coortheta, Coorphi, nodei, nodej)
    FileNodefreName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\Geolength\\ControlGroupNodefreGeolen{le}Simu{ST}beta{b}.txt".format(
        le=geo_length, ST=ExternalSimutime, b=beta)
    np.savetxt(FileNodefreName, node_fre)

    node_fre = np.delete(node_fre, [nodei, nodej])
    precisionsfre, recallsfre, _ = precision_recall_curve(Label_med, node_fre)
    AUCfrenodeij = auc(recallsfre, precisionsfre)
    print(AUCfrenodeij)


def frequency_controlgroup_PRAUC_givennodepair_diffgeolengthhighbeta(theta_A, phi_A, theta_B, phi_B, ExternalSimutime):
    """
        For a given node pair. Compared with the last one ,the beta is very high here
        :param theta_A:
        :param phi_A:
        :param theta_B:
        :param phi_B:
        :param ExternalSimutime:
        :return:
        """
    # Input data parameters
    N = 10000
    avg = 5
    beta = 100
    # random.seed(ExternalSimutime)
    rg = RandomGenerator(-12)
    for _ in range(random.randint(0, 100)):
        rg.ran1()

    # Network and coordinates
    G, Coortheta, Coorphi = SphericalSoftRGGwithGivenNode(N, avg, beta, rg, theta_A, phi_A, theta_B, phi_B)
    print("LinkNum:", G.number_of_edges())
    print("AveDegree:", G.number_of_edges() * 2 / G.number_of_nodes())
    print("ClusteringCoefficient:", nx.average_clustering(G))
    geo_length = distS2(theta_A, phi_A, theta_B, phi_B)
    print("Geo Length:", geo_length / math.pi)

    FileNetworkName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\Geolength\\HighbetaControlGroupnetworkGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    nx.write_edgelist(G, FileNetworkName)
    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\Geolength\\HighbetaControlGroupnetworkCoorGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    with open(FileNetworkCoorName, "w") as file:
        for data1, data2 in zip(Coortheta, Coorphi):
            file.write(f"{data1}\t{data2}\n")

    nodei = N - 2
    nodej = N - 1

    print("Node Geo distance", distS2(Coortheta[nodei], Coorphi[nodei], Coortheta[nodej], Coorphi[nodej]))
    # All shortest paths
    AllSP = nx.all_shortest_paths(G, nodei, nodej)
    AllSPlist = list(AllSP)
    print("SPnum", len(AllSPlist))
    print("SPlength", len(AllSPlist[0]))

    FileASPName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\Geolength\\HighbetaControlGroupASPGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    np.savetxt(FileASPName, AllSPlist, fmt="%i")

    # All shortest path node
    AllSPNode = set()
    for path in AllSPlist:
        AllSPNode.update(path)
    AllSPNode.discard(nodei)
    AllSPNode.discard(nodej)
    FileASPNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\Geolength\\HighbetaControlGroupASPNodeGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    np.savetxt(FileASPNodeName, list(AllSPNode), fmt="%i")
    tic = time.time()
    # Nearly shortest path node
    NSPNode, relevance = FindNearlySPNodes(G, nodei, nodej)
    print("time for finding NSP", time.time() - tic)
    print("NSP num", len(NSPNode))
    FileNSPNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\Geolength\\HighbetaControlGroupNSPNodeGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    np.savetxt(FileNSPNodeName, NSPNode, fmt="%i")
    FileNodeRelevanceName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\Geolength\\HighbetaControlGroupRelevanceGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    np.savetxt(FileNodeRelevanceName, relevance, fmt="%.3f")

    # Geodesic
    thetaSource = Coortheta[nodei]
    phiSource = Coorphi[nodei]
    thetaEnd = Coortheta[nodej]
    phiEnd = Coorphi[nodej]

    Geodistance = {}
    for NodeC in range(N):
        if NodeC in [nodei, nodej]:
            Geodistance[NodeC] = 0
        else:
            thetaMed = Coortheta[NodeC]
            phiMed = Coorphi[NodeC]
            dist = dist_to_geodesic_S2(thetaMed, phiMed, thetaSource, phiSource, thetaEnd, phiEnd)
            Geodistance[NodeC] = dist
    FileGeodistanceName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\Geolength\\HighbetaControlGroupGeoDistanceGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    np.savetxt(FileGeodistanceName, list(Geodistance.values()), fmt="%.8f")
    Geodistance = sorted(Geodistance.items(), key=lambda kv: (kv[1], kv[0]))
    Geodistance = Geodistance[:102]
    Top100closednode = [t[0] for t in Geodistance]
    Top100closednode = [n for n in Top100closednode if n not in [nodei, nodej]]

    FileTop100closedNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\Geolength\\HighbetaControlGroupTop100closedNodeGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    np.savetxt(FileTop100closedNodeName, Top100closednode, fmt="%i")

    # Create label array
    Label_med = np.zeros(N)
    Label_med[NSPNode] = 1  # True cases
    distance_med = np.zeros(N)

    # Calculate distances to geodesic
    for NodeC in range(0, N):
        if NodeC not in [nodei, nodej]:
            thetaMed = Coortheta[NodeC]
            phiMed = Coorphi[NodeC]
            dist = dist_to_geodesic_S2(thetaMed, phiMed, thetaSource, phiSource, thetaEnd, phiEnd)
            distance_med[NodeC] = dist

    # Remove source and target nodes from consideration
    Label_med = np.delete(Label_med, [nodei, nodej])
    distance_med = np.delete(distance_med, [nodei, nodej])
    distance_score = [1 / x for x in distance_med]
    # Calculate precision-recall curve and AUC
    precisions, recalls, _ = precision_recall_curve(Label_med, distance_score)
    AUCWithoutNornodeij = auc(recalls, precisions)
    print("PRAUC", AUCWithoutNornodeij)

    # Calculate precision-recall curve and AUC for control group
    node_fre = nodeNSPfrequency(N, avg, beta, rg, Coortheta, Coorphi, nodei, nodej)
    FileNodefreName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\Geolength\\HighbetaControlGroupNodefreGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    np.savetxt(FileNodefreName, node_fre)

    node_fre = np.delete(node_fre, [nodei, nodej])
    precisionsfre, recallsfre, _ = precision_recall_curve(Label_med, node_fre)
    AUCfrenodeij = auc(recallsfre, precisionsfre)
    print(AUCfrenodeij)

def PlotPRAUCFrequency():
    """
    PRAUC of using geo distance to predict nearly shortest path nodes
    50% Links are randomly removed when computing the nearly shortest paths nodes
    X is average clustering coefficient of each node, while the Y axis is the average degree
    :return:
    """

    beta_list = [[3.2781, 3.69375, 4.05, 4.7625, 5.57128906],
                 [3.21875, 3.575, 4.05, 4.525, 5.38085938],
                 [3.21875, 3.575, 4.05, 4.525, 5.38085938],
                 [3.21875, 3.575, 4.05, 4.525, 5.19042969],
                 [3.21875, 3.575, 4.05, 4.525, 5.38085938],
                 [3.1, 3.575, 4.05, 4.525, 5.19042969],
                 [3.1, 3.45625, 3.93125, 4.525, 5.19042969]]
    PRAUC_matrix = np.zeros((7, 5))
    PRAUC_std_matrix = np.zeros((7, 5))
    for EDindex in [0, 5]:
        ED_list = [5, 7, 10, 15, 20, 50, 100]  # Expected degrees
        ED = ED_list[EDindex]
        print("ED:", ED)

        for betaindex in [0, 4]:
            CC_list = [0.2, 0.25, 0.3, 0.35, 0.4]  # Clustering coefficients
            CC = CC_list[betaindex]
            print(CC)
            beta = beta_list[EDindex][betaindex]
            print(beta)
            PRAUC_list = []
            for ExternalSimutime in range(50):
                PRAUCName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\compare\\ControlFreAUCED{EDn}Beta{betan}PYSimu{ST}.txt".format(
                    EDn=ED, betan=beta, ST=ExternalSimutime)
                PRAUC_list_10times = np.loadtxt(PRAUCName)
                PRAUC_list.extend(PRAUC_list_10times)
            PRAUC_list = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PRAUC_list))
            mean_PRAUC = np.mean(PRAUC_list)
            PRAUC_matrix[EDindex][betaindex] = mean_PRAUC
            print(mean_PRAUC)
            PRAUC_std_matrix[EDindex][betaindex] = np.std(PRAUC_list)
    print(PRAUC_matrix)
    PRAUC_matrix = np.reshape(PRAUC_matrix[PRAUC_matrix!=0],(2,2))
    print(PRAUC_matrix)

    PRAUC_std_matrix = np.reshape(PRAUC_std_matrix[PRAUC_std_matrix != 0], (2, 2))
    print("std:", PRAUC_std_matrix)
    # plt.imshow(PRAUC_matrix, cmap="viridis", aspect="auto")
    # plt.colorbar()  # 添加颜色条
    # plt.title("Heatmap Example")
    # plt.xlabel("Column")
    # plt.ylabel("Row")
    #
    # # 显示热力图
    # plt.show()

    plt.figure()
    df = pd.DataFrame(PRAUC_matrix,
                      index=[5,50],  # DataFrame的行标签设置为大写字母
                      columns=[0.2,0.4])  # 设置DataFrame的列标签
    sns.heatmap(data=df, vmin=0, annot=True, fmt=".2f", cbar=True,
                cbar_kws={'label': 'AUCPR'})
    plt.title("Frequency")
    plt.xlabel("clustering coefficient")
    plt.ylabel("average degree")
    plt.savefig(
        "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\compare\\ControlFrePRAUCHeatmapNSP0_1LinkRemove.pdf",
        format='pdf', bbox_inches='tight', dpi=600)
    plt.show()


def PlotPRAUCFrequency2():
    """
    PRAUC of using geo distance to predict nearly shortest path nodes
    50% Links are randomly removed when computing the nearly shortest paths nodes
    X is average clustering coefficient of each node, while the Y axis is the average degree
    :return:
    """

    beta_list = [[3.2781, 3.69375, 4.05, 4.7625, 5.57128906],
                 [3.21875, 3.575, 4.05, 4.525, 5.38085938],
                 [3.21875, 3.575, 4.05, 4.525, 5.38085938],
                 [3.21875, 3.575, 4.05, 4.525, 5.19042969],
                 [3.21875, 3.575, 4.05, 4.525, 5.38085938],
                 [3.1, 3.575, 4.05, 4.525, 5.19042969],
                 [3.1, 3.45625, 3.93125, 4.525, 5.19042969]]
    PRAUC_matrix = np.zeros((7, 5))
    PRAUC_std_matrix = np.zeros((7, 5))
    for EDindex in [0, 5]:
        ED_list = [5, 7, 10, 15, 20, 50, 100]  # Expected degrees
        ED = ED_list[EDindex]
        print("ED:", ED)

        for betaindex in [0, 4]:
            CC_list = [0.2, 0.25, 0.3, 0.35, 0.4]  # Clustering coefficients
            CC = CC_list[betaindex]
            print(CC)
            beta = beta_list[EDindex][betaindex]
            print(beta)
            PRAUC_list = []
            for ExternalSimutime in range(50):
                PRAUCName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\compare\\AUCED{EDn}Beta{betan}PYSimu{ST}.txt".format(
                    EDn=ED, betan=beta, ST=ExternalSimutime)
                PRAUC_list_10times = np.loadtxt(PRAUCName)
                PRAUC_list.extend(PRAUC_list_10times)
            PRAUC_list = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PRAUC_list))
            mean_PRAUC = np.mean(PRAUC_list)
            PRAUC_matrix[EDindex][betaindex] = mean_PRAUC
            PRAUC_std_matrix[EDindex][betaindex] = np.std(PRAUC_list)
            print(mean_PRAUC)


    PRAUC_matrix = np.reshape(PRAUC_matrix[PRAUC_matrix != 0], (2, 2))
    PRAUC_std_matrix = np.reshape(PRAUC_std_matrix[PRAUC_std_matrix != 0], (2, 2))
    print("std:", PRAUC_std_matrix)
    # plt.imshow(PRAUC_matrix, cmap="viridis", aspect="auto")
    # plt.colorbar()  # 添加颜色条
    # plt.title("Heatmap Example")
    # plt.xlabel("Column")
    # plt.ylabel("Row")
    #
    # # 显示热力图
    # plt.show()
    plt.figure()
    df = pd.DataFrame(PRAUC_matrix,
                      index=[5,50],  # DataFrame的行标签设置为大写字母
                      columns=[0.2,0.4])  # 设置DataFrame的列标签
    sns.heatmap(data=df, vmin=0,vmax=0.35, annot=True, fmt=".2f", cbar=True,
                cbar_kws={'label': 'AUCPR'})
    plt.title("Distance of node from the geodesic")
    plt.xlabel("clustering coefficient")
    plt.ylabel("average degree")
    plt.savefig(
        "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\compare\\GeoDisPRAUCHeatmapNSP0_1LinkRemove.pdf",
        format='pdf', bbox_inches='tight', dpi=600)
    plt.show()

def plot_AUPRC_geo_vs_fre_diff_length_geo():
   # Data Source:"D:\data\geometric shortest path problem\SSRGG\PRAUC\ControlGroup\Geolength\100realization"
   #  x: Geo length : (8*math.pi/16, 9*math.pi/16), (4*math.pi/8, 5*math.pi/8), (2*math.pi/4, 3*math.pi/4), (3*math.pi/4, math.pi/4)
   #  y: ave AUPRC (errorbar) of predicted by geo distance from the geodesic and the frequency
   #  return: a figure

    Geolength_list = [(8*math.pi/16, 9*math.pi/16), (4*math.pi/8, 5*math.pi/8), (2*math.pi/4, 3*math.pi/4), (3*math.pi/4, math.pi/4)]
    N=10000
    x=[1,2,3,4]
    AUPRC_geodis = []
    AUPRC_fre = []
    std_AUPRC_geodis = []
    std_AUPRC_fre = []

    for Nodepairindex in range(4):
        AUPRC_geodis_foronegeo = []
        AUPRC_fre_foronegeo = []
        (theta_A, theta_B) = Geolength_list[Nodepairindex]
        geo_length = distS2(theta_A, 0, theta_B, 0)
        print("Geo Length:", geo_length / math.pi)
        for ExternalSimutime in range(50):
            if (Nodepairindex, ExternalSimutime) not in [(0, 2),(2,7),(2,25),(2,28),(3,38),(3,39)]:
                nodei = N-2
                nodej = N-1

                FileNSPNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\Geolength\\100realization\\ControlGroupNSPNodeGeolen{le}Simu{ST}.txt".format(
                    le=geo_length, ST=ExternalSimutime)
                NSPNode = np.loadtxt(FileNSPNodeName,dtype=int)
                Label_med = np.zeros(N)
                Label_med[NSPNode] = 1  # True cases

                # Calculate distances to geodesic
                FileGeodistanceName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\Geolength\\100realization\\ControlGroupGeoDistanceGeolen{le}Simu{ST}.txt".format(
                    le=geo_length, ST=ExternalSimutime)
                distance_med = np.loadtxt(FileGeodistanceName)

                # Remove source and target nodes from consideration
                Label_med = np.delete(Label_med, [nodei, nodej])
                distance_med = np.delete(distance_med, [nodei, nodej])
                distance_score = [1 / x for x in distance_med]
                # Calculate precision-recall curve and AUC
                precisions, recalls, _ = precision_recall_curve(Label_med, distance_score)
                AUCWithoutNornodeij = auc(recalls, precisions)
                AUPRC_geodis_foronegeo.append(AUCWithoutNornodeij)


                # load precision-recall curve and AUC for control group
                FileNodefreName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\Geolength\\100realization\\ControlGroupNodefreGeolen{le}Simu{ST}.txt".format(
                    le=geo_length, ST=ExternalSimutime)
                node_fre = np.loadtxt(FileNodefreName)
                node_fre = np.delete(node_fre, [nodei, nodej])
                precisionsfre, recallsfre, _ = precision_recall_curve(Label_med, node_fre)
                AUCfrenodeij = auc(recallsfre, precisionsfre)
                AUPRC_fre_foronegeo.append(AUCfrenodeij)
        AUPRC_geodis.append(np.mean(AUPRC_geodis_foronegeo))
        AUPRC_fre.append(np.mean(AUPRC_fre_foronegeo))
        std_AUPRC_geodis.append(np.std(AUPRC_geodis_foronegeo))
        std_AUPRC_fre.append(np.std(AUPRC_fre_foronegeo))


    x_labels = ['pi/16', 'pi/8', 'pi/4', 'pi/2']

    # 绘制带有误差条的图
    plt.errorbar(x, AUPRC_geodis, yerr=std_AUPRC_geodis, fmt='o', capsize=5, capthick=1, ecolor=(0, 0.4470, 0.7410), label='By GeoDistance')
    plt.errorbar(x, AUPRC_fre, yerr=std_AUPRC_fre, fmt='o', capsize=5, capthick=1, ecolor=(0.8500, 0.3250, 0.0980),
                 label='By Frequency')
    # 设置x轴标签
    plt.xticks(x, x_labels)

    # 添加标题和标签

    plt.xlabel('Length of the geodesic')
    plt.ylabel('AUPRC')
    plt.legend()

    plt.savefig(
        "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\Geolength\\100realization\\AUPRC_geo_vs_fre_diff_length_geo.pdf",
        format='pdf', bbox_inches='tight', dpi=600)
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Nodepairindex = sys.argv[1]
    # ExternalSimutime = sys.argv[2]
    #
    # Geolength_list = [(8*math.pi/16, 9*math.pi/16), (4*math.pi/8, 5*math.pi/8), (2*math.pi/4, 3*math.pi/4), (3*math.pi/4, math.pi/4)]
    # (theta_A, theta_B) = Geolength_list[int(Nodepairindex)]
    # frequency_controlgroup_PRAUC_givennodepair_diffgeolength(theta_A, 0, theta_B, 0, int(ExternalSimutime))

    frequency_controlgroup_PRAUC_givennodepair_diffgeolength(8*math.pi/16, 0, 9*math.pi/16, 0, int(0))

    # PlotPRAUCFrequency()
    # PlotPRAUCFrequency2()

    # frequency_controlgroup_PRAUC_bySP(0, 0, 0)
    # ED = sys.argv[1]
    # beta = sys.argv[2]
    # ExternalSimutime = sys.argv[3]
    # frequency_controlgroup_PRAUC_bySP(int(ED),int(beta),int(ExternalSimutime))

    # frequency_controlgroup_PRAUC_highbeta(0, 0, 0)
    # ED = sys.argv[1]
    # beta = sys.argv[2]
    # ExternalSimutime = sys.argv[3]
    # frequency_controlgroup_PRAUC_highbeta(int(ED), int(beta), int(ExternalSimutime))


    # plot_AUPRC_geo_vs_fre_diff_length_geo()


    # plot_frequency_controlgroup_PRAUC_bySP()

    # frequency_controlgroup_PRAUC_givennodepair_diffgeolengthhighbeta(4*math.pi/8, 0, 5*math.pi/8, 0, 0)
