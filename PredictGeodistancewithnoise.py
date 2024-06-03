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
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score

from NearlyShortestPathPredict import FindNearlySPNodes, FindNearlySPNodesRemoveSpecficLink, nodeNSPfrequency
from PredictGeodistanceVsRGG import NSPnodes_inRGG_with_coordinates
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


def PredictGeodistanceVsRGG_withnoise_givennodepair_difflength(noise_amplitude,theta_A, phi_A, theta_B, phi_B, ExternalSimutime):
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


    FileNetworkName = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\RGG\\Geolength\\PredictGeoVsRGGnetworkGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    nx.write_edgelist(G, FileNetworkName)
    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\RGG\\Geolength\\PredictGeoVsRGGnetworkCoorGeolen{le}Simu{ST}.txt".format(
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

    FileASPName = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\RGG\\Geolength\\PredictGeoVsRGGASPGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    np.savetxt(FileASPName, AllSPlist, fmt="%i")

    # All shortest path node
    AllSPNode = set()
    for path in AllSPlist:
        AllSPNode.update(path)
    AllSPNode.discard(nodei)
    AllSPNode.discard(nodej)
    FileASPNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\RGG\\Geolength\\PredictGeoVsRGGASPNodeGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    np.savetxt(FileASPNodeName, list(AllSPNode), fmt="%i")
    tic = time.time()
    # Nearly shortest path node(Actual Case)
    NSPNode, relevance = FindNearlySPNodes(G, nodei, nodej)
    print("time for finding NSP", time.time() - tic)
    print("NSP num", len(NSPNode))
    FileNSPNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\RGG\\Geolength\\PredictGeoVsRGGNSPNodeGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    np.savetxt(FileNSPNodeName, NSPNode, fmt="%i")
    FileNodeRelevanceName = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\RGG\\Geolength\\PredictGeoVsRGGRelevanceGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    np.savetxt(FileNodeRelevanceName, relevance, fmt="%.3f")

    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\RGG\\Geolength\\PredictGeoVsRGGnetworkCoorGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    with open(FileNetworkCoorName, "w") as file:
        for data1, data2 in zip(Coortheta, Coorphi):
            file.write(f"{data1}\t{data2}\n")

    # Create label array
    Label_med = np.zeros(N)
    Label_med[NSPNode] = 1  # True cases

    # add noise into theta and phi
    Coortheta = add_uniform_random_noise_to_coordinates(Coortheta, noise_amplitude,"theta")
    Coorphi = add_uniform_random_noise_to_coordinates(Coorphi, noise_amplitude,"phi")
    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\RGG\\Geolength\\PredictGeoVsRGGnetworkCoorwithnoiseGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    with open(FileNetworkCoorName, "w") as file:
        for data1, data2 in zip(Coortheta, Coorphi):
            file.write(f"{data1}\t{data2}\n")

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
    FileGeodistanceName = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\RGG\\Geolength\\PredictGeoVsRGGGeoDistancewithnoiseGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    np.savetxt(FileGeodistanceName, list(Geodistance.values()), fmt="%.8f")

    # Generate an RGG with the coordinates and predict it
    NSPNodeList_RGG = NSPnodes_inRGG_with_coordinates(N, avg, rg, Coortheta, Coorphi, nodei, nodej)
    Predicted_truecase_num = len(NSPNodeList_RGG)

    PredictNSPNodeList_RGG = np.zeros(N)
    PredictNSPNodeList_RGG[NSPNodeList_RGG] = 1  # True cases

    precision_RGG = precision_score(Label_med, PredictNSPNodeList_RGG)
    recall_RGG = recall_score(Label_med, PredictNSPNodeList_RGG)
    print("precsion_RGG:", precision_RGG)
    print("recall_RGG:", recall_RGG)

    # Predict nsp nodes use distance, where top Predicted_truecase_num nodes will be regarded as predicted nsp according to distance form the geodesic
    Geodistance = sorted(Geodistance.items(), key=lambda kv: (kv[1], kv[0]))
    Geodistance = Geodistance[:Predicted_truecase_num+2]
    Top100closednode = [t[0] for t in Geodistance]
    Top100closednode = [n for n in Top100closednode if n not in [nodei, nodej]]
    FileTop100closedNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\RGG\\Geolength\\PredictGeoVsRGGPredictedTrueNSPNodeByGeodistanceGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    np.savetxt(FileTop100closedNodeName, Top100closednode, fmt="%i")

    NSPNodeList_Geo = np.zeros(N)
    NSPNodeList_Geo[Top100closednode] = 1  # True cases
    precision_Geo = precision_score(Label_med, NSPNodeList_Geo)
    recall_Geo = recall_score(Label_med, NSPNodeList_Geo)
    print("precsion_Geo:", precision_Geo)
    print("recall_Geo:", recall_Geo)


def PredictGeodistanceVsRGG_withnoise(Edindex, betaindex, noiseindex, ExternalSimutime):
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

    noise_amplitude_list = [0.1, 0.5, 2]
    noise_amplitude = noise_amplitude_list[noiseindex]
    print("noise amplitude:", noise_amplitude)


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
    FileNetworkName = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\RGG\\NetworkED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta,no = noise_amplitude, ST=ExternalSimutime)
    nx.write_edgelist(G, FileNetworkName)
    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\RGG\\CoorED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta,no = noise_amplitude, ST=ExternalSimutime)
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
    filename_selecetednodepair = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\RGG\\SelecetedNodepairED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta,no = noise_amplitude, ST=ExternalSimutime)
    np.savetxt(filename_selecetednodepair, random_pairs, fmt="%i")

    # add noise into theta and phi
    CoorTheta = add_uniform_random_noise_to_coordinates(CoorTheta, noise_amplitude, "theta")
    CoorPhi = add_uniform_random_noise_to_coordinates(CoorPhi, noise_amplitude, "phi")
    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\FrequencyReconstruction\\CoorwithNoiseED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    with open(FileNetworkCoorName, "w") as file:
        for data1, data2 in zip(CoorTheta, CoorPhi):
            file.write(f"{data1}\t{data2}\n")


    for nodepair in random_pairs:
        count = count + 1
        print("Simunodepair:", count)
        nodei = nodepair[0]
        nodej = nodepair[1]

        tic = time.time()
        # Find nearly shortest path nodes
        NearlySPNodelist, _ = FindNearlySPNodes(G, nodei, nodej)
        NSPnum_nodepair.append(len(NearlySPNodelist))
        toc  = time.time()-tic
        print("NSP finding time:", toc)

        # Create label array
        Label_med = np.zeros(N)
        Label_med[NearlySPNodelist] = 1  # True cases

        thetaSource = CoorTheta[nodei]
        phiSource = CoorPhi[nodei]
        thetaEnd = CoorTheta[nodej]
        phiEnd = CoorPhi[nodej]
        geodistance_between_nodepair.append(distS2(thetaSource, phiSource, thetaEnd, phiEnd))

        Geodistance = {}
        for NodeC in range(N):
            if NodeC in [nodei, nodej]:
                Geodistance[NodeC] = 0
            else:
                thetaMed = CoorTheta[NodeC]
                phiMed = CoorPhi[NodeC]
                dist = dist_to_geodesic_S2(thetaMed, phiMed, thetaSource, phiSource, thetaEnd, phiEnd)
                Geodistance[NodeC] = dist

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

    precision_RGG_Name = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\RGG\\PrecisionRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta,no = noise_amplitude, ST=ExternalSimutime)
    np.savetxt(precision_RGG_Name, Precision_RGG_nodepair)

    recall_RGG_Name = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\RGG\\RecallRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta,no = noise_amplitude, ST=ExternalSimutime)
    np.savetxt(recall_RGG_Name, Recall_RGG_nodepair)

    precision_Geodis_Name = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\RGG\\PrecisionGeodisED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta,no = noise_amplitude, ST=ExternalSimutime)
    np.savetxt(precision_Geodis_Name, Precision_Geodis_nodepair)

    recall_Geodis_Name = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\RGG\\RecallGeodisED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta,no = noise_amplitude, ST=ExternalSimutime)
    np.savetxt(recall_Geodis_Name, Recall_Geodis_nodepair)

    NSPnum_nodepairName = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\RGG\\NSPNumED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta,no = noise_amplitude, ST=ExternalSimutime)
    np.savetxt(NSPnum_nodepairName, NSPnum_nodepair, fmt="%i")

    geodistance_between_nodepair_Name = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\RGG\\GeodistanceBetweenTwoNodesED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta,no = noise_amplitude, ST=ExternalSimutime)
    np.savetxt(geodistance_between_nodepair_Name, geodistance_between_nodepair)

    print("Mean Pre RGG:", np.mean(Precision_RGG_nodepair))
    print("Mean Recall RGG:", np.mean(Recall_RGG_nodepair))
    print("Mean Pre Geodistance:", np.mean(Precision_Geodis_nodepair))
    print("Mean Recall Geodistance:", np.mean(Recall_Geodis_nodepair))
    # print("Standard Deviation of AUC Without Normalization:", np.std(PRAUC_nodepair))


def PredictGeodistanceVsfrequency_withnoise_givennodepair_difflength(noise_amplitude,theta_A, phi_A, theta_B, phi_B, ExternalSimutime):
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

    FileNetworkName = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\FrequencyReconstruction\\Geolength\\ControlGroupnetworkGeolen{le}Simu{ST}beta{b}.txt".format(
        le=geo_length, ST=ExternalSimutime, b=beta)
    nx.write_edgelist(G, FileNetworkName)
    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\FrequencyReconstruction\\Geolength\\ControlGroupnetworkCoorGeolen{le}Simu{ST}beta{b}.txt".format(
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

    FileASPName = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\FrequencyReconstruction\\Geolength\\ControlGroupASPGeolen{le}Simu{ST}beta{b}.txt".format(
        le=geo_length, ST=ExternalSimutime, b=beta)
    np.savetxt(FileASPName, AllSPlist, fmt="%i")

    # All shortest path node
    AllSPNode = set()
    for path in AllSPlist:
        AllSPNode.update(path)
    AllSPNode.discard(nodei)
    AllSPNode.discard(nodej)
    FileASPNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\FrequencyReconstruction\\Geolength\\ControlGroupASPNodeGeolen{le}Simu{ST}beta{b}.txt".format(
        le=geo_length, ST=ExternalSimutime, b=beta)
    np.savetxt(FileASPNodeName, list(AllSPNode), fmt="%i")
    tic = time.time()
    # Nearly shortest path node
    NSPNode, relevance = FindNearlySPNodes(G, nodei, nodej)
    print("time for finding NSP", time.time() - tic)
    print("NSP num", len(NSPNode))
    FileNSPNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\FrequencyReconstruction\\Geolength\\ControlGroupNSPNodeGeolen{le}Simu{ST}beta{b}.txt".format(
        le=geo_length, ST=ExternalSimutime, b=beta)
    np.savetxt(FileNSPNodeName, NSPNode, fmt="%i")
    FileNodeRelevanceName = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\FrequencyReconstruction\\Geolength\\ControlGroupRelevanceGeolen{le}Simu{ST}beta{b}.txt".format(
        le=geo_length, ST=ExternalSimutime, b=beta)
    np.savetxt(FileNodeRelevanceName, relevance, fmt="%.3f")

    # Create label array
    Label_med = np.zeros(N)
    Label_med[NSPNode] = 1  # True cases

    # add noise into theta and phi
    Coortheta = add_uniform_random_noise_to_coordinates(Coortheta, noise_amplitude, "theta")
    Coorphi = add_uniform_random_noise_to_coordinates(Coorphi, noise_amplitude, "phi")
    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\FrequencyReconstruction\\Geolength\\PredictGeoVsRGGnetworkCoorwithnoiseGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    with open(FileNetworkCoorName, "w") as file:
        for data1, data2 in zip(Coortheta, Coorphi):
            file.write(f"{data1}\t{data2}\n")

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
    FileGeodistanceName = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\FrequencyReconstruction\\Geolength\\PredictGeoVsRGGGeoDistancewithnoiseGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    np.savetxt(FileGeodistanceName, list(Geodistance.values()), fmt="%.8f")

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
    print("PRAUC_geo", AUCWithoutNornodeij)

    # Calculate precision-recall curve and AUC for control group
    node_fre = nodeNSPfrequency(N, avg, beta, rg, Coortheta, Coorphi, nodei, nodej)
    FileNodefreName = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\FrequencyReconstruction\\Geolength\\ControlGroupNodefreGeolen{le}Simu{ST}beta{b}.txt".format(
        le=geo_length, ST=ExternalSimutime, b=beta)
    np.savetxt(FileNodefreName, node_fre)

    node_fre = np.delete(node_fre, [nodei, nodej])
    precisionsfre, recallsfre, _ = precision_recall_curve(Label_med, node_fre)
    AUCfrenodeij = auc(recallsfre, precisionsfre)
    print(AUCfrenodeij)


def PredictGeodistanceVsfrequency_withnoise(Edindex, betaindex, noiseindex, ExternalSimutime):
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

    noise_amplitude_list = [0.1, 0.5, 2]
    noise_amplitude = noise_amplitude_list[noiseindex]
    print("noise amplitude:", noise_amplitude)

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
    FileNetworkName = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\FrequencyReconstruction\\NetworkED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta,no = noise_amplitude, ST=ExternalSimutime)
    nx.write_edgelist(G, FileNetworkName)
    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\FrequencyReconstruction\\CoorED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta,no = noise_amplitude, ST=ExternalSimutime)
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
    filename_selecetednodepair = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\FrequencyReconstruction\\SelecetedNodepairED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta,no = noise_amplitude, ST=ExternalSimutime)
    np.savetxt(filename_selecetednodepair, random_pairs, fmt="%i")


    # add noise into theta and phi
    CoorTheta = add_uniform_random_noise_to_coordinates(CoorTheta, noise_amplitude, "theta")
    CoorPhi = add_uniform_random_noise_to_coordinates(CoorPhi, noise_amplitude, "phi")
    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\FrequencyReconstruction\\CoorwithNoiseED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta,no = noise_amplitude, ST=ExternalSimutime)
    with open(FileNetworkCoorName, "w") as file:
        for data1, data2 in zip(CoorTheta, CoorPhi):
            file.write(f"{data1}\t{data2}\n")


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

    PRAUCName = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\FrequencyReconstruction\\AUCED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta,no = noise_amplitude, ST=ExternalSimutime)
    np.savetxt(PRAUCName, PRAUC_nodepair)

    FrePRAUCName = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\FrequencyReconstruction\\ControlFreAUCED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta,no = noise_amplitude, ST=ExternalSimutime)
    np.savetxt(FrePRAUCName, PRAUC_fre_controlgroup_nodepair)

    NSPnum_nodepairName = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\FrequencyReconstruction\\NSPNumED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta,no = noise_amplitude, ST=ExternalSimutime)
    np.savetxt(NSPnum_nodepairName, NSPnum_nodepair, fmt="%i")

    geodistance_between_nodepair_Name = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\FrequencyReconstruction\\GeodistanceBetweenTwoNodesED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta,no = noise_amplitude, ST=ExternalSimutime)
    np.savetxt(geodistance_between_nodepair_Name, geodistance_between_nodepair)

    print("Mean AUC Without Normalization:", np.mean(PRAUC_nodepair))
    print("Mean AUC CONTROL:", np.mean(PRAUC_fre_controlgroup_nodepair))
    # print("Standard Deviation of AUC Without Normalization:", np.std(PRAUC_nodepair))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # PredictGeodistanceVsRGG_withnoise_givennodepair_difflength(0.1, 8*math.pi/16, 0, 9*math.pi/16, 0, int(0))

    # PredictGeodistanceVsRGG_withnoise(0, 0, 0, 0)
    # ED = sys.argv[1]
    # beta = sys.argv[2]
    # noise = sys.argv[3]
    # ExternalSimutime = sys.argv[4]
    # PredictGeodistanceVsRGG_withnoise(int(ED), int(beta), int(noise), int(ExternalSimutime))


    # PredictGeodistanceVsfrequency_withnoise_givennodepair_difflength(0.1, 8*math.pi/16, 0, 9*math.pi/16, 0, int(0))

    # TEST
    N = 10000
    nodei = N-2
    nodej = N-1
    theta_A = 8 * math.pi / 16
    phi_A = 0
    theta_B = 9 * math.pi / 16
    phi_B = 0
    ExternalSimutime = 0
    geo_length = distS2(theta_A, phi_A, theta_B, phi_B)
    beta = 4
    FileNSPNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\FrequencyReconstruction\\Geolength\\ControlGroupNSPNodeGeolen{le}Simu{ST}beta{b}.txt".format(
        le=geo_length, ST=ExternalSimutime, b=beta)
    NSPNode = np.loadtxt(FileNSPNodeName, dtype=int)

    # Create label array
    Label_med = np.zeros(N)
    Label_med[NSPNode] = 1  # True cases

    FileGeodistanceName = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\FrequencyReconstruction\\Geolength\\PredictGeoVsRGGGeoDistancewithnoiseGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    distance = np.loadtxt(FileGeodistanceName)

    # Remove source and target nodes from consideration
    Label_med = np.delete(Label_med, [nodei, nodej])
    distance_med = np.delete(distance, [nodei, nodej])
    distance_score = [1 / x for x in distance_med]
    # Calculate precision-recall curve and AUC
    precisions, recalls, _ = precision_recall_curve(Label_med, distance_score)
    AUCWithoutNornodeij = auc(recalls, precisions)
    print("PRAUC_geo", AUCWithoutNornodeij)

    # Calculate precision-recall curve and AUC for control group

    FileNodefreName = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\FrequencyReconstruction\\Geolength\\ControlGroupNodefreGeolen{le}Simu{ST}beta{b}.txt".format(
        le=geo_length, ST=ExternalSimutime, b=beta)
    node_fre = np.loadtxt(FileNodefreName)

    node_fre = np.delete(node_fre, [nodei, nodej])
    precisionsfre, recallsfre, _ = precision_recall_curve(Label_med, node_fre)
    AUCfrenodeij = auc(recallsfre, precisionsfre)
    print(AUCfrenodeij)



