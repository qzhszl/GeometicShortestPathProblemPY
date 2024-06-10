# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 6-6-2024
Generate the graph, remove links, blur node coordinates:  x = x + E(A), y = y + E(A),
where E ~ Unif(0,A), A is noise amplitude. Or do it in a more “kosher” way, uniformly place it within a 2D circle of radius A.

For the node pair ij:
	a) find shortest path nodes using distance to geodesic (with blurred node coordinates).
	b) find shortest path nodes by reconstructing the graph.

Use the same parameter combinations as before.
Vary noise magnitude A, see what happens to predictions.
It is for Euclidean soft random geometric graph
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
import pandas as pd

from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score

from NearlyShortestPathPredict import FindNearlySPNodesRemoveSpecficLink
from PredictGeodistanceVsRGGR2 import NSPnodes_inRGG_with_coordinatesR2
from R2SRGG import R2SRGG_withgivennodepair, distR2, dist_to_geodesic_R2, R2SRGG
from FrequencyControlGroupR2 import nodeNSPfrequencyR2
from SphericalSoftRandomGeomtricGraph import RandomGenerator
from main import find_nonzero_indices


def add_uniform_random_noise_to_coordinates_R2(lst, noise_amplitude):
    coor = []
    for x in lst:
        x_noise = x + random.uniform(-noise_amplitude, noise_amplitude)
        if x_noise > 1 or x_noise < 0:
            x_noise = x
        coor.append(x_noise)
    return coor


def PredictGeodistanceVsRGG_withnoise_givennodepair_difflengthR2(noise_amplitude, x_A, y_A, x_B, y_B, ExternalSimutime):
    """
    For a given node pair. This function is designed for the cluster
    :param x_A:
    :param y_A:
    :param x_B:
    :param y_B:
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
    G, Coorx, Coory = R2SRGG_withgivennodepair(N, avg, beta, rg, x_A, y_A, x_B, y_B)
    print("LinkNum:", G.number_of_edges())
    print("AveDegree:", G.number_of_edges() * 2 / G.number_of_nodes())
    print("ClusteringCoefficient:", nx.average_clustering(G))
    geo_length = distR2(x_A, y_A, x_B, y_B)
    print("Geo Length:", geo_length)

    FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Noise\\Geolength\\PredictGeoVsRGGnetworkGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    nx.write_edgelist(G, FileNetworkName)
    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Noise\\Geolength\\PredictGeoVsRGGnetworkCoorGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    with open(FileNetworkCoorName, "w") as file:
        for data1, data2 in zip(Coorx, Coory):
            file.write(f"{data1}\t{data2}\n")

    nodei = N - 2
    nodej = N - 1

    print("Node Geo distance", distR2(Coorx[nodei], Coory[nodei], Coorx[nodej], Coory[nodej]))
    # All shortest paths
    AllSP = nx.all_shortest_paths(G, nodei, nodej)
    AllSPlist = list(AllSP)
    print("SPnum", len(AllSPlist))
    print("SPlength", len(AllSPlist[0]))

    FileASPName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Noise\\Geolength\\PredictGeoVsRGGASPGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    np.savetxt(FileASPName, AllSPlist, fmt="%i")

    # All shortest path node
    AllSPNode = set()
    for path in AllSPlist:
        AllSPNode.update(path)
    AllSPNode.discard(nodei)
    AllSPNode.discard(nodej)
    FileASPNodeName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Noise\\Geolength\\PredictGeoVsRGGASPNodeGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    np.savetxt(FileASPNodeName, list(AllSPNode), fmt="%i")
    tic = time.time()
    # Nearly shortest path node(Actual Case)
    NSPNode, relevance = FindNearlySPNodesRemoveSpecficLink(G, nodei, nodej, Linkremoveratio=0.1)
    print("time for finding NSP", time.time() - tic)
    print("NSP num", len(NSPNode))
    FileNSPNodeName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Noise\\Geolength\\PredictGeoVsRGGNSPNodeGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    np.savetxt(FileNSPNodeName, NSPNode, fmt="%i")
    FileNodeRelevanceName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Noise\\Geolength\\PredictGeoVsRGGRelevanceGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    np.savetxt(FileNodeRelevanceName, relevance, fmt="%.3f")

    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Noise\\Geolength\\PredictGeoVsRGGnetworkCoorGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    with open(FileNetworkCoorName, "w") as file:
        for data1, data2 in zip(Coorx, Coory):
            file.write(f"{data1}\t{data2}\n")

    # Create label array
    Label_med = np.zeros(N)
    Label_med[NSPNode] = 1  # True cases

    # add noise into theta and phi
    Coorx = add_uniform_random_noise_to_coordinates_R2(Coorx, noise_amplitude)
    Coory = add_uniform_random_noise_to_coordinates_R2(Coory, noise_amplitude)
    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Noise\\Geolength\\PredictGeoVsRGGnetworkCoorwithnoiseGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    with open(FileNetworkCoorName, "w") as file:
        for data1, data2 in zip(Coorx, Coory):
            file.write(f"{data1}\t{data2}\n")

    # Geodesic
    thetaSource = Coorx[nodei]
    phiSource = Coory[nodei]
    thetaEnd = Coorx[nodej]
    phiEnd = Coory[nodej]

    Geodistance = {}
    for NodeC in range(N):
        if NodeC in [nodei, nodej]:
            Geodistance[NodeC] = 0
        else:
            thetaMed = Coorx[NodeC]
            phiMed = Coory[NodeC]
            dist,_ = dist_to_geodesic_R2(thetaMed, phiMed, thetaSource, phiSource, thetaEnd, phiEnd)
            Geodistance[NodeC] = dist
    FileGeodistanceName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Noise\\Geolength\\PredictGeoVsRGGGeoDistancewithnoiseGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    np.savetxt(FileGeodistanceName, list(Geodistance.values()), fmt="%.8f")

    # Generate an RGG with the coordinates and predict it
    NSPNodeList_RGG = NSPnodes_inRGG_with_coordinatesR2(N, avg, rg, Coorx, Coory, nodei, nodej)
    Predicted_truecase_num = len(NSPNodeList_RGG)

    PredictNSPNodeList_RGG = np.zeros(N)
    PredictNSPNodeList_RGG[NSPNodeList_RGG] = 1  # True cases

    precision_RGG = precision_score(Label_med, PredictNSPNodeList_RGG)
    recall_RGG = recall_score(Label_med, PredictNSPNodeList_RGG)
    print("precsion_RGG:", precision_RGG)
    print("recall_RGG:", recall_RGG)

    # Predict nsp nodes use distance, where top Predicted_truecase_num nodes will be regarded as predicted nsp according to distance form the geodesic
    Geodistance = sorted(Geodistance.items(), key=lambda kv: (kv[1], kv[0]))
    Geodistance = Geodistance[:Predicted_truecase_num + 2]
    Top100closednode = [t[0] for t in Geodistance]
    Top100closednode = [n for n in Top100closednode if n not in [nodei, nodej]]
    FileTop100closedNodeName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Noise\\Geolength\\PredictGeoVsRGGPredictedTrueNSPNodeByGeodistanceGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    np.savetxt(FileTop100closedNodeName, Top100closednode, fmt="%i")

    NSPNodeList_Geo = np.zeros(N)
    NSPNodeList_Geo[Top100closednode] = 1  # True cases
    precision_Geo = precision_score(Label_med, NSPNodeList_Geo)
    recall_Geo = recall_score(Label_med, NSPNodeList_Geo)
    print("precsion_Geo:", precision_Geo)
    print("recall_Geo:", recall_Geo)


def PredictGeodistanceVsRGG_withnoiseR2(Edindex, betaindex, noiseindex, ExternalSimutime):
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

    noise_amplitude_list = [0.001, 0.01, 0.1]
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

    G, Coorx, Coory = R2SRGG(N, ED, beta, rg)

    print("We have a network now!")
    FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Noise\\NetworkED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    nx.write_edgelist(G, FileNetworkName)
    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Noise\\CoorED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    with open(FileNetworkCoorName, "w") as file:
        for data1, data2 in zip(Coorx, Coory):
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
    filename_selecetednodepair = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Noise\\SelecetedNodepairED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(filename_selecetednodepair, random_pairs, fmt="%i")

    # add noise into theta and phi
    Coorx = add_uniform_random_noise_to_coordinates_R2(Coorx, noise_amplitude)
    Coory = add_uniform_random_noise_to_coordinates_R2(Coory, noise_amplitude)
    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\FrequencyReconstruction\\Noise\\CoorwithNoiseED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    with open(FileNetworkCoorName, "w") as file:
        for data1, data2 in zip(Coorx, Coory):
            file.write(f"{data1}\t{data2}\n")

    for nodepair in random_pairs:
        count = count + 1
        print("Simunodepair:", count)
        nodei = nodepair[0]
        nodej = nodepair[1]

        tic = time.time()
        # Find nearly shortest path nodes
        NearlySPNodelist, _ = FindNearlySPNodesRemoveSpecficLink(G, nodei, nodej, Linkremoveratio=0.1)
        NSPnum_nodepair.append(len(NearlySPNodelist))
        toc = time.time() - tic
        print("NSP finding time:", toc)

        # Create label array
        Label_med = np.zeros(N)
        Label_med[NearlySPNodelist] = 1  # True cases

        thetaSource = Coorx[nodei]
        phiSource = Coory[nodei]
        thetaEnd = Coorx[nodej]
        phiEnd = Coory[nodej]
        geodistance_between_nodepair.append(distR2(thetaSource, phiSource, thetaEnd, phiEnd))

        Geodistance = {}
        for NodeC in range(N):
            if NodeC in [nodei, nodej]:
                Geodistance[NodeC] = 0
            else:
                thetaMed = Coorx[NodeC]
                phiMed = Coory[NodeC]
                dist,_ = dist_to_geodesic_R2(thetaMed, phiMed, thetaSource, phiSource, thetaEnd, phiEnd)
                Geodistance[NodeC] = dist

        # Generate an RGG with the coordinates and predict it
        NSPNodeList_RGG = NSPnodes_inRGG_with_coordinatesR2(N, ED, rg, Coorx, Coory, nodei, nodej)
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

    precision_RGG_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Noise\\PrecisionRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(precision_RGG_Name, Precision_RGG_nodepair)

    recall_RGG_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Noise\\RecallRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(recall_RGG_Name, Recall_RGG_nodepair)

    precision_Geodis_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Noise\\PrecisionGeodisED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(precision_Geodis_Name, Precision_Geodis_nodepair)

    recall_Geodis_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Noise\\RecallGeodisED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(recall_Geodis_Name, Recall_Geodis_nodepair)

    NSPnum_nodepairName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Noise\\NSPNumED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(NSPnum_nodepairName, NSPnum_nodepair, fmt="%i")

    geodistance_between_nodepair_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Noise\\GeodistanceBetweenTwoNodesED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(geodistance_between_nodepair_Name, geodistance_between_nodepair)

    print("Mean Pre RGG:", np.mean(Precision_RGG_nodepair))
    print("Mean Recall RGG:", np.mean(Recall_RGG_nodepair))
    print("Mean Pre Geodistance:", np.mean(Precision_Geodis_nodepair))
    print("Mean Recall Geodistance:", np.mean(Recall_Geodis_nodepair))
    # print("Standard Deviation of AUC Without Normalization:", np.std(PRAUC_nodepair))


def plot_GeovsRGG_precsion_withnoiseR2():
    PRAUC_matrix = np.zeros((2, 2))
    PRAUC_std_matrix = np.zeros((2, 2))
    PRAUC_fre_matrix = np.zeros((2, 2))
    PRAUC_fre_std_matrix = np.zeros((2, 2))
    noise_amplitude = 0.5
    for EDindex in [0, 1]:
        ED_list = [5, 20]  # Expected degrees
        ED = ED_list[EDindex]
        print("ED:", ED)

        for betaindex in [0, 1]:
            beta_list = [4, 100]
            beta = beta_list[betaindex]
            print(beta)
            PRAUC_list = []
            PRAUC_fre_list = []
            for ExternalSimutime in range(20):
                precision_Geodis_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Noise\\PrecisionGeodisED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
                    EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
                PRAUC_list_10times = np.loadtxt(precision_Geodis_Name)
                PRAUC_list.extend(PRAUC_list_10times)

                precision_fre_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Noise\\PrecisionRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
                    EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
                PRAUC_fre_list_10times = np.loadtxt(precision_fre_Name)
                PRAUC_fre_list.extend(PRAUC_fre_list_10times)

            nonzero_indices_geo = find_nonzero_indices(PRAUC_list)
            # PRAUC_list = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PRAUC_list))
            PRAUC_list = [PRAUC_list[x] for x in nonzero_indices_geo]

            print("lenpre", len(PRAUC_list))
            mean_PRAUC = np.mean(PRAUC_list)

            PRAUC_matrix[EDindex][betaindex] = mean_PRAUC
            PRAUC_std_matrix[EDindex][betaindex] = np.std(PRAUC_list)
            print(mean_PRAUC)

            # PRAUC_fre_list = list(
            #     filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PRAUC_fre_list))
            PRAUC_fre_list = [PRAUC_fre_list[x] for x in nonzero_indices_geo]
            print(PRAUC_fre_list)
            print("lenPRE", len(PRAUC_fre_list))
            mean_fre_PRAUC = np.mean(PRAUC_fre_list)
            PRAUC_fre_matrix[EDindex][betaindex] = mean_fre_PRAUC
            PRAUC_fre_std_matrix[EDindex][betaindex] = np.std(PRAUC_list)
            print(mean_fre_PRAUC)

    plt.figure()
    df = pd.DataFrame(PRAUC_matrix,
                      index=[5, 20],  # DataFrame的行标签设置为大写字母
                      columns=[4, 100])  # 设置DataFrame的列标签
    sns.heatmap(data=df, vmin=0, vmax=0.8, annot=True, fmt=".2f", cbar=True,
                cbar_kws={'label': 'precision'})
    plt.title("Geo distance")
    plt.xlabel("beta")
    plt.ylabel("average degree")
    precision_Geodis_fig_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Noise\\PrecisionGeodisED{EDn}Beta{betan}Noise{no}PY.pdf".format(
        EDn=ED, betan=beta, no=noise_amplitude)
    plt.savefig(precision_Geodis_fig_Name,
                format='pdf', bbox_inches='tight', dpi=600)
    plt.close()

    plt.figure()
    df = pd.DataFrame(PRAUC_fre_matrix,
                      index=[5, 20],  # DataFrame的行标签设置为大写字母
                      columns=[4, 100])  # 设置DataFrame的列标签
    sns.heatmap(data=df, vmin=0, vmax=0.8, annot=True, fmt=".2f", cbar=True,
                cbar_kws={'label': 'precision'})
    plt.title("RGG")
    plt.xlabel("beta")
    plt.ylabel("average degree")

    precision_RGG_fig_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Noise\\PrecisionRGGED{EDn}Beta{betan}Noise{no}PY.pdf".format(
        EDn=ED, betan=beta, no=noise_amplitude)
    plt.savefig(
        precision_RGG_fig_Name,
        format='pdf', bbox_inches='tight', dpi=600)
    plt.close()


def plot_GeovsRGG_recall_withnoiseR2():
    noise_amplitude = 0.1
    PRAUC_matrix = np.zeros((2, 2))
    PRAUC_std_matrix = np.zeros((2, 2))
    PRAUC_fre_matrix = np.zeros((2, 2))
    PRAUC_fre_std_matrix = np.zeros((2, 2))

    for EDindex in [0, 1]:
        ED_list = [5, 20]  # Expected degrees
        ED = ED_list[EDindex]
        print("ED:", ED)

        for betaindex in [0, 1]:
            beta_list = [4, 100]
            beta = beta_list[betaindex]
            print(beta)
            PRAUC_list = []
            PRAUC_fre_list = []
            for ExternalSimutime in range(20):
                precision_Geodis_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Noise\\RecallGeodisED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
                    EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)

                PRAUC_list_10times = np.loadtxt(precision_Geodis_Name)
                PRAUC_list.extend(PRAUC_list_10times)

                precision_fre_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Noise\\RecallRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
                    EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
                PRAUC_fre_list_10times = np.loadtxt(precision_fre_Name)
                PRAUC_fre_list.extend(PRAUC_fre_list_10times)

            nonzero_indices_geo = find_nonzero_indices(PRAUC_list)
            # PRAUC_list = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PRAUC_list))
            PRAUC_list = [PRAUC_list[x] for x in nonzero_indices_geo]

            print("lenGEOturecase", len(PRAUC_list))
            mean_PRAUC = np.mean(PRAUC_list)

            PRAUC_matrix[EDindex][betaindex] = mean_PRAUC
            PRAUC_std_matrix[EDindex][betaindex] = np.std(PRAUC_list)
            print(mean_PRAUC)

            # PRAUC_fre_list = list(
            #     filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PRAUC_fre_list))
            PRAUC_fre_list = [PRAUC_fre_list[x] for x in nonzero_indices_geo]
            print(PRAUC_fre_list)
            print("lenRGGturecase", len(PRAUC_fre_list))
            mean_fre_PRAUC = np.mean(PRAUC_fre_list)
            PRAUC_fre_matrix[EDindex][betaindex] = mean_fre_PRAUC
            PRAUC_fre_std_matrix[EDindex][betaindex] = np.std(PRAUC_list)
            print(mean_fre_PRAUC)

    plt.figure()
    df = pd.DataFrame(PRAUC_matrix,
                      index=[5, 20],  # DataFrame的行标签设置为大写字母
                      columns=[4, 100])  # 设置DataFrame的列标签
    sns.heatmap(data=df, vmin=0, vmax=0.8, annot=True, fmt=".2f", cbar=True,
                cbar_kws={'label': 'recall'})
    plt.title("Geo distance")
    plt.xlabel("beta")
    plt.ylabel("average degree")
    recall_Geodis_fig_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Noise\\RecallGeodisED{EDn}Beta{betan}Noise{no}PY.pdf".format(
        EDn=ED, betan=beta, no=noise_amplitude)
    plt.savefig(
        recall_Geodis_fig_Name,
        format='pdf', bbox_inches='tight', dpi=600)
    plt.close()

    plt.figure()
    df = pd.DataFrame(PRAUC_fre_matrix,
                      index=[5, 20],  # DataFrame的行标签设置为大写字母
                      columns=[4, 100])  # 设置DataFrame的列标签
    sns.heatmap(data=df, vmin=0, vmax=0.8, annot=True, fmt=".2f", cbar=True,
                cbar_kws={'label': 'recall'})
    plt.title("RGG")
    plt.xlabel("beta")
    plt.ylabel("average degree")
    recall_RGG_fig_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Noise\\RecallRGGED{EDn}Beta{betan}Noise{no}PY.pdf".format(
        EDn=ED, betan=beta, no=noise_amplitude)
    plt.savefig(
        recall_RGG_fig_Name,
        format='pdf', bbox_inches='tight', dpi=600)
    plt.close()


def PredictGeodistanceVsfrequency_withnoise_givennodepair_difflength_R2(noise_amplitude, x_A, y_A, x_B, y_B,
                                                                        ExternalSimutime):
    """
        For a given node pair. This function is designed for the cluster
        :param x_A:
        :param y_A:
        :param x_B:
        :param y_B:
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
    G, Coorx, Coory = R2SRGG_withgivennodepair(N, avg, beta, rg, x_A, y_A, x_B, y_B)
    print("LinkNum:", G.number_of_edges())
    print("AveDegree:", G.number_of_edges() * 2 / G.number_of_nodes())
    print("ClusteringCoefficient:", nx.average_clustering(G))
    geo_length = distR2(x_A, y_A, x_B, y_B)
    print("Geo Length:", geo_length)

    FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\FrequencyReconstruction\\Noise\\Geolength\\ControlGroupnetworkGeolen{le}Simu{ST}beta{b}.txt".format(
        le=geo_length, ST=ExternalSimutime, b=beta)
    nx.write_edgelist(G, FileNetworkName)
    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\FrequencyReconstruction\\Noise\\Geolength\\ControlGroupnetworkCoorGeolen{le}Simu{ST}beta{b}.txt".format(
        le=geo_length, ST=ExternalSimutime, b=beta)
    with open(FileNetworkCoorName, "w") as file:
        for data1, data2 in zip(Coorx, Coory):
            file.write(f"{data1}\t{data2}\n")

    nodei = N - 2
    nodej = N - 1

    print("Node Geo distance", distR2(Coorx[nodei], Coory[nodei], Coorx[nodej], Coory[nodej]))
    # All shortest paths
    AllSP = nx.all_shortest_paths(G, nodei, nodej)
    AllSPlist = list(AllSP)
    print("SPnum", len(AllSPlist))
    print("SPlength", len(AllSPlist[0]))

    FileASPName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\FrequencyReconstruction\\Noise\\Geolength\\ControlGroupASPGeolen{le}Simu{ST}beta{b}.txt".format(
        le=geo_length, ST=ExternalSimutime, b=beta)
    np.savetxt(FileASPName, AllSPlist, fmt="%i")

    # All shortest path node
    AllSPNode = set()
    for path in AllSPlist:
        AllSPNode.update(path)
    AllSPNode.discard(nodei)
    AllSPNode.discard(nodej)
    FileASPNodeName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\FrequencyReconstruction\\Noise\\Geolength\\ControlGroupASPNodeGeolen{le}Simu{ST}beta{b}.txt".format(
        le=geo_length, ST=ExternalSimutime, b=beta)
    np.savetxt(FileASPNodeName, list(AllSPNode), fmt="%i")
    tic = time.time()
    # Nearly shortest path node
    NSPNode, relevance = FindNearlySPNodesRemoveSpecficLink(G, nodei, nodej, Linkremoveratio=0.1)
    print("time for finding NSP", time.time() - tic)
    print("NSP num", len(NSPNode))
    FileNSPNodeName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\FrequencyReconstruction\\Noise\\Geolength\\ControlGroupNSPNodeGeolen{le}Simu{ST}beta{b}.txt".format(
        le=geo_length, ST=ExternalSimutime, b=beta)
    np.savetxt(FileNSPNodeName, NSPNode, fmt="%i")
    FileNodeRelevanceName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\FrequencyReconstruction\\Noise\\Geolength\\ControlGroupRelevanceGeolen{le}Simu{ST}beta{b}.txt".format(
        le=geo_length, ST=ExternalSimutime, b=beta)
    np.savetxt(FileNodeRelevanceName, relevance, fmt="%.3f")

    # Create label array
    Label_med = np.zeros(N)
    Label_med[NSPNode] = 1  # True cases

    # add noise into theta and phi
    Coorx = add_uniform_random_noise_to_coordinates_R2(Coorx, noise_amplitude)
    Coory = add_uniform_random_noise_to_coordinates_R2(Coory, noise_amplitude)
    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\FrequencyReconstruction\\Noise\\Geolength\\PredictGeoVsRGGnetworkCoorwithnoiseGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    with open(FileNetworkCoorName, "w") as file:
        for data1, data2 in zip(Coorx, Coory):
            file.write(f"{data1}\t{data2}\n")

    # Geodesic
    thetaSource = Coorx[nodei]
    phiSource = Coory[nodei]
    thetaEnd = Coorx[nodej]
    phiEnd = Coory[nodej]

    Geodistance = {}
    for NodeC in range(N):
        if NodeC in [nodei, nodej]:
            Geodistance[NodeC] = 0
        else:
            thetaMed = Coorx[NodeC]
            phiMed = Coory[NodeC]
            dist,_ = dist_to_geodesic_R2(thetaMed, phiMed, thetaSource, phiSource, thetaEnd, phiEnd)
            Geodistance[NodeC] = dist
    FileGeodistanceName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\FrequencyReconstruction\\Noise\\Geolength\\PredictGeoVsRGGGeoDistancewithnoiseGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    np.savetxt(FileGeodistanceName, list(Geodistance.values()), fmt="%.8f")

    distance_med = np.zeros(N)

    # Calculate distances to geodesic
    for NodeC in range(0, N):
        if NodeC not in [nodei, nodej]:
            thetaMed = Coorx[NodeC]
            phiMed = Coory[NodeC]
            dist,_ = dist_to_geodesic_R2(thetaMed, phiMed, thetaSource, phiSource, thetaEnd, phiEnd)
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
    node_fre = nodeNSPfrequencyR2(N, avg, beta, rg, Coorx, Coory, nodei, nodej)
    FileNodefreName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\FrequencyReconstruction\\Noise\\Geolength\\ControlGroupNodefreGeolen{le}Simu{ST}beta{b}.txt".format(
        le=geo_length, ST=ExternalSimutime, b=beta)
    np.savetxt(FileNodefreName, node_fre)

    node_fre = np.delete(node_fre, [nodei, nodej])
    precisionsfre, recallsfre, _ = precision_recall_curve(Label_med, node_fre)
    AUCfrenodeij = auc(recallsfre, precisionsfre)
    print(AUCfrenodeij)


def test_result_PredictGeodistanceVsfrequency_withnoise_givennodepair_difflength():
    # TEST
    N = 10000
    nodei = N - 2
    nodej = N - 1
    theta_A = 8 * math.pi / 16
    phi_A = 0
    theta_B = 9 * math.pi / 16
    phi_B = 0
    ExternalSimutime = 0
    geo_length = distR2(theta_A, phi_A, theta_B, phi_B)
    beta = 4
    FileNSPNodeName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\FrequencyReconstruction\\Noise\\Geolength\\ControlGroupNSPNodeGeolen{le}Simu{ST}beta{b}.txt".format(
        le=geo_length, ST=ExternalSimutime, b=beta)
    NSPNode = np.loadtxt(FileNSPNodeName, dtype=int)

    # Create label array
    Label_med = np.zeros(N)
    Label_med[NSPNode] = 1  # True cases

    FileGeodistanceName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\FrequencyReconstruction\\Noise\\Geolength\\PredictGeoVsRGGGeoDistancewithnoiseGeolen{le}Simu{ST}.txt".format(
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

    FileNodefreName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\FrequencyReconstruction\\Noise\\Geolength\\ControlGroupNodefreGeolen{le}Simu{ST}beta{b}.txt".format(
        le=geo_length, ST=ExternalSimutime, b=beta)
    node_fre = np.loadtxt(FileNodefreName)

    node_fre = np.delete(node_fre, [nodei, nodej])
    precisionsfre, recallsfre, _ = precision_recall_curve(Label_med, node_fre)
    AUCfrenodeij = auc(recallsfre, precisionsfre)
    print(AUCfrenodeij)


def PredictGeodistanceVsfrequency_withnoise_R2(Edindex, betaindex, noiseindex, ExternalSimutime):
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

    noise_amplitude_list = [0.001, 0.01, 0.1]
    noise_amplitude = noise_amplitude_list[noiseindex]
    print("noise amplitude:", noise_amplitude)

    print("ExternalSimutime", ExternalSimutime)

    PRAUC_nodepair = []  # save the PRAUC for each node pair, we selected 100 node pair in total
    PRAUC_fre_controlgroup_nodepair = []  # save the PRAUC for each node pair, we selected 100 node pair in total
    NSPnum_nodepair = []  # save the Number of nearly shortest path for each node pair
    geodistance_between_nodepair = []  # save the geodeisc length between each node pair

    random.seed(ExternalSimutime)
    rg = RandomGenerator(-12)
    for i in range(random.randint(0, 100)):
        rg.ran1()

    G, Coorx, Coory = R2SRGG(N, ED, beta, rg)

    print("We have a network now!")
    FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\FrequencyReconstruction\\Noise\\NetworkED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    nx.write_edgelist(G, FileNetworkName)
    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\FrequencyReconstruction\\Noise\\CoorED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    with open(FileNetworkCoorName, "w") as file:
        for data1, data2 in zip(Coorx, Coory):
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
    filename_selecetednodepair = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\FrequencyReconstruction\\Noise\\SelecetedNodepairED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(filename_selecetednodepair, random_pairs, fmt="%i")

    # add noise into theta and phi
    Coorx = add_uniform_random_noise_to_coordinates_R2(Coorx, noise_amplitude)
    Coory = add_uniform_random_noise_to_coordinates_R2(Coory, noise_amplitude)
    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\FrequencyReconstruction\\Noise\\CoorwithNoiseED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    with open(FileNetworkCoorName, "w") as file:
        for data1, data2 in zip(Coorx, Coory):
            file.write(f"{data1}\t{data2}\n")

    for nodepair in random_pairs:
        count = count + 1
        print("Simunodepair:", count)
        nodei = nodepair[0]
        nodej = nodepair[1]

        thetaSource = Coorx[nodei]
        phiSource = Coory[nodei]
        thetaEnd = Coorx[nodej]
        phiEnd = Coory[nodej]
        geodistance_between_nodepair.append(distR2(thetaSource, phiSource, thetaEnd, phiEnd))

        # tic = time.time()
        # Find nearly shortest path nodes
        NearlySPNodelist, _ = FindNearlySPNodesRemoveSpecficLink(G, nodei, nodej, Linkremoveratio=0.1)
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
                thetaMed = Coorx[NodeC]
                phiMed = Coory[NodeC]
                dist,_ = dist_to_geodesic_R2(thetaMed, phiMed, thetaSource, phiSource, thetaEnd, phiEnd)
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
        node_fre = nodeNSPfrequencyR2(N, ED, beta, rg, Coorx, Coory, nodei, nodej)
        node_fre = np.delete(node_fre, [nodei, nodej])
        precisionsfre, recallsfre, _ = precision_recall_curve(Label_med, node_fre)
        AUCfrenodeij = auc(recallsfre, precisionsfre)
        PRAUC_fre_controlgroup_nodepair.append(AUCfrenodeij)

    # Calculate means and standard deviations of AUC
    # AUCWithoutnorMean = np.mean(PRAUC_nodepair[~np.isnan(PRAUC_nodepair)])
    # AUCWithoutnorStd = np.std(PRAUC_nodepair[~np.isnan(PRAUC_nodepair)])

    PRAUCName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\FrequencyReconstruction\\Noise\\AUCED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(PRAUCName, PRAUC_nodepair)

    FrePRAUCName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\FrequencyReconstruction\\Noise\\ControlFreAUCED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(FrePRAUCName, PRAUC_fre_controlgroup_nodepair)

    NSPnum_nodepairName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\FrequencyReconstruction\\Noise\\NSPNumED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(NSPnum_nodepairName, NSPnum_nodepair, fmt="%i")

    geodistance_between_nodepair_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\FrequencyReconstruction\\Noise\\GeodistanceBetweenTwoNodesED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(geodistance_between_nodepair_Name, geodistance_between_nodepair)

    print("Mean AUC Without Normalization:", np.mean(PRAUC_nodepair))
    print("Mean AUC CONTROL:", np.mean(PRAUC_fre_controlgroup_nodepair))
    # print("Standard Deviation of AUC Without Normalization:", np.std(PRAUC_nodepair))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # STEP 1
    # PredictGeodistanceVsRGG_withnoise_givennodepair_difflengthR2(0.1, 0.2, 0.2, 0.6, 0.6, int(0))


    # STEP 2
    # PredictGeodistanceVsRGG_withnoiseR2(0, 0, 0, 0)
    # ED = sys.argv[1]
    # beta = sys.argv[2]
    # noise = sys.argv[3]
    # ExternalSimutime = sys.argv[4]
    # PredictGeodistanceVsRGG_withnoiseR2(int(ED), int(beta), int(noise), int(ExternalSimutime))

    # plot_GeovsRGG_precsion_withnoise()
    # plot_GeovsRGG_recall_withnoise()

    # STEP 3
    PredictGeodistanceVsfrequency_withnoise_givennodepair_difflength_R2(0.01, 0.1, 0.2, 0.2, 0.6, int(0))

    # STEP 4
    # PredictGeodistanceVsfrequency_withnoise_R2(0,0,2,0)
    # ED = sys.argv[1]
    # beta = sys.argv[2]
    # noise = sys.argv[3]
    # ExternalSimutime = sys.argv[4]
    # PredictGeodistanceVsfrequency_withnoise_R2(int(ED), int(beta), int(noise), int(ExternalSimutime))
