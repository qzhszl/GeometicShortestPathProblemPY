# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 22-5-2024
Predict
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

from NearlyShortestPathPredict import FindNearlySPNodesRemoveSpecficLink
from R2RGG import RandomGeometricGraph
from R2SRGG import R2SRGG_withgivennodepair, distR2, dist_to_geodesic_R2, R2SRGG
from degree_Vs_radius_RGG import read_radius
from SphericalSoftRandomGeomtricGraph import RandomGenerator
from main import find_nonzero_indices


def NSPnodes_inRGG_with_coordinatesR2(N, avg, rg, CoorX, CoorY, nodei, nodej):
    """
    Given nodes(coordinates) of the SRGG.
    Generate a RGG with the given coordinates
    :return: Nearly SHORTEST PATH nodes in the corresponding RGG
    """
    r = read_radius(N,avg)
    G, _, _ = RandomGeometricGraph(N, avg, rg, radius=r, Coorx=CoorX, Coory=CoorY)
    NSPNodeList, _ = FindNearlySPNodesRemoveSpecficLink(G, nodei, nodej, Linkremoveratio=0.1)
    return NSPNodeList


def PredictGeodistanceVsRGG_givennodepair_difflengthR2(x_A, y_A, x_B, y_B, ExternalSimutime):
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


    FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Geolength\\PredictGeoVsRGGnetworkGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    nx.write_edgelist(G, FileNetworkName)
    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Geolength\\PredictGeoVsRGGnetworkCoorGeolen{le}Simu{ST}.txt".format(
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

    FileASPName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Geolength\\PredictGeoVsRGGASPGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    np.savetxt(FileASPName, AllSPlist, fmt="%i")

    # All shortest path node
    AllSPNode = set()
    for path in AllSPlist:
        AllSPNode.update(path)
    AllSPNode.discard(nodei)
    AllSPNode.discard(nodej)
    FileASPNodeName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Geolength\\PredictGeoVsRGGASPNodeGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    np.savetxt(FileASPNodeName, list(AllSPNode), fmt="%i")
    tic = time.time()
    # Nearly shortest path node
    NSPNode, relevance = FindNearlySPNodesRemoveSpecficLink(G, nodei, nodej, Linkremoveratio=0.1)
    print("time for finding NSP", time.time() - tic)
    print("NSP num", len(NSPNode))
    FileNSPNodeName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Geolength\\PredictGeoVsRGGNSPNodeGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    np.savetxt(FileNSPNodeName, NSPNode, fmt="%i")
    FileNodeRelevanceName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Geolength\\PredictGeoVsRGGRelevanceGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    np.savetxt(FileNodeRelevanceName, relevance, fmt="%.3f")

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
    FileGeodistanceName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Geolength\\PredictGeoVsRGGGeoDistanceGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    np.savetxt(FileGeodistanceName, list(Geodistance.values()), fmt="%.8f")


    # Create label array
    Label_med = np.zeros(N)
    Label_med[NSPNode] = 1  # True cases

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
    Geodistance = Geodistance[:Predicted_truecase_num+2]
    Top100closednode = [t[0] for t in Geodistance]
    Top100closednode = [n for n in Top100closednode if n not in [nodei, nodej]]
    FileTop100closedNodeName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\Geolength\\PredictGeoVsRGGPredictedTrueNSPNodeByGeodistanceGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    np.savetxt(FileTop100closedNodeName, Top100closednode, fmt="%i")

    NSPNodeList_Geo = np.zeros(N)
    NSPNodeList_Geo[Top100closednode] = 1  # True cases
    precision_Geo = precision_score(Label_med, NSPNodeList_Geo)
    recall_Geo = recall_score(Label_med, NSPNodeList_Geo)
    print("precsion_Geo:", precision_Geo)
    print("recall_Geo:", recall_Geo)


def PredictGeodistanceVsRGGR2(Edindex, betaindex, ExternalSimutime):
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

    G, Coorx, Coory = R2SRGG(N, ED, beta, rg)

    print("We have a network now!")
    FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\NetworkED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    nx.write_edgelist(G, FileNetworkName)
    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\CoorED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
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
    filename_selecetednodepair = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\SelecetedNodepairED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(filename_selecetednodepair, random_pairs, fmt="%i")

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

        tic = time.time()
        # Find nearly shortest path nodes
        NearlySPNodelist, _ = FindNearlySPNodesRemoveSpecficLink(G, nodei, nodej, Linkremoveratio=0.1)
        NSP_current_num = len(NearlySPNodelist)
        NSPnum_nodepair.append(NSP_current_num)
        toc  = time.time()-tic

        print("NSP finding time:", toc)
        print("NSP NUM:", NSP_current_num)

        if NSP_current_num != 0:
            Geodistance = {}
            for NodeC in range(N):
                if NodeC in [nodei, nodej]:
                    Geodistance[NodeC] = 0
                else:
                    thetaMed = Coorx[NodeC]
                    phiMed = Coory[NodeC]
                    dist,_ = dist_to_geodesic_R2(thetaMed, phiMed, thetaSource, phiSource, thetaEnd, phiEnd)
                    Geodistance[NodeC] = dist

            # Create label array
            Label_med = np.zeros(N)
            Label_med[NearlySPNodelist] = 1  # True cases

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

    precision_RGG_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\PrecisionRGGED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(precision_RGG_Name, Precision_RGG_nodepair)

    recall_RGG_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\RecallRGGED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(recall_RGG_Name, Recall_RGG_nodepair)

    precision_Geodis_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\PrecisionGeodisED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(precision_Geodis_Name, Precision_Geodis_nodepair)

    recall_Geodis_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\RecallGeodisED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(recall_Geodis_Name, Recall_Geodis_nodepair)

    NSPnum_nodepairName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\NSPNumED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(NSPnum_nodepairName, NSPnum_nodepair, fmt="%i")

    geodistance_between_nodepair_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\GeodistanceBetweenTwoNodesED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(geodistance_between_nodepair_Name, geodistance_between_nodepair)

    print("Mean Pre RGG:", np.mean(Precision_RGG_nodepair))
    print("Mean Recall RGG:", np.mean(Recall_RGG_nodepair))
    print("Mean Pre Geodistance:", np.mean(Precision_Geodis_nodepair))
    print("Mean Recall Geodistance:", np.mean(Recall_Geodis_nodepair))
    # print("Standard Deviation of AUC Without Normalization:", np.std(PRAUC_nodepair))



def plot_GeovsRGG_precsion():
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
            for ExternalSimutime in range(10):
                precision_Geodis_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\PrecisionGeodisED{EDn}Beta{betan}PYSimu{ST}.txt".format(
                    EDn=ED, betan=beta, ST=ExternalSimutime)

                PRAUC_list_10times = np.loadtxt(precision_Geodis_Name)
                PRAUC_list.extend(PRAUC_list_10times)

                precision_fre_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\PrecisionRGGED{EDn}Beta{betan}PYSimu{ST}.txt".format(
                    EDn=ED, betan=beta, ST=ExternalSimutime)
                PRAUC_fre_list_10times = np.loadtxt(precision_fre_Name)
                PRAUC_fre_list.extend(PRAUC_fre_list_10times)

            nonzero_indices_geo = find_nonzero_indices(PRAUC_list)
            # PRAUC_list = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PRAUC_list))
            PRAUC_list = [PRAUC_list[x] for x in nonzero_indices_geo]

            print("lenpre",len(PRAUC_list))
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
    plt.savefig(
        "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\GeoPrecisionHeatmapNSP0_1LinkRemove.pdf",
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
    plt.savefig(
        "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\RGGPrecisionHeatmapNSP0_1LinkRemove.pdf",
        format='pdf', bbox_inches='tight', dpi=600)
    plt.close()


def plot_GeovsRGG_recall():
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
            for ExternalSimutime in range(10):
                precision_Geodis_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\RecallGeodisED{EDn}Beta{betan}PYSimu{ST}.txt".format(
                    EDn=ED, betan=beta, ST=ExternalSimutime)

                PRAUC_list_10times = np.loadtxt(precision_Geodis_Name)
                PRAUC_list.extend(PRAUC_list_10times)

                precision_fre_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\RecallRGGED{EDn}Beta{betan}PYSimu{ST}.txt".format(
                    EDn=ED, betan=beta, ST=ExternalSimutime)
                PRAUC_fre_list_10times = np.loadtxt(precision_fre_Name)
                PRAUC_fre_list.extend(PRAUC_fre_list_10times)

            nonzero_indices_geo = find_nonzero_indices(PRAUC_list)
            # PRAUC_list = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PRAUC_list))
            PRAUC_list = [PRAUC_list[x] for x in nonzero_indices_geo]

            print("lenpre",len(PRAUC_list))
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
                cbar_kws={'label': 'recall'})
    plt.title("Geo distance")
    plt.xlabel("beta")
    plt.ylabel("average degree")
    plt.savefig(
        "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\GeoRecallHeatmapNSP0_1LinkRemove.pdf",
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
    plt.savefig(
        "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\CompareRGG\\RGGRecallHeatmapNSP0_1LinkRemove.pdf",
        format='pdf', bbox_inches='tight', dpi=600)
    plt.close()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # STEP 1
    # x_A = 0.2
    # y_A = 0.2
    # x_B = 0.2
    # y_B = 0.5
    # ExternalSimutime = 0
    # PredictGeodistanceVsRGG_givennodepair_difflengthR2(x_A, y_A, x_B, y_B, ExternalSimutime)

    # STEP2
    # PredictGeodistanceVsRGGR2(0, 0, 0)

    # STEP3
    ED = sys.argv[1]
    beta = sys.argv[2]
    ExternalSimutime = sys.argv[3]
    PredictGeodistanceVsRGGR2(int(ED), int(beta), int(ExternalSimutime))


