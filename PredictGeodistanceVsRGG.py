# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 22-5-2024
Predict
"""
import time

import numpy as np
import networkx as nx
import random
import math

from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score

from NearlyShortestPathPredict import FindNearlySPNodes, FindNearlySPNodesRemoveSpecficLink
from SphericalSoftRandomGeomtricGraph import RandomGenerator, SphericalRandomGeometricGraph, distS2, \
    SphericalRGGwithGivenNode, dist_to_geodesic_S2, SphericalSoftRGGwithGivenNode


def NSPnodes_inRGG_with_coordinates(N, avg, rg, Coortheta, Coorphi, nodei, nodej):
    """
    Given nodes(coordinates) of the SRGG.
    :return: SHORTEST PATH nodes in the corresponding shortest path nodes
    """
    G, coor1, coor2 = SphericalRandomGeometricGraph(N, avg, rg, Coortheta=Coortheta,
                                                    Coorphi=Coorphi, SaveNetworkPath=None)
    NSPNodeList, _ = FindNearlySPNodesRemoveSpecficLink(G, nodei, nodej, Linkremoveratio=0.1)
    return NSPNodeList


def PredictGeodistanceVsRGG_givennodepair_difflength(theta_A, phi_A, theta_B, phi_B, ExternalSimutime):
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

    FileNetworkName = "D:\\data\\geometric shortest path problem\\SphericalRGG\\Geolength\\PredictGeoVsRGGnetworkGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    nx.write_edgelist(G, FileNetworkName)
    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\SphericalRGG\\Geolength\\PredictGeoVsRGGnetworkCoorGeolen{le}Simu{ST}.txt".format(
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

    FileASPName = "D:\\data\\geometric shortest path problem\\SphericalRGG\\Geolength\\PredictGeoVsRGGASPGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    np.savetxt(FileASPName, AllSPlist, fmt="%i")

    # All shortest path node
    AllSPNode = set()
    for path in AllSPlist:
        AllSPNode.update(path)
    AllSPNode.discard(nodei)
    AllSPNode.discard(nodej)
    FileASPNodeName = "D:\\data\\geometric shortest path problem\\SphericalRGG\\Geolength\\PredictGeoVsRGGASPNodeGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    np.savetxt(FileASPNodeName, list(AllSPNode), fmt="%i")
    tic = time.time()
    # Nearly shortest path node
    NSPNode, relevance = FindNearlySPNodes(G, nodei, nodej)
    print("time for finding NSP", time.time() - tic)
    print("NSP num", len(NSPNode))
    FileNSPNodeName = "D:\\data\\geometric shortest path problem\\SphericalRGG\\Geolength\\PredictGeoVsRGGNSPNodeGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    np.savetxt(FileNSPNodeName, NSPNode, fmt="%i")
    FileNodeRelevanceName = "D:\\data\\geometric shortest path problem\\SphericalRGG\\Geolength\\PredictGeoVsRGGRelevanceGeolen{le}Simu{ST}.txt".format(
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
    FileGeodistanceName = "D:\\data\\geometric shortest path problem\\SphericalRGG\\Geolength\\PredictGeoVsRGGGeoDistanceGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    np.savetxt(FileGeodistanceName, list(Geodistance.values()), fmt="%.8f")


    # Create label array
    Label_med = np.zeros(N)
    Label_med[NSPNode] = 1  # True cases

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
    FileTop100closedNodeName = "D:\\data\\geometric shortest path problem\\SphericalRGG\\Geolength\\PredictGeoVsRGGPredictedTrueNSPNodeByGeodistanceGeolen{le}Simu{ST}.txt".format(
        le=geo_length, ST=ExternalSimutime)
    np.savetxt(FileTop100closedNodeName, Top100closednode, fmt="%i")

    NSPNodeList_Geo = np.zeros(N)
    NSPNodeList_Geo[Top100closednode] = 1  # True cases
    precision_Geo = precision_score(Label_med, NSPNodeList_Geo)
    recall_Geo = recall_score(Label_med, NSPNodeList_Geo)
    print("precsion_Geo:", precision_Geo)
    print("recall_Geo:", recall_Geo)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    PredictGeodistanceVsRGG_givennodepair_difflength(8*math.pi/16,0, 9*math.pi/16,0,0)

