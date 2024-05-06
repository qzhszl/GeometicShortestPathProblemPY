# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 1-5-2024
"""
import random
import math
import time

import networkx as nx
from sklearn.metrics import precision_recall_curve, auc

from NearlyShortestPathPredict import FindNearlySPNodes
from SphericalSoftRandomGeomtricGraph import RandomGenerator, SphericalSoftRGGwithGivenNode, SphericalSoftRGG, \
    dist_to_geodesic_S2, distS2
import numpy as np
import matplotlib.pyplot as plt


def VisualizationDataOfSPNSPGEO():
    # Input data parameters
    N = 10000
    avg = 5
    beta = 3.575
    rg = RandomGenerator(-12)
    for _ in range(random.randint(0,100)):
        rg.ran1()

    # Network and coordinates
    G,Coortheta,Coorphi = SphericalSoftRGGwithGivenNode(N,avg,beta,rg,math.pi/4,0,3*math.pi/8,0)
    print("LinkNum:",G.number_of_edges())
    print("AveDegree:", G.number_of_edges()*2/G.number_of_nodes())
    print("ClusteringCoefficient:",nx.average_clustering(G))

    FileNetworkName = "D:\\data\\geometric shortest path problem\\SSRGG\\VisualizeSPNSPDe\\testPYnetwork.txt"
    nx.write_edgelist(G, FileNetworkName)
    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\SSRGG\\VisualizeSPNSPDe\\testPYnetworkCoor.txt"
    with open(FileNetworkCoorName, "w") as file:
        for data1, data2 in zip(Coortheta, Coorphi):
            file.write(f"{data1}\t{data2}\n")

    nodei = 1
    nodej = 102

    print("Node Geo distance",distS2(Coortheta[nodei], Coorphi[nodei], Coortheta[nodej], Coorphi[nodej]))
    # All shortest paths
    AllSP = nx.all_shortest_paths(G, nodei, nodej)
    AllSPlist = list(AllSP)
    print("SPnum",len(AllSPlist))
    print("SPlength",len(AllSPlist[0]))

    FileASPName = "D:\\data\\geometric shortest path problem\\SSRGG\\VisualizeSPNSPDe\\testPYASP.txt"
    np.savetxt(FileASPName, AllSPlist, fmt="%i")

    # All shortest path node
    AllSPNode = set()
    for path in AllSPlist:
        AllSPNode.update(path)
    AllSPNode.discard(nodei)
    AllSPNode.discard(nodej)
    FileASPNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\VisualizeSPNSPDe\\testPYASPNode.txt"
    np.savetxt(FileASPNodeName, list(AllSPNode), fmt="%i")

    # Nearly shortest path node
    NSPNode, relevance = FindNearlySPNodes(G, nodei, nodej)
    print("NSP num", len(NSPNode))
    FileNSPNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\VisualizeSPNSPDe\\testPYNSPNode.txt"
    np.savetxt(FileNSPNodeName, NSPNode, fmt="%i")
    FileNodeRelevanceName = "D:\\data\\geometric shortest path problem\\SSRGG\\VisualizeSPNSPDe\\testPYRelevance.txt"
    np.savetxt(FileNodeRelevanceName, relevance, fmt="%.3f")

    # Geodesic
    thetaSource = Coortheta[nodei]
    phiSource = Coorphi[nodei]
    thetaEnd = Coortheta[nodej]
    phiEnd = Coorphi[nodej]

    Geodistance={}
    for NodeC in range(N):
        if NodeC in [nodei,nodej]:
            Geodistance[NodeC] = 0
        else:
            thetaMed = Coortheta[NodeC]
            phiMed = Coorphi[NodeC]
            dist = dist_to_geodesic_S2(thetaMed, phiMed, thetaSource, phiSource, thetaEnd, phiEnd)
            Geodistance[NodeC] = dist
    FileGeodistanceName = "D:\\data\\geometric shortest path problem\\SSRGG\\VisualizeSPNSPDe\\testPYGeoDistance.txt"
    np.savetxt(FileGeodistanceName, list(Geodistance.values()), fmt="%.8f")
    Geodistance = sorted(Geodistance.items(), key=lambda kv: (kv[1], kv[0]))
    Geodistance = Geodistance[:102]
    Top100closednode = [t[0] for t in Geodistance]
    Top100closednode = [n for n in Top100closednode if n not in [nodei,nodej]]

    FileTop100closedNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\VisualizeSPNSPDe\\testPYFileTop100closedNode.txt"
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
    print("PRAUC",AUCWithoutNornodeij)


def PlotVisualizationDataOfSPNSPGEO():
    FileASPNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\VisualizeSPNSPDe\\testPYASPNode.txt"
    p_all_node_list = np.loadtxt(FileASPNodeName)
    p_all_node_list = p_all_node_list.astype(int)

    FileNSPNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\VisualizeSPNSPDe\\testPYNSPNode.txt"
    nsp_node_List = np.loadtxt(FileNSPNodeName)
    nsp_node_List = nsp_node_List.astype(int)

    FileNodeRelevanceName = "D:\\data\\geometric shortest path problem\\SSRGG\\VisualizeSPNSPDe\\testPYRelevance.txt"
    relevance_list = np.loadtxt(FileNodeRelevanceName)
    # print(relevance_list)

    FileGeodistanceName = "D:\\data\\geometric shortest path problem\\SSRGG\\VisualizeSPNSPDe\\testPYGeoDistance.txt"
    geo_distance = np.loadtxt(FileGeodistanceName)

    FileTop100closedNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\VisualizeSPNSPDe\\testPYFileTop100closedNode.txt"
    Top100closednode = np.loadtxt(FileTop100closedNodeName)
    top100_closed_node_list =  Top100closednode.astype(int)

    x_asp = relevance_list[p_all_node_list]
    y_asp = geo_distance[p_all_node_list]

    x_nsp = relevance_list[nsp_node_List]
    y_nsp = geo_distance[nsp_node_List]

    x_close = relevance_list[top100_closed_node_list]
    y_close = geo_distance[top100_closed_node_list]

    # plt.figure(figsize=(12, 8))
    # plt.hist(geo_distance,bins=60)
    # plt.title('histogram of geodistance')
    # plt.show()


    # Plot settings
    plt.figure(figsize=(12, 8))
    plt.scatter(relevance_list, geo_distance,marker="o", edgecolors=(0, 0.4470, 0.7410),facecolors="none", label='General node')

    # Additional scatter plots with customized markers
    scatter_size = 100
    plt.scatter(x_asp, y_asp, s=scatter_size, marker='*', edgecolors=(0.8500, 0.3250, 0.0980), facecolors="none",
                label='Shortest path node')
    plt.scatter(x_nsp, y_nsp, s=scatter_size, marker='s', edgecolors=(0.4940, 0.1840, 0.5560),facecolors="none",
                label='Nearly shortest path node')
    plt.scatter(x_close, y_close, s=scatter_size, marker='^', edgecolors=(0.4660, 0.6740, 0.1880),facecolors="none",
                label='Top 100 nodes along the geodesic')

    # Configure plot
    plt.xlabel('Node Relevance', fontsize=30)
    plt.ylabel('Deviation', fontsize=30)
    # plt.xscale('log')
    plt.legend()

    # Save the plot as an EPS file
    picname = 'D:\\data\\geometric shortest path problem\\SSRGG\\VisualizeSPNSPDe\\testPYVisualizeSPNSPDe.eps'
    plt.savefig(picname, format='eps', dpi=600)

    plt.show()  # Display the plot


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # tic = time.time()
    VisualizationDataOfSPNSPGEO()
    # print(time.time()-tic)
    PlotVisualizationDataOfSPNSPGEO()





