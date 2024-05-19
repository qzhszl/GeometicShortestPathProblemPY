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
    dist_to_geodesic_S2, distS2, loadNodeSSRGG
from sklearn.metrics import precision_recall_curve, auc
import sys
import seaborn as sns
import pandas as pd


# Function to find nodes that frequently appear in the shortest paths
def FindNearlySPNodes(G, nodei, nodej, RelevanceSimTimes=1000):
    """
    Main function to find nearly shortest paths, where 10% links are removed when finding the NSP
    :param G:
    :param nodei:
    :param nodej:
    :param RelevanceSimTimes:
    :return: NSPlist and relevance for each node
    """
    N = G.number_of_nodes()  # Total number of nodes in the graph
    Noderelevance = np.zeros(N)  # Initialize node relevance

    # Simulate the removal of random edges and calculate shortest paths
    for Simutime in range(RelevanceSimTimes):
        # print("NSP Simutime:",Simutime)
        ShuffleTable = np.random.rand(G.number_of_edges())  # Random numbers for shuffle
        H = G.copy()  # Create a copy of the graph
        edges_to_remove = [e for e, shuffle_value in zip(G.edges, ShuffleTable) if shuffle_value < 0.1]
        H.remove_edges_from(edges_to_remove)  # Remove edges with shuffle value < 0.5
        # time3 = time.time()

        # Find all shortest paths between nodei and nodej
        try:
            # timespecial = time.time()
            # shortest_paths = nx.all_shortest_paths(H, nodei, nodej)
            # print("timeallsp",time.time()-timespecial)
            # print("pathlength", sum(1 for _ in shortest_paths))
            shortest_paths = nx.all_shortest_paths(H, nodei, nodej)
            # time30 = time.time()
            # print("timeallsppathonly0", time30 - time3)
            PNodeList = set()  # Use a set to keep unique nodes
            count = 0
            for path in shortest_paths:
                PNodeList.update(path)
                count += 1
                if count > 1000000:
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


def FindNearlySPNodesRemoveSpecficLink(G, nodei, nodej, Linkremoveratio=0.1, RelevanceSimTimes=1000):
    """
    Main function to find nearly shortest paths, where Linkremoveratio%(default is 10%) links are removed when finding the NSP
   :param G:
   :param nodei:
   :param nodej:
   :param RelevanceSimTimes:
   :return: NSPlist and relevance for each node
   """
    N = G.number_of_nodes()  # Total number of nodes in the graph
    Noderelevance = np.zeros(N)  # Initialize node relevance

    # Simulate the removal of random edges and calculate shortest paths
    for Simutime in range(RelevanceSimTimes):
        # print("NSP Simutime:",Simutime)
        ShuffleTable = np.random.rand(G.number_of_edges())  # Random numbers for shuffle
        H = G.copy()  # Create a copy of the graph
        edges_to_remove = [e for e, shuffle_value in zip(G.edges, ShuffleTable) if shuffle_value < Linkremoveratio]
        H.remove_edges_from(edges_to_remove)  # Remove edges with shuffle value < 0.5
        # time3 = time.time()

        # Find all shortest paths between nodei and nodej
        try:
            # timespecial = time.time()
            # shortest_paths = nx.all_shortest_paths(H, nodei, nodej)
            # print("timeallsp",time.time()-timespecial)
            # print("pathlength", sum(1 for _ in shortest_paths))
            shortest_paths = nx.all_shortest_paths(H, nodei, nodej)
            # time30 = time.time()
            # print("timeallsppathonly0", time30 - time3)
            PNodeList = set()  # Use a set to keep unique nodes
            count = 0
            for path in shortest_paths:
                PNodeList.update(path)
                count += 1
                if count > 1000000:
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
    """
    A test experiment for finding nearly shortest path nodes
    :return:
    """
    # Demo 1, use a and
    rg = RandomGenerator(-12)  # Seed initialization
    N = 10000
    avg = 100
    beta = 5.19042969
    tic = time.time()
    print(tic)
    G, Coortheta, Coorphi = SphericalSoftRGGwithGivenNode(N, avg, beta, rg, math.pi / 2, 0, math.pi / 2, 1)
    toc1 = time.time()
    print("Genenrate a graph time:", time.time() - tic)
    print("CLU:", nx.average_clustering(G))
    nodei = N - 1
    nodej = N - 2
    NSP, NSPrelevance = FindNearlySPNodes(G, nodei, nodej, RelevanceSimTimes=1000)
    print("NSP nodes Num:", len(NSP))
    print("Finding NSP time:", time.time() - toc1)
    plt.hist(NSPrelevance)
    plt.show()


def relevance_vs_link_remove():
    N = 10000
    avg = 5
    beta = 3.5
    rg = RandomGenerator(-12)
    for _ in range(random.randint(0,100)):
        rg.ran1()

    # Network and coordinates
    G, Coortheta, Coorphi = SphericalSoftRGGwithGivenNode(N, avg, beta, rg, math.pi / 4, 0, 3 * math.pi / 8, 0)
    print("LinkNum:", G.number_of_edges())
    print("AveDegree:", G.number_of_edges() * 2 / G.number_of_nodes())
    print("ClusteringCoefficient:", nx.average_clustering(G))

    FileNetworkName = "D:\\data\\geometric shortest path problem\\SSRGG\\relevance\\data20240515\\testPYnetwork.txt"
    nx.write_edgelist(G, FileNetworkName)
    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\SSRGG\\relevance\\data20240515\\testPYnetworkCoor.txt"
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

    FileASPName = "D:\\data\\geometric shortest path problem\\SSRGG\\relevance\\data20240515\\testPYASP.txt"
    np.savetxt(FileASPName, AllSPlist, fmt="%i")

    # All shortest path node
    AllSPNode = set()
    for path in AllSPlist:
        AllSPNode.update(path)
    AllSPNode.discard(nodei)
    AllSPNode.discard(nodej)
    FileASPNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\relevance\\data20240515\\testPYASPNode.txt"
    np.savetxt(FileASPNodeName, list(AllSPNode), fmt="%i")

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
    FileGeodistanceName = "D:\\data\\geometric shortest path problem\\SSRGG\\relevance\\data20240515\\testPYGeoDistance.txt"
    np.savetxt(FileGeodistanceName, list(Geodistance.values()), fmt="%.8f")
    Geodistance = sorted(Geodistance.items(), key=lambda kv: (kv[1], kv[0]))
    Geodistance = Geodistance[:102]
    Top100closednode = [t[0] for t in Geodistance]
    Top100closednode = [n for n in Top100closednode if n not in [nodei, nodej]]

    FileTop100closedNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\relevance\\data20240515\\testPYFileTop100closedNode.txt"
    np.savetxt(FileTop100closedNodeName, Top100closednode, fmt="%i")

    # Nearly shortest path node
    for Linkremoveratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
        print("RemoveLINK:", Linkremoveratio)
        NSPNode, relevance = FindNearlySPNodesRemoveSpecficLink(G, nodei, nodej, Linkremoveratio=Linkremoveratio)
        print("NSP num", len(NSPNode))
        FileNSPNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\relevance\\data20240515\\testPYNSPNodeLinkRemoveRatio{ratio}.txt".format(
            ratio=Linkremoveratio)
        np.savetxt(FileNSPNodeName, NSPNode, fmt="%i")
        FileNodeRelevanceName = "D:\\data\\geometric shortest path problem\\SSRGG\\relevance\\data20240515\\testPYRelevance{ratio}.txt".format(
            ratio=Linkremoveratio)
        np.savetxt(FileNodeRelevanceName, relevance, fmt="%.8f")

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


def plot_relevance_vs_link_remove():
    relevance_matrix = np.zeros((10000,5))
    Linkremoveratio_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    for Linkremoveratioindex in range(5):
        Linkremoveratio = Linkremoveratio_list[Linkremoveratioindex]
        FileNodeRelevanceName = "D:\\data\\geometric shortest path problem\\SSRGG\\relevance\\testPYRelevance{ratio}.txt".format(
            ratio=Linkremoveratio)
        relevance_matrix[:,Linkremoveratioindex] = np.loadtxt(FileNodeRelevanceName)

    FileNSPNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\relevance\\testPYASPNode.txt"
    ASPlist = np.loadtxt(FileNSPNodeName, dtype=int)


    for i in range(5):
        for j in range(i+1,5):
            print(i,j)
            plt.figure()
            scatter_size = 100
            plt.scatter(relevance_matrix[:,i], relevance_matrix[:,j], marker="o", edgecolors=(0, 0.4470, 0.7410), facecolors="none",
                        label='General node')
            plt.scatter(relevance_matrix[ASPlist,i], relevance_matrix[ASPlist,j], s=scatter_size, marker='*', edgecolors=(0.8500, 0.3250, 0.0980), facecolors="none",
                        label='All shortest path nodes')
            xlabelname = "relevance by removing {ix}0% links".format(ix=i+1)
            plt.xlabel(xlabelname)
            ylabelname = "relevance by removing {jx}0% links".format(jx=j+1)
            plt.ylabel(ylabelname)
            picfilename = "D:\\data\\geometric shortest path problem\\SSRGG\\relevance\\ScatterCor0{ix}vs0{jx}.pdf".format(ix=i+1,jx=j+1)
            plt.legend()
            plt.savefig(picfilename,
                format='pdf', bbox_inches='tight', dpi=600)
            plt.show()

    # print(ASPlist)
    # plt.figure()
    # ASPnode_relevance_change = relevance_matrix[ASPlist,:]
    # for i in range(len(ASPlist)):
    #     plt.plot([1,2,3,4,5], ASPnode_relevance_change[i])
    # plt.xticks([1,2,3,4,5],["10%","20%","30%","40%","50%"])
    # plt.show()





    # plt.show()


    # the correlation heatmap
    # data = pd.DataFrame(relevance_matrix, columns=["10%", "20%", "30%", "40%","50%"])
    # # 计算相关性矩阵
    # correlation_matrix = data.corr()
    # print(correlation_matrix)  # 显示相关性矩阵
    # # 绘制热力图
    # plt.figure(figsize=(8, 6))  # 设置图形大小
    # sns.heatmap(correlation_matrix, annot=True, cmap="Blues", fmt=".2f", linewidths=0.5, cbar=True,
    #             cbar_kws={'label': 'Pearson Correlation'})
    # # 设置标题和轴标签
    # plt.title("Correlation Heatmap")
    # plt.xlabel("Links removed when finding nearly shortest path nodes")
    # plt.ylabel("Links removed when finding nearly shortest path nodes")
    # plt.savefig(
    #     "D:\\data\\geometric shortest path problem\\SSRGG\\relevance\\PearsonHeatmap.pdf",
    #     format='pdf', bbox_inches='tight', dpi=600)
    # # 显示图形
    # plt.show()

def test_two_brunch_in_relevance_with_linkremoval():
    # load data
    FileNetworkName = "D:\\data\\geometric shortest path problem\\SSRGG\\relevance\\testPYnetwork.txt"
    G = loadNodeSSRGG(10000, FileNetworkName)

    degreedic = dict(sorted(dict(G.degree()).items()))
    degreelist = [val for (node, val) in degreedic.items()]

    relevance_matrix = np.zeros((10000, 5))
    Linkremoveratio_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    for Linkremoveratioindex in range(5):
        Linkremoveratio = Linkremoveratio_list[Linkremoveratioindex]
        FileNodeRelevanceName = "D:\\data\\geometric shortest path problem\\SSRGG\\relevance\\testPYRelevance{ratio}.txt".format(
            ratio=Linkremoveratio)
        relevance_matrix[:, Linkremoveratioindex] = np.loadtxt(FileNodeRelevanceName)

    FileASPNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\relevance\\testPYASPNode.txt"
    ASPlist = np.loadtxt(FileASPNodeName, dtype=int)
    FileNSPNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\relevance\\testPYNSPNodeLinkRemoveRatio0.5.txt"
    NSPlist = np.loadtxt(FileNSPNodeName, dtype=int)


    # plot figure
    plt.figure()
    scatter_size = 100
    # plt.scatter(relevance_matrix[:,4], relevance_matrix[:,0], marker="o", edgecolors=(0, 0.4470, 0.7410),
    #             facecolors="none",
    #             label='General node')
    plt.scatter(relevance_matrix[:,0], degreelist, marker="o", edgecolors=(0, 0.4470, 0.7410),
                facecolors="none",
                label='General node')
    plt.scatter(relevance_matrix[ASPlist, 0], [degreelist[i] for i in ASPlist], s=scatter_size, marker='*',
                edgecolors=(0.8500, 0.3250, 0.0980), facecolors="none",
                label='All shortest path nodes')
    plt.scatter(relevance_matrix[NSPlist, 0], [degreelist[i] for i in NSPlist], s=scatter_size, marker='s', edgecolors=(0.4940, 0.1840, 0.5560), facecolors="none",
                label='NSP 50% link removal')

    xlabelname = "relevance by removing {jx}0% links".format(jx=0 + 1)
    plt.xlabel(xlabelname)
    ylabelname = "Degree"
    plt.ylabel(ylabelname)
    plt.legend()

    picfilename = "D:\\data\\geometric shortest path problem\\SSRGG\\relevance\\ScatterDegreeCor0{ix}vs0{jx}.pdf".format(
        ix=0 + 1, jx=4 + 1)
    plt.savefig(picfilename,
                format='pdf', bbox_inches='tight', dpi=600)
    # plt.show()


    # 3dfigure
    x = relevance_matrix[:,0]
    y = relevance_matrix[:,4]
    z = np.zeros_like(x)
    dx = np.ones_like(x)*0.001
    dy = np.ones_like(y)*0.001
    dz = degreelist

    # 创建 3D 图形对象
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制柱状图
    ax.bar3d(x, y, z, dx, dy, dz, edgecolor='k')

    # 添加透明圆圈
    for xi, yi, zi in zip(x, y, dz):
        ax.scatter3D(xi, yi, zi, color='none', edgecolor=(0, 0.4470, 0.7410), s=100)
    for xi, yi, zi in zip([x[i] for i in ASPlist], [y[i] for i in ASPlist], [dz[i] for i in ASPlist]):
        ax.scatter3D(xi, yi, zi, color='none',marker='*',edgecolor=(0.8500, 0.3250, 0.0980), s=100)

    # 设置图形标题和轴标签
    # ax.set_title('3D Bar Plot with Scatter Points')
    ax.set_xlabel('relevance by removing 10% links')
    ax.set_ylabel('relevance by removing 50% links')
    ax.set_zlabel('Degree')
    picfilename = "D:\\data\\geometric shortest path problem\\SSRGG\\relevance\\d3ScatterDegreeCor0{ix}vs0{jx}.pdf".format(
        ix=0 + 1, jx=4 + 1)
    plt.savefig(picfilename,
                format='pdf', bbox_inches='tight', dpi=600)
    # 显示图形
    plt.show()


def test_two_brunch_in_relevance_with_linkremoval_betweenness():
    # load data
    FileNetworkName = "D:\\data\\geometric shortest path problem\\SSRGG\\relevance\\testPYnetwork.txt"
    G = loadNodeSSRGG(10000, FileNetworkName)

    betweenness_dict = nx.betweenness_centrality(G)
    degreelist = [betweenness_dict[i] for i in range(G.number_of_nodes())]
    FileBcName = "D:\\data\\geometric shortest path problem\\SSRGG\\relevance\\testPYbetweenness.txt"
    np.savetxt(FileBcName, degreelist)

    relevance_matrix = np.zeros((10000, 5))
    Linkremoveratio_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    for Linkremoveratioindex in range(5):
        Linkremoveratio = Linkremoveratio_list[Linkremoveratioindex]
        FileNodeRelevanceName = "D:\\data\\geometric shortest path problem\\SSRGG\\relevance\\testPYRelevance{ratio}.txt".format(
            ratio=Linkremoveratio)
        relevance_matrix[:, Linkremoveratioindex] = np.loadtxt(FileNodeRelevanceName)

    FileASPNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\relevance\\testPYASPNode.txt"
    ASPlist = np.loadtxt(FileASPNodeName, dtype=int)
    FileNSPNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\relevance\\testPYNSPNodeLinkRemoveRatio0.5.txt"
    NSPlist = np.loadtxt(FileNSPNodeName, dtype=int)

    # plot figure
    plt.figure()
    scatter_size = 100
    # plt.scatter(relevance_matrix[:,4], relevance_matrix[:,0], marker="o", edgecolors=(0, 0.4470, 0.7410),
    #             facecolors="none",
    #             label='General node')
    plt.scatter(relevance_matrix[:, 0], degreelist, marker="o", edgecolors=(0, 0.4470, 0.7410),
                facecolors="none",
                label='General node')
    plt.scatter(relevance_matrix[ASPlist, 0], [degreelist[i] for i in ASPlist], s=scatter_size, marker='*',
                edgecolors=(0.8500, 0.3250, 0.0980), facecolors="none",
                label='All shortest path nodes')
    plt.scatter(relevance_matrix[NSPlist, 0], [degreelist[i] for i in NSPlist], s=scatter_size, marker='s',
                edgecolors=(0.4940, 0.1840, 0.5560), facecolors="none",
                label='NSP 50% link removal')

    xlabelname = "Betweenness"
    plt.xlabel(xlabelname)
    ylabelname = "relevance by removing {jx}0% links".format(jx=0 + 1)
    plt.ylabel(ylabelname)
    plt.legend()

    picfilename = "D:\\data\\geometric shortest path problem\\SSRGG\\relevance\\ScatterDegreeCor0{ix}vs0{jx}.pdf".format(
        ix=0 + 1, jx=4 + 1)
    plt.savefig(picfilename,
                format='pdf', bbox_inches='tight', dpi=600)
    # plt.show()

    # 3dfigure
    x = relevance_matrix[:, 0]
    y = relevance_matrix[:, 4]
    z = np.zeros_like(x)
    dx = np.ones_like(x) * 0.001
    dy = np.ones_like(y) * 0.001
    dz = degreelist

    # 创建 3D 图形对象
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制柱状图
    ax.bar3d(x, y, z, dx, dy, dz, edgecolor='k')

    # 添加透明圆圈
    for xi, yi, zi in zip(x, y, dz):
        ax.scatter3D(xi, yi, zi, color='none', edgecolor=(0, 0.4470, 0.7410), s=100)
    for xi, yi, zi in zip([x[i] for i in ASPlist], [y[i] for i in ASPlist], [dz[i] for i in ASPlist]):
        ax.scatter3D(xi, yi, zi, color='none', marker='*', edgecolor=(0.8500, 0.3250, 0.0980), s=100)

    # 设置图形标题和轴标签
    # ax.set_title('3D Bar Plot with Scatter Points')
    ax.set_xlabel('relevance by removing 10% links')
    ax.set_ylabel('relevance by removing 50% links')
    ax.set_zlabel('betweenness')
    picfilename = "D:\\data\\geometric shortest path problem\\SSRGG\\relevance\\d3ScatterDegreeCor0{ix}vs0{jx}.pdf".format(
        ix=0 + 1, jx=4 + 1)
    plt.savefig(picfilename,
                format='pdf', bbox_inches='tight', dpi=600)
    # 显示图形
    plt.show()


def test_two_brunch_in_relevance_with_hopcount():
    # load data
    FileNetworkName = "D:\\data\\geometric shortest path problem\\SSRGG\\relevance\\testPYnetwork.txt"
    G = loadNodeSSRGG(10000, FileNetworkName)
    N = G.number_of_nodes()
    hopcountlength_list=[]
    for nodeindex in range(N):
        if nx.has_path(G, nodeindex, N-2):
            d1 = nx.shortest_path_length(G, nodeindex, N-2)
            d2 = nx.shortest_path_length(G, nodeindex, N-1)
            hopcountlength_list.append(min(d1, d2))
        else:
            hopcountlength_list.append(0)

    relevance_matrix = np.zeros((10000, 5))
    Linkremoveratio_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    for Linkremoveratioindex in range(5):
        Linkremoveratio = Linkremoveratio_list[Linkremoveratioindex]
        FileNodeRelevanceName = "D:\\data\\geometric shortest path problem\\SSRGG\\relevance\\testPYRelevance{ratio}.txt".format(
            ratio=Linkremoveratio)
        relevance_matrix[:, Linkremoveratioindex] = np.loadtxt(FileNodeRelevanceName)

    FileASPNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\relevance\\testPYASPNode.txt"
    ASPlist = np.loadtxt(FileASPNodeName, dtype=int)
    FileNSPNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\relevance\\testPYNSPNodeLinkRemoveRatio0.5.txt"
    NSPlist = np.loadtxt(FileNSPNodeName, dtype=int)


    # plot figure
    plt.figure()
    scatter_size = 100
    # plt.scatter(relevance_matrix[:,4], relevance_matrix[:,0], marker="o", edgecolors=(0, 0.4470, 0.7410),
    #             facecolors="none",
    #             label='General node')
    plt.scatter(relevance_matrix[:,0], hopcountlength_list, marker="o", edgecolors=(0, 0.4470, 0.7410),
                facecolors="none",
                label='General node')
    plt.scatter(relevance_matrix[ASPlist, 0], [hopcountlength_list[i] for i in ASPlist], s=scatter_size, marker='*',
                edgecolors=(0.8500, 0.3250, 0.0980), facecolors="none",
                label='All shortest path nodes')
    plt.scatter(relevance_matrix[NSPlist, 0], [hopcountlength_list[i] for i in NSPlist], s=scatter_size, marker='s', edgecolors=(0.4940, 0.1840, 0.5560), facecolors="none",
                label='NSP 50% link removal')

    xlabelname = "relevance by removing {jx}0% links".format(jx=0 + 1)
    plt.xlabel(xlabelname)
    ylabelname = "Hopcount"
    plt.ylabel(ylabelname)
    plt.yticks([0,3,6,9,12,15,18])
    plt.legend()

    picfilename = "D:\\data\\geometric shortest path problem\\SSRGG\\relevance\\ScatterHopCor0{ix}vs0{jx}.pdf".format(
        ix=0 + 1, jx=4 + 1)
    plt.savefig(picfilename,
                format='pdf', bbox_inches='tight', dpi=600)
    # plt.show()


    # 3dfigure
    x = relevance_matrix[:,0]
    y = relevance_matrix[:,4]
    z = np.zeros_like(x)
    dx = np.ones_like(x)*0.001
    dy = np.ones_like(y)*0.001
    dz = hopcountlength_list

    # 创建 3D 图形对象
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制柱状图
    ax.bar3d(x, y, z, dx, dy, dz, edgecolor='k')

    # 添加透明圆圈
    for xi, yi, zi in zip(x, y, dz):
        ax.scatter3D(xi, yi, zi, color='none', edgecolor=(0, 0.4470, 0.7410), s=100)
    for xi, yi, zi in zip([x[i] for i in ASPlist], [y[i] for i in ASPlist], [dz[i] for i in ASPlist]):
        ax.scatter3D(xi, yi, zi, color='none',marker='*',edgecolor=(0.8500, 0.3250, 0.0980), s=100)

    # 设置图形标题和轴标签
    # ax.set_title('3D Bar Plot with Scatter Points')
    ax.set_xlabel('relevance by removing 10% links')
    ax.set_ylabel('relevance by removing 50% links')
    ax.set_zlabel('Hopcount')
    ax.set_zticks([0, 3, 6, 9, 12, 15, 18])
    picfilename = "D:\\data\\geometric shortest path problem\\SSRGG\\relevance\\d3ScatterHopCor0{ix}vs0{jx}.pdf".format(
        ix=0 + 1, jx=4 + 1)
    plt.savefig(picfilename,
                format='pdf', bbox_inches='tight', dpi=600)
    # 显示图形
    plt.show()


def testPRAUCvsPositiveCase():
    """
    Test how PRAUC change with the increment of positive case. Though the distance/score are the same,
    the PRAUC will increase with the increase of actual postive case
    :return:
    """
    fig, axs = plt.subplots(2)
    Label_med = [0, 0, 0, 0, 1, 0, 1, 0]
    distance_score = [1, 2, 3, 1.5, 0.14, 0.13, 0.12, 0.11]
    distance_score = [1 / x for x in distance_score]
    precisions, recalls, _ = precision_recall_curve(Label_med, distance_score)
    print(precisions)
    print(recalls)
    AUCWithoutNornodeij = auc(recalls, precisions)
    print(AUCWithoutNornodeij)
    axs[0].plot(recalls, precisions)

    Label_med = [0, 0, 0, 1, 1, 0, 1, 0]
    distance_score = [1, 2, 3, 1.5, 0.14, 0.13, 0.12, 0.11]
    distance_score = [1 / x for x in distance_score]
    precisions, recalls, _ = precision_recall_curve(Label_med, distance_score)
    print(precisions)
    print(recalls)
    AUCWithoutNornodeij = auc(recalls, precisions)
    print(AUCWithoutNornodeij)
    axs[1].plot(recalls, precisions)
    plt.show()




def GeodiscPRAUC(Edindex, betaindex, ExternalSimutime):
    """
    :param ED: average degree
    :param beta: parameter to control the clustering coefficient
    :return: ED and beta
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
    NSPnum_nodepair = []  # save the Number of nearly shortest path for each node pair
    geodistance_between_nodepair = []  # save the geodeisc length between each node pair

    rg = RandomGenerator(-12)
    for i in range(random.randint(0, 100)):
        rg.ran1()

    G, CoorTheta, CoorPhi = SphericalSoftRGG(N, ED, beta, rg)
    while abs(nx.average_clustering(G) - CC) > 0.1:
        G, CoorTheta, CoorPhi = SphericalSoftRGG(N, ED, beta, rg)
    print("We have a network now!")
    FileNetworkName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\NSP0_1LinkRemove\\NetworkED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    nx.write_edgelist(G, FileNetworkName)
    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\NSP0_1LinkRemove\\CoorED{EDn}Beta{betan}PYSimu{ST}.txt".format(
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
    filename_selecetednodepair = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\NSP0_1LinkRemove\\SelecetedNodepairED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(filename_selecetednodepair, random_pairs, fmt="%i")

    for nodepair in random_pairs:
        count = count + 1
        print(count, "Simu")
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

    # Calculate means and standard deviations of AUC
    # AUCWithoutnorMean = np.mean(PRAUC_nodepair[~np.isnan(PRAUC_nodepair)])
    # AUCWithoutnorStd = np.std(PRAUC_nodepair[~np.isnan(PRAUC_nodepair)])

    PRAUCName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\NSP0_1LinkRemove\\AUCED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(PRAUCName, PRAUC_nodepair)

    NSPnum_nodepairName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\NSP0_1LinkRemove\\NSPNumED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(NSPnum_nodepairName, NSPnum_nodepair, fmt="%i")

    geodistance_between_nodepair_Name = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\NSP0_1LinkRemove\\GeodistanceBetweenTwoNodesED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(geodistance_between_nodepair_Name, geodistance_between_nodepair)

    print("Mean AUC Without Normalization:", np.mean(PRAUC_nodepair))
    print("Standard Deviation of AUC Without Normalization:", np.std(PRAUC_nodepair))


def PlotPRAUC():
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
    for EDindex in range(7):
        ED_list = [5, 7, 10, 15, 20, 50, 100]  # Expected degrees
        ED = ED_list[EDindex]
        print("ED:", ED)

        for betaindex in range(5):
            CC_list = [0.2, 0.25, 0.3, 0.35, 0.4]  # Clustering coefficients
            CC = CC_list[betaindex]
            print(CC)
            beta = beta_list[EDindex][betaindex]
            print(beta)
            PRAUC_list = []
            for ExternalSimutime in range(9):
                PRAUCName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\NSP0_1LinkRemove\\AUCED{EDn}Beta{betan}PYSimu{ST}.txt".format(
                    EDn=ED, betan=beta, ST=ExternalSimutime)
                PRAUC_list_10times = np.loadtxt(PRAUCName)
                PRAUC_list.extend(PRAUC_list_10times)
            PRAUC_list = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PRAUC_list))
            mean_PRAUC = np.mean(PRAUC_list)
            PRAUC_matrix[EDindex][betaindex] = mean_PRAUC
            print(mean_PRAUC)
            print(PRAUC_matrix)
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
                      index=[ED_list],  # DataFrame的行标签设置为大写字母
                      columns=CC_list)  # 设置DataFrame的列标签
    sns.heatmap(data=df, vmin=0, annot=True, fmt=".2f", cbar=True,
                cbar_kws={'label': 'PRAUC'})
    # plt.title("50% links are removed when computing Nearly Shortest Path Node")
    plt.xlabel("clustering coefficient")
    plt.ylabel("average degree")
    plt.savefig(
        "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\NSP0_1LinkRemove\\PRAUCHeatmapNSP0_1LinkRemove.pdf",
        format='pdf', bbox_inches='tight', dpi=600)
    plt.show()


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
                PRAUCName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\NSP0_1LinkRemove\\ControlFreAUCED{EDn}Beta{betan}PYSimu{ST}.txt".format(
                    EDn=ED, betan=beta, ST=ExternalSimutime)
                PRAUC_list_10times = np.loadtxt(PRAUCName)
                PRAUC_list.extend(PRAUC_list_10times)
            PRAUC_list = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PRAUC_list))
            mean_PRAUC = np.mean(PRAUC_list)
            PRAUC_matrix[EDindex][betaindex] = mean_PRAUC
            print(mean_PRAUC)
            print(PRAUC_matrix)
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
                      index=[ED_list],  # DataFrame的行标签设置为大写字母
                      columns=CC_list)  # 设置DataFrame的列标签
    sns.heatmap(data=df, vmin=0, annot=True, fmt=".2f", cbar=True,
                cbar_kws={'label': 'PRAUC'})
    plt.title("Frequency")
    plt.xlabel("clustering coefficient")
    plt.ylabel("average degree")
    plt.savefig(
        "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\NSP0_1LinkRemove\\ControlFrePRAUCHeatmapNSP0_1LinkRemove.pdf",
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
                PRAUCName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\NSP0_1LinkRemove\\AUCED{EDn}Beta{betan}PYSimu{ST}.txt".format(
                    EDn=ED, betan=beta, ST=ExternalSimutime)
                PRAUC_list_10times = np.loadtxt(PRAUCName)
                PRAUC_list.extend(PRAUC_list_10times)
            PRAUC_list = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PRAUC_list))
            mean_PRAUC = np.mean(PRAUC_list)
            PRAUC_matrix[EDindex][betaindex] = mean_PRAUC
            print(mean_PRAUC)
            print(PRAUC_matrix)
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
                      index=[ED_list],  # DataFrame的行标签设置为大写字母
                      columns=CC_list)  # 设置DataFrame的列标签
    sns.heatmap(data=df, vmin=0, annot=True, fmt=".2f", cbar=True,
                cbar_kws={'label': 'PRAUC'})
    plt.title("Frequency")
    plt.xlabel("clustering coefficient")
    plt.ylabel("average degree")
    plt.savefig(
        "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\NSP0_1LinkRemove\\GeoDisPRAUCHeatmapNSP0_1LinkRemove.pdf",
        format='pdf', bbox_inches='tight', dpi=600)
    plt.show()


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
        FreNodeList, _ = FindNearlySPNodesRemoveSpecficLink(H, nodei, nodej, Linkremoveratio=0.1)
        for node in FreNodeList:
            NodeFrequency[node] += 1
        print(time.time()-tic)
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
    FileNetworkName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\Compare\\NetworkED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    nx.write_edgelist(G, FileNetworkName)
    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\Compare\\CoorED{EDn}Beta{betan}PYSimu{ST}.txt".format(
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
    filename_selecetednodepair = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\Compare\\SelecetedNodepairED{EDn}Beta{betan}PYSimu{ST}.txt".format(
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

    PRAUCName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\Compare\\AUCED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(PRAUCName, PRAUC_nodepair)

    FrePRAUCName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\Compare\\ControlFreAUCED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(FrePRAUCName, PRAUC_fre_controlgroup_nodepair)

    NSPnum_nodepairName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\Compare\\NSPNumED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(NSPnum_nodepairName, NSPnum_nodepair, fmt="%i")

    geodistance_between_nodepair_Name = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\Compare\\GeodistanceBetweenTwoNodesED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(geodistance_between_nodepair_Name, geodistance_between_nodepair)

    print("Mean AUC Without Normalization:", np.mean(PRAUC_nodepair))
    print("Mean AUC CONTROL:", np.mean(PRAUC_fre_controlgroup_nodepair))
    # print("Standard Deviation of AUC Without Normalization:", np.std(PRAUC_nodepair))


def frequency_controlgroup_PRAUC_givennodepair():
     # Input data parameters
    N = 10000
    avg = 5
    beta = 3.69375
    rg = RandomGenerator(-12)
    # for _ in range(random.randint(0,100)):
    #     rg.ran1()

    # Network and coordinates
    G,Coortheta,Coorphi = SphericalSoftRGGwithGivenNode(N,avg,beta,rg,math.pi/4,0,3*math.pi/8,0)
    print("LinkNum:",G.number_of_edges())
    print("AveDegree:", G.number_of_edges()*2/G.number_of_nodes())
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
    components=[]
    largest_component=[]
    nodes=[]
    unique_pairs=[]
    unique_pairs=[]
    # test end here

    nodei = N-2
    nodej = N-1

    print("Node Geo distance",distS2(Coortheta[nodei], Coorphi[nodei], Coortheta[nodej], Coorphi[nodej]))
    # All shortest paths
    AllSP = nx.all_shortest_paths(G, nodei, nodej)
    AllSPlist = list(AllSP)
    print("SPnum",len(AllSPlist))
    print("SPlength",len(AllSPlist[0]))

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
    print("time for finding NSP", time.time()-tic)
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

    Geodistance={}
    for NodeC in range(N):
        if NodeC in [nodei,nodej]:
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
    Top100closednode = [n for n in Top100closednode if n not in [nodei,nodej]]

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
    print("PRAUC",AUCWithoutNornodeij)

    # Calculate precision-recall curve and AUC for control group
    node_fre = nodeNSPfrequency(N, avg, beta, rg, Coortheta, Coorphi, nodei, nodej)
    FileNodefreName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\ControlGroupNodefre.txt"
    np.savetxt(FileNodefreName, node_fre)

    node_fre = np.delete(node_fre, [nodei, nodej])
    precisionsfre, recallsfre, _ = precision_recall_curve(Label_med, node_fre)
    AUCfrenodeij = auc(recallsfre, precisionsfre)
    print(AUCfrenodeij)


def frequency_controlgroup_PRAUC_givennodepair_diffgeolength(theta_A,phi_A,theta_B,phi_B):
    # Input data parameters
    N = 10000
    avg = 5
    beta = 3.69375
    random.seed()
    rg = RandomGenerator(-12)
    for _ in range(random.randint(0,100)):
        rg.ran1()

    # Network and coordinates
    G, Coortheta, Coorphi = SphericalSoftRGGwithGivenNode(N, avg, beta, rg, theta_A, phi_A, theta_B, phi_B)
    print("LinkNum:", G.number_of_edges())
    print("AveDegree:", G.number_of_edges() * 2 / G.number_of_nodes())
    print("ClusteringCoefficient:", nx.average_clustering(G))
    geo_length = distS2(theta_A, phi_A, theta_B, phi_B)
    print("Geo Length:", geo_length/math.pi)


    FileNetworkName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\Geolength\\ControlGroupnetworkGeolen{le}.txt".format(le = geo_length)
    nx.write_edgelist(G, FileNetworkName)
    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\Geolength\\ControlGroupnetworkCoorGeolen{le}.txt".format(le = geo_length)
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

    FileASPName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\Geolength\\ControlGroupASPGeolen{le}.txt".format(le = geo_length)
    np.savetxt(FileASPName, AllSPlist, fmt="%i")

    # All shortest path node
    AllSPNode = set()
    for path in AllSPlist:
        AllSPNode.update(path)
    AllSPNode.discard(nodei)
    AllSPNode.discard(nodej)
    FileASPNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\Geolength\\ControlGroupASPNodeGeolen{le}.txt".format(le = geo_length)
    np.savetxt(FileASPNodeName, list(AllSPNode), fmt="%i")
    tic = time.time()
    # Nearly shortest path node
    NSPNode, relevance = FindNearlySPNodes(G, nodei, nodej)
    print("time for finding NSP", time.time() - tic)
    print("NSP num", len(NSPNode))
    FileNSPNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\Geolength\\ControlGroupNSPNodeGeolen{le}.txt".format(le = geo_length)
    np.savetxt(FileNSPNodeName, NSPNode, fmt="%i")
    FileNodeRelevanceName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\Geolength\\ControlGroupRelevanceGeolen{le}.txt".format(le = geo_length)
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
    FileGeodistanceName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\Geolength\\ControlGroupGeoDistanceGeolen{le}.txt".format(le = geo_length)
    np.savetxt(FileGeodistanceName, list(Geodistance.values()), fmt="%.8f")
    Geodistance = sorted(Geodistance.items(), key=lambda kv: (kv[1], kv[0]))
    Geodistance = Geodistance[:102]
    Top100closednode = [t[0] for t in Geodistance]
    Top100closednode = [n for n in Top100closednode if n not in [nodei, nodej]]

    FileTop100closedNodeName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\Geolength\\ControlGroupTop100closedNodeGeolen{le}.txt".format(le = geo_length)
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
    FileNodefreName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\ControlGroup\\Geolength\\ControlGroupNodefreGeolen{le}.txt".format(le = geo_length)
    np.savetxt(FileNodefreName, node_fre)

    node_fre = np.delete(node_fre, [nodei, nodej])
    precisionsfre, recallsfre, _ = precision_recall_curve(Label_med, node_fre)
    AUCfrenodeij = auc(recallsfre, precisionsfre)
    print(AUCfrenodeij)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # TestFindNSPnodes()


    # ED = sys.argv[1]
    # beta = sys.argv[2]
    # ExternalSimutime = sys.argv[3]
    # GeodiscPRAUC(int(ED),int(beta),int(ExternalSimutime))



    # GeodiscPRAUC(0,0,0)



    # PlotPRAUC()



    # relevance_vs_link_remove()



    # plot_relevance_vs_link_remove()




    # test_two_brunch_in_relevance_with_linkremoval()


    # test_two_brunch_in_relevance_with_linkremoval_betweenness()


    # test_two_brunch_in_relevance_with_hopcount()


    # frequency_controlgroup_PRAUC_givennodepair()


    # ED = sys.argv[1]
    # beta = sys.argv[2]
    # ExternalSimutime = sys.argv[3]
    # frequency_controlgroup_PRAUC(int(ED),int(beta),int(ExternalSimutime))


    # frequency_controlgroup_PRAUC(0, 0, 0)
    # frequency_controlgroup_PRAUC(0,0,0)


    for (theta_A,theta_B) in [(8*math.pi/16, 9*math.pi/16),(4*math.pi/8, 5*math.pi/8),(2*math.pi/4, 3*math.pi/4),(3*math.pi/4, math.pi/4)]:
        geo_length = distS2(theta_A, 0, theta_B, 0)
        print("Geo Length:", geo_length/math.pi)
        frequency_controlgroup_PRAUC_givennodepair_diffgeolength(theta_A, 0, theta_B, 0)







