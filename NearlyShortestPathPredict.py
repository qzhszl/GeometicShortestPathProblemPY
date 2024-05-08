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
    dist_to_geodesic_S2, distS2
from sklearn.metrics import precision_recall_curve, auc
import sys
import seaborn as sns
import pandas as pd


# Function to find nodes that frequently appear in the shortest paths
def FindNearlySPNodes(G, nodei, nodej, RelevanceSimTimes=1000):
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
    avg = 100
    beta = 5.19042969
    tic = time.time()
    print(tic)
    G, Coortheta, Coorphi = SphericalSoftRGGwithGivenNode(N, avg, beta, rg, math.pi / 2, 0, math.pi / 2, 1)
    toc1 = time.time()
    print("Genenrate a graph time:", time.time() - tic)
    print("CLU:",nx.average_clustering(G))
    nodei = N - 1
    nodej = N - 2
    NSP, NSPrelevance = FindNearlySPNodes(G, nodei, nodej, RelevanceSimTimes=1000)
    print("NSP nodes Num:",len(NSP))
    print("Finding NSP time:", time.time() - toc1)
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
    beta_list = [[3.2781,3.69375,4.05,4.7625,5.57128906],
                [3.21875,3.575,4.05,4.525,5.38085938],
                [3.21875,3.575,4.05,4.525,5.38085938],
                [3.21875,3.575,4.05,4.525,5.19042969],
                [3.21875,3.575,4.05,4.525,5.38085938],
                [3.1,3.575,4.05,4.525,5.19042969],
                [3.1,3.45625,3.93125,4.525,5.19042969]]
    beta = beta_list[Edindex][betaindex]
    print("beta:", beta)
    PRAUC_nodepair = [] # save the PRAUC for each node pair, we selected 100 node pair in total
    NSPnum_nodepair = []  # save the Number of nearly shortest path for each node pair
    geodistance_between_nodepair = [] # save the geodeisc length between each node pair

    rg = RandomGenerator(-12)
    for i in range(random.randint(0,100)):
        rg.ran1()

    G, CoorTheta, CoorPhi = SphericalSoftRGG(N, ED, beta, rg)
    while abs(nx.average_clustering(G)-CC)>0.1:
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
    filename_selecetednodepair = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\NSP0_1LinkRemove\\SelecetedNodepairED{EDn}Beta{betan}PYSimu{ST}.txt".format(EDn=ED, betan=beta,ST=ExternalSimutime)
    np.savetxt(filename_selecetednodepair, random_pairs, fmt="%i")

    for nodepair in random_pairs:
        count = count + 1
        print(count,"Simu")
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
        distance_score = [1/x for x in distance_med]
        # Calculate precision-recall curve and AUC
        precisions, recalls, _ = precision_recall_curve(Label_med, distance_score)
        AUCWithoutNornodeij = auc(recalls, precisions)

        # Store AUC values
        PRAUC_nodepair.append(AUCWithoutNornodeij)

    # Calculate means and standard deviations of AUC
    # AUCWithoutnorMean = np.mean(PRAUC_nodepair[~np.isnan(PRAUC_nodepair)])
    # AUCWithoutnorStd = np.std(PRAUC_nodepair[~np.isnan(PRAUC_nodepair)])

    PRAUCName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\NSP0_1LinkRemove\\AUCED{EDn}Beta{betan}PYSimu{ST}.txt".format(EDn=ED, betan=beta,ST=ExternalSimutime)
    np.savetxt(PRAUCName, PRAUC_nodepair)

    NSPnum_nodepairName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\NSP0_1LinkRemove\\NSPNumED{EDn}Beta{betan}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(NSPnum_nodepairName, NSPnum_nodepair,fmt="%i")

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
    PRAUC_matrix = np.zeros((7,5))
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
            PRAUC_list =[]
            for ExternalSimutime in range(9):
                PRAUCName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\NSP0_1LinkRemove\\AUCED{EDn}Beta{betan}PYSimu{ST}.txt".format(EDn=ED, betan=beta,ST=ExternalSimutime)
                PRAUC_list_10times = np.loadtxt(PRAUCName)
                PRAUC_list.extend(PRAUC_list_10times)
            PRAUC_list = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PRAUC_list))
            mean_PRAUC = np.mean(PRAUC_list)
            PRAUC_matrix[EDindex][betaindex]=mean_PRAUC
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
    sns.heatmap(data=df,vmin=0,annot=True,fmt=".2f",cbar=True,
            cbar_kws={'label': 'PRAUC'})
    # plt.title("50% links are removed when computing Nearly Shortest Path Node")
    plt.xlabel("clustering coefficient")
    plt.ylabel("average degree")
    plt.savefig("D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\NSP0_1LinkRemove\\PRAUCHeatmapNSP0_1LinkRemove.pdf", format='pdf', bbox_inches='tight', dpi=600)
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # TestFindNSPnodes()

    # ED = sys.argv[1]
    # beta = sys.argv[2]
    # ExternalSimutime = sys.argv[3]

    # GeodiscPRAUC(int(ED),int(beta),int(ExternalSimutime))


    # GeodiscPRAUC(0,0,0)

    # PlotPRAUC()
    # length_NSP()
    Label_med = [1,1,1,1,1,1,1,1]
    distance_score=[1,2,3,4,5,6,7,8]
    precisions, recalls, _ = precision_recall_curve(Label_med, distance_score)
    print(precisions)
    print(recalls)
    AUCWithoutNornodeij = auc(recalls, precisions)
    print(AUCWithoutNornodeij)





