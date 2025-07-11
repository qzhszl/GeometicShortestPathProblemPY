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
import os
import shutil
import sys
import time

import numpy as np
import networkx as nx
import random
# import math

# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score

from PredictGeodistanceVsRGGR2 import SPnodes_inRGG_with_coordinatesR2
from R2SRGG import R2SRGG_withgivennodepair, distR2, dist_to_geodesic_R2, R2SRGG, loadSRGGandaddnode
from PredictGeodistancewithnoiseR2 import add_uniform_random_noise_to_coordinates_R2
from FrequencyControlGroupR2 import nodeSPfrequency_loaddata_R2, nodeSPfrequency_loaddata_R2_clu
from degree_Vs_radius_RGG import degree_vs_radius
from SphericalSoftRandomGeomtricGraph import RandomGenerator
from main import find_nonzero_indices, find_nonnan_indices, all_shortest_path_node, find_top_n_values, \
    find_k_connected_node_pairs


def generate_r2SRGG_mothernetwork(Edindex, betaindex):
    # generate 100 SRGG FOR EACH ED, beta and the amplitude of node
    # N = 10000
    # ED_list = [5, 10, 20, 40]  # Expected degrees
    # ED = ED_list[Edindex]
    # print("ED:", ED)
    #
    # beta_list = [2.1, 4, 8, 16, 32, 64, 128]
    # beta = beta_list[betaindex]
    # print("beta:", beta)
    # noise_amplitude =0
    # print("noise_amplitude:", noise_amplitude)

    N = 10000
    ED_list = [2, 3.5, 100, 1000, N - 1]  # Expected degrees
    ED = ED_list[Edindex]
    print("ED:", ED)

    beta_list = [2.1, 4, 8, 32, 128]
    beta = beta_list[betaindex]
    print("beta:", beta)

    rg = RandomGenerator(-12)
    rseed = random.randint(0, 100)
    print(rseed)
    for i in range(rseed):
        rg.ran1()

    noise_amplitude = 0
    G, Coorx, Coory = R2SRGG(N, ED, beta, rg)

    real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
    print("real ED:", real_avg)
    # print("clu:", nx.average_clustering(G))
    # components = list(nx.connected_components(G))
    # largest_component = max(components, key=len)
    # print("LCC", len(largest_component))

    FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\Noise\\NetworkOriginalED{EDn}Beta{betan}Noise{no}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude)
    nx.write_edgelist(G, FileNetworkName)

    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\Noise\\CoorED{EDn}Beta{betan}Noise{no}mothernetwork.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude)
    with open(FileNetworkCoorName, "w") as file:
        for data1, data2 in zip(Coorx, Coory):
            file.write(f"{data1}\t{data2}\n")

    # Coorx = add_uniform_random_noise_to_coordinates_R2(Coorx, noise_amplitude)
    # Coory = add_uniform_random_noise_to_coordinates_R2(Coory, noise_amplitude)
    # FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\Noise\\CoorwithNoiseED{EDn}Beta{betan}Noise{no}.txt".format(
    #     EDn=ED, betan=beta, no=noise_amplitude)
    # with open(FileNetworkCoorName, "w") as file:
    #     for data1, data2 in zip(Coorx, Coory):
    #         file.write(f"{data1}\t{data2}\n")

def generate_r2SRGG_withdiffinput_clu(Edindex, betaindex, noise_amplitude):
    # generate 100 SRGG FOR EACH ED, beta and the amplitude of node
    N = 10000
    # ED_list = [2, 3.5, 100, 1000, N - 1]  # Expected degrees
    ED_list = [2, 4, 8, 16, 32, 64, 128]  # Expected degrees
    ED = ED_list[Edindex]
    print("ED:", ED)

    # beta_list = [2.1, 4, 8, 32, 128]
    beta_list = [4,8,128]
    beta = beta_list[betaindex]
    print("beta:", beta)

    print("noise_amplitude:", noise_amplitude)

    rg = RandomGenerator(-12)
    rseed = random.randint(0, 100)
    print(rseed)
    for i in range(rseed):
        rg.ran1()

    # FileNetworkName = "/home/ytian10/GSPP/SSRGGpy/R2/NoiseNotinUnit/EuclideanSoftRGGnetwork/NetworkOriginalED{EDn}Beta{betan}Noise0.txt".format(
    #     EDn=ED, betan=beta)
    # G = loadSRGGandaddnode(N,FileNetworkName)

    Coorx=[]
    Coory=[]
    FileNetworkCoorName = "/home/zqiu1/network/CoorED{EDn}Beta{betan}Noise0mothernetwork.txt".format(
        EDn=ED, betan=beta)
    with open(FileNetworkCoorName, "r") as file:
        for line in file:
            if line.startswith("#"):
                continue
            data = line.strip().split("\t")  # 使用制表符分割
            Coorx.append(float(data[0]))
            Coory.append(float(data[1]))


    # real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
    # print("real ED:", real_avg)
    # print("clu:", nx.average_clustering(G))
    # components = list(nx.connected_components(G))
    # largest_component = max(components, key=len)
    # print("LCC",len(largest_component))


    Coorx = add_uniform_random_noise_to_coordinates_R2(Coorx, noise_amplitude)
    Coory = add_uniform_random_noise_to_coordinates_R2(Coory, noise_amplitude)
    FileNetworkCoorName = "/home/zqiu1/network/CoorwithNoiseED{EDn}Beta{betan}Noise{no}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude)
    with open(FileNetworkCoorName, "w") as file:
        for data1, data2 in zip(Coorx, Coory):
            file.write(f"{data1}\t{data2}\n")

    for ExternalSimutime in range(100):
        print(ExternalSimutime)
        H, Coorx1, Coory1 = R2SRGG(N, ED, beta, rg, Coorx, Coory)
        FileNetworkName = "/home/zqiu1/network/NetworkwithNoiseED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
            EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
        nx.write_edgelist(H, FileNetworkName)

def generate_r2SRGG_withdiffinput(Edindex, betaindex, noise_amplitude):
    # generate 100 SRGG FOR EACH ED, beta and the amplitude of node
    N = 10000
    ED_list = [5, 10, 20, 40]  # Expected degrees
    ED = ED_list[Edindex]
    print("ED:", ED)

    beta_list = [2.1, 4, 8, 16,32,64, 128]
    beta = beta_list[betaindex]
    print("beta:", beta)

    print("noise_amplitude:", noise_amplitude)

    rg = RandomGenerator(-12)
    rseed = random.randint(0, 100)
    print(rseed)
    for i in range(rseed):
        rg.ran1()

    FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\Noise\\NetworkOriginalED{EDn}Beta{betan}Noise0.txt".format(
        EDn=ED, betan=beta)
    G = loadSRGGandaddnode(N,FileNetworkName)

    Coorx=[]
    Coory=[]
    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\Noise\\CoorED{EDn}Beta{betan}Noise0mothernetwork.txt".format(
        EDn=ED, betan=beta)
    with open(FileNetworkCoorName, "r") as file:
        for line in file:
            if line.startswith("#"):
                continue
            data = line.strip().split("\t")  # 使用制表符分割
            Coorx.append(float(data[0]))
            Coory.append(float(data[1]))


    real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
    print("real ED:", real_avg)
    # print("clu:", nx.average_clustering(G))
    # components = list(nx.connected_components(G))
    # largest_component = max(components, key=len)
    # print("LCC",len(largest_component))


    Coorx = add_uniform_random_noise_to_coordinates_R2(Coorx, noise_amplitude)
    Coory = add_uniform_random_noise_to_coordinates_R2(Coory, noise_amplitude)
    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\Noise\\CoorwithNoiseED{EDn}Beta{betan}Noise{no}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude)
    with open(FileNetworkCoorName, "w") as file:
        for data1, data2 in zip(Coorx, Coory):
            file.write(f"{data1}\t{data2}\n")

    for ExternalSimutime in range(100):
        print(ExternalSimutime)
        H, Coorx1, Coory1 = R2SRGG(N, ED, beta, rg, Coorx, Coory)
        FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\Noise\\NetworkwithNoiseED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
            EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
        nx.write_edgelist(H, FileNetworkName)


def predict_geodistance_Vs_reconstructionRGG_SRGG_withnoise_SP_R2(Edindex, betaindex, noiseindex, ExternalSimutime):
    """
    :param Edindex: average degree
    :param betaindex: parameter to control the clustering coefficient
    :return: PRAUC control and test simu for diff ED and beta
    4 combination of ED and beta
    ED = 5 and 20 while beta = 4 and 100
    """
    N = 10000
    ED_list = [5, 10, 20, 40]  # Expected degrees
    ED = ED_list[Edindex]
    print("ED:", ED)

    beta_list = [2.1, 4, 8, 16,32,64, 128]
    beta = beta_list[betaindex]
    print("beta:", beta)

    noise_amplitude_list = [0, 0.001, 0.01, 0.1, 0.5]
    noise_amplitude = noise_amplitude_list[noiseindex]
    print("noise amplitude:", noise_amplitude)

    Precision_Geodis_nodepair = []
    Recall_Geodis_nodepair = []
    Precision_RGG_nodepair = []  # save the precision_RGG for each node pair, we selected 100 node pair in total
    Recall_RGG_nodepair = []  # we selected 100 node pair in total
    Precision_SRGG_nodepair = []
    Recall_SRGG_nodepair = []

    SPnum_nodepair = []  # save the Number of nearly shortest path for each node pair
    geodistance_between_nodepair = []  # save the geodeisc length between each node pair

    random.seed(ExternalSimutime)
    rg = RandomGenerator(-12)
    for i in range(random.randint(0, 100)):
        rg.ran1()

    FileOriNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\Noise\\NetworkOriginalED{EDn}Beta{betan}Noise{no}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude)
    G = loadSRGGandaddnode(N, FileOriNetworkName)

    real_avg = 2*nx.number_of_edges(G)/nx.number_of_nodes(G)
    print("real ED:", real_avg)
    realradius = degree_vs_radius(N, real_avg)

    # load coordinates with noise
    Coorx = []
    Coory = []
    FileOriNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\Noise\\CoorED{EDn}Beta{betan}Noise{no}mothernetwork.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude)
    with open(FileOriNetworkCoorName, "r") as file:
        for line in file:
            if line.startswith("#"):
                continue
            data = line.strip().split("\t")  # 使用制表符分割
            Coorx.append(float(data[0]))
            Coory.append(float(data[1]))

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
    filename_selecetednodepair = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise\\SelecetedNodepairED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(filename_selecetednodepair, random_pairs, fmt="%i")

    for nodepair in random_pairs:
        count = count + 1
        print("Simunodepair:", count)
        nodei = nodepair[0]
        nodej = nodepair[1]

        tic = time.time()

        # Find shortest path nodes
        SPNodelist = all_shortest_path_node(G, nodei, nodej)
        SPnodenum = len(SPNodelist)

        SPnum_nodepair.append(SPnodenum)

        Predicted_truecase_num = SPnodenum
        toc = time.time() - tic
        print("SP finding time:", toc)
        print("SP num:", SPnodenum)

        # Create label array
        Label_med = np.zeros(N)
        Label_med[SPNodelist] = 1  # True cases

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
        SPNodeList_RGG = SPnodes_inRGG_with_coordinatesR2(N, real_avg, realradius,rg, Coorx, Coory, nodei, nodej)
        # toc2 = time.time() - toc
        # print("RGG generate time:", toc2)

        PredictNSPNodeList_RGG = np.zeros(N)
        PredictNSPNodeList_RGG[SPNodeList_RGG] = 1  # True cases

        precision_RGG = precision_score(Label_med, PredictNSPNodeList_RGG)
        recall_RGG = recall_score(Label_med, PredictNSPNodeList_RGG)

        # Store precision and recall values for RGG
        Precision_RGG_nodepair.append(precision_RGG)
        Recall_RGG_nodepair.append(recall_RGG)


        # Predict sp nodes use distance, where top Predicted_truecase_num nodes will be regarded as predicted nsp according to distance form the geodesic
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


        # Predict sp nodes using reconstruction of SRGG
        node_fre = nodeSPfrequency_loaddata_R2(N, ED, beta, noise_amplitude, nodei, nodej)
        _, SPnode_predictedbySRGG = find_top_n_values(node_fre, Predicted_truecase_num)
        SPNodeList_SRGG = np.zeros(N)
        SPNodeList_SRGG[SPnode_predictedbySRGG] = 1  # True cases
        precision_SRGG = precision_score(Label_med, SPNodeList_SRGG)
        recall_SRGG = recall_score(Label_med, SPNodeList_SRGG)
        # Store precision and recall values
        Precision_SRGG_nodepair.append(precision_SRGG)
        Recall_SRGG_nodepair.append(recall_SRGG)



    # Calculate means and standard deviations of AUC
    # AUCWithoutnorMean = np.mean(PRAUC_nodepair[~np.isnan(PRAUC_nodepair)])
    # AUCWithoutnorStd = np.std(PRAUC_nodepair[~np.isnan(PRAUC_nodepair)])

    precision_RGG_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise\\PrecisionRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(precision_RGG_Name, Precision_RGG_nodepair)

    recall_RGG_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise\\RecallRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(recall_RGG_Name, Recall_RGG_nodepair)

    precision_Geodis_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise\\PrecisionGeodisED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(precision_Geodis_Name, Precision_Geodis_nodepair)

    recall_Geodis_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise\\RecallGeodisED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(recall_Geodis_Name, Recall_Geodis_nodepair)

    precision_SRGG_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise\\PrecisionSRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(precision_SRGG_Name, Precision_SRGG_nodepair)

    recall_SRGG_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise\\RecallSRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(recall_SRGG_Name, Recall_SRGG_nodepair)


    NSPnum_nodepairName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise\\NSPNumED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(NSPnum_nodepairName, SPnum_nodepair, fmt="%i")

    geodistance_between_nodepair_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise\\GeodistanceBetweenTwoNodesED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(geodistance_between_nodepair_Name, geodistance_between_nodepair)

    print("Mean Pre RGG:", np.mean(Precision_RGG_nodepair))
    print("Mean Recall RGG:", np.mean(Recall_RGG_nodepair))
    print("Mean Pre SRGG:", np.mean(Precision_SRGG_nodepair))
    print("Mean Recall SRGG:", np.mean(Recall_SRGG_nodepair))
    print("Mean Pre Geodistance:", np.mean(Precision_Geodis_nodepair))
    print("Mean Recall Geodistance:", np.mean(Recall_Geodis_nodepair))

    # print("Standard Deviation of AUC Without Normalization:", np.std(PRAUC_nodepair))


def predict_geodistance_Vs_reconstructionRGG_SRGG_withnoise_SP_R2_clu(Edindex, betaindex, noiseindex, ExternalSimutime):
    """
    :param Edindex: average degree
    :param betaindex: parameter to control the clustering coefficient
    :return: PRAUC control and test simu for diff ED and beta
    4 combination of ED and beta
    ED = 5 and 20 while beta = 4 and 100
    """
    N = 10000
    # ED_list = [2, 3.5, 5, 10, 100, 1000, N - 1]  # Expected degrees
    ED_list = [2, 4, 8, 16, 32, 64, 128,256,512,1024]
    ED = ED_list[Edindex]

    # beta_list = [2.1, 4, 8, 16, 32, 64, 128]
    beta_list = [4,8,128]
    beta = beta_list[betaindex]

    noise_amplitude_list = [0, 0.0005,0.001, 0.005,0.01,0.05, 0.1, 0.5,1]
    noise_amplitude = noise_amplitude_list[noiseindex]
    print(fr"inputpara:ED:{ED},beta:{beta},noise:{noise_amplitude},simu:{ExternalSimutime}",flush=True)

    Precision_Geodis_nodepair = []
    Recall_Geodis_nodepair = []
    Precision_RGG_nodepair = []  # save the precision_RGG for each node pair, we selected 100 node pair in total
    Recall_RGG_nodepair = []  # we selected 100 node pair in total
    Precision_SRGG_nodepair = []
    Recall_SRGG_nodepair = []

    SPnum_nodepair = []  # save the Number of nearly shortest path for each node pair
    geodistance_between_nodepair = []  # save the geodeisc length between each node pair

    random.seed(ExternalSimutime)
    rg = RandomGenerator(-12)
    for i in range(random.randint(0, 100)):
        rg.ran1()

    # # source_folder = "/home/ytian10/GSPP/SSRGGpy/R2/NoiseNotinUnit/EuclideanSoftRGGnetwork/"
    # source_folder = "/shares/bulk/ytian10/"
    # destination_folder = "/home/zqiu1/network/"
    # network_template = "NetworkOriginalED{EDn}Beta{betan}Noise0.txt"
    # networkcoordinate_template = "CoorwithNoiseED{EDn}Beta{betan}Noise{no}.txt"

    FileOriNetworkName = "/home/zqiu1/network/NetworkOriginalED{EDn}Beta{betan}Noise0.txt".format(
        EDn=ED, betan=beta)
    G = loadSRGGandaddnode(N, FileOriNetworkName)

    # load coordinates with noise
    Coorx = []
    Coory = []
    FileOriNetworkCoorName = "/home/zqiu1/network/CoorwithNoiseED{EDn}Beta{betan}Noise{no}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude)
    with open(FileOriNetworkCoorName, "r") as file:
        for line in file:
            if line.startswith("#"):
                continue
            data = line.strip().split("\t")  # 使用制表符分割
            Coorx.append(float(data[0]))
            Coory.append(float(data[1]))


    real_avg = 2*nx.number_of_edges(G)/nx.number_of_nodes(G)
    print("real ED:", real_avg, flush=True)
    realradius = degree_vs_radius(N, real_avg)

    nodepair_num = 5

    # Random select nodepair_num nodes in the largest connected component
    unique_pairs = find_k_connected_node_pairs(G, nodepair_num)
    # filename_selecetednodepair = "/home/zqiu1/network/selected_node_pair_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
    #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    # np.savetxt(filename_selecetednodepair, unique_pairs, fmt="%i")
    count = 0

    for nodepair in unique_pairs:
        count = count + 1
        print("Simunodepair:", count,flush=True)
        nodei = nodepair[0]
        nodej = nodepair[1]

        # tic = time.time()

        # Find shortest path nodes
        SPNodelist = all_shortest_path_node(G, nodei, nodej)
        SPnodenum = len(SPNodelist)

        SPnum_nodepair.append(SPnodenum)

        Predicted_truecase_num = SPnodenum
        # toc = time.time() - tic
        # print("SP finding time:", toc)
        # print("SP num:", SPnodenum)

        # Create label array
        Label_med = np.zeros(N)
        Label_med[SPNodelist] = 1  # True cases

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
        SPNodeList_RGG = SPnodes_inRGG_with_coordinatesR2(N, real_avg, realradius,rg, Coorx, Coory, nodei, nodej)
        # toc2 = time.time() - toc
        # print("RGG generate time:", toc2)

        PredictNSPNodeList_RGG = np.zeros(N)
        PredictNSPNodeList_RGG[SPNodeList_RGG] = 1  # True cases

        precision_RGG = precision_score(Label_med, PredictNSPNodeList_RGG)
        recall_RGG = recall_score(Label_med, PredictNSPNodeList_RGG)

        # Store precision and recall values for RGG
        Precision_RGG_nodepair.append(precision_RGG)
        Recall_RGG_nodepair.append(recall_RGG)
        print("PreRGG:", Precision_RGG_nodepair,flush=True)


        # Predict sp nodes use distance, where top Predicted_truecase_num nodes will be regarded as predicted nsp according to distance form the geodesic
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
        print("PreGeo:",Precision_Geodis_nodepair,flush=True)


        # Predict sp nodes using reconstruction of SRGG
        node_fre = nodeSPfrequency_loaddata_R2_clu(N, ED, beta, noise_amplitude, nodei, nodej)
        _, SPnode_predictedbySRGG = find_top_n_values(node_fre, Predicted_truecase_num)
        SPNodeList_SRGG = np.zeros(N)
        SPNodeList_SRGG[SPnode_predictedbySRGG] = 1  # True cases
        precision_SRGG = precision_score(Label_med, SPNodeList_SRGG)
        recall_SRGG = recall_score(Label_med, SPNodeList_SRGG)
        # Store precision and recall values
        Precision_SRGG_nodepair.append(precision_SRGG)
        Recall_SRGG_nodepair.append(recall_SRGG)
        print("PRESRGG:", Precision_SRGG_nodepair,flush=True)



    # Calculate means and standard deviations of AUC
    # AUCWithoutnorMean = np.mean(PRAUC_nodepair[~np.isnan(PRAUC_nodepair)])
    # AUCWithoutnorStd = np.std(PRAUC_nodepair[~np.isnan(PRAUC_nodepair)])

    # local file path: D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise\\
    precision_RGG_Name = "/home/zqiu1/data/PrecisionRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(precision_RGG_Name, Precision_RGG_nodepair)

    recall_RGG_Name = "/home/zqiu1/data/RecallRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(recall_RGG_Name, Recall_RGG_nodepair)

    precision_Geodis_Name = "/home/zqiu1/data/PrecisionGeodisED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(precision_Geodis_Name, Precision_Geodis_nodepair)

    recall_Geodis_Name = "/home/zqiu1/data/RecallGeodisED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(recall_Geodis_Name, Recall_Geodis_nodepair)

    precision_SRGG_Name = "/home/zqiu1/data/PrecisionSRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(precision_SRGG_Name, Precision_SRGG_nodepair)

    recall_SRGG_Name = "/home/zqiu1/data/RecallSRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(recall_SRGG_Name, Recall_SRGG_nodepair)


    NSPnum_nodepairName = "/home/zqiu1/data/NSPNumED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(NSPnum_nodepairName, SPnum_nodepair, fmt="%i")

    geodistance_between_nodepair_Name = "/home/zqiu1/data/GeodistanceBetweenTwoNodesED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(geodistance_between_nodepair_Name, geodistance_between_nodepair)

    print("Mean Pre RGG:", np.mean(Precision_RGG_nodepair),flush=True)
    print("Mean Recall RGG:", np.mean(Recall_RGG_nodepair),flush=True)
    print("Mean Pre SRGG:", np.mean(Precision_SRGG_nodepair),flush=True)
    print("Mean Recall SRGG:", np.mean(Recall_SRGG_nodepair),flush=True)
    print("Mean Pre Geodistance:", np.mean(Precision_Geodis_nodepair),flush=True)
    print("Mean Recall Geodistance:", np.mean(Recall_Geodis_nodepair),flush=True)

    # print("Standard Deviation of AUC Without Normalization:", np.std(PRAUC_nodepair))


# def plot_predict_geodistance_Vs_reconstructionRGG_SRGG_withnoise_SP_R2_clu(Edindex, betaindex):
#     # plot PRECISION
#     ED_list = [5, 10, 20, 40]  # Expected degrees
#     ED = ED_list[Edindex]
#     beta_list = [2.1, 4, 8, 16,32,64, 128]
#     beta = beta_list[betaindex]
#     RGG_precision_list_all_ave = []
#     SRGG_precision_list_all_ave = []
#     Geo_precision_list_all_ave = []
#     RGG_precision_list_all_std = []
#     SRGG_precision_list_all_std = []
#     Geo_precision_list_all_std = []
#     exemptionlist = []
#
#     for noise_amplitude in [0, 0.001,0.01,0.1, 1]:
#         PrecisonRGG_specificnoise = []
#         for ExternalSimutime in range(20):
#             try:
#                 precision_RGG_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise\\PrecisionRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
#                     EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
#                 Precison_RGG_5_times = np.loadtxt(precision_RGG_Name)
#                 PrecisonRGG_specificnoise.extend(Precison_RGG_5_times)
#             except FileNotFoundError:
#                 exemptionlist.append((ED,beta,noise_amplitude,ExternalSimutime))
#         # nonzero_indices_geo = find_nonzero_indices(PrecisonRGG_specificnoise)
#         # PrecisonRGG_specificnoise = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PrecisonRGG_specificnoise))
#         # PrecisonRGG_specificnoise = [PrecisonRGG_specificnoise[x] for x in nonzero_indices_geo]
#         RGG_precision_list_all_ave.append(np.mean(PrecisonRGG_specificnoise))
#         RGG_precision_list_all_std.append(np.std(PrecisonRGG_specificnoise))
#     # print("lenpre", len(PrecisonRGG_specificnoise))
#         PrecisonSRGG_specificnoise = []
#         for ExternalSimutime in range(20):
#             try:
#                 precision_SRGG_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise\\PrecisionSRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
#                     EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
#                 Precison_SRGG_5_times = np.loadtxt(precision_SRGG_Name)
#                 PrecisonSRGG_specificnoise.extend(Precison_SRGG_5_times)
#             except FileNotFoundError:
#                 pass
#         # nonzero_indices_geo = find_nonzero_indices(PrecisonRGG_specificnoise)
#         # PrecisonRGG_specificnoise = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PrecisonRGG_specificnoise))
#         # PrecisonRGG_specificnoise = [PrecisonRGG_specificnoise[x] for x in nonzero_indices_geo]
#         SRGG_precision_list_all_ave.append(np.mean(PrecisonSRGG_specificnoise))
#         SRGG_precision_list_all_std.append(np.std(PrecisonSRGG_specificnoise))
#         # print("lenpre", len(PrecisonRGG_specificnoise))
#
#         PrecisonGeodis_specificnoise = []
#         for ExternalSimutime in range(20):
#             try:
#                 precision_Geodis_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise\\PrecisionGeodisED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
#                     EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
#                 Precison_Geodis_5_times = np.loadtxt(precision_Geodis_Name)
#                 PrecisonGeodis_specificnoise.extend(Precison_Geodis_5_times)
#             except FileNotFoundError:
#                 pass
#         # nonzero_indices_geo = find_nonzero_indices(PrecisonRGG_specificnoise)
#         # PrecisonRGG_specificnoise = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PrecisonRGG_specificnoise))
#         # PrecisonRGG_specificnoise = [PrecisonRGG_specificnoise[x] for x in nonzero_indices_geo]
#         Geo_precision_list_all_ave.append(np.mean(PrecisonGeodis_specificnoise))
#         Geo_precision_list_all_std.append(np.std(PrecisonGeodis_specificnoise))
#         # print("lenpre", len(PrecisonRGG_specificnoise))
#
#
#     fig, ax = plt.subplots()
#     # Data
#     y1 = RGG_precision_list_all_ave
#     y2 = SRGG_precision_list_all_ave
#     y3 = Geo_precision_list_all_ave
#
#     # X axis labels
#     # x_labels = ['0', '0.001', '0.01']
#     x_labels = ['0', '0.001', '0.01']
#
#     # X axis positions for each bar group
#     x = np.arange(len(x_labels))
#
#     # Width of each bar
#     width = 0.2
#
#     # Plotting the bars
#     bar1 = ax.bar(x - width, y1, width, label='RGG',yerr=RGG_precision_list_all_std, capsize=5)
#     bar2 = ax.bar(x, y2, width, label='SRGG', yerr=SRGG_precision_list_all_std, capsize=5)
#     bar3 = ax.bar(x + width, y3, width, label='Geo distance', yerr=Geo_precision_list_all_std, capsize=5)
#
#
#     # Adding labels and title
#     ax.set_ylim(0,1)
#     ax.set_xlabel('noise amplitude')
#     ax.set_ylabel('precision')
#     title_name = "beta:{beta_n}, E[D]:{ed_n}".format(ed_n=ED, beta_n = beta)
#     ax.set_title(title_name)
#     ax.set_xticks(x)
#     ax.set_xticklabels(x_labels)
#     ax.legend()
#
#     # Display the plot
#     plt.show()
#     figname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise\\PrecisionGeoVsRGGSRGGED{EDn}Beta{betan}N.pdf".format(
#                 EDn=ED, betan=beta)
#
#     fig.savefig(figname, format='pdf', bbox_inches='tight', dpi=600)
#     plt.close()
#     print(exemptionlist)


# def plot_predict_geodistance_Vs_reconstructionRGG_SRGG_withnoise_SP_R2_clu2(Edindex, betaindex):
#     # plot recall
#     ED_list = [5, 10, 20, 40]  # Expected degrees
#     ED = ED_list[Edindex]
#     beta_list = [2.1, 4, 8, 16,32,64, 128]
#     beta = beta_list[betaindex]
#     RGG_precision_list_all_ave = []
#     SRGG_precision_list_all_ave = []
#     Geo_precision_list_all_ave = []
#     RGG_precision_list_all_std = []
#     SRGG_precision_list_all_std = []
#     Geo_precision_list_all_std = []
#     exemptionlist = []
#
#     for noise_amplitude in [0, 0.001, 0.01, 0.1, 0.5]:
#         PrecisonRGG_specificnoise = []
#         for ExternalSimutime in range(20):
#             try:
#                 precision_RGG_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise\\RecallRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
#                     EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
#                 Precison_RGG_5_times = np.loadtxt(precision_RGG_Name)
#                 PrecisonRGG_specificnoise.extend(Precison_RGG_5_times)
#             except:
#                 pass
#         # nonzero_indices_geo = find_nonzero_indices(PrecisonRGG_specificnoise)
#         # PrecisonRGG_specificnoise = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PrecisonRGG_specificnoise))
#         # PrecisonRGG_specificnoise = [PrecisonRGG_specificnoise[x] for x in nonzero_indices_geo]
#         RGG_precision_list_all_ave.append(np.mean(PrecisonRGG_specificnoise))
#         RGG_precision_list_all_std.append(np.std(PrecisonRGG_specificnoise))
#         # print("lenpre", len(PrecisonRGG_specificnoise))
#         PrecisonSRGG_specificnoise = []
#         for ExternalSimutime in range(20):
#             try:
#                 precision_SRGG_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise\\RecallSRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
#                     EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
#                 Precison_SRGG_5_times = np.loadtxt(precision_SRGG_Name)
#                 PrecisonSRGG_specificnoise.extend(Precison_SRGG_5_times)
#             except:
#                 pass
#         # nonzero_indices_geo = find_nonzero_indices(PrecisonRGG_specificnoise)
#         # PrecisonRGG_specificnoise = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PrecisonRGG_specificnoise))
#         # PrecisonRGG_specificnoise = [PrecisonRGG_specificnoise[x] for x in nonzero_indices_geo]
#         SRGG_precision_list_all_ave.append(np.mean(PrecisonSRGG_specificnoise))
#         SRGG_precision_list_all_std.append(np.std(PrecisonSRGG_specificnoise))
#         # print("lenpre", len(PrecisonRGG_specificnoise))
#
#         PrecisonGeodis_specificnoise = []
#         for ExternalSimutime in range(20):
#             try:
#                 precision_Geodis_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise\\RecallGeodisED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
#                     EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
#                 Precison_Geodis_5_times = np.loadtxt(precision_Geodis_Name)
#                 PrecisonGeodis_specificnoise.extend(Precison_Geodis_5_times)
#             except:
#                 pass
#         # nonzero_indices_geo = find_nonzero_indices(PrecisonRGG_specificnoise)
#         # PrecisonRGG_specificnoise = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PrecisonRGG_specificnoise))
#         # PrecisonRGG_specificnoise = [PrecisonRGG_specificnoise[x] for x in nonzero_indices_geo]
#         Geo_precision_list_all_ave.append(np.mean(PrecisonRGG_specificnoise))
#         Geo_precision_list_all_std.append(np.std(PrecisonGeodis_specificnoise))
#         # print("lenpre", len(PrecisonRGG_specificnoise))
#
#     fig, ax = plt.subplots()
#     # Data
#     y1 = RGG_precision_list_all_ave
#     y2 = SRGG_precision_list_all_ave
#     y3 = Geo_precision_list_all_ave
#
#     # X axis labels
#     x_labels = ['0', '0.001', '0.01', '0.1', '0.5']
#
#     # X axis positions for each bar group
#     x = np.arange(len(x_labels))
#
#     # Width of each bar
#     width = 0.2
#
#     # Plotting the bars
#     bar1 = ax.bar(x - width, y1, width, label='RGG', yerr=RGG_precision_list_all_std, capsize=5)
#     bar2 = ax.bar(x, y2, width, label='SRGG', yerr=SRGG_precision_list_all_std, capsize=5)
#     bar3 = ax.bar(x + width, y3, width, label='Geo distance', yerr=Geo_precision_list_all_std, capsize=5)
#
#     # Adding labels and title
#     ax.set_xlabel('noise amplitude')
#     ax.set_ylabel('recall')
#     title_name = "beta:{beta_n], E[D]: {ed_n}".format(ed_n=ED, beta_n=beta)
#     ax.set_title(title_name)
#     ax.set_xticks(x)
#     ax.set_xticklabels(x_labels)
#     ax.legend()
#
#     # Display the plot
#     plt.show()
#     figname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise\\RecallGeoVsRGGSRGGED{EDn}Beta{betan}N.png".format(
#         EDn=ED, betan=beta)
#     plt.savefig(figname, format='png', bbox_inches='tight', dpi=600)
#     plt.close()


def check_data_wehavenow():
    ED_list = [5, 10, 20, 40]  # Expected degrees

    beta_list = [2.1, 4, 8, 16, 32, 64, 128]

    exemptionlist = []
    for ED_index in range(4):
        for beta_index in range(6):
            ED = ED_list[ED_index]
            beta = beta_list[beta_index]
            for noise_amplitude in [0]:
                for ExternalSimutime in range(20):
                    try:
                        precision_RGG_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise0\\PrecisionSRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
                            EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
                        Precison_RGG_5_times = np.loadtxt(precision_RGG_Name)
                    except FileNotFoundError:
                        exemptionlist.append((ED, beta, noise_amplitude, ExternalSimutime))
    print(exemptionlist)
    np.savetxt("notrun.txt",exemptionlist)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # STEP 1 generate a lot of SRGG and SRGG with noise
    # for Edindex in range(5):
    #     for betaindex in range(5):
    #         generate_r2SRGG_mothernetwork(Edindex, betaindex)
    # for Edindex in range(4):
    #     for betaindex in range(7):
    #         for noise_amplitude in [1]:
    #         # for noise_amplitude in [0, 0.001,0.01,0.1]:
    #             generate_r2SRGG_withdiffinput(Edindex, betaindex, noise_amplitude)

    # STEP 1.2 generate a lot of SRGG and SRGG with noise for cluster
    # ED = sys.argv[1]
    # beta = sys.argv[2]
    # noise_amplitude_index = sys.argv[3]
    # noise_amplitude_vec =  [0, 0.001, 0.01, 0.1, 1]
    # noise_amplitude = noise_amplitude_vec[int(noise_amplitude_index)]
    # generate_r2SRGG_withdiffinput_clu(int(ED), int(beta), noise_amplitude)


    # STEP 2.1 test and run the simu
    # predict_geodistance_Vs_reconstructionRGG_SRGG_withnoise_SP_R2(1, 3, 0, 0)

    # # STEP 2.2
    ED = sys.argv[1]
    beta = sys.argv[2]
    noise = sys.argv[3]
    ExternalSimutime = sys.argv[4]
    predict_geodistance_Vs_reconstructionRGG_SRGG_withnoise_SP_R2_clu(int(ED), int(beta), int(noise), int(ExternalSimutime))

    # STEP 2.3
    # check_data_wehavenow()
    # NOTRUNLIST = np.loadtxt("notrun.txt")
    # i = sys.argv[1]
    # ED = NOTRUNLIST[int(i)][0]
    # beta = NOTRUNLIST[int(i)][1]
    # noise = NOTRUNLIST[int(i)][2]
    # ExternalSimutime = NOTRUNLIST[int(i)][3]
    # predict_geodistance_Vs_reconstructionRGG_SRGG_withnoise_SP_R2_clu(int(ED), int(beta), int(noise), int(ExternalSimutime))


    # STEP 3 plot the figure
    # for Edindex in range(2):
    #     for betaindex in range(2):
    #         plot_predict_geodistance_Vs_reconstructionRGG_SRGG_withnoise_SP_R2_clu(Edindex, betaindex)
    # for (ED, beta, noise_amplitude, ExternalSimutime) in [(20, 4, 0, 0),(20, 4, 0, 1),(20, 4, 0, 3)]:
    #     precision_Geodis_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise\\PrecisionGeodisED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
    #         EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    #     try:
    #         Precison_Geodis_5_times = np.loadtxt(precision_Geodis_Name)
    #         print(Precison_Geodis_5_times)
    #     except FileNotFoundError:
    #         pass


