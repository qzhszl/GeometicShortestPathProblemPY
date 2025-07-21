# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 6-6-2024
Generate the graph, remove links, blur node coordinates:  x = x + E(A), y = y + E(A),
where E ~ Unif(0,A), A is noise amplitude. Or do it in a more “kosher” way, uniformly place it within a 2D circle of radius A.

For the node pair ij:
	a) test if the neighbour of the node is more likely to become the shortest path node
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
import json
from collections import Counter
# import math

# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score

from PredictGeodistanceVsRGGR2 import SPnodes_inRGG_with_coordinatesR2
from R2SRGG import R2SRGG_withgivennodepair, distR2, dist_to_geodesic_R2, R2SRGG, loadSRGGandaddnode
from PredictGeodistancewithnoiseR2 import add_uniform_random_noise_to_coordinates_R2
from FrequencyControlGroupR2 import nodeSPfrequency_loaddata_R2, nodeSPfrequency_loaddata_R2_clu
from R2RGG import RandomGeometricGraph
from degree_Vs_radius_RGG import degree_vs_radius
from SphericalSoftRandomGeomtricGraph import RandomGenerator
from main import find_nonzero_indices, find_nonnan_indices, all_shortest_path_node, find_top_n_values, \
    find_k_connected_node_pairs, find_all_connected_node_pairs


def select_data_and_nodepair():
    # check the data we have and selected node pair has specific hocpount
    ED = 8
    beta = 4
    noise_amplitude = 0

    exemptionlist = []
    N = 10000
    FileOriNetworkName = "F:\\SRGGnetwork20250512\\NetworkOriginalED{EDn}Beta{betan}Noise0.txt".format(
        EDn=ED, betan=beta)
    G = loadSRGGandaddnode(N, FileOriNetworkName)

    real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
    print("real ED:", real_avg)

    Coorx = []
    Coory = []
    FileOriNetworkCoorName = "F:\\SRGGnetwork20250512\\CoorED{EDn}Beta{betan}Noise{no}mothernetwork.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude)
    with open(FileOriNetworkCoorName, "r") as file:
        for line in file:
            if line.startswith("#"):
                continue
            data = line.strip().split("\t")  # 使用制表符分割
            Coorx.append(float(data[0]))
            Coory.append(float(data[1]))

    hop_vec = []
    node_pair_list = find_k_connected_node_pairs(G,100000)
    # node_pair_list = find_all_connected_node_pairs(G)
    for node_pair in node_pair_list:
        nodei =  node_pair[0]
        nodej = node_pair[1]
        hop_vec.append(nx.shortest_path_length(G,nodei,nodej))

    count = Counter(hop_vec)
    print("频数统计结果：", count)

    # # 2. 找出频率最高的元素
    most_common_hop, freq = count.most_common(1)[0]
    print("出现次数最多的 hop 值为：", most_common_hop, "，频率为：", freq)

    hopc = 13
    # 3. 筛选对应的 node_pair
    selected_pairs = [pair for hop, pair in zip(hop_vec, node_pair_list) if hop == hopc]

    filename_selecetednodepair = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\predictcenternodevssidenode\\SelecetedNodepairED{EDn}Beta{betan}Noise{no}h{h}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude,h = hopc)
    np.savetxt(filename_selecetednodepair, selected_pairs, fmt="%i")


    # for ExternalSimutime in range(100):
    #     FileOriNetworkName = "F:\\SRGGnetwork20250512\\NetworkwithNoiseED{EDn}Beta{betan}Noise{noise_amplitude}PYSimu{ExternalSimutime}.txt".format(
    #         EDn=ED, betan=beta, noise_amplitude=noise_amplitude, ExternalSimutime=ExternalSimutime)
    #     if os.path.exists(FileOriNetworkName):
    #         print("文件存在：", FileOriNetworkName)
    #     else:
    #         exemptionlist.append((ED, beta, noise_amplitude, ExternalSimutime))
    # print(exemptionlist)


def nodeSPfrequency_loaddata_R2_clu_side_center(N, ED, beta, noise_amplitude, nodei, nodej):
    """
        Given nodes of the SRGG.
        For i = 1 to 100 independent realizations:
    	Reconstruct G_i using original connection probabilities. Find shortest path nodes SPi.
        Characterize each node by the frequency it belong to the shortest path.
        Use this frequency to computer AUPRC.
        :return: Frequency to predict NSP
        """
    NodeFrequency_center = np.zeros(N)  # Initialize node frequency
    NodeFrequency_side = np.zeros(N)
    for i in range(100):
        # print("fresimu time:", i)
        try:
            FileNetworkName = "/work/zqiu1/NetworkwithNoiseED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
                EDn=ED, betan=beta, no=noise_amplitude, ST=i)
            H = loadSRGGandaddnode(N, FileNetworkName)
            sp_center,sp_side = shortest_path_node_center_side_part(H, nodei, nodej)
        except:
            sp_center=[]
            sp_side =[]
        for node in sp_center:
            NodeFrequency_center[node] += 1
        for node in sp_side:
            NodeFrequency_side[node] += 1
    return NodeFrequency_center,NodeFrequency_side


def SPnodes_inRGG_with_coordinatesR2_side_center(G, nodei, nodej):
    """
    Given nodes(coordinates) of the SRGG.
    Generate a RGG with the given coordinates
    :return: Nearly SHORTEST PATH nodes in the corresponding RGG
    """

    if  nx.has_path(G, nodei, nodej):
        SP_center,SP_side = shortest_path_node_center_side_part(G, nodei, nodej)
    else:
        SP_center = []
        SP_side = []
    return SP_center,SP_side


def shortest_path_node_center_side_part(G, nodei, nodej):
    """
    return: one shortest path is divided into two parts, SP nodes center and SP nodes sides
    """
    shortest_paths = nx.shortest_path(G, nodei, nodej)
    shortest_paths = shortest_paths[1:-1]
    SPlength = len(shortest_paths)
    num = int(SPlength/4)
    SP_center =shortest_paths[num:num*3]
    SP_side = shortest_paths[0:num]+shortest_paths[3*num:]
    return SP_center, SP_side



def predict_center_vs_side_geodistance_Vs_reconstructionRGG_SRGG_withnoise_SP_R2_clu(Edindex, betaindex, noiseindex, ExternalSimutime):
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

    Precision_Geodis_nodepair_center = []
    Recall_Geodis_nodepair_center = []
    Precision_RGG_nodepair_center = []  # save the precision_RGG for each node pair, we selected 100 node pair in total
    Recall_RGG_nodepair_center = []  # we selected 100 node pair in total
    Precision_SRGG_nodepair_center = []
    Recall_SRGG_nodepair_center = []

    Precision_Geodis_nodepair_side = []
    Recall_Geodis_nodepair_side = []
    Precision_RGG_nodepair_side = []  # save the precision_RGG for each node pair, we selected 100 node pair in total
    Recall_RGG_nodepair_side = []  # we selected 100 node pair in total
    Precision_SRGG_nodepair_side = []
    Recall_SRGG_nodepair_side = []

    SPnum_nodepair = []  # save the Number of nearly shortest path for each node pair
    geodistance_between_nodepair = []  # save the geodeisc length between each node pair

    random.seed(ExternalSimutime)
    rg = RandomGenerator(-12)
    for i in range(random.randint(0, 100)):
        rg.ran1()

    # # source_folder = "/home/ytian10/GSPP/SSRGGpy/R2/NoiseNotinUnit/EuclideanSoftRGGnetwork/"
    # source_folder = "/shares/bulk/ytian10/"
    # destination_folder = "F:\\SRGGnetwork20250512\\network/"
    # network_template = "NetworkOriginalED{EDn}Beta{betan}Noise0.txt"
    # networkcoordinate_template = "CoorwithNoiseED{EDn}Beta{betan}Noise{no}.txt"

    FileOriNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\predictcenternodevssidenode\\NetworkOriginalED{EDn}Beta{betan}Noise0.txt".format(
        EDn=ED, betan=beta)
    G = loadSRGGandaddnode(N, FileOriNetworkName)

    # load coordinates with noise
    Coorx = []
    Coory = []
    FileOriNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\predictcenternodevssidenode\\CoorwithNoiseED{EDn}Beta{betan}Noise{no}.txt".format(EDn=ED, betan=beta, no=noise_amplitude)
    with open(FileOriNetworkCoorName, "r") as file:
        for line in file:
            if line.startswith("#"):
                continue
            data = line.strip().split("\t")  # 使用制表符分割
            Coorx.append(float(data[0]))
            Coory.append(float(data[1]))


    real_avg = 2*nx.number_of_edges(G)/nx.number_of_nodes(G)
    print("real ED:", real_avg, flush=True)
    # realradius = degree_vs_radius(N, real_avg)
    realradius = 0.014130866694903842



    G_RGG,_,_  = RandomGeometricGraph(N, real_avg, rg, radius=realradius, Coorx=Coorx, Coory=Coory)

    if G_RGG.has_node(3851):
        print(f"节点 {3851} 存在于图中")
    else:
        print(f"节点 {3851} 不存在于图中")

    # Random select nodepair_num nodes in the largest connected component
    if ED ==4:
        filename_selecetednodepair = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\predictcenternodevssidenode\\SelecetedNodepairED{EDn}Beta{betan}Noise{no}h33.txt".format(
            Nn=N, EDn=ED, betan=beta, no=noise_amplitude)
    else:
        filename_selecetednodepair = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\predictcenternodevssidenode\\SelecetedNodepairED{EDn}Beta{betan}Noise{no}h13.txt".format(
            Nn=N, EDn=ED, betan=beta, no=noise_amplitude)
    selected_node_pairs = np.loadtxt(filename_selecetednodepair, dtype=int)
    nodepair_num = 50
    selected_node_pairs = selected_node_pairs.tolist()
    count = 0
    unique_pairs = selected_node_pairs[nodepair_num*ExternalSimutime:nodepair_num*ExternalSimutime+ nodepair_num]

    for nodepair in unique_pairs:
        count = count + 1
        print("Simunodepair:", count,flush=True)
        nodei = nodepair[0]
        nodej = nodepair[1]

        # tic = time.time()

        # Find shortest path nodes
        sp_center,sp_side = shortest_path_node_center_side_part(G, nodei, nodej)

        SPnodenum = len(sp_center)

        SPnum_nodepair.append(SPnodenum)

        Predicted_truecase_num = SPnodenum
        # toc = time.time() - tic
        # print("SP finding time:", toc)
        # print("SP num:", SPnodenum)

        # Create label array
        Label_med = np.zeros(N)
        Label_med[sp_center] = 1  # True cases for center node

        Label_med_side= np.zeros(N)
        Label_med_side[sp_side] = 1  # True cases for side node


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
        SPNodeList_RGG_center,SPNodeList_RGG_side = SPnodes_inRGG_with_coordinatesR2_side_center(G_RGG, nodei, nodej)
        # toc2 = time.time() - toc
        # print("RGG generate time:", toc2)

        PredictNSPNodeList_RGG = np.zeros(N)
        PredictNSPNodeList_RGG[SPNodeList_RGG_center] = 1  # True cases

        precision_RGG = precision_score(Label_med, PredictNSPNodeList_RGG)
        recall_RGG = recall_score(Label_med, PredictNSPNodeList_RGG)

        PredictNSPNodeList_RGG_side = np.zeros(N)
        PredictNSPNodeList_RGG_side[SPNodeList_RGG_side] = 1  # True cases
        precision_RGG_side = precision_score(Label_med_side, PredictNSPNodeList_RGG_side)
        recall_RGG_side = recall_score(Label_med_side, PredictNSPNodeList_RGG_side)


        # Store precision and recall values for RGG
        Precision_RGG_nodepair_center.append(precision_RGG)
        Recall_RGG_nodepair_center.append(recall_RGG)
        print("PreRGG:", Precision_RGG_nodepair_center,flush=True)
        Precision_RGG_nodepair_side.append(precision_RGG_side)
        Recall_RGG_nodepair_side.append(recall_RGG_side)


        # Predict sp nodes use distance, where top Predicted_truecase_num nodes will be regarded as predicted nsp according to distance form the geodesic
        Geodistance = sorted(Geodistance.items(), key=lambda kv: (kv[1], kv[0]))
        Geodistance = Geodistance[:Predicted_truecase_num + 2]
        Top100closednode = [t[0] for t in Geodistance]
        Top100closednode = [n for n in Top100closednode if n not in [nodei, nodej]]
        NSPNodeList_Geo = np.zeros(N)
        NSPNodeList_Geo[Top100closednode] = 1  # True cases
        precision_Geo = precision_score(Label_med, NSPNodeList_Geo)
        recall_Geo = recall_score(Label_med, NSPNodeList_Geo)

        precision_Geo_side = precision_score(Label_med_side, NSPNodeList_Geo)
        recall_Geo_side = recall_score(Label_med_side, NSPNodeList_Geo)

        # Store precision and recall values
        Precision_Geodis_nodepair_center.append(precision_Geo)
        Recall_Geodis_nodepair_center.append(recall_Geo)
        print("PreGeo:",Precision_Geodis_nodepair_center,flush=True)
        Precision_Geodis_nodepair_side.append(precision_Geo_side)
        Recall_Geodis_nodepair_side.append(recall_Geo_side)

        # Predict sp nodes using reconstruction of SRGG
        node_fre_center, node_fre_side = nodeSPfrequency_loaddata_R2_clu_side_center(N, ED, beta, noise_amplitude, nodei, nodej)
        _, SPnode_predictedbySRGG = find_top_n_values(node_fre_center, Predicted_truecase_num)
        SPNodeList_SRGG = np.zeros(N)
        SPNodeList_SRGG[SPnode_predictedbySRGG] = 1  # True cases
        precision_SRGG = precision_score(Label_med, SPNodeList_SRGG)
        recall_SRGG = recall_score(Label_med, SPNodeList_SRGG)

        _, SPnode_predictedbySRGG_side = find_top_n_values(node_fre_side, Predicted_truecase_num)
        SPNodeList_SRGG_side = np.zeros(N)
        SPNodeList_SRGG_side[SPnode_predictedbySRGG_side] = 1  # True cases
        precision_SRGG_side = precision_score(Label_med_side, SPNodeList_SRGG_side)
        recall_SRGG_side = recall_score(Label_med_side, SPNodeList_SRGG_side)
        # Store precision and recall values
        Precision_SRGG_nodepair_center.append(precision_SRGG)
        Recall_SRGG_nodepair_center.append(recall_SRGG)
        print("PRESRGG:", Precision_SRGG_nodepair_center,flush=True)
        Precision_SRGG_nodepair_side.append(precision_SRGG_side)
        Recall_SRGG_nodepair_side.append(recall_SRGG_side)


    # Calculate means and standard deviations of AUC
    # AUCWithoutnorMean = np.mean(PRAUC_nodepair[~np.isnan(PRAUC_nodepair)])
    # AUCWithoutnorStd = np.std(PRAUC_nodepair[~np.isnan(PRAUC_nodepair)])

    # local file path: D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise\\
    precision_RGG_Name = "F:\\SRGGnetwork20250512\\data/PrecisionRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}_center.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(precision_RGG_Name, Precision_RGG_nodepair_center)

    recall_RGG_Name = "F:\\SRGGnetwork20250512\\data/RecallRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}_center.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(recall_RGG_Name, Recall_RGG_nodepair_center)

    precision_Geodis_Name = "F:\\SRGGnetwork20250512\\data/PrecisionGeodisED{EDn}Beta{betan}Noise{no}PYSimu{ST}_center.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(precision_Geodis_Name, Precision_Geodis_nodepair_center)

    recall_Geodis_Name = "F:\\SRGGnetwork20250512\\data/RecallGeodisED{EDn}Beta{betan}Noise{no}PYSimu{ST}_center.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(recall_Geodis_Name, Recall_Geodis_nodepair_center)

    precision_SRGG_Name = "F:\\SRGGnetwork20250512\\data/PrecisionSRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}_center.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(precision_SRGG_Name, Precision_SRGG_nodepair_center)

    recall_SRGG_Name = "F:\\SRGGnetwork20250512\\data/RecallSRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}_center.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(recall_SRGG_Name, Recall_SRGG_nodepair_center)


    precision_RGG_Name_side = "F:\\SRGGnetwork20250512\\data/PrecisionRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}_side.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(precision_RGG_Name_side, Precision_RGG_nodepair_side)

    recall_RGG_Name_side = "F:\\SRGGnetwork20250512\\data/RecallRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}_side.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(recall_RGG_Name_side, Recall_RGG_nodepair_side)

    precision_Geodis_Name_side = "F:\\SRGGnetwork20250512\\data/PrecisionGeodisED{EDn}Beta{betan}Noise{no}PYSimu{ST}_side.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(precision_Geodis_Name_side, Precision_Geodis_nodepair_side)

    recall_Geodis_Name_side = "F:\\SRGGnetwork20250512\\data/RecallGeodisED{EDn}Beta{betan}Noise{no}PYSimu{ST}_side.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(recall_Geodis_Name_side, Recall_Geodis_nodepair_side)

    precision_SRGG_Name_side = "F:\\SRGGnetwork20250512\\data/PrecisionSRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}_side.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(precision_SRGG_Name_side, Precision_SRGG_nodepair_side)

    recall_SRGG_Name_side = "F:\\SRGGnetwork20250512\\data/RecallSRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}_side.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(recall_SRGG_Name_side, Recall_SRGG_nodepair_side)



    NSPnum_nodepairName = "F:\\SRGGnetwork20250512\\data/NSPNumED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(NSPnum_nodepairName, SPnum_nodepair, fmt="%i")

    geodistance_between_nodepair_Name = "F:\\SRGGnetwork20250512\\data/GeodistanceBetweenTwoNodesED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(geodistance_between_nodepair_Name, geodistance_between_nodepair)

    print("Mean Pre RGG:", np.mean(Precision_RGG_nodepair_center),flush=True)
    print("Mean Recall RGG:", np.mean(Recall_RGG_nodepair_center),flush=True)
    print("Mean Pre SRGG:", np.mean(Precision_SRGG_nodepair_center),flush=True)
    print("Mean Recall SRGG:", np.mean(Recall_SRGG_nodepair_center),flush=True)
    print("Mean Pre Geodistance:", np.mean(Precision_Geodis_nodepair_center),flush=True)
    print("Mean Recall Geodistance:", np.mean(Recall_Geodis_nodepair_center),flush=True)

    # print("Standard Deviation of AUC Without Normalization:", np.std(PRAUC_nodepair))




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # STEP 1 generate a lot of SRGG and SRGG with noise
    # select_data_and_nodepair()

    # STEP 2
    # ED = sys.argv[1]
    # beta = sys.argv[2]
    # noise = sys.argv[3]
    # ExternalSimutime = sys.argv[4]
    # predict_center_vs_side_geodistance_Vs_reconstructionRGG_SRGG_withnoise_SP_R2_clu(int(ED), int(beta), int(noise),
    #                                                                                  int(ExternalSimutime))

    predict_center_vs_side_geodistance_Vs_reconstructionRGG_SRGG_withnoise_SP_R2_clu(2, 0, 0,
                                                                                     0)






