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


def predict_neighbournode_SRGG_withnoise_SP_R2_clu(Edindex, betaindex, noiseindex):
    """
    :param Edindex: average degree
    :param betaindex: parameter to control the clustering coefficient
    :return: PRAUC control and test simu for diff ED and beta
    4 combination of ED and beta
    ED = 5 and 20 while beta = 4 and 100
    """
    N = 1000
    # ED_list = [2, 3.5, 5, 10, 100, 1000, N - 1]  # Expected degrees
    ED_list = [2, 4, 8, 16, 32, 64, 128,256,512,1024]
    ED = ED_list[Edindex]

    # beta_list = [2.1, 4, 8, 16, 32, 64, 128]
    beta_list = [4,8,128]
    beta = beta_list[betaindex]

    noise_amplitude_list = [0, 0.0005,0.001, 0.005,0.01,0.05, 0.1, 0.5,1]
    noise_amplitude = noise_amplitude_list[noiseindex]
    print(fr"inputpara:ED:{ED},beta:{beta},noise:{noise_amplitude}",flush=True)


    random.seed(ED*beta)
    rg = RandomGenerator(-12)
    for i in range(random.randint(0, 100)):
        rg.ran1()

    G, Coorx, Coory = R2SRGG(N, ED, beta, rg)
    real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
    print("real ED:", real_avg, flush=True)
    if noise_amplitude>0:
        Coorx = add_uniform_random_noise_to_coordinates_R2(Coorx, noise_amplitude)
        Coory = add_uniform_random_noise_to_coordinates_R2(Coory, noise_amplitude)

    # load coordinates with noise


    nodepair_num = 100

    # Random select nodepair_num nodes in the largest connected component
    unique_pairs = find_k_connected_node_pairs(G, nodepair_num)
    # filename_selecetednodepair = "/home/zqiu1/network/selected_node_pair_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
    #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    # np.savetxt(filename_selecetednodepair, unique_pairs, fmt="%i")
    count = 0
    results = []
    for nodepair in unique_pairs:
        count = count + 1
        print("Simunodepair:", count,flush=True)

        try:
            nodei = nodepair[0]
            nodej = nodepair[1]

            thetaSource = Coorx[nodei]
            phiSource = Coory[nodei]
            thetaEnd = Coorx[nodej]
            phiEnd = Coory[nodej]

            # tic = time.time()

            SP_list = []
            R2distance_list = []

            distance = nx.shortest_path_length(G, nodei, nodej)

            if distance>1:
                # Find all the neighbours of node i and node j
                neighbors_i = set(G.neighbors(nodei))
                neighbors_j = set(G.neighbors(nodej))

                # Union of both neighbor sets
                combined_neighbors = neighbors_i.union(neighbors_j)

                combined_neighbors = list(combined_neighbors)
                # 预先计算所有节点到 nodei 和 nodej 的最短路径长度
                lengths_from_nodei = nx.single_source_shortest_path_length(G, nodei)
                lengths_from_nodej = nx.single_source_shortest_path_length(G, nodej)

                for nodek in combined_neighbors:
                    # compute distance for node k to geodesic
                    thetaMed = Coorx[nodek]
                    phiMed = Coory[nodek]
                    dist, _ = dist_to_geodesic_R2(thetaMed, phiMed, thetaSource, phiSource, thetaEnd, phiEnd)
                    R2distance_list.append(dist)
                    # compute if the node number is the shortest path node
                    d1 = lengths_from_nodei.get(nodek)
                    d2 = lengths_from_nodej.get(nodek)
                    if d1 is not None and d2 is not None and d1 + d2 == distance:
                        SP_list.append(nodek)

                result = {
                    "node_pair": [nodei,nodej],
                    "combined_neighbors": combined_neighbors,
                    "SP_list": SP_list,
                    "R2distance_list": R2distance_list
                }
            results.append(result)
        except nx.NetworkXNoPath:
            continue

    # with open("/home/zqiu1/data/neighbour_prediction_results.json", "w") as f:
    #     json.dump(results, f, indent=2)
    with open("neighbour_prediction_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # with open("results.json", "r") as f:
    #     results = json.load(f)

    ratios = []

    for item in results:
        sp_len = len(item["SP_list"])
        neighbor_len = len(item["combined_neighbors"])

        if neighbor_len > 0:
            ratio = sp_len / neighbor_len
            ratios.append(ratio)
        # 如果 neighbor_len 为 0，跳过该项，避免除以 0

    # 求均值
    if ratios:
        average_ratio = sum(ratios) / len(ratios)
        print("ave ratio:", average_ratio)
    else:
        print("no data error")





# def check_data_wehavenow():
#     ED_list = [5, 10, 20, 40]  # Expected degrees
#
#     beta_list = [2.1, 4, 8, 16, 32, 64, 128]
#
#     exemptionlist = []
#     for ED_index in range(4):
#         for beta_index in range(6):
#             ED = ED_list[ED_index]
#             beta = beta_list[beta_index]
#             for noise_amplitude in [0]:
#                 for ExternalSimutime in range(20):
#                     try:
#                         precision_RGG_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise0\\PrecisionSRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
#                             EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
#                         Precison_RGG_5_times = np.loadtxt(precision_RGG_Name)
#                     except FileNotFoundError:
#                         exemptionlist.append((ED, beta, noise_amplitude, ExternalSimutime))
#     print(exemptionlist)
#     np.savetxt("notrun.txt",exemptionlist)

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
    # ED = sys.argv[1]
    # beta = sys.argv[2]
    # noise = sys.argv[3]
    ED =0
    beta =0
    noise = 0

    predict_neighbournode_SRGG_withnoise_SP_R2_clu(int(ED), int(beta), int(noise))

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




