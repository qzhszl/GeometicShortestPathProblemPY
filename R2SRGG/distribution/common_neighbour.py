# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 2024/11/12
"""
import numpy as np
import networkx as nx
import random
import json
import math
import sys

# from R2SRGG import R2SRGG, distR2, dist_to_geodesic_R2, loadSRGGandaddnode, R2SRGG_withgivennodepair,dist_to_geodesic_perpendicular_R2,RandomGenerator
from R2SRGG.R2SRGG import R2SRGG, distR2, dist_to_geodesic_R2, loadSRGGandaddnode, R2SRGG_withgivennodepair, \
    dist_to_geodesic_perpendicular_R2,RandomGenerator
from main import all_shortest_path_node, find_k_connected_node_pairs, find_all_connected_node_pairs

def common_neighbour_generator(coorx,coory):



def neighbour_distance_with_beta_one_graph_clu(beta_index,ExternalSimutime):
    """
    From the results from neighbour_distance_beta(), we observed that the deviation of common neighbours grows with
    the increment of beta, which contradict the results according to the deviation of the shortest path
    The function are investigating why this contradiction appear.
    The main idea is:
    1. freeze the coordinates of the graph
    2. place i and j in (0.49),(0.5),(0.5),(0.5)
    3. for every beta, get the list of the common neighbours and see what will change
    4. do deviation for both "from a point to the geodesic" and "from a point to the line that geodesic belongs to(perpendicular distance)"
    :return:
    """
    N = 10000
    ED = 10
    betavec = [2.2,2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4,5,6,7, 8,9,10, 16, 32, 64, 128,256,512]
    Geodistance_index = 0
    distance_list = [[0.49, 0.5, 0.5, 0.5], [0.25, 0.25, 0.3, 0.3], [0.25, 0.25, 0.3, 0.3], [0.25, 0.25, 0.5, 0.5],
                     [0.25, 0.25, 0.75, 0.75]]
    x_A = distance_list[Geodistance_index][0]
    y_A = distance_list[Geodistance_index][1]
    x_B = distance_list[Geodistance_index][2]
    y_B = distance_list[Geodistance_index][3]
    geodesic_distance_AB = round(x_B - x_A, 2)
    rg = RandomGenerator(-12)
    rseed = random.randint(0, 100)
    for i in range(rseed):
        rg.ran1()

    inputbeta_network = 2.2
    INputExternalSimutime = 1
    network_index = 2
    beta = betavec[beta_index]
    inputED_network = 5
    print("beta:",beta)
    # load initial network

    filefolder_name = "/home/zqiu1/GSPP/SSRGGpy/R2/distribution/NetworkSRGG/"
    # filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\givendistance\\"
    coorx = []
    coory = []
    FileNetworkCoorName = filefolder_name + "network_coordinates_N{Nn}ED{EDn}beta{betan}xA{xA}yA{yA}xB{xB}yB{yB}Simu{simu}networktime{nt}.txt".format(
        Nn=N, EDn=inputED_network, betan=inputbeta_network, xA=x_A, yA=y_A, xB=x_B, yB=y_B, simu=INputExternalSimutime, nt=network_index)

    with open(FileNetworkCoorName, "r") as file:
        for line in file:
            if line.startswith("#"):
                continue
            data = line.strip().split("\t")  # 使用制表符分割
            coorx.append(float(data[0]))
            coory.append(float(data[1]))
    common_neighbors_dic = {}
    deviations_for_a_nodepair_dic = {}
    connectedornot_dic = {}
    for simu_times in range(1000):
        print(simu_times)
        G, coorx, coory = R2SRGG_withgivennodepair(N, ED, beta, rg, x_A, y_A, x_B, y_B, coorx, coory)
        if nx.has_path(G,N-1,N-2):
            connectedornot_dic[simu_times] = 1
        else:
            connectedornot_dic[simu_times] = 0
        common_neighbors, deviations_for_a_nodepair = compute_common_neighbour_deviation(G, coorx, coory, N)
        print("node",common_neighbors)
        print("dev", deviations_for_a_nodepair)
        common_neighbors_dic[simu_times] = common_neighbors
        deviations_for_a_nodepair_dic[simu_times] = deviations_for_a_nodepair


    filefolder_name2 = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\neighbour_distance\\"
    common_neigthbour_name = filefolder_name2 + "common_neigthbour_list_N{Nn}ED{EDn}beta{betan}xA{xA}yA{yA}xB{xB}yB{yB}Simu{simu}.json".format(
        Nn=N, EDn=ED, betan=beta, xA=x_A, yA=y_A, xB=x_B, yB=y_B, simu=ExternalSimutime)
    with open(common_neigthbour_name, 'w') as file:
        json.dump({str(k): v for k, v in common_neighbors_dic.items()}, file)

    deviations_name = filefolder_name2 + "common_neigthbour_deviationlist_N{Nn}ED{EDn}beta{betan}xA{xA}yA{yA}xB{xB}yB{yB}Simu{simu}.json".format(
        Nn=N, EDn=ED, betan=beta, xA=x_A, yA=y_A, xB=x_B, yB=y_B, simu=ExternalSimutime)
    with open(deviations_name, 'w') as file:
        json.dump({str(k): v for k, v in deviations_for_a_nodepair_dic.items()}, file)

    connected_deviations_name = filefolder_name2 + "common_neigthbour_connection_list_N{Nn}ED{EDn}beta{betan}xA{xA}yA{yA}xB{xB}yB{yB}Simu{simu}.json".format(
        Nn=N, EDn=ED, betan=beta, xA=x_A, yA=y_A, xB=x_B, yB=y_B, simu=ExternalSimutime)
    with open(connected_deviations_name, 'w') as file:
        json.dump({str(k): v for k, v in connectedornot_dic.items()}, file)
