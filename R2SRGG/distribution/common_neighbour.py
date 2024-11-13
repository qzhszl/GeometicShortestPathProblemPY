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

def common_neighbour_generator(N, avg, beta, rg, Coorx, Coory):
    assert beta > 2
    assert avg > 0
    assert N > 1

    R = 2.0  # manually tuned value
    alpha = (2 * N / avg * R * R) * (math.pi / (math.sin(2 * math.pi / beta) * beta))
    alpha = math.sqrt(alpha)
    s = []
    t = []
    # Assign coordinates
    xx = Coorx
    yy = Coory

    # make connections
    for i in range(N-2,N):
        for j in range(i):
            dist = math.sqrt((xx[i] - xx[j]) ** 2 + (yy[i] - yy[j]) ** 2)
            assert dist > 0
            try:
                prob = 1 / (1 + math.exp(beta * math.log(alpha * dist)))
            except:
                prob = 0
            if rg.ran1() < prob:
                s.append(i)
                t.append(j)
    # Create graph and remove self-loops
    G = nx.Graph()
    G.add_edges_from(zip(s, t))
    if not G.has_node(9999):
        G.add_node(9999)

    # 检查节点 9998 是否在图中，如果不在则添加
    if not G.has_node(9998):
        G.add_node(9998)
    # if G.number_of_nodes() < N:
    #     ExpectedNodeList = [i for i in range(0, N)]
    #     Nodelist = list(G.nodes)
    #     difference = [item for item in ExpectedNodeList if item not in Nodelist]
    #     G.add_nodes_from(difference)
    return G, xx, yy


def compute_common_neighbour_deviation(G, Coorx, Coory, N):
    nodei = N - 2
    nodej = N - 1
    # Find the common neighbours
    common_neighbors = list(nx.common_neighbors(G, nodei, nodej))
    if common_neighbors:
        xSource = Coorx[nodei]
        ySource = Coory[nodei]
        xEnd = Coorx[nodej]
        yEnd = Coory[nodej]
        # length_geodesic.append(distR2(xSource, ySource, xEnd, yEnd)) # for test
        # Compute deviation for the shortest path of each node pair
        deviations_for_a_nodepair = []
        for SPnode in common_neighbors:
            xMed = Coorx[SPnode]
            yMed = Coory[SPnode]
            # dist, _ = dist_to_geodesic_R2(xMed, yMed, xSource, ySource, xEnd, yEnd)
            dist, _ = dist_to_geodesic_perpendicular_R2(xMed, yMed, xSource, ySource, xEnd, yEnd)
            deviations_for_a_nodepair.append(dist)
    else:
        deviations_for_a_nodepair = []
    return common_neighbors, deviations_for_a_nodepair


def neighbour_distance_ED_beta_one_graph(ED_index, beta_index,ExternalSimutime):
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
    ED_vec = [2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    betavec = [2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64, 128, 256, 512]
    ED = ED_vec[ED_index]
    Geodistance_index = 0
    distance_list = [[0.491, 0.5, 0.509, 0.5],[0.25, 0.5, 0.75, 0.5]]
    x_A = distance_list[Geodistance_index][0]
    y_A = distance_list[Geodistance_index][1]
    x_B = distance_list[Geodistance_index][2]
    y_B = distance_list[Geodistance_index][3]
    geodesic_distance_AB = round(x_B - x_A, 2)
    rg = RandomGenerator(-12)
    rseed = random.randint(0, 10000)
    for i in range(rseed):
        rg.ran1()

    inputbeta_network = 2.2
    INputExternalSimutime = 1
    network_index = 2
    beta = betavec[beta_index]
    inputED_network = 5
    print("ED:", ED)
    print("beta:",beta)
    # load initial network

    # filefolder_name = "/home/zqiu1/GSPP/SSRGGpy/R2/distribution/NetworkSRGG/"
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\givendistance\\"
    coorx = []
    coory = []
    FileNetworkCoorName = filefolder_name + "network_coordinates_N{Nn}xA{xA}yA{yA}xB{xB}yB{yB}.txt".format(
        Nn=10000, xA=0.491, yA=0.5, xB=0.509, yB=0.5)

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
        # print(simu_times)
        G, coorx, coory = common_neighbour_generator(N, ED, beta, rg, coorx, coory)
        if nx.has_path(G,N-1,N-2):
            connectedornot_dic[simu_times] = 1
        else:
            connectedornot_dic[simu_times] = 0
        common_neighbors, deviations_for_a_nodepair = compute_common_neighbour_deviation(G, coorx, coory, N)
        # print("node",common_neighbors)
        # print("dev", deviations_for_a_nodepair)
        common_neighbors_dic[simu_times] = common_neighbors
        deviations_for_a_nodepair_dic[simu_times] = deviations_for_a_nodepair


    filefolder_name2 = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\neighbour_distance\\commonneighbourmodel\\"
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


def neighbour_distance_ED_beta_one_graph_centerO(ED_index, beta_index, ExternalSimutime):
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
    ED_vec = [2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    betavec = [2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64, 128, 256, 512]
    ED = ED_vec[ED_index]
    Geodistance_index = 0
    distance_list = [[-0.005, 0, 0.005, 0]]
    x_A = distance_list[Geodistance_index][0]
    y_A = distance_list[Geodistance_index][1]
    x_B = distance_list[Geodistance_index][2]
    y_B = distance_list[Geodistance_index][3]
    geodesic_distance_AB = round(x_B - x_A, 2)
    rg = RandomGenerator(-12)
    rseed = random.randint(0, 10000)
    for i in range(rseed):
        rg.ran1()

    beta = betavec[beta_index]
    print("ED:", ED)
    print("beta:", beta)
    # load initial network

    # filefolder_name = "/home/zqiu1/GSPP/SSRGGpy/R2/distribution/NetworkSRGG/"
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\givendistance\\"
    coorx = []
    coory = []
    FileNetworkCoorName = filefolder_name + "network_coordinates_N{Nn}xA{xA}yA{yA}xB{xB}yB{yB}centero.txt".format(
        Nn=10000, xA=x_A, yA=y_A, xB=x_B, yB=y_B)
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
        # print(simu_times)
        G, coorx, coory = common_neighbour_generator(N, ED, beta, rg, coorx, coory)
        if nx.has_path(G, N - 1, N - 2):
            connectedornot_dic[simu_times] = 1
        else:
            connectedornot_dic[simu_times] = 0
        common_neighbors, deviations_for_a_nodepair = compute_common_neighbour_deviation(G, coorx, coory, N)
        # print("node",common_neighbors)
        # print("dev", deviations_for_a_nodepair)
        common_neighbors_dic[simu_times] = common_neighbors
        deviations_for_a_nodepair_dic[simu_times] = deviations_for_a_nodepair

    filefolder_name2 = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\neighbour_distance\\commonneighbourmodel\\"
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

if __name__ == '__main__':
    # rg = RandomGenerator(-12)
    # rseed = random.randint(0, 10000)
    # for i in range(rseed):
    #     rg.ran1()
    # filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\givendistance\\"
    # coorx = []
    # coory = []
    # FileNetworkCoorName = filefolder_name + "network_coordinates_N{Nn}xA{xA}yA{yA}xB{xB}yB{yB}.txt".format(
    #     Nn=10000, xA=0.491, yA=0.5, xB=0.509, yB=0.5)
    #
    # with open(FileNetworkCoorName, "r") as file:
    #     for line in file:
    #         if line.startswith("#"):
    #             continue
    #         data = line.strip().split("\t")  # 使用制表符分割
    #         coorx.append(float(data[0]))
    #         coory.append(float(data[1]))
    # common_neighbour_generator(N=10000, avg=10, beta=4, rg=rg, Coorx=coorx, Coory=coory)
    ED_vec = [2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    betavec = [2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64, 128, 256, 512]
    # for ED in range(23):
    #     for beta in range(22):
    #         neighbour_distance_ED_beta_one_graph_centerO(ED, beta, 0)
    for ED in range(23):
        beta_index = 9
        neighbour_distance_ED_beta_one_graph_centerO(ED, beta_index, 0)

