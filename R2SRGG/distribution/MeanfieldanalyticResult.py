# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 10-10-2024
This .m is simulation for Maksim's mean field explanation.
We generate two nodes with fixed coordinates, find all their common neighbours
"""
import numpy as np
import networkx as nx
import random
import json
import math
import sys

from R2SRGG import R2SRGG, distR2, dist_to_geodesic_R2, loadSRGGandaddnode, R2SRGG_withgivennodepair,dist_to_geodesic_perpendicular_R2
# from R2SRGG.R2SRGG import R2SRGG, distR2, dist_to_geodesic_R2, loadSRGGandaddnode, R2SRGG_withgivennodepair, \
#     dist_to_geodesic_perpendicular_R2
from SphericalSoftRandomGeomtricGraph import RandomGenerator
from main import all_shortest_path_node, find_k_connected_node_pairs, find_all_connected_node_pairs

def test_clustering():
    """
    :return: For networkx, c_i for isolated node is 0!
    """
    G = nx.Graph()
    # Add nodes
    G.add_nodes_from([1, 2, 3, 4])
    # Add links (edges)
    edges = [(1, 2), (2, 3), (1, 3),(1,5),(2,5)]
    G.add_edges_from(edges)
    print(nx.clustering(G))
    print("clu:", nx.average_clustering(G))
    common_neighbors = list(nx.common_neighbors(G, 1, 2))
    print(common_neighbors)


def neighbour_distance_inlargeSRGG_clu_cc_givennodepair(N, ED, beta, cc, rg, ExternalSimutime, geodesic_distance_AB,x_A,y_A,x_B,y_B,target_ED):
    """
    :param N:
    :param ED:
    :param ExternalSimutime:
    :return:
    for each node pair, we record the ave,max,min of distance from the shortest path to the geodesic,
    length of the geo distances.
    The generated network, the selected node pair and all the deviation of both shortest path and baseline nodes will be recorded.
    """
    if N> ED:
        deviation_vec = []  # deviation of all shortest path nodes for all node pairs
        baseline_deviation_vec = []  # deviation of all shortest path nodes for all node pairs
        # For each node pair:
        ave_deviation = []
        max_deviation = []
        min_deviation = []
        ave_baseline_deviation =[]
        length_geodesic = []
        hopcount_vec = []
        SPnodenum_vec =[]

        # load a network

        # Randomly generate 10 networks
        Network_generate_time = 10

        for network in range(Network_generate_time):
            # N = 100 # FOR TEST
            G, Coorx, Coory = R2SRGG_withgivennodepair(N, ED, beta, rg, x_A, y_A, x_B, y_B)
            real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
            print("real ED:", real_avg)
            ave_clu = nx.average_clustering(G)
            print("clu:", ave_clu)
            components = list(nx.connected_components(G))
            largest_component = max(components, key=len)
            LCC_number = len(largest_component)
            print("LCC", LCC_number)
            nodei = N-2
            nodej = N-1

            # Find the common neighbours
            common_neighbors = list(nx.common_neighbors(G, nodei, nodej))
            SPnodenum_vec.append(len(common_neighbors))
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
                    dist, _ = dist_to_geodesic_R2(xMed, yMed, xSource, ySource, xEnd, yEnd)
                    deviations_for_a_nodepair.append(dist)

                deviation_vec = deviation_vec+deviations_for_a_nodepair

                ave_deviation.append(np.mean(deviations_for_a_nodepair))
                max_deviation.append(max(deviations_for_a_nodepair))
                min_deviation.append(min(deviations_for_a_nodepair))

        deviation_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\neighbour_distance\\Givendistancedeviation_neighbour_nodes_N{Nn}ED{EDn}CG{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
            Nn = N,EDn=target_ED, betan=cc, ST=ExternalSimutime, Geodistance = geodesic_distance_AB)
        np.savetxt(deviation_vec_name, deviation_vec)
        # For each node pair:
        ave_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\neighbour_distance\\Givendistanceave_neighbour_nodes_deviation_N{Nn}ED{EDn}CG{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
            Nn = N,EDn=target_ED, betan=cc, ST=ExternalSimutime, Geodistance = geodesic_distance_AB)
        np.savetxt(ave_deviation_name, ave_deviation)
        max_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\neighbour_distance\\Givendistancemax_neighbour_nodes_deviation_N{Nn}ED{EDn}CG{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
            Nn = N,EDn=target_ED, betan=cc, ST=ExternalSimutime, Geodistance = geodesic_distance_AB)
        np.savetxt(max_deviation_name, max_deviation)
        min_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\neighbour_distance\\Givendistancemin_neighbour_nodes_deviation_N{Nn}ED{EDn}CG{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
            Nn = N,EDn=target_ED, betan=cc, ST=ExternalSimutime, Geodistance = geodesic_distance_AB)
        np.savetxt(min_deviation_name, min_deviation)
        SPnodenum_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\neighbour_distance\\Givendistanceneighbournodenum_N{Nn}ED{EDn}CG{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
            Nn = N,EDn=target_ED, betan=cc, ST=ExternalSimutime, Geodistance = geodesic_distance_AB)
        np.savetxt(SPnodenum_vec_name, SPnodenum_vec,fmt="%i")


def neighbour_distance(network_size_index, average_degree_index, cc_index, Geodistance_index ,ExternalSimutime):
    # for control cc
    #----------------------------------------------------------------------------------------------------
    smallED_matrix = [[2.6, 2.6, 2.6, 2.6, 2.6, 0, 0],
                      [4.3, 4, 4, 4, 4, 0],
                      [5.6, 5.4, 5.2, 5.2, 5.2, 5.2, 5.2],
                      [7.5, 6.7, 6.5, 6.5, 6.5, 6.5]]
    smallbeta_matrix = [[3.1, 4.5, 6, 300, 0, 0],
                        [2.7, 3.5, 4.7, 7.6, 300, 0],
                        [2.7, 3.4, 4.3, 5.7, 11, 300],
                        [2.55, 3.2, 4, 5.5, 8.5, 300]]

    Nvec = [10, 100, 200, 500, 1000, 10000]
    kvec = list(range(2, 20)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    cc_vec = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    betavec = [2.55, 3.2, 3.99, 5.15, 7.99, 300]

    distance_list = [[0.49,0.5,0.5,0.5],[0.25, 0.25, 0.3, 0.3],[0.25, 0.25, 0.3, 0.3], [0.25, 0.25, 0.5, 0.5], [0.25, 0.25, 0.75, 0.75]]
    x_A = distance_list[Geodistance_index][0]
    y_A = distance_list[Geodistance_index][1]
    x_B = distance_list[Geodistance_index][2]
    y_B = distance_list[Geodistance_index][3]
    geodesic_distance_AB = round(x_B - x_A, 4)

    random.seed(ExternalSimutime)
    N = Nvec[network_size_index]
    target_ED = kvec[average_degree_index]
    if average_degree_index<4:
        ED = smallED_matrix[average_degree_index][cc_index]
        if ED == 0:
            raise RuntimeError("Not exist")
        beta = smallbeta_matrix[average_degree_index][cc_index]
    else:
        ED = kvec[average_degree_index]
        beta = betavec[cc_index]

    C_G = cc_vec[cc_index]
    print("input para:", (N, ED, beta,C_G,geodesic_distance_AB))


    rg = RandomGenerator(-12)
    rseed = random.randint(0, 100)
    for i in range(rseed):
        rg.ran1()

    # for large network, we only generate one network and randomly selected 1,000 node pair.
    # for small network, we generate 100 networks and selected all the node pair in the LCC
    if N > 100:
        neighbour_distance_inlargeSRGG_clu_cc_givennodepair(N, ED, beta, C_G,rg, ExternalSimutime,geodesic_distance_AB,x_A,y_A,x_B,y_B,target_ED)
    # else:
    #     # Random select nodepair_num nodes in the largest connected component
    #     distance_insmallSRGG(N, ED, beta, rg, ExternalSimutime)
    print("ok")


def neighbour_distance_inlargeSRGG_clu_beta_givennodepair(N, ED, beta, rg, ExternalSimutime, geodesic_distance_AB,x_A,y_A,x_B,y_B):
    """
    :param N:
    :param ED:
    :param ExternalSimutime:
    :return:
    for each node pair, we record the ave,max,min of distance from the shortest path to the geodesic,
    length of the geo distances.
    The generated network, the selected node pair and all the deviation of both shortest path and baseline nodes will be recorded.
    """
    if N> ED:
        deviation_vec = []  # deviation of all shortest path nodes for all node pairs
        baseline_deviation_vec = []  # deviation of all shortest path nodes for all node pairs
        # For each node pair:
        ave_deviation = []
        max_deviation = []
        min_deviation = []
        ave_baseline_deviation =[]
        length_geodesic = []
        hopcount_vec = []
        SPnodenum_vec =[]

        # load a network

        # Randomly generate 10 networks
        Network_generate_time = 10

        for network in range(Network_generate_time):
            # N = 100 # FOR TEST
            G, Coorx, Coory = R2SRGG_withgivennodepair(N, ED, beta, rg, x_A, y_A, x_B, y_B)
            real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
            print("real ED:", real_avg)
            ave_clu = nx.average_clustering(G)
            print("clu:", ave_clu)
            components = list(nx.connected_components(G))
            largest_component = max(components, key=len)
            LCC_number = len(largest_component)
            print("LCC", LCC_number)
            nodei = N-2
            nodej = N-1

            # Find the common neighbours
            common_neighbors = list(nx.common_neighbors(G, nodei, nodej))
            SPnodenum_vec.append(len(common_neighbors))
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
                    dist, _ = dist_to_geodesic_R2(xMed, yMed, xSource, ySource, xEnd, yEnd)
                    deviations_for_a_nodepair.append(dist)

                deviation_vec = deviation_vec+deviations_for_a_nodepair

                ave_deviation.append(np.mean(deviations_for_a_nodepair))
                max_deviation.append(max(deviations_for_a_nodepair))
                min_deviation.append(min(deviations_for_a_nodepair))

        deviation_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\neighbour_distance\\Givendistancedeviation_neighbour_nodes_N{Nn}ED{EDn}beta{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
            Nn = N,EDn=ED, betan=beta, ST=ExternalSimutime, Geodistance = geodesic_distance_AB)
        np.savetxt(deviation_vec_name, deviation_vec)
        # For each node pair:
        ave_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\neighbour_distance\\Givendistanceave_neighbour_nodes_deviation_N{Nn}ED{EDn}beta{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
            Nn = N,EDn=ED, betan=beta, ST=ExternalSimutime, Geodistance = geodesic_distance_AB)
        np.savetxt(ave_deviation_name, ave_deviation)
        max_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\neighbour_distance\\Givendistancemax_neighbour_nodes_deviation_N{Nn}ED{EDn}beta{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
            Nn = N,EDn=ED, betan=beta, ST=ExternalSimutime, Geodistance = geodesic_distance_AB)
        np.savetxt(max_deviation_name, max_deviation)
        min_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\neighbour_distance\\Givendistancemin_neighbour_nodes_deviation_N{Nn}ED{EDn}beta{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
            Nn = N,EDn=ED, betan=beta, ST=ExternalSimutime, Geodistance = geodesic_distance_AB)
        np.savetxt(min_deviation_name, min_deviation)
        SPnodenum_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\neighbour_distance\\Givendistanceneighbournodenum_N{Nn}ED{EDn}beta{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
            Nn = N,EDn=ED, betan=beta, ST=ExternalSimutime, Geodistance = geodesic_distance_AB)
        np.savetxt(SPnodenum_vec_name, SPnodenum_vec, fmt="%i")


def neighbour_distance_beta(network_size_index, average_degree_index, cc_index, Geodistance_index, ExternalSimutime):
    # for control beta
    # ----------------------------------------------------------------------------------------------------
    Nvec = [10, 100, 200, 500, 1000, 10000]
    N = Nvec[network_size_index]
    kvec = [2, 5, 10, 20, 100]
    betavec = [2.2, 4, 8, 16, 32, 64, 128]
    # betavec = [2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8]

    distance_list = [[0.49, 0.5, 0.5, 0.5], [0.25, 0.25, 0.3, 0.3], [0.25, 0.25, 0.3, 0.3], [0.25, 0.25, 0.5, 0.5],
                     [0.25, 0.25, 0.75, 0.75]]
    x_A = distance_list[Geodistance_index][0]
    y_A = distance_list[Geodistance_index][1]
    x_B = distance_list[Geodistance_index][2]
    y_B = distance_list[Geodistance_index][3]
    geodesic_distance_AB = round(x_B - x_A, 4)

    random.seed(ExternalSimutime)
    ED = kvec[average_degree_index]
    beta = betavec[cc_index]

    print("input para:", (N, ED, beta, geodesic_distance_AB))

    rg = RandomGenerator(-12)
    rseed = random.randint(0, 100)
    for i in range(rseed):
        rg.ran1()

    # for large network, we only generate one network and randomly selected 1,000 node pair.
    # for small network, we generate 100 networks and selected all the node pair in the LCC
    if N > 100:
        neighbour_distance_inlargeSRGG_clu_beta_givennodepair(N, ED, beta, rg, ExternalSimutime,
                                                            geodesic_distance_AB, x_A, y_A, x_B, y_B)
    # else:
    #     # Random select nodepair_num nodes in the largest connected component
    #     distance_insmallSRGG(N, ED, beta, rg, ExternalSimutime)
    print("ok")

    # Press the green button in the gutter to run the script.


def compute_common_neighbour_deviation(G,Coorx,Coory,N):
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


def neighbour_distance_with_beta_one_graph():
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
    ED = 5
    betavec = [2.2, 4, 8, 16, 32, 64, 128]
    # betavec = [2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8]
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

    beta = 2.2
    ExternalSimutime = 1
    network_index = 2

    # load initial network
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\givendistance\\"
    FileNetworkName = filefolder_name + "network_N{Nn}ED{EDn}beta{betan}xA{xA}yA{yA}xB{xB}yB{yB}Simu{simu}networktime{nt}.txt".format(
        Nn=N, EDn=ED, betan=beta, xA=x_A, yA=y_A, xB=x_B, yB=y_B, simu=ExternalSimutime, nt=network_index)
    G = loadSRGGandaddnode(N,FileNetworkName)
    coorx = []
    coory = []
    FileNetworkCoorName = filefolder_name + "network_coordinates_N{Nn}ED{EDn}beta{betan}xA{xA}yA{yA}xB{xB}yB{yB}Simu{simu}networktime{nt}.txt".format(
        Nn=N, EDn=ED, betan=beta, xA=x_A, yA=y_A, xB=x_B, yB=y_B, simu=ExternalSimutime, nt=network_index)
    print(FileNetworkCoorName)
    with open(FileNetworkCoorName, "r") as file:
        for line in file:
            if line.startswith("#"):
                continue
            data = line.strip().split("\t")  # 使用制表符分割
            coorx.append(float(data[0]))
            coory.append(float(data[1]))

    common_neighbors, deviations_for_a_nodepair = compute_common_neighbour_deviation(G, coorx, coory, N)
    print(common_neighbors)
    print(deviations_for_a_nodepair)

    for beta_index in range(len(betavec)):
        G, coorx, coory = R2SRGG_withgivennodepair(N, ED, beta, rg, x_A, y_A, x_B, y_B, coorx, coory)
        print(coorx[N-1], coory[N-1])
        print(coorx[N-2], coory[N-2])
        common_neighbors, deviations_for_a_nodepair = compute_common_neighbour_deviation(G, coorx, coory, N)
        print(common_neighbors)
        print(deviations_for_a_nodepair)


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


if __name__ == '__main__':
    # network_size_index = 4
    # average_degree_index = 2
    # beta_index = 1
    # external_simu_time = 0
    # distance_inSRGG(network_size_index, average_degree_index, beta_index, external_simu_time)

    # for N_index in range(4):
    #     for ED_index in range(24):
    #         for beta_index in range(7):
    #             distance_inSRGG(N_index, ED_index, beta_index, 0)


    # i = sys.argv[1]
    # exemptionlist = np.loadtxt("/home/zqiu1/GSPP/SSRGGpy/R2/distribution/notrun.txt")
    # notrun_pair = exemptionlist[int(i)]
    # ED = notrun_pair[1]
    # beta = notrun_pair[2]
    # ExternalSimutime = notrun_pair[3]
    # kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    # betavec = [2.1, 4, 8, 16, 32, 64, 128]
    # ED_index = kvec.index(notrun_pair[1])
    # beta_index = betavec.index(notrun_pair[2])
    # distance_inSRGG_clu(7, int(ED_index), int(beta_index), int(ExternalSimutime))

    # test the code

    # distance_inSRGG_withEDCC(5, int(4), int(0), int(0), int(0))
    # run the code
    # ED = sys.argv[1]
    # cc_index = sys.argv[2]
    # Geodistance_index = sys.argv[3]
    # ExternalSimutime = sys.argv[4]
    # distance_inSRGG_withEDCC(5, int(ED), int(cc_index), int(Geodistance_index), int(ExternalSimutime))
    #


    """
    test code
    """
    # r = math.sqrt(1/9999/math.pi)
    # print(r)
    # test_clustering()
    # neighbour_distance(5,0,0,0,0)

    # """
    # Run code
    # """
    # ED = sys.argv[1]
    # cc_index = sys.argv[2]
    # Geodistance_index = sys.argv[3]
    # ExternalSimutime = sys.argv[4]
    # neighbour_distance(5, int(ED), int(cc_index), int(Geodistance_index), int(ExternalSimutime))

    # """
    # Run code locally
    # """
    # cc_index = 2
    # Geodistance_index = 0
    # for ED in range(28):
    #     for ExternalSimutime in range(10):
    #         neighbour_distance(5, int(ED), int(cc_index), int(Geodistance_index), int(ExternalSimutime))

    # neighbour_distance_beta(5,0,0,0,0)
    # """
    # Run code beta
    # """
    # ED = sys.argv[1]
    # cc_index = sys.argv[2]
    # Geodistance_index = sys.argv[3]
    # ExternalSimutime = sys.argv[4]
    # neighbour_distance_beta(5, int(ED), int(cc_index), int(Geodistance_index), int(ExternalSimutime))

    """
    check neighbour distance 
    """
    # neighbour_distance_with_beta_one_graph()
    # N = 10000
    # for k in range(2,10):
    #     print(math.sqrt(k/(math.pi*N)))
    """
    check neighbour distance for clu
    """
    betaindex = sys.argv[1]
    simu_index = sys.argv[2]
    # betaindex = 0
    neighbour_distance_with_beta_one_graph_clu(int(betaindex), int(simu_index)) # beta index change from 0 to 19

