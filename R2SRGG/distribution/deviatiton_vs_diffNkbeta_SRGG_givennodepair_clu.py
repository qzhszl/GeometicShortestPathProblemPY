# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 26-9-2024
The SRGG are given node pair
We will generate SRGG, compute the ave,max,min of distance from the shortest path to the geodesic(deviation) for selected node pairs
We also record the deviation for randomly selected nodes as a baseline

For small graph, we generate 100 graphs
for each graph, we record the real average degree, LCC number, clustering coefficient
for each node pair, we only record the ave,max,min of distance from the shortest path to the geodesic,
length of the geo distances and randomly select some nodes and record their deviation

For large network, we only generate 1 graph and randomly selected 100 node pairs.
The generated network, the selected node pair and all the deviation of both shortest path and baseline nodes will be recorded.

This script is collecting data for investigating
1. distribution of the ave, max and min deviation and compare them with random selected nodes
2.the local optimum of the average distance from the shortest path to the geodesic in SRGG.
We have already seen that with the increase of the average degree, the average distance goes up, then down, then up agian.
We also want to study what will happen if we change the beta, i.e. the parameter that determines clustering coefficient
"""
import itertools

import numpy as np
import networkx as nx
import random
import math
import sys

from R2SRGG import R2SRGG, distR2, dist_to_geodesic_R2, loadSRGGandaddnode, R2SRGG_withgivennodepair
# from R2SRGG.R2SRGG import R2SRGG, distR2, dist_to_geodesic_R2, loadSRGGandaddnode, R2SRGG_withgivennodepair
from SphericalSoftRandomGeomtricGraph import RandomGenerator
from main import all_shortest_path_node, find_k_connected_node_pairs, find_all_connected_node_pairs


# def generate_r2SRGG():
#     rg = RandomGenerator(-12)
#     rseed = random.randint(0, 100)
#     print(rseed)
#     for i in range(rseed):
#         rg.ran1()
#
#     Nvec = [200, 500, 1000, 10000]
#     kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
#     betavec = [2.1, 4, 8, 16, 32, 64, 128]
#
#     for N in Nvec:
#         for ED in kvec:
#             for beta in betavec:
#
#                 G, Coorx, Coory = R2SRGG(N, ED, beta, rg)
#                 real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
#                 print("input para:", (N,ED,beta))
#                 print("real ED:", real_avg)
#                 print("clu:", nx.average_clustering(G))
#                 components = list(nx.connected_components(G))
#                 largest_component = max(components, key=len)
#                 print("LCC", len(largest_component))
#
#                 FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\network_N{Nn}ED{EDn}Beta{betan}.txt".format(
#                     Nn=N, EDn=ED, betan=beta)
#                 nx.write_edgelist(G, FileNetworkName)
#
#                 FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt".format(
#                     Nn=N, EDn=ED, betan=beta)
#                 with open(FileNetworkCoorName, "w") as file:
#                     for data1, data2 in zip(Coorx, Coory):
#                         file.write(f"{data1}\t{data2}\n")


def distance_insmallSRGG(N, ED, beta, rg, ExternalSimutime):
    """
    :param N:
    :param ED:
    :param beta:
    :param rg:
    :return:
    for each graph, we record the real average degree, LCC number, clustering coefficient
    for each node pair, we only record the ave,max,min of distance from the shortest path to the geodesic,
    length of the geo distances.
    """
    if N> ED:
        # For graph:
        real_ave_degree = []
        LCC_num = []
        clustering_coefficient = []
        # For each node pair:
        ave_deviation = []
        max_deviation = []
        min_deviation = []
        ave_baseline_deviation =[]
        length_geodesic = []
        SPnodenum_vec =[]
        simu_times = 100
        for simu_index in range(simu_times):
            G, Coorx, Coory = R2SRGG(N, ED, beta, rg)
            try:
                real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
            except:
                flag = 0
                while flag == 0:
                    G, Coorx, Coory = R2SRGG(N, ED, beta, rg)
                    if nx.number_of_edges(G) > 0:
                        flag=1
                        real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)

            print("real ED:", real_avg)
            real_ave_degree.append(real_avg)
            ave_clu = nx.average_clustering(G)
            print("clu:",ave_clu)
            clustering_coefficient.append(ave_clu)
            components = list(nx.connected_components(G))
            largest_component = max(components, key=len)
            LCC_number = len(largest_component)
            print("LCC", LCC_number)
            LCC_num.append(LCC_number)

            # pick up all the node pairs in the LCC and save them in the unique_pairs
            unique_pairs = find_all_connected_node_pairs(G)
            count = 0
            for node_pair in unique_pairs:
                count = count+1
                nodei = node_pair[0]
                nodej = node_pair[1]
                # Find the shortest path nodes
                SPNodelist = all_shortest_path_node(G, nodei, nodej)
                SPnodenum = len(SPNodelist)
                SPnodenum_vec.append(SPnodenum)
                if SPnodenum>0:
                    xSource = Coorx[nodei]
                    ySource = Coory[nodei]
                    xEnd = Coorx[nodej]
                    yEnd = Coory[nodej]
                    length_geodesic.append(distR2(xSource, ySource, xEnd, yEnd))
                    # Compute deviation for the shortest path of each node pair
                    deviations_for_a_nodepair = []
                    for SPnode in SPNodelist:
                        xMed = Coorx[SPnode]
                        yMed = Coory[SPnode]
                        dist, _ = dist_to_geodesic_R2(xMed, yMed, xSource, ySource, xEnd, yEnd)
                        deviations_for_a_nodepair.append(dist)
                    ave_deviation.append(np.mean(deviations_for_a_nodepair))
                    max_deviation.append(max(deviations_for_a_nodepair))
                    min_deviation.append(min(deviations_for_a_nodepair))

                    baseline_deviations_for_a_nodepair = []
                    # compute baseline's deviation
                    filtered_numbers = [num for num in range(N) if num not in [nodei,nodej]]
                    base_line_node_index = random.sample(filtered_numbers,SPnodenum)

                    for SPnode in base_line_node_index:
                        xMed = Coorx[SPnode]
                        yMed = Coory[SPnode]
                        dist, _ = dist_to_geodesic_R2(xMed, yMed, xSource, ySource, xEnd, yEnd)
                        baseline_deviations_for_a_nodepair.append(dist)
                    ave_baseline_deviation.append(np.mean(baseline_deviations_for_a_nodepair))


        real_ave_degree_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\real_ave_degree_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(real_ave_degree_name, real_ave_degree)
        LCC_num_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\LCC_num_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(LCC_num_name, LCC_num, fmt="%i")
        clustering_coefficient_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\clustering_coefficient_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(clustering_coefficient_name, clustering_coefficient)
        # For each node pair:
        ave_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(ave_deviation_name, ave_deviation)
        max_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\max_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(max_deviation_name, max_deviation)
        min_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\min_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(min_deviation_name, min_deviation)
        ave_baseline_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\ave_baseline_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(ave_baseline_deviation_name, ave_baseline_deviation)
        length_geodesic_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\length_geodesic_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(length_geodesic_name, length_geodesic)
        SPnodenum_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\SPnodenum_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(SPnodenum_vec_name, SPnodenum_vec, fmt="%i")


def distance_inlargeSRGG_clu_cc_givennodepair(N, ED, beta, cc, rg, ExternalSimutime, geodesic_distance_AB,x_A,y_A,x_B,y_B,target_ED):
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
            # Find the shortest path nodes
            if nx.has_path(G, nodei, nodej):
                SPNodelist = all_shortest_path_node(G, nodei, nodej)
                hopcount_vec.append(nx.shortest_path_length(G, nodei, nodej))
                SPnodenum = len(SPNodelist)
                SPnodenum_vec.append(SPnodenum)
                if SPnodenum>0:
                    xSource = Coorx[nodei]
                    ySource = Coory[nodei]
                    xEnd = Coorx[nodej]
                    yEnd = Coory[nodej]
                    length_geodesic.append(distR2(xSource, ySource, xEnd, yEnd))
                    # Compute deviation for the shortest path of each node pair
                    deviations_for_a_nodepair = []
                    for SPnode in SPNodelist:
                        xMed = Coorx[SPnode]
                        yMed = Coory[SPnode]
                        dist, _ = dist_to_geodesic_R2(xMed, yMed, xSource, ySource, xEnd, yEnd)
                        deviations_for_a_nodepair.append(dist)

                    deviation_vec = deviation_vec+deviations_for_a_nodepair

                    ave_deviation.append(np.mean(deviations_for_a_nodepair))
                    max_deviation.append(max(deviations_for_a_nodepair))
                    min_deviation.append(min(deviations_for_a_nodepair))

                    baseline_deviations_for_a_nodepair = []
                    # compute baseline's deviation
                    filtered_numbers = [num for num in range(N) if num not in [nodei,nodej]]
                    base_line_node_index = random.sample(filtered_numbers,SPnodenum)

                    for SPnode in base_line_node_index:
                        xMed = Coorx[SPnode]
                        yMed = Coory[SPnode]
                        dist, _ = dist_to_geodesic_R2(xMed, yMed, xSource, ySource, xEnd, yEnd)
                        baseline_deviations_for_a_nodepair.append(dist)
                    ave_baseline_deviation.append(np.mean(baseline_deviations_for_a_nodepair))
                    baseline_deviation_vec = baseline_deviation_vec + baseline_deviations_for_a_nodepair

        deviation_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\Givendistancedeviation_shortest_path_nodes_N{Nn}ED{EDn}CG{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
            Nn = N,EDn=target_ED, betan=cc, ST=ExternalSimutime, Geodistance = geodesic_distance_AB)
        np.savetxt(deviation_vec_name, deviation_vec)
        baseline_deviation_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\Givendistancedeviation_baseline_nodes_num_N{Nn}ED{EDn}CG{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
            Nn = N,EDn=target_ED, betan=cc, ST=ExternalSimutime, Geodistance = geodesic_distance_AB)
        np.savetxt(baseline_deviation_vec_name, baseline_deviation_vec)
        # For each node pair:
        ave_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\Givendistanceave_deviation_N{Nn}ED{EDn}CG{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
            Nn = N,EDn=target_ED, betan=cc, ST=ExternalSimutime, Geodistance = geodesic_distance_AB)
        np.savetxt(ave_deviation_name, ave_deviation)
        max_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\Givendistancemax_deviation_N{Nn}ED{EDn}CG{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
            Nn = N,EDn=target_ED, betan=cc, ST=ExternalSimutime, Geodistance = geodesic_distance_AB)
        np.savetxt(max_deviation_name, max_deviation)
        min_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\Givendistancemin_deviation_N{Nn}ED{EDn}CG{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
            Nn = N,EDn=target_ED, betan=cc, ST=ExternalSimutime, Geodistance = geodesic_distance_AB)
        np.savetxt(min_deviation_name, min_deviation)
        ave_baseline_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\Givendistanceave_baseline_deviation_N{Nn}ED{EDn}CG{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
            Nn = N,EDn=target_ED, betan=cc, ST=ExternalSimutime, Geodistance = geodesic_distance_AB)
        np.savetxt(ave_baseline_deviation_name, ave_baseline_deviation)
        length_geodesic_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\Givendistancelength_geodesic_N{Nn}ED{EDn}CG{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
            Nn = N,EDn=target_ED, betan=cc, ST=ExternalSimutime, Geodistance = geodesic_distance_AB)
        np.savetxt(length_geodesic_name, length_geodesic)
        SPnodenum_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\GivendistanceSPnodenum_N{Nn}ED{EDn}CG{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
            Nn = N,EDn=target_ED, betan=cc, ST=ExternalSimutime, Geodistance = geodesic_distance_AB)
        np.savetxt(SPnodenum_vec_name, SPnodenum_vec,fmt="%i")
        hopcount_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\Givendistancehopcount_sp_N{Nn}ED{EDn}CG{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
            Nn = N,EDn=target_ED, betan=cc, ST=ExternalSimutime, Geodistance = geodesic_distance_AB)
        np.savetxt(hopcount_Name, hopcount_vec)

def distance_inSRGG_withEDCC(network_size_index, average_degree_index, cc_index, Geodistance_index ,ExternalSimutime):
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

    distance_list = [[0.25, 0.25, 0.3, 0.3], [0.25, 0.25, 0.5, 0.5], [0.25, 0.25, 0.75, 0.75]]
    x_A = distance_list[Geodistance_index][0]
    y_A = distance_list[Geodistance_index][1]
    x_B = distance_list[Geodistance_index][2]
    y_B = distance_list[Geodistance_index][3]
    geodesic_distance_AB = x_B - x_A

    random.seed(ExternalSimutime)
    N = Nvec[network_size_index]

    if average_degree_index<4:
        ED = smallED_matrix[average_degree_index][cc_index]
        if ED == 0:
            raise RuntimeError("Not exist")
        beta = smallbeta_matrix[average_degree_index][cc_index]
    else:
        ED = kvec[average_degree_index]
        beta = betavec[cc_index]
    target_ED = kvec[average_degree_index]
    C_G = cc_vec[cc_index]
    print("input para:", (N, ED, beta,C_G,geodesic_distance_AB))

    rg = RandomGenerator(-12)
    rseed = random.randint(0, 100)
    for i in range(rseed):
        rg.ran1()

    # for large network, we only generate one network and randomly selected 1,000 node pair.
    # for small network, we generate 100 networks and selected all the node pair in the LCC
    if N > 100:
        distance_inlargeSRGG_clu_cc_givennodepair(N, ED, beta, C_G,rg, ExternalSimutime,geodesic_distance_AB,x_A,y_A,x_B,y_B,target_ED)
    else:
        # Random select nodepair_num nodes in the largest connected component
        distance_insmallSRGG(N, ED, beta, rg, ExternalSimutime)
    print("ok")
    # Press the green button in the gutter to run the script.


def distance_inlargeSRGG_clu_beta_givennodepair(N, ED, beta, rg, ExternalSimutime, geodesic_distance_AB, x_A, y_A, x_B,
                                                y_B):
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
        filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\GivenGeodistance\\"
        filefolder_name = "/work/zqiu1/"
        # Randomly generate 10 networks
        Network_generate_time = 20

        for network_index in range(Network_generate_time):
            # N = 100 # FOR TEST
            G, Coorx, Coory = R2SRGG_withgivennodepair(N, ED, beta, rg, x_A, y_A, x_B, y_B)
            # FileNetworkName = filefolder_name + "network_N{Nn}ED{EDn}beta{betan}xA{xA}yA{yA}xB{xB}yB{yB}Simu{simu}networktime{nt}.txt".format(
            #     Nn=N, EDn=ED, betan=beta, xA=x_A, yA=y_A, xB=x_B, yB=y_B, simu=ExternalSimutime, nt=network_index)
            # nx.write_edgelist(G, FileNetworkName)
            #
            # FileNetworkCoorName = filefolder_name + "network_coordinates_N{Nn}ED{EDn}beta{betan}xA{xA}yA{yA}xB{xB}yB{yB}Simu{simu}networktime{nt}.txt".format(
            #     Nn=N, EDn=ED, betan=beta, xA=x_A, yA=y_A, xB=x_B, yB=y_B, simu=ExternalSimutime, nt=network_index)
            # with open(FileNetworkCoorName, "w") as file:
            #     for data1, data2 in zip(Coorx, Coory):
            #         file.write(f"{data1}\t{data2}\n")

            real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
            print("real ED:", real_avg)
            # ave_clu = nx.average_clustering(G)
            # print("clu:", ave_clu)
            # components = list(nx.connected_components(G))
            # largest_component = max(components, key=len)
            # LCC_number = len(largest_component)
            # print("LCC", LCC_number)
            nodei = N-2
            nodej = N-1
            # Find the shortest path nodes
            if nx.has_path(G, nodei, nodej):
                SPNodelist = all_shortest_path_node(G, nodei, nodej)
                hopcount_vec.append(nx.shortest_path_length(G, nodei, nodej))
                SPnodenum = len(SPNodelist)
                SPnodenum_vec.append(SPnodenum)
                if SPnodenum>0:
                    xSource = Coorx[nodei]
                    ySource = Coory[nodei]
                    xEnd = Coorx[nodej]
                    yEnd = Coory[nodej]
                    length_geodesic.append(distR2(xSource, ySource, xEnd, yEnd))
                    # Compute deviation for the shortest path of each node pair
                    deviations_for_a_nodepair = []
                    for SPnode in SPNodelist:
                        xMed = Coorx[SPnode]
                        yMed = Coory[SPnode]
                        dist, _ = dist_to_geodesic_R2(xMed, yMed, xSource, ySource, xEnd, yEnd)
                        deviations_for_a_nodepair.append(dist)

                    deviation_vec = deviation_vec+deviations_for_a_nodepair

                    ave_deviation.append(np.mean(deviations_for_a_nodepair))
                    max_deviation.append(max(deviations_for_a_nodepair))
                    min_deviation.append(min(deviations_for_a_nodepair))

                    baseline_deviations_for_a_nodepair = []
                    # compute baseline's deviation
                    filtered_numbers = [num for num in range(N) if num not in [nodei,nodej]]
                    base_line_node_index = random.sample(filtered_numbers,SPnodenum)

                    for SPnode in base_line_node_index:
                        xMed = Coorx[SPnode]
                        yMed = Coory[SPnode]
                        dist, _ = dist_to_geodesic_R2(xMed, yMed, xSource, ySource, xEnd, yEnd)
                        baseline_deviations_for_a_nodepair.append(dist)
                    ave_baseline_deviation.append(np.mean(baseline_deviations_for_a_nodepair))
                    baseline_deviation_vec = baseline_deviation_vec + baseline_deviations_for_a_nodepair


        deviation_vec_name = filefolder_name+ "Givendistancedeviation_shortest_path_nodes_N{Nn}ED{EDn}beta{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
            Nn = N,EDn=ED, betan=beta, ST=ExternalSimutime, Geodistance = geodesic_distance_AB)
        np.savetxt(deviation_vec_name, deviation_vec)
        baseline_deviation_vec_name = filefolder_name+"Givendistancedeviation_baseline_nodes_num_N{Nn}ED{EDn}beta{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
            Nn = N,EDn=ED, betan=beta, ST=ExternalSimutime, Geodistance = geodesic_distance_AB)
        np.savetxt(baseline_deviation_vec_name, baseline_deviation_vec)
        # For each node pair:
        ave_deviation_name = filefolder_name+"Givendistanceave_deviation_N{Nn}ED{EDn}beta{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
            Nn = N,EDn=ED, betan=beta, ST=ExternalSimutime, Geodistance = geodesic_distance_AB)
        np.savetxt(ave_deviation_name, ave_deviation)
        max_deviation_name = filefolder_name+"Givendistancemax_deviation_N{Nn}ED{EDn}beta{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
            Nn = N,EDn=ED, betan=beta, ST=ExternalSimutime, Geodistance = geodesic_distance_AB)
        np.savetxt(max_deviation_name, max_deviation)
        min_deviation_name = filefolder_name+"Givendistancemin_deviation_N{Nn}ED{EDn}beta{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
            Nn = N,EDn=ED, betan=beta, ST=ExternalSimutime, Geodistance = geodesic_distance_AB)
        np.savetxt(min_deviation_name, min_deviation)
        ave_baseline_deviation_name = filefolder_name+"Givendistanceave_baseline_deviation_N{Nn}ED{EDn}beta{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
            Nn = N,EDn=ED, betan=beta, ST=ExternalSimutime, Geodistance = geodesic_distance_AB)
        np.savetxt(ave_baseline_deviation_name, ave_baseline_deviation)
        length_geodesic_name = filefolder_name+"Givendistancelength_geodesic_N{Nn}ED{EDn}beta{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
            Nn = N,EDn=ED, betan=beta, ST=ExternalSimutime, Geodistance = geodesic_distance_AB)
        np.savetxt(length_geodesic_name, length_geodesic)
        SPnodenum_vec_name = filefolder_name+"GivendistanceSPnodenum_N{Nn}ED{EDn}beta{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
            Nn = N,EDn=ED, betan=beta, ST=ExternalSimutime, Geodistance = geodesic_distance_AB)
        np.savetxt(SPnodenum_vec_name, SPnodenum_vec,fmt="%i")
        hopcount_Name = filefolder_name+"Givendistancehopcount_sp_N{Nn}ED{EDn}beta{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
            Nn = N,EDn=ED, betan=beta, ST=ExternalSimutime, Geodistance = geodesic_distance_AB)
        np.savetxt(hopcount_Name, hopcount_vec)


def distance_inSRGG_withavgbeta(network_size_index, average_degree_index, beta_index, Geodistance_index, ExternalSimutime):
    Nvec = [10, 100, 200, 500, 1000, 10000]
    # kvec = list(range(2, 20)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    kvec = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]  # log uniformly distributed k
    kvec = [49,56,64,81,92,104]
    kvec = [586,663,750,959,1085,1228]
    betavec = [2.2, 4, 8, 16, 32,128]

    # avg_vec = [2, 5, 10, 20, 50, 100]
    # beta_vec = [2.2, 2.4, 2.6, 2.8] + list(range(3, 16))

    # distance_list = [[0.49, 0.5, 0.5, 0.5], [0.25, 0.25, 0.3, 0.3], [0.25, 0.25, 0.5, 0.5], [0.25, 0.25, 0.75, 0.75]]
    # distance_list = [[0.49, 0.5, 0.5, 0.5], [0.25, 0.25, 0.75, 0.75]]
    distance_list = [[0.491, 0.5, 0.509, 0.5],[0.25, 0.5, 0.75, 0.5],[0.45, 0.5, 0.55, 0.5]]
    x_A = distance_list[Geodistance_index][0]
    y_A = distance_list[Geodistance_index][1]
    x_B = distance_list[Geodistance_index][2]
    y_B = distance_list[Geodistance_index][3]
    geodesic_distance_AB = x_B - x_A
    geodesic_distance_AB = round(geodesic_distance_AB,2)

    random.seed(ExternalSimutime)
    N = Nvec[network_size_index]


    ED = kvec[average_degree_index]
    beta = betavec[beta_index]
    print("input para:", (N, ED, beta,geodesic_distance_AB,ExternalSimutime))

    rg = RandomGenerator(-12)
    rseed = random.randint(0, 100)
    for i in range(rseed):
        rg.ran1()

    # for large network, we only generate one network and randomly selected 1,000 node pair.
    # for small network, we generate 100 networks and selected all the node pair in the LCC
    if N > 100:
        distance_inlargeSRGG_clu_beta_givennodepair(N, ED, beta, rg, ExternalSimutime, geodesic_distance_AB, x_A,
                                                    y_A, x_B, y_B)
    else:
        # Random select nodepair_num nodes in the largest connected component
        distance_insmallSRGG(N, ED, beta, rg, ExternalSimutime)
    print("ok")
    # Press the green button in the gutter to run the script.
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

    # """
    # test and run the code for different exact ED and clustering coefficient
    # """
    # # test the code
    # # distance_inSRGG_withEDCC(5, int(4), int(0), int(0), int(0))
    # # run the code
    # ED = sys.argv[1]
    # cc_index = sys.argv[2]
    # Geodistance_index = sys.argv[3]
    # ExternalSimutime = sys.argv[4]
    # distance_inSRGG_withEDCC(5, int(ED), int(cc_index), int(Geodistance_index), int(ExternalSimutime))


    """
    test and run the code for different input ED and beta
    """
    # test the code
    # distance_inSRGG_withavgbeta(5, int(4), int(0), int(0), int(0))
    # run the code
    ED = sys.argv[1]
    cc_index = sys.argv[2]
    Geodistance_index = sys.argv[3]
    ExternalSimutime = sys.argv[4]
    distance_inSRGG_withavgbeta(5, int(ED), int(cc_index), int(Geodistance_index), int(ExternalSimutime))









