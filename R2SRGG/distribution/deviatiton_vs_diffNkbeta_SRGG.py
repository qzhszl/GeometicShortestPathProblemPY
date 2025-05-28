# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 20-8-2024
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

from R2SRGG.R2SRGG import R2SRGG, distR2, dist_to_geodesic_R2, loadSRGGandaddnode, R2SRGG_withgivennodepair
from SphericalSoftRandomGeomtricGraph import RandomGenerator
from main import all_shortest_path_node, find_k_connected_node_pairs, find_all_connected_node_pairs, hopcount_node


def generate_r2SRGG():
    rg = RandomGenerator(-12)
    rseed = random.randint(0, 100)
    print(rseed)
    for i in range(rseed):
        rg.ran1()

    # Nvec = [200, 500, 1000, 10000]
    Nvec = [1000]
    # kvec = np.arange(2, 6.1, 0.2)
    kvec = [2, 4, 6, 11, 20, 34, 61, 108, 190, 336, 595, 1051, 1857, 3282, 5800, 10250, 18116, 32016, 56582,
            99999]  # for N = 10^5
    # kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    betavec = [2.2, 4, 8, 16, 32, 64, 128]

    # betavec = [2.2, 2.4, 2.5, 2.6, 2.8, 3, 3.25, 3.5, 3.75, 5, 6, 7]
    # betavec = [4]
    kvec = [1465694]  # for N = 10^4
    # kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]

    kvec = [1425, 2033, 2900, 4139, 5909, 8430, 12039, 17177, 24510, 34968, 49887, 71168]  # this is for N = 1000

    betavec = [4]

    for N in Nvec:
        for ED in kvec:
            ED =  round(ED,1)
            for beta in betavec:
                G, Coorx, Coory = R2SRGG(N, ED, beta, rg)
                real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
                print("input para:", (N,ED,beta))
                print("real ED:", real_avg)
                # print("clu:", nx.average_clustering(G))
                # components = list(nx.connected_components(G))
                # largest_component = max(components, key=len)
                # print("LCC", len(largest_component))

                FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\inputavgbeta\\network_N{Nn}ED{EDn}Beta{betan}.txt".format(
                    Nn=N, EDn=ED, betan=beta)
                nx.write_edgelist(G, FileNetworkName)

                FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\inputavgbeta\\network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt".format(
                    Nn=N, EDn=ED, betan=beta)
                with open(FileNetworkCoorName, "w") as file:
                    for data1, data2 in zip(Coorx, Coory):
                        file.write(f"{data1}\t{data2}\n")


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
        count_vec =[]
        # For each node pair:
        ave_deviation = []
        max_deviation = []
        min_deviation = []
        ave_baseline_deviation =[]
        length_geodesic = []
        SP_hopcount = []
        max_dev_node_hopcount = []
        min_dev_node_hopcount = []

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
                nodei = node_pair[0]
                nodej = node_pair[1]
                # Find the shortest path nodes
                SPNodelist = all_shortest_path_node(G, nodei, nodej)
                SPnodenum = len(SPNodelist)
                SPnodenum_vec.append(SPnodenum)

                if SPnodenum>0:
                    # hopcount of the SP
                    SP_hopcount_fornodepair = nx.shortest_path_length(G,nodei,nodej)
                    SP_hopcount.append(SP_hopcount_fornodepair)

                    xSource = Coorx[nodei]
                    ySource = Coory[nodei]
                    xEnd = Coorx[nodej]
                    yEnd = Coory[nodej]
                    length_geodesic.append(distR2(xSource, ySource, xEnd, yEnd))
                    # Compute deviation for the shortest path of each node pair
                    deviations_for_a_nodepair = []
                    hop_for_a_nodepair = []
                    for SPnode in SPNodelist:
                        xMed = Coorx[SPnode]
                        yMed = Coory[SPnode]
                        dist, _ = dist_to_geodesic_R2(xMed, yMed, xSource, ySource, xEnd, yEnd)
                        deviations_for_a_nodepair.append(dist)
                        # hop = hopcount_node(G, nodei, nodej, SPnode)
                        # hop_for_a_nodepair.append(hop)
                    ave_deviation.append(np.mean(deviations_for_a_nodepair))
                    max_deviation.append(max(deviations_for_a_nodepair))
                    min_deviation.append(min(deviations_for_a_nodepair))

                    # max hopcount
                    max_value = max(deviations_for_a_nodepair)
                    max_index = deviations_for_a_nodepair.index(max_value)
                    maxhop_node_index = SPNodelist[max_index]
                    max_dev_node_hopcount.append(nx.shortest_path_length(G, nodei, maxhop_node_index))

                    # min hopcount
                    min_value = min(deviations_for_a_nodepair)
                    min_index = deviations_for_a_nodepair.index(min_value)
                    minhop_node_index = SPNodelist[min_index]
                    min_dev_node_hopcount.append(nx.shortest_path_length(G, nodei, minhop_node_index))

                    count = count + 1

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
            count_vec.append(count)

        # real_ave_degree_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\real_ave_degree_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        #     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        # np.savetxt(real_ave_degree_name, real_ave_degree)
        # LCC_num_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\LCC_num_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        #     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        # np.savetxt(LCC_num_name, LCC_num, fmt="%i")
        # clustering_coefficient_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\clustering_coefficient_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        #     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        # np.savetxt(clustering_coefficient_name, clustering_coefficient)
        # # For each node pair:
        # ave_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        #     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        # np.savetxt(ave_deviation_name, ave_deviation)
        # max_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\max_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        #     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        # np.savetxt(max_deviation_name, max_deviation)
        # min_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\min_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        #     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        # np.savetxt(min_deviation_name, min_deviation)
        # ave_baseline_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\ave_baseline_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        #     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        # np.savetxt(ave_baseline_deviation_name, ave_baseline_deviation)
        # length_geodesic_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\length_geodesic_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        #     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        # np.savetxt(length_geodesic_name, length_geodesic)
        # SPnodenum_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\SPnodenum_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        #     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        # np.savetxt(SPnodenum_vec_name, SPnodenum_vec, fmt="%i")
        # nodepairs_for_eachgraph_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\nodepairs_for_eachgraph_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
        # np.savetxt(nodepairs_for_eachgraph_vec_name, count_vec, fmt="%i")

        SPhopcount_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\formaxhop\\distancetosinglenode\\SPhopcount_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(SPhopcount_name, SP_hopcount, fmt="%i")

        max_dev_node_hopcount_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\formaxhop\\distancetosinglenode\\max_dev_node_hopcount_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(max_dev_node_hopcount_name, max_dev_node_hopcount, fmt="%i")

        SPhopcount_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\formaxhop\\distancetosinglenode\\min_SPhopcount_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(SPhopcount_name, SP_hopcount, fmt="%i")

        min_dev_node_hopcount_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\formaxhop\\distancetosinglenode\\min_dev_node_hopcount_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(min_dev_node_hopcount_name, min_dev_node_hopcount, fmt="%i")


def distance_insmallSRGG_local(N, ED, beta, rg):
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

    # For graph:
    real_ave_degree = []
    LCC_num = []
    SLCC_num = []
    clustering_coefficient = []
    count_vec =[]
    # For each node pair:
    ave_deviation = []
    max_deviation = []
    min_deviation = []
    ave_baseline_deviation =[]
    length_geodesic = []
    SP_hopcount = []
    max_dev_node_hopcount = []
    SPnodenum_vec =[]
    simu_times = 100
    for simu_index in range(simu_times):
        G, Coorx, Coory = R2SRGG(N, ED, beta, rg)
        real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)

        # print("real ED:", real_avg)
        real_ave_degree.append(real_avg)

        # LCC and the second LCC
        connected_components = sorted(nx.connected_components(G), key=len, reverse=True)
        if len(connected_components) > 1:
            largest_largest_component = connected_components[0]
            largest_largest_size = len(largest_largest_component)
            LCC_num.append(largest_largest_size)
            # ?????????????????
            second_largest_component = connected_components[1]
            second_largest_size = len(second_largest_component)
            SLCC_num.append(second_largest_size)


        # pick up all the node pairs in the LCC and save them in the unique_pairs
        unique_pairs = find_all_connected_node_pairs(G)
        count = 0
        for node_pair in unique_pairs:
            nodei = node_pair[0]
            nodej = node_pair[1]
            # Find the shortest path nodes
            SPNodelist = all_shortest_path_node(G, nodei, nodej)
            SPnodenum = len(SPNodelist)
            SPnodenum_vec.append(SPnodenum)

            if SPnodenum>0:
                # hopcount of the SP
                SP_hopcount_fornodepair = nx.shortest_path_length(G,nodei,nodej)
                SP_hopcount.append(SP_hopcount_fornodepair)

                xSource = Coorx[nodei]
                ySource = Coory[nodei]
                xEnd = Coorx[nodej]
                yEnd = Coory[nodej]
                length_geodesic.append(distR2(xSource, ySource, xEnd, yEnd))
                # Compute deviation for the shortest path of each node pair
                deviations_for_a_nodepair = []
                hop_for_a_nodepair = []
                for SPnode in SPNodelist:
                    xMed = Coorx[SPnode]
                    yMed = Coory[SPnode]
                    dist, _ = dist_to_geodesic_R2(xMed, yMed, xSource, ySource, xEnd, yEnd)
                    deviations_for_a_nodepair.append(dist)
                    # hop = hopcount_node(G, nodei, nodej, SPnode)
                    # hop_for_a_nodepair.append(hop)
                ave_deviation.append(np.mean(deviations_for_a_nodepair))
                max_deviation.append(max(deviations_for_a_nodepair))
                min_deviation.append(min(deviations_for_a_nodepair))

                max_value = max(deviations_for_a_nodepair)
                max_index = deviations_for_a_nodepair.index(max_value)
                maxhop_node_index = SPNodelist[max_index]
                max_dev_node_hopcount.append(hopcount_node(G, nodei, nodej, maxhop_node_index))

                count = count + 1

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
        count_vec.append(count)
    print(np.mean(real_ave_degree))
    real_ave_degree_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\real_ave_degree_N{Nn}ED{EDn}Beta{betan}.txt".format(
        Nn = N, EDn=ED, betan=beta)
    np.savetxt(real_ave_degree_name, real_ave_degree)
    # LCC_num_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\LCC_num_N{Nn}ED{EDn}Beta{betan}.txt".format(
    #     Nn = N, EDn=ED, betan=beta)
    # np.savetxt(LCC_num_name, LCC_num, fmt="%i")
    clustering_coefficient_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\clustering_coefficient_N{Nn}ED{EDn}Beta{betan}.txt".format(
        Nn = N, EDn=ED, betan=beta)
    # np.savetxt(clustering_coefficient_name, clustering_coefficient)
    # For each node pair:
    ave_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\ave_deviation_N{Nn}ED{EDn}Beta{betan}.txt".format(
        Nn = N, EDn=ED, betan=beta)
    np.savetxt(ave_deviation_name, ave_deviation)
    max_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\max_deviation_N{Nn}ED{EDn}Beta{betan}.txt".format(
        Nn = N, EDn=ED, betan=beta)
    # np.savetxt(max_deviation_name, max_deviation)
    min_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\min_deviation_N{Nn}ED{EDn}Beta{betan}.txt".format(
        Nn = N, EDn=ED, betan=beta)
    # np.savetxt(min_deviation_name, min_deviation)
    ave_baseline_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\ave_baseline_deviation_N{Nn}ED{EDn}Beta{betan}.txt".format(
        Nn = N, EDn=ED, betan=beta)
    # np.savetxt(ave_baseline_deviation_name, ave_baseline_deviation)
    length_geodesic_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\length_geodesic_N{Nn}ED{EDn}Beta{betan}.txt".format(
        Nn = N, EDn=ED, betan=beta)
    np.savetxt(length_geodesic_name, length_geodesic)
    SPnodenum_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\SPnodenum_N{Nn}ED{EDn}Beta{betan}.txt".format(
        Nn = N, EDn=ED, betan=beta)
    # np.savetxt(SPnodenum_vec_name, SPnodenum_vec, fmt="%i")
    nodepairs_for_eachgraph_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\nodepairs_for_eachgraph_N{Nn}ED{EDn}Beta{betan}.txt".format(
        Nn=N, EDn=ED, betan=beta)
    np.savetxt(nodepairs_for_eachgraph_vec_name, count_vec, fmt="%i")

    SPhopcount_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\SPhopcount_N{Nn}ED{EDn}Beta{betan}.txt".format(
        Nn=N, EDn=ED, betan=beta)
    np.savetxt(SPhopcount_name, SP_hopcount, fmt="%i")
    # max_dev_node_hopcount_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\max_dev_node_hopcount_N{Nn}ED{EDn}Beta{betan}.txt".format(
    #     Nn=N, EDn=ED, betan=beta)
    # np.savetxt(max_dev_node_hopcount_name, max_dev_node_hopcount, fmt="%i")

    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\"
    LCCname = filefolder_name + "LCC_2LCC_N{Nn}ED{EDn}beta{betan}.txt".format(
        Nn=N, EDn=ED, betan=beta)
    with open(LCCname, "w") as file:
        file.write("# LCC\tSECLCC\n")
        for name, age in zip(LCC_num, SLCC_num):
            file.write(f"{name}\t{age}\n")


def distance_inlargeSRGG(N,ED,beta,rg, ExternalSimutime):
    """
    :param N:
    :param ED:
    :param beta:
    :param rg:
    :param ExternalSimutime:
    :return:
    for each node pair, we record the ave,max,min of distance from the shortest path to the geodesic,
    length of the geo distances.
    The generated network, the selected node pair and all the deviation of both shortest path and baseline nodes will be recorded.
    """
    deviation_vec = []  # deviation of all shortest path nodes for all node pairs
    baseline_deviation_vec = []  # deviation of all shortest path nodes for all node pairs
    # For each node pair:
    ave_deviation = []
    max_deviation = []
    min_deviation = []
    ave_baseline_deviation = []
    length_geodesic = []
    hopcount_vec = []
    SPnodenum_vec = []
    LCC_vec = []
    second_vec = []
    max_dev_node_hopcount = []
    min_dev_node_hopcount = []

    # load a network
    # try:
    #     FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\inputavgbeta\\network_N{Nn}ED{EDn}Beta{betan}.txt".format(
    #         Nn=N, EDn=ED, betan=beta)
    #     G = loadSRGGandaddnode(N, FileNetworkName)
    #     # load coordinates with noise
    #     Coorx = []
    #     Coory = []
    #
    #     FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\inputavgbeta\\network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt".format(
    #         Nn=N, EDn=ED, betan=beta)
    #     with open(FileNetworkCoorName, "r") as file:
    #         for line in file:
    #             if line.startswith("#"):
    #                 continue
    #             data = line.strip().split("\t")  # 使用制表符分割
    #             Coorx.append(float(data[0]))
    #             Coory.append(float(data[1]))
    # except:
    #     print("no network")
    #     G, Coorx, Coory = R2SRGG(N, ED, beta, rg)
        # FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\inputavgbeta\\network_N{Nn}ED{EDn}Beta{betan}.txt".format(
        #     Nn=N, EDn=ED, betan=beta)
        # nx.write_edgelist(G, FileNetworkName)
        # FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\inputavgbeta\\network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt".format(
        #     Nn=N, EDn=ED, betan=beta)
        # with open(FileNetworkCoorName, "w") as file:
        #     for data1, data2 in zip(Coorx, Coory):
        #         file.write(f"{data1}\t{data2}\n")
    G, Coorx, Coory = R2SRGG(N, ED, beta, rg)

    real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
    print("real ED:", real_avg)

    # ave_clu = nx.average_clustering(G)
    # print("clu:", ave_clu)

    components = list(nx.connected_components(G))
    largest_component = max(components, key=len)
    LCC_number = len(largest_component)
    print("LCC", LCC_number)

    # Randomly choose 100 connectede node pairs
    nodepair_num = 100000
    unique_pairs = find_k_connected_node_pairs(G, nodepair_num)
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\formaxhop\\distancetosinglenode\\"
    # filename_selecetednodepair = filefolder_name+"selected_node_pair_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
    #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    # np.savetxt(filename_selecetednodepair, unique_pairs, fmt="%i")

    # connected_components = sorted(nx.connected_components(G), key=len, reverse=True)
    # if len(connected_components) > 1:
    #     largest_largest_component = connected_components[0]
    #     largest_largest_size = len(largest_largest_component)
    #     LCC_vec.append(largest_largest_size)
    #     # 获取第二大连通分量的节点集合和大小
    #     second_largest_component = connected_components[1]
    #     second_largest_size = len(second_largest_component)
    #     second_vec.append(second_largest_size)
    # if ExternalSimutime == 0:
    #     LCCname = filefolder_name + "LCC_2LCC_N{Nn}ED{EDn}beta{betan}.txt".format(
    #         Nn=N, EDn=ED, betan=beta)
    #     # with open(LCCname, "w") as file:
    #     #     file.write("# LCC\tSECLCC\n")  # 使用制表符分隔列
    #     #     # 写入数据
    #     #     for name, age in zip(LCC_vec, second_vec):
    #     #         file.write(f"{name}\t{age}\n")

    for node_pair in unique_pairs:
        print("node_pair:", node_pair)
        nodei = node_pair[0]
        nodej = node_pair[1]
        # Find the shortest path nodes
        SPNodelist = all_shortest_path_node(G, nodei, nodej)
        hopcount_vec.append(nx.shortest_path_length(G, nodei, nodej))
        SPnodenum = len(SPNodelist)
        SPnodenum_vec.append(SPnodenum)
        if SPnodenum > 0:
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

            deviation_vec = deviation_vec + deviations_for_a_nodepair

            ave_deviation.append(np.mean(deviations_for_a_nodepair))
            # max_deviation.append(max(deviations_for_a_nodepair))
            # min_deviation.append(min(deviations_for_a_nodepair))

            # max hopcount
            max_value = max(deviations_for_a_nodepair)
            max_index = deviations_for_a_nodepair.index(max_value)
            maxhop_node_index = SPNodelist[max_index]
            max_dev_node_hopcount.append(nx.shortest_path_length(G, nodei, maxhop_node_index))

            # min hopcount
            min_value = min(deviations_for_a_nodepair)
            min_index = deviations_for_a_nodepair.index(min_value)
            minhop_node_index = SPNodelist[min_index]
            min_dev_node_hopcount.append(nx.shortest_path_length(G, nodei, minhop_node_index))


            # baseline_deviations_for_a_nodepair = []
            # # compute baseline's deviation
            # filtered_numbers = [num for num in range(N) if num not in [nodei, nodej]]
            # base_line_node_index = random.sample(filtered_numbers, SPnodenum)
            #
            # for SPnode in base_line_node_index:
            #     xMed = Coorx[SPnode]
            #     yMed = Coory[SPnode]
            #     dist, _ = dist_to_geodesic_R2(xMed, yMed, xSource, ySource, xEnd, yEnd)
            #     baseline_deviations_for_a_nodepair.append(dist)
            # ave_baseline_deviation.append(np.mean(baseline_deviations_for_a_nodepair))
            # baseline_deviation_vec = baseline_deviation_vec + baseline_deviations_for_a_nodepair

    deviation_vec_name = filefolder_name+"deviation_shortest_path_nodes_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(deviation_vec_name, deviation_vec)
    # baseline_deviation_vec_name = filefolder_name+"deviation_baseline_nodes_num_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
    #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    # np.savetxt(baseline_deviation_vec_name, baseline_deviation_vec)
    # For each node pair:
    ave_deviation_name = filefolder_name+"ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(ave_deviation_name, ave_deviation)
    # max_deviation_name = filefolder_name+"max_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
    #     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
    # np.savetxt(max_deviation_name, max_deviation)
    # min_deviation_name = filefolder_name+"min_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
    #     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
    # np.savetxt(min_deviation_name, min_deviation)
    # ave_baseline_deviation_name = filefolder_name+"ave_baseline_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
    #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    # np.savetxt(ave_baseline_deviation_name, ave_baseline_deviation)
    length_geodesic_name = filefolder_name+"length_geodesic_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(length_geodesic_name, length_geodesic)
    SPnodenum_vec_name = filefolder_name+"SPnodenum_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(SPnodenum_vec_name, SPnodenum_vec, fmt="%i")
    hopcount_Name = filefolder_name+"min_hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(hopcount_Name, hopcount_vec)
    hopcount_Name = filefolder_name + "hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(hopcount_Name, hopcount_vec)
    max_dev_node_hopcount_name = filefolder_name+"max_dev_node_hopcount_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(max_dev_node_hopcount_name, max_dev_node_hopcount, fmt="%i")
    min_dev_node_hopcount_name = filefolder_name+"min_dev_node_hopcount_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(min_dev_node_hopcount_name, min_dev_node_hopcount, fmt="%i")




def distance_inSRGG(network_size_index, average_degree_index, beta_index, ExternalSimutime):
    Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    # kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    # betavec = [2.1, 4, 8, 16, 32, 64, 128]
    # betavec = [2.2, 2.4, 2.5, 2.6, 2.8, 3, 3.25, 3.5, 3.75, 5, 6, 7, 10, 12]
    # kvec = [1425, 2033, 2900, 4139, 5909, 8430, 12039, 17177, 24510, 34968,49887, 71168]
    kvec = [5]
    betavec = [4]


    random.seed(ExternalSimutime)
    N = Nvec[network_size_index]
    ED = kvec[average_degree_index]
    beta = betavec[beta_index]
    print("input para:", (N, ED, beta))

    rg = RandomGenerator(-12)
    rseed = random.randint(0, 100)
    for i in range(rseed):
        rg.ran1()

    # for large network, we only generate one network and randomly selected 1,000 node pair.
    # for small network, we generate 100 networks and selected all the node pair in the LCC

    if N>100:
        distance_inlargeSRGG(N, ED, beta, rg, ExternalSimutime)
    else:
        # Random select nodepair_num nodes in the largest connected component
        distance_insmallSRGG(N, ED, beta, rg, ExternalSimutime)

# def distance_inSRGG_realED(network_size_index, average_degree_index, beta_index, ExternalSimutime):
#     Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
#     kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
#     betavec = [2.1, 4, 8, 16, 32, 64, 128]
#
#     random.seed(ExternalSimutime)
#     N = Nvec[network_size_index]
#     ED = kvec[average_degree_index]
#     beta = betavec[beta_index]
#     print("input para:", (N, ED, beta))
#
#     rg = RandomGenerator(-12)
#     rseed = random.randint(0, 100)
#     for i in range(rseed):
#         rg.ran1()
#
#     # for large network, we only generate one network and randomly selected 1,000 node pair.
#     # for small network, we generate 100 networks and selected all the node pair in the LCC
#     if N > 100:
#         distance_inlargeSRGG(N, ED, beta, ExternalSimutime)
#     else:
#         # Random select nodepair_num nodes in the largest connected component
#         distance_insmallSRGG(N, ED, beta, rg, ExternalSimutime)

def shortest_path_length_hopcount_fornetwork(N,beta):
    # avg
    kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    for beta in [beta]:
        for ED in kvec:
            hopcount_for_one_graph = []
            nodenum_for_one_graph = []
            FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\network_N{Nn}ED{EDn}Beta{betan}.txt".format(
                Nn=N, EDn=ED, betan=beta)
            G = loadSRGGandaddnode(N, FileNetworkName)
            real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
            clustering_coefficient = nx.average_clustering(G)
            for ExternalSimutime in range(10):
                SPnodenum_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\SPnodenum_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                SPnodenum_vec = np.loadtxt(SPnodenum_vec_name, dtype=int)
                nodenum_for_one_graph = nodenum_for_one_graph+list(SPnodenum_vec)
                filename_selecetednodepair = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\selected_node_pair_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                nodepair = np.loadtxt(filename_selecetednodepair, dtype=int)
                for [nodei,nodej] in nodepair:
                    hopcount_for_one_graph.append(nx.shortest_path_length(G, source=nodei, target=nodej))

            hopcount_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\hopcount_sp_ED{EDn}Beta{betan}.txt".format(
                EDn=ED, betan=beta)
            np.savetxt(hopcount_Name, hopcount_for_one_graph)
            spnodenum_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\SPnodenum_ED{EDn}Beta{betan}.txt".format(
                EDn=ED, betan=beta)
            np.savetxt(spnodenum_Name, nodenum_for_one_graph)


def generate_proper_network(N, input_ED, beta_index):
    # for 10000 node
    betavec = [2.55, 3.2, 3.99, 5.15, 7.99, 300]
    beta = betavec[beta_index]
    cc = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    C_G = cc[beta_index]

    kvec = list(range(6, 20)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    ED_bound = input_ED*0.05

    min_ED = 1
    max_ED = N-1
    count = 0
    rg = RandomGenerator(-12)
    G, Coorx, Coory = R2SRGG(N, input_ED, beta, rg)
    real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
    ED= input_ED
    while abs(input_ED-real_avg)>ED_bound and count < 20:
        count = count + 1
        if input_ED-real_avg>0:
            min_ED = ED
            ED = min_ED+0.5*(max_ED-min_ED)
        else:
            max_ED = ED
            ED = min_ED + 0.5 * (max_ED - min_ED)
        G, Coorx, Coory = R2SRGG(N, ED, beta, rg)
        real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)

    print("input para:", (N, input_ED, beta))
    print("real ED:", real_avg)
    print("clu:", nx.average_clustering(G))
    components = list(nx.connected_components(G))
    largest_component = max(components, key=len)
    print("LCC", len(largest_component))
    if count < 20:
        FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\cleanwithEDCC\\network_N{Nn}ED{EDn}CC{betan}.txt".format(
            Nn=N, EDn=input_ED, betan=C_G)
        nx.write_edgelist(G, FileNetworkName)

        FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\cleanwithEDCC\\network_coordinates_N{Nn}ED{EDn}CC{betan}.txt".format(
            Nn=N, EDn=input_ED, betan=C_G)
        with open(FileNetworkCoorName, "w") as file:
            for data1, data2 in zip(Coorx, Coory):
                file.write(f"{data1}\t{data2}\n")
    return ED


def generate_proper_network_withgivendistances(N, input_ED_index,beta_index,Geodistance_index):
    """
    fix distance: A,B (0.25,0.25) (0.75,0.75)
    (0.25,0.25) (0.5,0.5)
    (0.25,0.25) (0.3,0.3)
    :return:
    """
    distance_list = [[0.25, 0.25, 0.3, 0.3], [0.25, 0.25, 0.5, 0.5], [0.25, 0.25, 0.75, 0.75]]
    x_A = distance_list[Geodistance_index][0]
    y_A = distance_list[Geodistance_index][1]
    x_B = distance_list[Geodistance_index][2]
    y_B = distance_list[Geodistance_index][3]
    geodesic_distance_AB = x_B-x_A
    # BETA for 10000 node
    betavec = [2.55, 3.2, 3.99, 5.15, 7.99, 300]
    beta = betavec[beta_index]
    cc = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    C_G = cc[beta_index]

    kvec = list(range(2, 20)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    input_ED = kvec[input_ED_index]
    ED_bound = input_ED * 0.05

    min_ED = 1
    max_ED = N - 1
    count = 0
    rg = RandomGenerator(-12)
    G, Coorx, Coory = R2SRGG_withgivennodepair(N, input_ED, beta, rg,x_A,y_A,x_B,y_B)
    real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
    ED = input_ED
    while abs(input_ED - real_avg) > ED_bound and count < 20:
        count = count + 1
        if input_ED - real_avg > 0:
            min_ED = ED
            ED = min_ED + 0.5 * (max_ED - min_ED)
        else:
            max_ED = ED
            ED = min_ED + 0.5 * (max_ED - min_ED)
            pass
        G, Coorx, Coory = R2SRGG_withgivennodepair(N, ED, beta, rg, x_A, y_A, x_B, y_B)
        real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)

    print("input para:", (N, input_ED, beta))
    print("real ED:", real_avg)
    print("clu:", nx.average_clustering(G))
    components = list(nx.connected_components(G))
    largest_component = max(components, key=len)
    print("LCC", len(largest_component))
    FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\cleanwithEDCC\\Givendistancenetwork_N{Nn}ED{EDn}CC{betan}Geodistance{Geodistance}.txt".format(
        Nn=N, EDn=input_ED, betan=C_G, Geodistance = geodesic_distance_AB)
    nx.write_edgelist(G, FileNetworkName)

    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\cleanwithEDCC\\Givendistancenetwork_coordinates_N{Nn}ED{EDn}CC{betan}Geodistance{Geodistance}.txt".format(
        Nn=N, EDn=input_ED, betan=C_G, Geodistance = geodesic_distance_AB)
    with open(FileNetworkCoorName, "w") as file:
        for data1, data2 in zip(Coorx, Coory):
            file.write(f"{data1}\t{data2}\n")


    # Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # generate_r2SRGG()
    """
    run simulations for large networks(N = 1000, N>10000 will be put on the cluster)
    step1
    """
    # network_size_index = 2
    # average_degree_index = 7
    # beta_index = 1
    # external_simu_time = 0
    # distance_inSRGG(network_size_index, average_degree_index, beta_index, external_simu_time)
    """
    run simulations for large networks(N = 1000, N>10000 will be put on the cluster)
    step1
    """
    # kvec = [4,16,64]
    kvec = [5]
    # kvec = [1425, 2033, 2900, 4139, 5909, 8430, 12039, 17177, 24510, 34968,49887, 71168]
    betavec = [4]
    for N_index in [7]:
        for ED_index in range(len(kvec)):
            for beta_index in range(1):
                distance_inSRGG(N_index, ED_index, beta_index, 0)



    # rg = RandomGenerator(-12)
    # rseed = random.randint(0, 100)
    # for i in range(rseed):
    #     rg.ran1()
    # distance_inlargeSRGG(1000, 5, 4, rg, 0)

    """
    run simulations for small networks(N = 100)
    step1
    """
    # rg = RandomGenerator(-12)
    # kvec = [10, 12, 15, 18, 22, 27, 33, 40, 49, 60, 73, 89, 99]

    # kvec = [2, 3, 4, 5, 6, 9, 11, 15, 20, 27, 37, 49, 65, 87, 117, 156, 209, 279, 373, 499]
    # N = 500

    # kvec = np.arange(1, 2.3, 0.1)
    # kvec = [round(a, 1) for a in kvec]
    # N = 50
    # # kvec =kvec +[9, 11, 15, 20, 27, 37, 49]
    # betavec = [4]
    # for ED in kvec:
    #     for beta in betavec:
    #         print(ED, beta)
    #         distance_insmallSRGG_local(N, ED, beta, rg)


    # N = 100
    # kvec = [113, 149, 198, 260, 340, 446, 584, 762, 993, 1292, 1690, 2276, 3142, 4339]
    # betavec = [4, 8, 16, 32, 64, 128,2.2]
    # for ED in kvec:
    #     for beta in betavec:
    #         print(ED,beta)
    #         distance_insmallSRGG_local(N, ED, beta, rg)

    # ED = 9
    # beta = 4

    # for N_index in [0,3]:
    #     for ED_index in range(24):
    #         for beta_index in range(14):
    #             distance_inSRGG(N_index, ED_index, beta_index, 0)


    """
    ## generate_proper_network(N, ED)
    """
    # EDdic = {}
    # C_G_vec = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    # betavec = [2.55, 3.2, 3.99, 5.15, 7.99, 300]
    # kvec = list(range(6, 20)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    # print(len(kvec))
    # count = 0
    # for beta in range(len(betavec)):
    #     count = count+1
    #     ED_VEC = []
    #     for ED_input in kvec:
    #         real_input_ED = generate_proper_network(10000, ED_input, beta)
    #         ED_VEC.append(real_input_ED)
    #     print(ED_VEC)
    #     EDdic[C_G_vec[count]] = ED_VEC
    # print(EDdic)


    # ED = sys.argv[1]
    # cc_index = sys.argv[2]
    # generate_proper_network(10000, int(ED), int(cc_index))
    # N = 10000
    # exceptionlist =[]
    # for C_G in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    #     for input_ED in list(range(2, 20)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]:
    #         try:
    #             FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\cleanwithEDCC\\network_coordinates_N{Nn}ED{EDn}CC{betan}.txt".format(
    #                 Nn=N, EDn=input_ED, betan=C_G)
    #             a = np.loadtxt(FileNetworkCoorName)
    #         except:
    #             exceptionlist.append((C_G,input_ED))
    # print(exceptionlist)

    # generate ER graph with fixed distances
    # step 1:
    # kvec = list(range(2, 20)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    # betavec = [2.55, 3.2, 3.99, 5.15, 7.99, 300]
    # kvec = list(range(2, 20)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]

    # for beta in betavec:
    #     for ED_input in [10]:
    #         rg = RandomGenerator(-12)
    #         G, xx, yy = R2SRGG_withgivennodepair(10000, ED_input, beta, rg, 0.25, 0.25, 0.3, 0.3, Coorx=None, Coory=None, SaveNetworkPath=None)
    #         real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
    #         print("real ED:", real_avg)
    #         ave_clu = nx.average_clustering(G)
    #         print("clu:", ave_clu)
    #         components = list(nx.connected_components(G))
    #         largest_component = max(components, key=len)
    #         LCC_number = len(largest_component)
    #         print("LCC", LCC_number)

    # step 2:
    # generate_proper_network_withgivendistances(100, 0, 0, 0)


    """
    generate proper cc and ED network
    """
    # betavec = [2.55, 3.2, 3.99, 5.15, 7.99, 300]
    # conclusion at this stage is that the data after degree> 6 is clean, but not before
    # N = 10000
    # input_ED = 2
    # C_G = 0.3
    # rg = RandomGenerator(-12)
    # ED = 2.6
    # beta = 6
    # G, Coorx, Coory = R2SRGG(N, ED, beta, rg)
    # real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
    # print("input para:", (N, ED, beta))
    # print("real ED:", real_avg)
    # print("clu:", nx.average_clustering(G))
    # components = list(nx.connected_components(G))
    # largest_component = max(components, key=len)
    # print("LCC", len(largest_component))
    #
    # FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\cleanwithEDCC\\smallED\\network_N{Nn}ED{EDn}CC{betan}.txt".format(
    #     Nn=N, EDn=input_ED, betan=C_G)
    # nx.write_edgelist(G, FileNetworkName)
    #
    # FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\cleanwithEDCC\\smallED\\network_coordinates_N{Nn}ED{EDn}CC{betan}.txt".format(
    #     Nn=N, EDn=input_ED, betan=C_G)
    # with open(FileNetworkCoorName, "w") as file:
    #     for data1, data2 in zip(Coorx, Coory):
    #         file.write(f"{data1}\t{data2}\n")

    """
    generate proper cc and ED network for fixed distance node pair
    """
    # cc = 0.2
    # distance_list = [[0.25, 0.25, 0.3, 0.3], [0.25, 0.25, 0.5, 0.5], [0.25, 0.25, 0.75, 0.75]]
    # betavec = [2.55]
    # cc_vec = [0.1]
    # ED_vec = [7.5]
    # N = 10000
    # rg = RandomGenerator(-12)
    #
    # for Geodistance_index in range(3):
    #     x_A = distance_list[Geodistance_index][0]
    #     y_A = distance_list[Geodistance_index][1]
    #     x_B = distance_list[Geodistance_index][2]
    #     y_B = distance_list[Geodistance_index][3]
    #     geodesic_distance_AB = x_B - x_A
    #     for simuindex in range(4):
    #         ED = ED_vec[simuindex]
    #         beta = betavec[simuindex]
    #         G, Coorx, Coory = R2SRGG_withgivennodepair(N, ED, beta, rg, x_A, y_A, x_B, y_B)
    #         real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
    #         print("input para:", (N, ED, beta,geodesic_distance_AB))
    #         print("real ED:", real_avg)
    #         print("clu:", nx.average_clustering(G))
    #         components = list(nx.connected_components(G))
    #         largest_component = max(components, key=len)
    #         print("LCC", len(largest_component))
    #
    #         FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\cleanwithEDCC\\GivenDistance\\Givendistancenetwork_N{Nn}ED{EDn}CC{betan}Geodistance{Geodistance}.txt".format(
    #             Nn=N, EDn=ED, betan=cc, Geodistance = geodesic_distance_AB)
    #         nx.write_edgelist(G, FileNetworkName)
    #
    #         FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\cleanwithEDCC\\GivenDistance\\Givendistancenetwork_coordinates_N{Nn}ED{EDn}CC{betan}Geodistance{Geodistance}.txt".format(
    #             Nn=N, EDn=ED, betan=cc, Geodistance = geodesic_distance_AB)
    #         with open(FileNetworkCoorName, "w") as file:
    #             for data1, data2 in zip(Coorx, Coory):
    #                 file.write(f"{data1}\t{data2}\n")
    # rg = RandomGenerator(-12)
    # rseed = random.randint(0, 100)
    # for i in range(rseed):
    #     rg.ran1()
    # N = 10000
    # ED = 5
    # beta = 2.2
    # x_A = 0.495
    # y_A = 0.5
    # x_B = 0.505
    # y_B = 0.5
    # ExternalSimutime = 0
    # network_index = 0
    # filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\GivenGeodistance\\"
    # # Randomly generate 10 networks
    #
    # G, Coorx, Coory = R2SRGG_withgivennodepair(N, ED, beta, rg, x_A, y_A, x_B, y_B)
    # FileNetworkName = filefolder_name + "network_N{Nn}ED{EDn}beta{betan}xA{xA}yA{yA}xB{xB}yB{yB}Simu{simu}networktime{nt}.txt".format(
    #     Nn=N, EDn=ED, betan=beta, xA=x_A, yA=y_A, xB=x_B, yB=y_B, simu=ExternalSimutime, nt=network_index)
    # nx.write_edgelist(G, FileNetworkName)
    #
    # FileNetworkCoorName = filefolder_name + "network_coordinates_N{Nn}ED{EDn}beta{betan}xA{xA}yA{yA}xB{xB}yB{yB}Simu{simu}networktime{nt}.txt".format(
    #     Nn=N, EDn=ED, betan=beta, xA=x_A, yA=y_A, xB=x_B, yB=y_B, simu=ExternalSimutime, nt=network_index)
    # with open(FileNetworkCoorName, "w") as file:
    #     for data1, data2 in zip(Coorx, Coory):
    #         file.write(f"{data1}\t{data2}\n")

    """
    test a network
    """
    # rg = RandomGenerator(-12)
    # rseed = random.randint(0, 100)
    # for i in range(rseed):
    #     rg.ran1()
    # G, Coorx, Coory = R2SRGG(10000, 10, 4, rg)
    # real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
    # print("real ED:", real_avg)