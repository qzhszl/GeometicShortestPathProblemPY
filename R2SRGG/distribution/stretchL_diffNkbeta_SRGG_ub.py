# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 15-9-2025

For large network, we only generate multiple graphs and randomly selected 1000 node pairs.
The generated network, the selected node pair and all the deviation of both shortest path and baseline nodes will be recorded.

This script is collecting data for investigating
1. what we need: ave link length; hopcount, ave link lengths on shortest path
2. the relation between the length of the shortest path L = <d_e><h> versus degree
"""
import itertools

import numpy as np
import networkx as nx
import random
import math
import sys
import os
import shutil
import multiprocessing as mp

from R2SRGG.R2SRGG import R2SRGG, distR2, dist_to_geodesic_R2, loadSRGGandaddnode, R2SRGG_withlinkweight
from SphericalSoftRandomGeomtricGraph import RandomGenerator
from main import find_k_connected_node_pairs, find_all_connected_node_pairs, hopcount_node


def compute_edge_Euclidean_length(nodes, nodet, Coorx, Coory):
    xSource = Coorx[nodes]
    ySource = Coory[nodes]
    xEnd = Coorx[nodet]
    yEnd = Coory[nodet]
    edge_length = distR2(xSource, ySource, xEnd, yEnd)
    return edge_length


def stretch_in_multiple_largenetworks_oneSP_clu(N, ED, beta, rg, ExternalSimutime):
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
    # LCC_num = []
    # clustering_coefficient = []
    count_vec = []
    ave_graphlinklength_vec = []
    # std_length_edge_vec = []

    # For each node pair:
    ave_deviation = []
    max_deviation = []
    min_deviation = []
    ave_baseline_deviation = []
    length_geodesic = []
    SP_hopcount = []
    max_dev_node_hopcount = []
    SPnodenum_vec = []
    ave_edge_length = []
    length_edge_vec = []

    folder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\localmin_hunter\\"
    # folder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\1hopdiff\\"

    # folder_name = "/home/qzh/data/"

    simu_times = 100
    for simu_index in range(simu_times):
        G, linkweight_vec, Coorx, Coory = R2SRGG_withlinkweight(N, ED, beta, rg)
        real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)

        print("real ED:", real_avg)
        real_ave_degree.append(real_avg)

        # compute the edge length of the networks
        ave_graphlinklength_vec.append(np.mean(linkweight_vec))

        # ave_clu = nx.average_clustering(G)
        # print("clu:", ave_clu)
        # clustering_coefficient.append(ave_clu)
        # components = list(nx.connected_components(G))
        # largest_component = max(components, key=len)
        # LCC_number = len(largest_component)
        # print("LCC", LCC_number)
        # LCC_num.append(LCC_number)

        # pick up all the node pairs in the LCC and save them in the unique_pairs
        nodepair_num = 1000
        unique_pairs = find_k_connected_node_pairs(G, nodepair_num)
        count = 0

        for node_pair in unique_pairs:
            nodei = node_pair[0]
            nodej = node_pair[1]
            # Find the shortest path nodes
            SPNodelist = nx.shortest_path(G, nodei, nodej)
            SPnodenum = len(SPNodelist) - 2
            # SPnodenum_vec.append(SPnodenum)

            # hopcount of the SP
            SP_hopcount.append(SPnodenum + 1)

            if SPnodenum + 1 > 0:  # for deviation, we restrict ourself for hopcount>2: SPnodenum+1 > 1, but not for the stretch(hop =1 are also included:SPnodenum+1 > 0)
                # compute the length of the edges on the shortest path
                length_edge_for_anodepair = []
                shortest_path_edges = list(zip(SPNodelist[:-1], SPNodelist[1:]))
                for (nodes, nodet) in shortest_path_edges:
                    d_E = compute_edge_Euclidean_length(nodes, nodet, Coorx, Coory)
                    length_edge_for_anodepair.append(d_E)
                length_edge_vec = length_edge_vec + length_edge_for_anodepair
                ave_edge_length.append(np.mean(length_edge_for_anodepair))

                # # # compute the deviation
                # xSource = Coorx[nodei]
                # ySource = Coory[nodei]
                # xEnd = Coorx[nodej]
                # yEnd = Coory[nodej]
                # length_geodesic.append(distR2(xSource, ySource, xEnd, yEnd))
                # # Compute deviation for the shortest path of each node pair
                # deviations_for_a_nodepair = []
                # hop_for_a_nodepair = []
                # for SPnode in SPNodelist[1:len(SPNodelist)-1]:
                #     xMed = Coorx[SPnode]
                #     yMed = Coory[SPnode]
                #     dist, _ = dist_to_geodesic_R2(xMed, yMed, xSource, ySource, xEnd, yEnd)
                #     deviations_for_a_nodepair.append(dist)
                #     # hop = hopcount_node(G, nodei, nodej, SPnode)
                #     # hop_for_a_nodepair.append(hop)
                # ave_deviation.append(np.mean(deviations_for_a_nodepair))
                # max_deviation.append(max(deviations_for_a_nodepair))
                # min_deviation.append(min(deviations_for_a_nodepair))

                # max_value = max(deviations_for_a_nodepair)
                # max_index = deviations_for_a_nodepair.index(max_value)
                # maxhop_node_index = SPNodelist[max_index]
                # max_dev_node_hopcount.append(hopcount_node(G, nodei, nodej, maxhop_node_index))

                count = count + 1

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
        count_vec.append(count)

    # For each node graph:
    real_ave_degree_name = folder_name + "real_ave_degree_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(real_ave_degree_name, real_ave_degree)

    ave_graphlinklength_Name = folder_name + "ave_graph_edge_length_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(ave_graphlinklength_Name, ave_graphlinklength_vec)

    # LCC_num_name = folder_name+"LCC_num_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
    #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    # np.savetxt(LCC_num_name, LCC_num, fmt="%i")
    # clustering_coefficient_name = folder_name+"clustering_coefficient_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
    #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    # np.savetxt(clustering_coefficient_name, clustering_coefficient)

    # For each node pair:
    # ave_deviation_name = folder_name+"ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
    #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    # np.savetxt(ave_deviation_name, ave_deviation)
    # max_deviation_name = folder_name+"max_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
    #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    # np.savetxt(max_deviation_name, max_deviation)
    # min_deviation_name = folder_name+"min_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
    #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    # np.savetxt(min_deviation_name, min_deviation)

    # ave_baseline_deviation_name = folder_name+"ave_baseline_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
    #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    # np.savetxt(ave_baseline_deviation_name, ave_baseline_deviation)

    # length_geodesic_name = folder_name + "length_geodesic_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
    #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    # np.savetxt(length_geodesic_name, length_geodesic)

    # SPnodenum_vec_name = folder_name+"SPnodenum_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
    #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    # np.savetxt(SPnodenum_vec_name, SPnodenum_vec, fmt="%i")

    nodepairs_for_eachgraph_vec_name = folder_name + "nodepairs_for_eachgraph_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(nodepairs_for_eachgraph_vec_name, count_vec, fmt="%i")

    hopcount_Name = folder_name + "hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(hopcount_Name, SP_hopcount, fmt="%i")

    edgelength_name = folder_name + "edgelength_N{Nn}ED{EDn}beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(edgelength_name, length_edge_vec)

    aveedgelength_name = folder_name + "ave_edgelength_N{Nn}ED{EDn}beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(aveedgelength_name, ave_edge_length)

    # max_dev_node_hopcount_name = folder_name+"max_dev_node_hopcount_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
    #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    # np.savetxt(max_dev_node_hopcount_name, max_dev_node_hopcount, fmt="%i")


def stretch_inlargeSRGG_oneSP_clu(N, ED, beta, rg, ExternalSimutime):
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
    ave_edge_length = []
    ave_baseline_deviation = []
    length_geodesic = []
    length_edge_vec = []
    hopcount_vec = []
    max_dev_node_hopcount = []
    corresponding_sp_max_dev_node_hopcount = []
    SPnodenum_vec = []
    LCC_vec = []
    second_vec = []
    delta_vec = []  # delta is the Euclidean geometric distance between two nodes i,k, where i,k is the neighbours of j

    folder_name1 = "/home/qzh/network/"
    folder_name = "/home/qzh/data/"
    try:
        FileNetworkName = folder_name1 + "network_N{Nn}ED{EDn}Beta{betan}.txt".format(
            Nn=N, EDn=ED, betan=beta)
        G = loadSRGGandaddnode(N, FileNetworkName)
        # load coordinates with noise
        Coorx = []
        Coory = []

        FileNetworkCoorName = folder_name1 + "network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt".format(
            Nn=N, EDn=ED, betan=beta)
        with open(FileNetworkCoorName, "r") as file:
            for line in file:
                if line.startswith("#"):
                    continue
                data = line.strip().split("\t")  # 使用制表符分割
                Coorx.append(float(data[0]))
                Coory.append(float(data[1]))
    except:
        G, Coorx, Coory = R2SRGG(N, ED, beta, rg)
        FileNetworkName = folder_name1 + +"network_N{Nn}ED{EDn}Beta{betan}.txt".format(
            Nn=N, EDn=ED, betan=beta)
        nx.write_edgelist(G, FileNetworkName)
        FileNetworkCoorName = folder_name1 + +"network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt".format(
            Nn=N, EDn=ED, betan=beta)
        with open(FileNetworkCoorName, "w") as file:
            for data1, data2 in zip(Coorx, Coory):
                file.write(f"{data1}\t{data2}\n")

    # G, Coorx, Coory = R2SRGG(N, ED, beta, rg)
    # #     # if ExternalSimutime == 0:
    # FileNetworkName = folder_name+"network_N{Nn}ED{EDn}Beta{betan}.txt".format(
    #     Nn=N, EDn=ED, betan=beta)
    # nx.write_edgelist(G, FileNetworkName)
    # FileNetworkCoorName = folder_name+"network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt".format(
    #     Nn=N, EDn=ED, betan=beta)
    # with open(FileNetworkCoorName, "w") as file:
    #     for data1, data2 in zip(Coorx, Coory):
    #         file.write(f"{data1}\t{data2}\n")

    real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
    print("real ED:", real_avg)

    real_avg_name = folder_name + "real_avg_N{Nn}ED{EDn}beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(real_avg_name, [real_avg])

    # Randomly choose 100 connectede node pairs
    nodepair_num = 10000
    unique_pairs = find_k_connected_node_pairs(G, nodepair_num)
    # filename_selecetednodepair = folder_name+"selected_node_pair_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
    #     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
    # np.savetxt(filename_selecetednodepair, unique_pairs, fmt="%i")

    # LCC and the second LCC
    # connected_components = sorted(nx.connected_components(G), key=len, reverse=True)
    # if len(connected_components) > 1:
    #     largest_largest_component = connected_components[0]
    #     largest_largest_size = len(largest_largest_component)
    #     LCC_vec.append(largest_largest_size)
    #     # ?????????????????
    #     second_largest_component = connected_components[1]
    #     second_largest_size = len(second_largest_component)
    #     second_vec.append(second_largest_size)
    # if ExternalSimutime==0:
    #     filefolder_name = folder_name+""
    #     LCCname = filefolder_name + "LCC_2LCC_N{Nn}ED{EDn}beta{betan}.txt".format(
    #         Nn=N, EDn=ED, betan=beta)
    #     with open(LCCname, "w") as file:
    #         file.write("# LCC\tSECLCC\n")
    #         for name, age in zip(LCC_vec, second_vec):
    #             file.write(f"{name}\t{age}\n")
    count = 0
    for node_pair in unique_pairs:
        count = count + 1
        print(f"{count}node_pair:{node_pair}")

        nodei = node_pair[0]
        nodej = node_pair[1]
        # Find the shortest path nodes
        try:
            SPNodelist = nx.shortest_path(G, nodei, nodej)
            SPnodenum = len(SPNodelist) - 2
            SPnodenum_vec.append(SPnodenum)
            if SPnodenum > 0:
                hopcount_vec.append(nx.shortest_path_length(G, nodei, nodej))

                # compute the length of the edges
                length_edge_for_anodepair = []
                shortest_path_edges = list(zip(SPNodelist[:-1], SPNodelist[1:]))
                for (nodes, nodet) in shortest_path_edges:
                    d_E = compute_edge_Euclidean_length(nodes, nodet, Coorx, Coory)
                    length_edge_for_anodepair.append(d_E)
                length_edge_vec = length_edge_vec + length_edge_for_anodepair
                ave_edge_length.append(np.mean(length_edge_for_anodepair))

                # compute the deviation
                xSource = Coorx[nodei]
                ySource = Coory[nodei]
                xEnd = Coorx[nodej]
                yEnd = Coory[nodej]
                length_geodesic.append(distR2(xSource, ySource, xEnd, yEnd))
                # Compute deviation for the shortest path of each node pair
                deviations_for_a_nodepair = []
                for SPnode in SPNodelist[1:len(SPNodelist) - 1]:
                    xMed = Coorx[SPnode]
                    yMed = Coory[SPnode]
                    dist, _ = dist_to_geodesic_R2(xMed, yMed, xSource, ySource, xEnd, yEnd)
                    deviations_for_a_nodepair.append(dist)

                deviation_vec = deviation_vec + deviations_for_a_nodepair

                ave_deviation.append(np.mean(deviations_for_a_nodepair))
                # max_deviation.append(max(deviations_for_a_nodepair))
                # min_deviation.append(min(deviations_for_a_nodepair))

                # Compute delta for the shortest path of each node pair:
                delta_for_a_nodepair = []
                for i in range(len(SPNodelist) - 2):  # 计算相隔节点的距离
                    node1 = SPNodelist[i]
                    node2 = SPNodelist[i + 2]

                    delta = distR2(Coorx[node1], Coory[node1], Coorx[node2], Coory[node2])
                    delta_for_a_nodepair.append(delta)

                delta_vec = delta_vec + delta_for_a_nodepair

                # max hopcount
                # max_value = max(deviations_for_a_nodepair)
                # max_index = deviations_for_a_nodepair.index(max_value)
                # maxhop_node_index = SPNodelist[max_index]
                # max_dev_node_hopcount.append(hopcount_node(G, nodei, nodej, maxhop_node_index))
                # corresponding_sp_max_dev_node_hopcount.append(nx.shortest_path_length(G, nodei, nodej))

                # baseline: random selected
                # baseline_deviations_for_a_nodepair = []
                # # compute baseline's deviation
                # filtered_numbers = [num for num in range(N) if num not in [nodei,nodej]]
                # base_line_node_index = random.sample(filtered_numbers,SPnodenum)
                #
                # for SPnode in base_line_node_index:
                #     xMed = Coorx[SPnode]
                #     yMed = Coory[SPnode]
                #     dist, _ = dist_to_geodesic_R2(xMed, yMed, xSource, ySource, xEnd, yEnd)
                #     baseline_deviations_for_a_nodepair.append(dist)
                # ave_baseline_deviation.append(np.mean(baseline_deviations_for_a_nodepair))
                # baseline_deviation_vec = baseline_deviation_vec + baseline_deviations_for_a_nodepair
        except:
            pass
    deviation_vec_name = folder_name + "deviation_shortest_path_nodes_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(deviation_vec_name, deviation_vec)
    # baseline_deviation_vec_name = folder_name+"deviation_baseline_nodes_num_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
    #     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
    # np.savetxt(baseline_deviation_vec_name, baseline_deviation_vec)
    # For each node pair:
    ave_deviation_name = folder_name + "ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(ave_deviation_name, ave_deviation)
    # max_deviation_name = folder_name+"max_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
    #     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
    # np.savetxt(max_deviation_name, max_deviation)
    # min_deviation_name = folder_name+"min_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
    #     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
    # np.savetxt(min_deviation_name, min_deviation)
    # ave_baseline_deviation_name = folder_name+"ave_baseline_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
    #     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
    # np.savetxt(ave_baseline_deviation_name, ave_baseline_deviation)
    length_geodesic_name = folder_name + "length_geodesic_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(length_geodesic_name, length_geodesic)
    SPnodenum_vec_name = folder_name + "SPnodenum_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(SPnodenum_vec_name, SPnodenum_vec, fmt="%i")
    hopcount_Name = folder_name + "hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(hopcount_Name, hopcount_vec, fmt="%i")
    delta_Name = folder_name + "delta_N{Nn}ED{EDn}beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(delta_Name, delta_vec)

    edgelength_name = folder_name + "edgelength_N{Nn}ED{EDn}beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(edgelength_name, length_edge_vec)

    aveedgelength_name = folder_name + "ave_edgelength_N{Nn}ED{EDn}beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(aveedgelength_name, ave_edge_length)

    # max_dev_node_hopcount_name = folder_name+"max_dev_node_hopcount_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
    #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    # np.savetxt(max_dev_node_hopcount_name, max_dev_node_hopcount, fmt="%i")
    # max_dev_node_hopcount_name2 = folder_name+"sphopcountmax_dev_node_hopcount_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
    #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    # np.savetxt(max_dev_node_hopcount_name2, corresponding_sp_max_dev_node_hopcount, fmt="%i")


def stretch_inSRGG_oneSP(network_size, input_expected_degree, beta, ExternalSimutime):
    """
    Only one shortest path is chosen to test how the deviation changes with different network parameters
    :param network_size_index:
    :param average_degree_index:
    :param beta_index:
    :param ExternalSimutime:
    :return:
    """

    # random.seed(ExternalSimutime)
    N = network_size
    # if N ==1000:
    #     kvec = [2, 3, 4, 5, 8, 12, 20, 31, 49, 77, 120, 188, 296, 468, 739, 1166, 1836, 2842] # for N = 1000
    # else:
    #     kvec = [2, 3, 4, 6, 11, 20, 37, 67, 121, 218, 392, 705, 1267, 2275, 4086, 7336, 13169, 23644,29999]  # for N = 1000
    ED = input_expected_degree
    beta = beta
    print("input para:", (N, ED, beta), flush=True)

    rg = RandomGenerator(-12)
    rseed = random.randint(0, 100)
    for i in range(rseed):
        rg.ran1()

    # for large network, we only generate one network and randomly selected 1,000 node pair.
    # for small network, we generate 100 networks and selected 1000 node pairs in the LCC
    stretch_in_multiple_largenetworks_oneSP_clu(N, ED, beta, rg, ExternalSimutime)


def compute_stretch():
    tasks = []
    # tasks = [(8, 128, 0.0005,0), (8, 128, 0.005,0), (16, 128, 0.005,0), (16, 128, 0.5,0), (32, 128, 0.005,0), (32, 128, 0.5,0),
    #          (64, 128, 0.0005,0), (64, 128, 0.005,0), (64, 128, 0.05,0), (64, 128, 0.5,0), (128, 128, 0.0005,0),
    #          (128, 128, 0.005,0), (128, 128, 0.05,0), (128, 128, 0.5,0)]

    # for simutime in range(10):
    #     tasks.append((32, 128, 0.1, simutime))
    # for simutime in range(10):
    #     tasks.append((32, 8, 1, simutime))
    # tasks = [(32, 8, 0.1, 0), (64, 4, 1, 1), (64, 4, 1, 2), (64, 4, 1, 3), (64, 4, 1, 4), (64, 4, 1, 5), (64, 4, 1, 6),
    #          (64, 4, 1, 7), (64, 4, 1, 8), (64, 4, 1, 9)]

    # # simu1: diff N
    # # Nvec = [10, 22, 46, 100, 215, 464, 1000, 2154, 4642, 10000]
    # Nvec = [100, 215, 464, 1000, 2154, 4642, 10000]
    # beta_vec = [1024]
    # input_ED_vec = [10]
    # for N in Nvec:
    #     for inputED in input_ED_vec:
    #         for beta in beta_vec:
    #             tasks.append((N,inputED, beta,0))
    # with mp.Pool(processes=4) as pool:
    #     pool.starmap(distance_inSRGG_oneSP, tasks)
    #
    # # simu2: diff beta
    # Nvec = [100, 1000, 10000]
    # beta_vec = [2.2, 3.0, 4.2, 5.9, 8.3, 11.7, 16.5, 23.2, 32.7, 46.1, 64.9, 91.5, 128.9, 181.7, 256]
    # input_ED_vec = [10]
    # for N in Nvec:
    #     for inputED in input_ED_vec:
    #         for beta in beta_vec:
    #             tasks.append((N, inputED, beta, 0))
    #
    # with mp.Pool(processes=4) as pool:
    #     pool.starmap(distance_inSRGG_oneSP, tasks)

    # simu3: diff ED

    # # Nvec = [100, 215, 464, 1000, 2154, 4642,10000]
    # Nvec = [215]
    # beta_vec=[1024]
    kvec_dict = {
        100: [2, 3, 5, 8, 12, 18, 29, 45, 70, 109, 169, 264, 412, 642, 1000],
        215: [2, 3, 5, 9, 14, 24, 39, 63, 104, 170, 278, 455, 746, 1221, 2000],
        464: [2, 3, 6, 10, 18, 30, 52, 89, 154, 265, 456, 785, 1350, 2324, 4000],
        1000: [2, 4, 7, 12, 21, 39, 70, 126, 229, 414, 748, 1353, 2446, 4424, 8000],
        2154: [2, 4, 7, 14, 27, 52, 99, 190, 364, 697, 1335, 2558, 4902, 9393, 18000],
        4642: [2, 4, 8, 16, 33, 67, 135, 272, 549, 1107, 2234, 4506, 9091, 18340, 37000],
        10000: [2.2, 2.8, 3.0, 3.4, 3.8, 4.4, 6.0, 7.0, 8.0, 9.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389,
                2276,
                3727, 6105,
                9999, 16479, 27081, 44767, 73534, 121205, 199999]}
    #
    # for N in Nvec:
    #     if N ==10:
    #         input_ED_vec = list(range(2, 10)) + [10, 12, 15, 18, 22, 27, 33, 40, 49, 60, 73, 89, 99]  # FOR N =10
    #     else:
    #         input_ED_vec = kvec_dict[N]
    #     for inputED in input_ED_vec:
    #         for beta in beta_vec:
    #             tasks.append((N, inputED, beta, 0))

    # simu4: diff ED for catching local minumum
    # Nvec = [464, 1000, 2154, 4642,10000]
    Nvec = [215]
    beta_vec = [1024]
    #
    # # analytic_k = {215: range(24,104+1,2), 464: 80, 1000: 104, 2154: 134, 4642: 173, 10000: 224}
    k_dict = {215: list(range(24, 104 + 1, 2)), 464: list(range(30, 154 + 1, 2)), 1000: list(range(39, 229 + 1, 2)),
              2154: list(
                  range(52, 364 + 1, 2)), 4642: list(range(67, 272 + 1, 2)), 10000: list(range(118, 316 + 1, 2))}

    # k_dict = {681: list(range(40,164 + 1, 2)), 1468: list(range(50,240 + 1, 2)), 3156: list(range(72,384 + 1, 2)), 6803: list(range(87,
    #     295+ 1, 2)), 14683: list(range(140,340 + 1, 2))}

    # Nvec = [681, 1468, 3156, 6803, 14683]
    # Nvec = [681, 1468,3156]
    for N in Nvec:
        input_ED_vec = k_dict[N]
        for inputED in input_ED_vec:
            for beta in beta_vec:
                tasks.append((N, inputED, beta, 0))

    # simu4: diff ED for catching 1-hop difference
    # Nvec = [10000]
    # for N in Nvec:
    #     input_ED_vec = kvec_dict[N]
    #     for inputED in input_ED_vec:
    #         for beta in [1024]:
    #             tasks.append((N, inputED, beta, 0))

    with mp.Pool(processes=2) as pool:
        pool.starmap(stretch_inSRGG_oneSP, tasks)

    # Press the green button in the gutter to run the script.


def generate_ED_log_unifrom(start_point, end_point, point_number):
    points = np.logspace(np.log10(start_point), np.log10(end_point), point_number)
    points_int_vec = [round(i) for i in points]
    return points_int_vec


def compute_proper_ed():
    kvec_dict_forsmallbeta = {
        464: generate_ED_log_unifrom(2,1000000,12),
        681: generate_ED_log_unifrom(2, 1000000, 15),
        1000: generate_ED_log_unifrom(2,1000000,15),
        1468: generate_ED_log_unifrom(2, 1000000, 15),
        2154: generate_ED_log_unifrom(2,1000000,15),
        3156: generate_ED_log_unifrom(2, 1000000, 15),
        4642: generate_ED_log_unifrom(2,1000000,15),
        6803: generate_ED_log_unifrom(2, 1000000, 15),
        10000: generate_ED_log_unifrom(2,1000000,15)}
    print(kvec_dict_forsmallbeta)

    # rg = RandomGenerator(-12)
    # rseed = random.randint(0, 100)
    # for i in range(rseed):
    #     rg.ran1()
    # Nvec = [464,681,1000,1468,2154,3156,4642]
    # for N in Nvec:
    #     G,coorx,coory = R2SRGG(N,10000000,2.1,rg)
    #     real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
    #     print(real_avg)


    # points_int = generate_ED_log_unifrom()



if __name__ == '__main__':
    """
    Step1 run simulations with different beta and input AVG for one sp case
    """

    # compute_stretch()

    # """
    # SMALL NETWROK CHECK
    # """
    # G = nx.Graph()  # 无向图；用 nx.DiGraph() 表示有向图
    #
    # # 添加带权边（权重可选）
    # G.add_weighted_edges_from([
    #     ('A', 'B', 1),
    #     ('B', 'C', 2),
    #     ('A', 'C', 5),
    #     ('C', 'D', 1),
    #     ('B', 'D', 4)
    # ])
    #
    # # 求从 A 到 D 的最短路径（按权重）
    # shortest_path = nx.shortest_path(G, source='A', target='D', weight='weight')
    #
    #
    # SPnodenum = len(shortest_path) - 2
    # print(SPnodenum+1)
    # # SPnodenum_vec.append(SPnodenum)
    #
    #
    # # 提取路径中的所有边
    # shortest_path_edges = list(zip(shortest_path[:-1], shortest_path[1:]))
    # for (nodei, nodej) in shortest_path_edges:
    #     print(nodei)
    #     print(nodej)
    #
    # print("最短路径:", shortest_path)
    # print("路径上的所有边:", shortest_path_edges)


    # k_dict = {681: list(range(40,164 + 1, 2)), 1468: list(range(50,240 + 1, 2)), 3156: list(range(72,384 + 1, 2)), 6803: list(range(87,
    #         295+ 1, 2)), 14683: list(range(140,340 + 1, 2))}
    #
    # for key, item in k_dict.items():
    #     print(key)
    #     print(len(item))

    # kvec_dict = {
    #     100: [2, 3, 5, 8, 12, 18, 29, 45, 70, 109, 169, 264, 412, 642, 1000],
    #     215: [2, 3, 5, 9, 14, 24, 39, 63, 104, 170, 278, 455, 746, 1221, 2000],
    #     464: [2, 3, 6, 10, 18, 30, 52, 89, 154, 265, 456, 785, 1350, 2324, 4000],
    #     1000: [2, 4, 7, 12, 21, 39, 70, 126, 229, 414, 748, 1353, 2446, 4424, 8000],
    #     2154: [2, 4, 7, 14, 27, 52, 99, 190, 364, 697, 1335, 2558, 4902, 9393, 18000],
    #     4642: [2, 4, 8, 16, 33, 67, 135, 272, 549, 1107, 2234, 4506, 9091, 18340, 37000],
    #     10000: [2.2, 2.8, 3.0, 3.4, 3.8, 4.4, 6.0, 7.0, 8.0, 9.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389,
    #             2276,
    #             3727, 6105,
    #             9999, 16479, 27081, 44767, 73534, 121205, 199999]}
    #

    compute_proper_ed()


