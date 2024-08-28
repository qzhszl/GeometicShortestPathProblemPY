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

from R2SRGG import R2SRGG, distR2, dist_to_geodesic_R2, loadSRGGandaddnode
from SphericalSoftRandomGeomtricGraph import RandomGenerator
from main import all_shortest_path_node, find_k_connected_node_pairs, find_all_connected_node_pairs


def generate_r2SRGG():
    rg = RandomGenerator(-12)
    rseed = random.randint(0, 100)
    print(rseed)
    for i in range(rseed):
        rg.ran1()

    Nvec = [200, 500, 1000, 10000]
    kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    betavec = [2.1, 4, 8, 16, 32, 64, 128]

    for N in Nvec:
        for ED in kvec:
            for beta in betavec:

                G, Coorx, Coory = R2SRGG(N, ED, beta, rg)
                real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
                print("input para:", (N,ED,beta))
                print("real ED:", real_avg)
                print("clu:", nx.average_clustering(G))
                components = list(nx.connected_components(G))
                largest_component = max(components, key=len)
                print("LCC", len(largest_component))

                FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\network_N{Nn}ED{EDn}Beta{betan}.txt".format(
                    Nn=N, EDn=ED, betan=beta)
                nx.write_edgelist(G, FileNetworkName)

                FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt".format(
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


def distance_inlargeSRGG(N,ED,beta,ExternalSimutime):
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
    if N> ED:
        deviation_vec = []  # deviation of all shortest path nodes for all node pairs
        baseline_deviation_vec = []  # deviation of all shortest path nodes for all node pairs
        # For each node pair:
        ave_deviation = []
        max_deviation = []
        min_deviation = []
        ave_baseline_deviation =[]
        length_geodesic = []
        SPnodenum_vec =[]

        # load a network
        FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\network_N{Nn}ED{EDn}Beta{betan}.txt".format(
            Nn=N, EDn=ED, betan=beta)
        G = loadSRGGandaddnode(N, FileNetworkName)
        # load coordinates with noise
        Coorx = []
        Coory = []

        FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt".format(
            Nn=N, EDn=ED, betan=beta)
        with open(FileNetworkCoorName, "r") as file:
            for line in file:
                if line.startswith("#"):
                    continue
                data = line.strip().split("\t")  # 使用制表符分割
                Coorx.append(float(data[0]))
                Coory.append(float(data[1]))


        real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
        print("real ED:", real_avg)

        ave_clu = nx.average_clustering(G)
        print("clu:",ave_clu)

        components = list(nx.connected_components(G))
        largest_component = max(components, key=len)
        LCC_number = len(largest_component)
        print("LCC", LCC_number)

        # Randomly choose 100 connectede node pairs
        nodepair_num = 100
        unique_pairs = find_k_connected_node_pairs(G, nodepair_num)
        filename_selecetednodepair = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\selected_node_pair_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(filename_selecetednodepair, unique_pairs, fmt="%i")
        components = []
        largest_component = []

        for node_pair in unique_pairs:
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

        deviation_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\deviation_shortest_path_nodes_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(deviation_vec_name, deviation_vec)
        baseline_deviation_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\deviation_baseline_nodes_num_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(baseline_deviation_vec_name, baseline_deviation_vec)
        # For each node pair:
        ave_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(ave_deviation_name, ave_deviation)
        max_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\max_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(max_deviation_name, max_deviation)
        min_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\min_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(min_deviation_name, min_deviation)
        ave_baseline_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\ave_baseline_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(ave_baseline_deviation_name, ave_baseline_deviation)
        length_geodesic_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\length_geodesic_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(length_geodesic_name, length_geodesic)
        SPnodenum_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\SPnodenum_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(SPnodenum_vec_name, SPnodenum_vec,fmt="%i")


def distance_inlargeSRGG_clu(N,ED,beta,ExternalSimutime):
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
    if N> ED:
        deviation_vec = []  # deviation of all shortest path nodes for all node pairs
        baseline_deviation_vec = []  # deviation of all shortest path nodes for all node pairs
        # For each node pair:
        ave_deviation = []
        max_deviation = []
        min_deviation = []
        ave_baseline_deviation =[]
        length_geodesic = []
        SPnodenum_vec =[]

        # load a network
        FileNetworkName = "/home/zqiu1/GSPP/SSRGGpy/R2/distribution/NetworkSRGG/network_N{Nn}ED{EDn}Beta{betan}.txt".format(
            Nn=N, EDn=ED, betan=beta)
        G = loadSRGGandaddnode(N, FileNetworkName)
        # load coordinates with noise
        Coorx = []
        Coory = []

        FileNetworkCoorName = "/home/zqiu1/GSPP/SSRGGpy/R2/distribution/NetworkSRGG/network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt".format(
            Nn=N, EDn=ED, betan=beta)
        with open(FileNetworkCoorName, "r") as file:
            for line in file:
                if line.startswith("#"):
                    continue
                data = line.strip().split("\t")  # 使用制表符分割
                Coorx.append(float(data[0]))
                Coory.append(float(data[1]))


        real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
        print("real ED:", real_avg)

        ave_clu = nx.average_clustering(G)
        print("clu:",ave_clu)

        components = list(nx.connected_components(G))
        largest_component = max(components, key=len)
        LCC_number = len(largest_component)
        print("LCC", LCC_number)

        # Randomly choose 100 connectede node pairs
        nodepair_num = 10
        unique_pairs = find_k_connected_node_pairs(G, nodepair_num)
        filename_selecetednodepair = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\selected_node_pair_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(filename_selecetednodepair, unique_pairs, fmt="%i")
        components = []
        largest_component = []

        for node_pair in unique_pairs:
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

        deviation_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\deviation_shortest_path_nodes_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(deviation_vec_name, deviation_vec)
        baseline_deviation_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\deviation_baseline_nodes_num_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(baseline_deviation_vec_name, baseline_deviation_vec)
        # For each node pair:
        ave_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(ave_deviation_name, ave_deviation)
        max_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\max_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(max_deviation_name, max_deviation)
        min_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\min_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(min_deviation_name, min_deviation)
        ave_baseline_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\ave_baseline_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(ave_baseline_deviation_name, ave_baseline_deviation)
        length_geodesic_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\length_geodesic_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(length_geodesic_name, length_geodesic)
        SPnodenum_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\SPnodenum_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(SPnodenum_vec_name, SPnodenum_vec,fmt="%i")


def distance_inSRGG(network_size_index, average_degree_index, beta_index, ExternalSimutime):
    Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    betavec = [2.1, 4, 8, 16, 32, 64, 128]

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
        distance_inlargeSRGG(N, ED, beta, ExternalSimutime)
    else:
        # Random select nodepair_num nodes in the largest connected component
        distance_insmallSRGG(N, ED, beta, rg, ExternalSimutime)

def distance_inSRGG_clu(network_size_index, average_degree_index, beta_index, ExternalSimutime):
    Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    betavec = [2.1, 4, 8, 16, 32, 64, 128]

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
    if N > 100:
        distance_inlargeSRGG_clu(N, ED, beta, ExternalSimutime)
    else:
        # Random select nodepair_num nodes in the largest connected component
        distance_insmallSRGG(N, ED, beta, rg, ExternalSimutime)

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

    ED = sys.argv[1]
    beta = sys.argv[2]
    ExternalSimutime = sys.argv[3]
    distance_inSRGG_clu(7, int(ED), int(beta), int(ExternalSimutime))




