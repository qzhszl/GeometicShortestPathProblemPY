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
import os
import shutil

from R2SRGG import R2SRGG, distR2, dist_to_geodesic_R2, loadSRGGandaddnode
from SphericalSoftRandomGeomtricGraph import RandomGenerator
from main import all_shortest_path_node, find_k_connected_node_pairs, find_all_connected_node_pairs, hopcount_node


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
                data = line.strip().split("\t")  # ???????
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


def distance_inlargeSRGG_clu(N,ED,beta,rg,ExternalSimutime):
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
        hopcount_vec = []
        max_dev_node_hopcount = []
        corresponding_sp_max_dev_node_hopcount = []
        SPnodenum_vec =[]
        LCC_vec =[]
        second_vec = []

        source_folder = "/home/zqiu1/GSPP_SRGG_Dev/NetworkSRGG/"
        destination_folder = "/work/zqiu1/"
        network_template = "network_N{Nn}ED{EDn}Beta{betan}.txt"
        networkcoordinate_template  = "network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt"

        # load or generate a network
        try:
            FileNetworkName = "/work/zqiu1/network_N{Nn}ED{EDn}Beta{betan}.txt".format(
                Nn=N, EDn=ED, betan=beta)
            G = loadSRGGandaddnode(N, FileNetworkName)
            # load coordinates with noise
            Coorx = []
            Coory = []

            FileNetworkCoorName = "/work/zqiu1/network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt".format(
                Nn=N, EDn=ED, betan=beta)
            with open(FileNetworkCoorName, "r") as file:
                for line in file:
                    if line.startswith("#"):
                        continue
                    data = line.strip().split("\t")
                    Coorx.append(float(data[0]))
                    Coory.append(float(data[1]))
        except ValueError:
            os.makedirs(destination_folder, exist_ok=True)
            source_file = source_folder+ network_template.format(Nn=N, EDn=ED, betan=beta)
            destination_file = destination_folder+ network_template.format(Nn=N, EDn=ED, betan=beta)
            shutil.copy(source_file, destination_file)
            print(f"Copied: {source_file} -> {destination_file}")
            source_file = source_folder + networkcoordinate_template.format(Nn=N, EDn=ED, betan=beta)
            destination_file = destination_folder + networkcoordinate_template.format(Nn=N, EDn=ED, betan=beta)
            shutil.copy(source_file, destination_file)
            print(f"Copied: {source_file} -> {destination_file}")
        except:
            G, Coorx, Coory = R2SRGG(N, ED, beta, rg)
            if ExternalSimutime == 0:
                FileNetworkName = "/work/zqiu1/network_N{Nn}ED{EDn}Beta{betan}.txt".format(
                    Nn=N, EDn=ED, betan=beta)
                nx.write_edgelist(G, FileNetworkName)
                FileNetworkCoorName = "/work/zqiu1/network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt".format(
                    Nn=N, EDn=ED, betan=beta)
                with open(FileNetworkCoorName, "w") as file:
                    for data1, data2 in zip(Coorx, Coory):
                        file.write(f"{data1}\t{data2}\n")

        real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
        print("real ED:", real_avg)

        # Randomly choose 100 connectede node pairs
        nodepair_num = 50
        unique_pairs = find_k_connected_node_pairs(G, nodepair_num)
        filename_selecetednodepair = "/work/zqiu1/selected_node_pair_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(filename_selecetednodepair, unique_pairs, fmt="%i")

        connected_components = sorted(nx.connected_components(G), key=len, reverse=True)
        if len(connected_components) > 1:
            largest_largest_component = connected_components[0]
            largest_largest_size = len(largest_largest_component)
            LCC_vec.append(largest_largest_size)
            # ?????????????????
            second_largest_component = connected_components[1]
            second_largest_size = len(second_largest_component)
            second_vec.append(second_largest_size)
        if ExternalSimutime==0:
            filefolder_name = "/work/zqiu1/"
            LCCname = filefolder_name + "LCC_2LCC_N{Nn}ED{EDn}beta{betan}.txt".format(
                Nn=N, EDn=ED, betan=beta)
            with open(LCCname, "w") as file:
                file.write("# LCC\tSECLCC\n")  # ????????
                # ????
                for name, age in zip(LCC_vec, second_vec):
                    file.write(f"{name}\t{age}\n")

        for node_pair in unique_pairs:
            print("node_pair:",node_pair)
            nodei = node_pair[0]
            nodej = node_pair[1]
            # Find the shortest path nodes
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

                max_value = max(deviations_for_a_nodepair)
                max_index = deviations_for_a_nodepair.index(max_value)
                maxhop_node_index = SPNodelist[max_index]
                max_dev_node_hopcount.append(hopcount_node(G, nodei, nodej, maxhop_node_index))
                corresponding_sp_max_dev_node_hopcount.append(nx.shortest_path_length(G, nodei, nodej))


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

        deviation_vec_name = "/work/zqiu1/deviation_shortest_path_nodes_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(deviation_vec_name, deviation_vec)
        baseline_deviation_vec_name = "/work/zqiu1/deviation_baseline_nodes_num_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(baseline_deviation_vec_name, baseline_deviation_vec)
        # For each node pair:
        ave_deviation_name = "/work/zqiu1/ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(ave_deviation_name, ave_deviation)
        max_deviation_name = "/work/zqiu1/max_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(max_deviation_name, max_deviation)
        min_deviation_name = "/work/zqiu1/min_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(min_deviation_name, min_deviation)
        ave_baseline_deviation_name = "/work/zqiu1/ave_baseline_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(ave_baseline_deviation_name, ave_baseline_deviation)
        length_geodesic_name = "/work/zqiu1/length_geodesic_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(length_geodesic_name, length_geodesic)
        SPnodenum_vec_name = "/work/zqiu1/SPnodenum_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(SPnodenum_vec_name, SPnodenum_vec,fmt="%i")
        hopcount_Name = "/work/zqiu1/hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(hopcount_Name, hopcount_vec)

        max_dev_node_hopcount_name = "/work/zqiu1/max_dev_node_hopcount_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(max_dev_node_hopcount_name, max_dev_node_hopcount, fmt="%i")
        max_dev_node_hopcount_name2 = "/work/zqiu1/sphopcountmax_dev_node_hopcount_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(max_dev_node_hopcount_name2, corresponding_sp_max_dev_node_hopcount, fmt="%i")

def distance_inlargeSRGG_clu_cc(N, ED, beta, cc, ExternalSimutime):
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
        FileNetworkName = "/home/zqiu1/GSPP/SSRGGpy/R2/distribution/NetworkSRGG/network_N{Nn}ED{EDn}CC{betan}.txt".format(
            Nn=N, EDn=ED, betan=cc)
        G = loadSRGGandaddnode(N, FileNetworkName)
        # load coordinates with noise
        Coorx = []
        Coory = []

        FileNetworkCoorName = "/home/zqiu1/GSPP/SSRGGpy/R2/distribution/NetworkSRGG/network_coordinates_N{Nn}ED{EDn}CC{betan}.txt".format(
            Nn=N, EDn=ED, betan=cc)
        with open(FileNetworkCoorName, "r") as file:
            for line in file:
                if line.startswith("#"):
                    continue
                data = line.strip().split("\t")  # ???????
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
        hopcount_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\hopcount_sp_ED{EDn}Beta{betan}Simu{ST}.txt".format(
            EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(hopcount_Name, hopcount_vec)




def distance_inSRGG(network_size_index, average_degree_index, beta_index, ExternalSimutime):
    Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    betavec = [2.2, 4, 8, 16, 32, 64, 128]

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
    # kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    # kvec = [2,2.5,3,3.5,4,4.5,5,5.5,6]
    # kvec = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    # kvec = np.arange(2, 6.1, 0.2)
    # kvec = [round(a, 1) for a in kvec]
    kvec = np.arange(6.5, 9.6, 0.5)
    kvec = [round(a, 1) for a in kvec]
    kvec2 = [5, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    kvec = kvec + kvec2
    kvec = [5,10,20]

    # kvec = [5,20]
    # betavec = [2.2, 4, 8, 16, 32, 64, 128]
    betavec = [2.3, 2.4, 2.5, 2.6, 2.8, 3, 3.25, 3.5, 3.75, 5, 6, 7]
    betavec = [2.2, 3.0, 4.2, 5.9, 8.3, 11.7, 16.5, 23.2, 32.7, 46.1, 64.9, 91.5, 128.9, 181.7, 256]

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
        distance_inlargeSRGG_clu(N, ED, beta,rg, ExternalSimutime)
    else:
        # Random select nodepair_num nodes in the largest connected component
        distance_insmallSRGG(N, ED, beta, rg, ExternalSimutime)

if __name__ == '__main__':
    # network_size_index = 4
    # average_degree_index = 2
    # beta_index = 1
    # external_simu_time = 0
    # distance_inSRGG(network_size_index, average_degree_index, beta_index, external_simu_time)

    """
    Step 2 Run the simulation
    """
    ED = sys.argv[1]
    beta_index = sys.argv[2]
    ExternalSimutime = sys.argv[3]
    distance_inSRGG_clu(7, int(ED), int(beta_index), int(ExternalSimutime))

