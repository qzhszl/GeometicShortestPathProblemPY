# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 16-7-2025

For large network, we only generate 1 graph and randomly selected 1000 node pairs.
The generated network, the selected node pair and all the deviation of both shortest path and baseline nodes will be recorded.

This script is collecting data for investigating
1. ave deviation
2. the relation between the length of the shortest path L = <d_e><h> versus deviation as the shortest path change
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


from R2SRGG import R2SRGG, distR2, dist_to_geodesic_R2, loadSRGGandaddnode
from SphericalSoftRandomGeomtricGraph import RandomGenerator
from main import all_shortest_path_node, find_k_connected_node_pairs, find_all_connected_node_pairs, hopcount_node


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
#
#
# def distance_insmallSRGG(N, ED, beta, rg, ExternalSimutime):
#     """
#     :param N:
#     :param ED:
#     :param beta:
#     :param rg:
#     :return:
#     for each graph, we record the real average degree, LCC number, clustering coefficient
#     for each node pair, we only record the ave,max,min of distance from the shortest path to the geodesic,
#     length of the geo distances.
#     """
#     if N> ED:
#         # For graph:
#         real_ave_degree = []
#         LCC_num = []
#         clustering_coefficient = []
#         # For each node pair:
#         ave_deviation = []
#         max_deviation = []
#         min_deviation = []
#         ave_baseline_deviation =[]
#         length_geodesic = []
#         SPnodenum_vec =[]
#         simu_times = 100
#         for simu_index in range(simu_times):
#             G, Coorx, Coory = R2SRGG(N, ED, beta, rg)
#             try:
#                 real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
#             except:
#                 flag = 0
#                 while flag == 0:
#                     G, Coorx, Coory = R2SRGG(N, ED, beta, rg)
#                     if nx.number_of_edges(G) > 0:
#                         flag=1
#                         real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
#
#             print("real ED:", real_avg)
#             real_ave_degree.append(real_avg)
#             ave_clu = nx.average_clustering(G)
#             print("clu:",ave_clu)
#             clustering_coefficient.append(ave_clu)
#             components = list(nx.connected_components(G))
#             largest_component = max(components, key=len)
#             LCC_number = len(largest_component)
#             print("LCC", LCC_number)
#             LCC_num.append(LCC_number)
#
#             # pick up all the node pairs in the LCC and save them in the unique_pairs
#             unique_pairs = find_all_connected_node_pairs(G)
#             count = 0
#             for node_pair in unique_pairs:
#                 count = count+1
#                 nodei = node_pair[0]
#                 nodej = node_pair[1]
#                 # Find the shortest path nodes
#                 SPNodelist = all_shortest_path_node(G, nodei, nodej)
#                 SPnodenum = len(SPNodelist)
#                 SPnodenum_vec.append(SPnodenum)
#                 if SPnodenum>0:
#                     xSource = Coorx[nodei]
#                     ySource = Coory[nodei]
#                     xEnd = Coorx[nodej]
#                     yEnd = Coory[nodej]
#                     length_geodesic.append(distR2(xSource, ySource, xEnd, yEnd))
#                     # Compute deviation for the shortest path of each node pair
#                     deviations_for_a_nodepair = []
#                     for SPnode in SPNodelist:
#                         xMed = Coorx[SPnode]
#                         yMed = Coory[SPnode]
#                         dist, _ = dist_to_geodesic_R2(xMed, yMed, xSource, ySource, xEnd, yEnd)
#                         deviations_for_a_nodepair.append(dist)
#                     ave_deviation.append(np.mean(deviations_for_a_nodepair))
#                     max_deviation.append(max(deviations_for_a_nodepair))
#                     min_deviation.append(min(deviations_for_a_nodepair))
#
#                     baseline_deviations_for_a_nodepair = []
#                     # compute baseline's deviation
#                     filtered_numbers = [num for num in range(N) if num not in [nodei,nodej]]
#                     base_line_node_index = random.sample(filtered_numbers,SPnodenum)
#
#                     for SPnode in base_line_node_index:
#                         xMed = Coorx[SPnode]
#                         yMed = Coory[SPnode]
#                         dist, _ = dist_to_geodesic_R2(xMed, yMed, xSource, ySource, xEnd, yEnd)
#                         baseline_deviations_for_a_nodepair.append(dist)
#                     ave_baseline_deviation.append(np.mean(baseline_deviations_for_a_nodepair))
#
#
#         real_ave_degree_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\real_ave_degree_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#             Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         np.savetxt(real_ave_degree_name, real_ave_degree)
#         LCC_num_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\LCC_num_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#             Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         np.savetxt(LCC_num_name, LCC_num, fmt="%i")
#         clustering_coefficient_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\clustering_coefficient_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#             Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         np.savetxt(clustering_coefficient_name, clustering_coefficient)
#         # For each node pair:
#         ave_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#             Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         np.savetxt(ave_deviation_name, ave_deviation)
#         max_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\max_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#             Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         np.savetxt(max_deviation_name, max_deviation)
#         min_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\min_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#             Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         np.savetxt(min_deviation_name, min_deviation)
#         ave_baseline_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\ave_baseline_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#             Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         np.savetxt(ave_baseline_deviation_name, ave_baseline_deviation)
#         length_geodesic_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\length_geodesic_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#             Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         np.savetxt(length_geodesic_name, length_geodesic)
#         SPnodenum_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\SPnodenum_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#             Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         np.savetxt(SPnodenum_vec_name, SPnodenum_vec, fmt="%i")
#
#
# def distance_inlargeSRGG(N,ED,beta,ExternalSimutime):
#     """
#     :param N:
#     :param ED:
#     :param beta:
#     :param rg:
#     :param ExternalSimutime:
#     :return:
#     for each node pair, we record the ave,max,min of distance from the shortest path to the geodesic,
#     length of the geo distances.
#     The generated network, the selected node pair and all the deviation of both shortest path and baseline nodes will be recorded.
#     """
#     if N> ED:
#         deviation_vec = []  # deviation of all shortest path nodes for all node pairs
#         baseline_deviation_vec = []  # deviation of all shortest path nodes for all node pairs
#         # For each node pair:
#         ave_deviation = []
#         max_deviation = []
#         min_deviation = []
#         ave_baseline_deviation =[]
#         length_geodesic = []
#         SPnodenum_vec =[]
#
#         # load a network
#         FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\network_N{Nn}ED{EDn}Beta{betan}.txt".format(
#             Nn=N, EDn=ED, betan=beta)
#         G = loadSRGGandaddnode(N, FileNetworkName)
#         # load coordinates with noise
#         Coorx = []
#         Coory = []
#
#         FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt".format(
#             Nn=N, EDn=ED, betan=beta)
#         with open(FileNetworkCoorName, "r") as file:
#             for line in file:
#                 if line.startswith("#"):
#                     continue
#                 data = line.strip().split("\t")  # ???????
#                 Coorx.append(float(data[0]))
#                 Coory.append(float(data[1]))
#
#
#         real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
#         print("real ED:", real_avg)
#
#         ave_clu = nx.average_clustering(G)
#         print("clu:",ave_clu)
#
#         components = list(nx.connected_components(G))
#         largest_component = max(components, key=len)
#         LCC_number = len(largest_component)
#         print("LCC", LCC_number)
#
#         # Randomly choose 100 connectede node pairs
#         nodepair_num = 100
#         unique_pairs = find_k_connected_node_pairs(G, nodepair_num)
#         filename_selecetednodepair = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\selected_node_pair_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#             Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         np.savetxt(filename_selecetednodepair, unique_pairs, fmt="%i")
#         components = []
#         largest_component = []
#
#         for node_pair in unique_pairs:
#             nodei = node_pair[0]
#             nodej = node_pair[1]
#             # Find the shortest path nodes
#             SPNodelist = all_shortest_path_node(G, nodei, nodej)
#             SPnodenum = len(SPNodelist)
#             SPnodenum_vec.append(SPnodenum)
#             if SPnodenum>0:
#                 xSource = Coorx[nodei]
#                 ySource = Coory[nodei]
#                 xEnd = Coorx[nodej]
#                 yEnd = Coory[nodej]
#                 length_geodesic.append(distR2(xSource, ySource, xEnd, yEnd))
#                 # Compute deviation for the shortest path of each node pair
#                 deviations_for_a_nodepair = []
#                 for SPnode in SPNodelist:
#                     xMed = Coorx[SPnode]
#                     yMed = Coory[SPnode]
#                     dist, _ = dist_to_geodesic_R2(xMed, yMed, xSource, ySource, xEnd, yEnd)
#                     deviations_for_a_nodepair.append(dist)
#
#                 deviation_vec = deviation_vec+deviations_for_a_nodepair
#
#                 ave_deviation.append(np.mean(deviations_for_a_nodepair))
#                 max_deviation.append(max(deviations_for_a_nodepair))
#                 min_deviation.append(min(deviations_for_a_nodepair))
#
#                 baseline_deviations_for_a_nodepair = []
#                 # compute baseline's deviation
#                 filtered_numbers = [num for num in range(N) if num not in [nodei,nodej]]
#                 base_line_node_index = random.sample(filtered_numbers,SPnodenum)
#
#                 for SPnode in base_line_node_index:
#                     xMed = Coorx[SPnode]
#                     yMed = Coory[SPnode]
#                     dist, _ = dist_to_geodesic_R2(xMed, yMed, xSource, ySource, xEnd, yEnd)
#                     baseline_deviations_for_a_nodepair.append(dist)
#                 ave_baseline_deviation.append(np.mean(baseline_deviations_for_a_nodepair))
#                 baseline_deviation_vec = baseline_deviation_vec + baseline_deviations_for_a_nodepair
#
#         deviation_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\deviation_shortest_path_nodes_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#             Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         np.savetxt(deviation_vec_name, deviation_vec)
#         baseline_deviation_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\deviation_baseline_nodes_num_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#             Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         np.savetxt(baseline_deviation_vec_name, baseline_deviation_vec)
#         # For each node pair:
#         ave_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#             Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         np.savetxt(ave_deviation_name, ave_deviation)
#         max_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\max_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#             Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         np.savetxt(max_deviation_name, max_deviation)
#         min_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\min_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#             Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         np.savetxt(min_deviation_name, min_deviation)
#         ave_baseline_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\ave_baseline_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#             Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         np.savetxt(ave_baseline_deviation_name, ave_baseline_deviation)
#         length_geodesic_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\length_geodesic_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#             Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         np.savetxt(length_geodesic_name, length_geodesic)
#         SPnodenum_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\SPnodenum_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#             Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         np.savetxt(SPnodenum_vec_name, SPnodenum_vec,fmt="%i")
#
#
# def distance_inlargeSRGG_clu(N,ED,beta,rg,ExternalSimutime):
#     """
#     :param N:
#     :param ED:
#     :param beta:
#     :param rg:
#     :param ExternalSimutime:
#     :return:
#     for each node pair, we record the ave,max,min of distance from the shortest path to the geodesic,
#     length of the geo distances.
#     The generated network, the selected node pair and all the deviation of both shortest path and baseline nodes will be recorded.
#     """
#     if N> ED:
#         deviation_vec = []  # deviation of all shortest path nodes for all node pairs
#         baseline_deviation_vec = []  # deviation of all shortest path nodes for all node pairs
#         # For each node pair:
#         ave_deviation = []
#         max_deviation = []
#         min_deviation = []
#         ave_baseline_deviation =[]
#         length_geodesic = []
#         hopcount_vec = []
#         max_dev_node_hopcount = []
#         corresponding_sp_max_dev_node_hopcount = []
#         SPnodenum_vec =[]
#         LCC_vec =[]
#         second_vec = []
#
#         source_folder = "/home/zqiu1/GSPP_SRGG_Dev/NetworkSRGG/"
#         # ?????
#         destination_folder = "/work/zqiu1/"
#         network_template = "network_N{Nn}ED{EDn}Beta{betan}.txt"
#         networkcoordinate_template  = "network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt"
#
#         # load or generate a network
#         try:
#             FileNetworkName = "/work/zqiu1/network_N{Nn}ED{EDn}Beta{betan}.txt".format(
#                 Nn=N, EDn=ED, betan=beta)
#             G = loadSRGGandaddnode(N, FileNetworkName)
#             # load coordinates with noise
#             Coorx = []
#             Coory = []
#
#             FileNetworkCoorName = "/work/zqiu1/network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt".format(
#                 Nn=N, EDn=ED, betan=beta)
#             with open(FileNetworkCoorName, "r") as file:
#                 for line in file:
#                     if line.startswith("#"):
#                         continue
#                     data = line.strip().split("\t")
#                     Coorx.append(float(data[0]))
#                     Coory.append(float(data[1]))
#         # except:
#         #     os.makedirs(destination_folder, exist_ok=True)
#         #     source_file = source_folder+ network_template.format(Nn=N, EDn=ED, betan=beta)
#         #     destination_file = destination_folder+ network_template.format(Nn=N, EDn=ED, betan=beta)
#         #     shutil.copy(source_file, destination_file)
#         #     print(f"Copied: {source_file} -> {destination_file}")
#         #     source_file = source_folder + networkcoordinate_template.format(Nn=N, EDn=ED, betan=beta)
#         #     destination_file = destination_folder + networkcoordinate_template.format(Nn=N, EDn=ED, betan=beta)
#         #     shutil.copy(source_file, destination_file)
#         #     print(f"Copied: {source_file} -> {destination_file}")
#         #
#         #     FileNetworkName = "/work/zqiu1/network_N{Nn}ED{EDn}Beta{betan}.txt".format(
#         #         Nn=N, EDn=ED, betan=beta)
#         #     G = loadSRGGandaddnode(N, FileNetworkName)
#         #     # load coordinates with noise
#         #     Coorx = []
#         #     Coory = []
#         #
#         #     FileNetworkCoorName = "/work/zqiu1/network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt".format(
#         #         Nn=N, EDn=ED, betan=beta)
#         #     with open(FileNetworkCoorName, "r") as file:
#         #         for line in file:
#         #             if line.startswith("#"):
#         #                 continue
#         #             data = line.strip().split("\t")
#         #             Coorx.append(float(data[0]))
#         #             Coory.append(float(data[1]))
#
#         except:
#             G, Coorx, Coory = R2SRGG(N, ED, beta, rg)
#             # if ExternalSimutime == 0:
#             #     FileNetworkName = "/work/zqiu1/network_N{Nn}ED{EDn}Beta{betan}.txt".format(
#             #         Nn=N, EDn=ED, betan=beta)
#             #     nx.write_edgelist(G, FileNetworkName)
#             #     FileNetworkCoorName = "/work/zqiu1/network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt".format(
#             #         Nn=N, EDn=ED, betan=beta)
#             #     with open(FileNetworkCoorName, "w") as file:
#             #         for data1, data2 in zip(Coorx, Coory):
#             #             file.write(f"{data1}\t{data2}\n")
#
#         real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
#         print("real ED:", real_avg)
#
#         # Randomly choose 100 connectede node pairs
#         nodepair_num = 100
#         unique_pairs = find_k_connected_node_pairs(G, nodepair_num)
#         filename_selecetednodepair = "/work/zqiu1/selected_node_pair_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#             Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         np.savetxt(filename_selecetednodepair, unique_pairs, fmt="%i")
#
#         connected_components = sorted(nx.connected_components(G), key=len, reverse=True)
#         if len(connected_components) > 1:
#             largest_largest_component = connected_components[0]
#             largest_largest_size = len(largest_largest_component)
#             LCC_vec.append(largest_largest_size)
#             # ?????????????????
#             second_largest_component = connected_components[1]
#             second_largest_size = len(second_largest_component)
#             second_vec.append(second_largest_size)
#         if ExternalSimutime==0:
#             filefolder_name = "/work/zqiu1/"
#             LCCname = filefolder_name + "LCC_2LCC_N{Nn}ED{EDn}beta{betan}.txt".format(
#                 Nn=N, EDn=ED, betan=beta)
#             with open(LCCname, "w") as file:
#                 file.write("# LCC\tSECLCC\n")
#                 for name, age in zip(LCC_vec, second_vec):
#                     file.write(f"{name}\t{age}\n")
#
#         for node_pair in unique_pairs:
#             print("node_pair:",node_pair)
#             nodei = node_pair[0]
#             nodej = node_pair[1]
#             # Find the shortest path nodes
#             SPNodelist = all_shortest_path_node(G, nodei, nodej)
#             hopcount_vec.append(nx.shortest_path_length(G, nodei, nodej))
#             SPnodenum = len(SPNodelist)
#             SPnodenum_vec.append(SPnodenum)
#             if SPnodenum>0:
#                 xSource = Coorx[nodei]
#                 ySource = Coory[nodei]
#                 xEnd = Coorx[nodej]
#                 yEnd = Coory[nodej]
#                 length_geodesic.append(distR2(xSource, ySource, xEnd, yEnd))
#                 # Compute deviation for the shortest path of each node pair
#                 deviations_for_a_nodepair = []
#                 for SPnode in SPNodelist:
#                     xMed = Coorx[SPnode]
#                     yMed = Coory[SPnode]
#                     dist, _ = dist_to_geodesic_R2(xMed, yMed, xSource, ySource, xEnd, yEnd)
#                     deviations_for_a_nodepair.append(dist)
#
#                 deviation_vec = deviation_vec+deviations_for_a_nodepair
#
#                 ave_deviation.append(np.mean(deviations_for_a_nodepair))
#                 max_deviation.append(max(deviations_for_a_nodepair))
#                 min_deviation.append(min(deviations_for_a_nodepair))
#
#                 # max hopcount
#                 # max_value = max(deviations_for_a_nodepair)
#                 # max_index = deviations_for_a_nodepair.index(max_value)
#                 # maxhop_node_index = SPNodelist[max_index]
#                 # max_dev_node_hopcount.append(hopcount_node(G, nodei, nodej, maxhop_node_index))
#                 # corresponding_sp_max_dev_node_hopcount.append(nx.shortest_path_length(G, nodei, nodej))
#
#                 # baseline: random selected
#                 # baseline_deviations_for_a_nodepair = []
#                 # # compute baseline's deviation
#                 # filtered_numbers = [num for num in range(N) if num not in [nodei,nodej]]
#                 # base_line_node_index = random.sample(filtered_numbers,SPnodenum)
#                 #
#                 # for SPnode in base_line_node_index:
#                 #     xMed = Coorx[SPnode]
#                 #     yMed = Coory[SPnode]
#                 #     dist, _ = dist_to_geodesic_R2(xMed, yMed, xSource, ySource, xEnd, yEnd)
#                 #     baseline_deviations_for_a_nodepair.append(dist)
#                 # ave_baseline_deviation.append(np.mean(baseline_deviations_for_a_nodepair))
#                 # baseline_deviation_vec = baseline_deviation_vec + baseline_deviations_for_a_nodepair
#
#         deviation_vec_name = "/work/zqiu1/deviation_shortest_path_nodes_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#             Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         np.savetxt(deviation_vec_name, deviation_vec)
#         # baseline_deviation_vec_name = "/work/zqiu1/deviation_baseline_nodes_num_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#         #     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         # np.savetxt(baseline_deviation_vec_name, baseline_deviation_vec)
#         # For each node pair:
#         ave_deviation_name = "/work/zqiu1/ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#             Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         np.savetxt(ave_deviation_name, ave_deviation)
#         # max_deviation_name = "/work/zqiu1/max_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#         #     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         # np.savetxt(max_deviation_name, max_deviation)
#         # min_deviation_name = "/work/zqiu1/min_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#         #     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         # np.savetxt(min_deviation_name, min_deviation)
#         # ave_baseline_deviation_name = "/work/zqiu1/ave_baseline_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#         #     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         # np.savetxt(ave_baseline_deviation_name, ave_baseline_deviation)
#         length_geodesic_name = "/work/zqiu1/length_geodesic_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#             Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         np.savetxt(length_geodesic_name, length_geodesic)
#         SPnodenum_vec_name = "/work/zqiu1/SPnodenum_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#             Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         np.savetxt(SPnodenum_vec_name, SPnodenum_vec,fmt="%i")
#         hopcount_Name = "/work/zqiu1/hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
#                     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         np.savetxt(hopcount_Name, hopcount_vec)
#
#         # max_dev_node_hopcount_name = "/work/zqiu1/max_dev_node_hopcount_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#         #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         # np.savetxt(max_dev_node_hopcount_name, max_dev_node_hopcount, fmt="%i")
#         # max_dev_node_hopcount_name2 = "/work/zqiu1/sphopcountmax_dev_node_hopcount_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#         #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         # np.savetxt(max_dev_node_hopcount_name2, corresponding_sp_max_dev_node_hopcount, fmt="%i")
#
# def distance_inlargeSRGG_clu_cc(N, ED, beta, cc, ExternalSimutime):
#     """
#     :param N:
#     :param ED:
#     :param ExternalSimutime:
#     :return:
#     for each node pair, we record the ave,max,min of distance from the shortest path to the geodesic,
#     length of the geo distances.
#     The generated network, the selected node pair and all the deviation of both shortest path and baseline nodes will be recorded.
#     """
#     if N> ED:
#         deviation_vec = []  # deviation of all shortest path nodes for all node pairs
#         baseline_deviation_vec = []  # deviation of all shortest path nodes for all node pairs
#         # For each node pair:
#         ave_deviation = []
#         max_deviation = []
#         min_deviation = []
#         ave_baseline_deviation =[]
#         length_geodesic = []
#         hopcount_vec = []
#         SPnodenum_vec =[]
#
#         # load a network
#         FileNetworkName = "/home/zqiu1/GSPP/SSRGGpy/R2/distribution/NetworkSRGG/network_N{Nn}ED{EDn}CC{betan}.txt".format(
#             Nn=N, EDn=ED, betan=cc)
#         G = loadSRGGandaddnode(N, FileNetworkName)
#         # load coordinates with noise
#         Coorx = []
#         Coory = []
#
#         FileNetworkCoorName = "/home/zqiu1/GSPP/SSRGGpy/R2/distribution/NetworkSRGG/network_coordinates_N{Nn}ED{EDn}CC{betan}.txt".format(
#             Nn=N, EDn=ED, betan=cc)
#         with open(FileNetworkCoorName, "r") as file:
#             for line in file:
#                 if line.startswith("#"):
#                     continue
#                 data = line.strip().split("\t")  # ???????
#                 Coorx.append(float(data[0]))
#                 Coory.append(float(data[1]))
#
#
#         real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
#         print("real ED:", real_avg)
#
#         ave_clu = nx.average_clustering(G)
#         print("clu:",ave_clu)
#
#         components = list(nx.connected_components(G))
#         largest_component = max(components, key=len)
#         LCC_number = len(largest_component)
#         print("LCC", LCC_number)
#
#         # Randomly choose 100 connectede node pairs
#         nodepair_num = 10
#         unique_pairs = find_k_connected_node_pairs(G, nodepair_num)
#         filename_selecetednodepair = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\selected_node_pair_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#             Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         np.savetxt(filename_selecetednodepair, unique_pairs, fmt="%i")
#         components = []
#         largest_component = []
#
#         for node_pair in unique_pairs:
#             nodei = node_pair[0]
#             nodej = node_pair[1]
#             # Find the shortest path nodes
#             SPNodelist = all_shortest_path_node(G, nodei, nodej)
#             hopcount_vec.append(nx.shortest_path_length(G, nodei, nodej))
#             SPnodenum = len(SPNodelist)
#             SPnodenum_vec.append(SPnodenum)
#             if SPnodenum>0:
#                 xSource = Coorx[nodei]
#                 ySource = Coory[nodei]
#                 xEnd = Coorx[nodej]
#                 yEnd = Coory[nodej]
#                 length_geodesic.append(distR2(xSource, ySource, xEnd, yEnd))
#                 # Compute deviation for the shortest path of each node pair
#                 deviations_for_a_nodepair = []
#                 for SPnode in SPNodelist:
#                     xMed = Coorx[SPnode]
#                     yMed = Coory[SPnode]
#                     dist, _ = dist_to_geodesic_R2(xMed, yMed, xSource, ySource, xEnd, yEnd)
#                     deviations_for_a_nodepair.append(dist)
#
#                 deviation_vec = deviation_vec+deviations_for_a_nodepair
#
#                 ave_deviation.append(np.mean(deviations_for_a_nodepair))
#                 max_deviation.append(max(deviations_for_a_nodepair))
#                 min_deviation.append(min(deviations_for_a_nodepair))
#
#                 baseline_deviations_for_a_nodepair = []
#                 # compute baseline's deviation
#                 filtered_numbers = [num for num in range(N) if num not in [nodei,nodej]]
#                 base_line_node_index = random.sample(filtered_numbers,SPnodenum)
#
#                 for SPnode in base_line_node_index:
#                     xMed = Coorx[SPnode]
#                     yMed = Coory[SPnode]
#                     dist, _ = dist_to_geodesic_R2(xMed, yMed, xSource, ySource, xEnd, yEnd)
#                     baseline_deviations_for_a_nodepair.append(dist)
#                 ave_baseline_deviation.append(np.mean(baseline_deviations_for_a_nodepair))
#                 baseline_deviation_vec = baseline_deviation_vec + baseline_deviations_for_a_nodepair
#
#         deviation_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\deviation_shortest_path_nodes_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#             Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         np.savetxt(deviation_vec_name, deviation_vec)
#         baseline_deviation_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\deviation_baseline_nodes_num_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#             Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         np.savetxt(baseline_deviation_vec_name, baseline_deviation_vec)
#         # For each node pair:
#         ave_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#             Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         np.savetxt(ave_deviation_name, ave_deviation)
#         max_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\max_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#             Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         np.savetxt(max_deviation_name, max_deviation)
#         min_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\min_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#             Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         np.savetxt(min_deviation_name, min_deviation)
#         ave_baseline_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\ave_baseline_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#             Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         np.savetxt(ave_baseline_deviation_name, ave_baseline_deviation)
#         length_geodesic_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\length_geodesic_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#             Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         np.savetxt(length_geodesic_name, length_geodesic)
#         SPnodenum_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\SPnodenum_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#             Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
#         np.savetxt(SPnodenum_vec_name, SPnodenum_vec,fmt="%i")
#         hopcount_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\hopcount_sp_ED{EDn}Beta{betan}Simu{ST}.txt".format(
#             EDn=ED, betan=beta, ST=ExternalSimutime)
#         np.savetxt(hopcount_Name, hopcount_vec)
#
#
# def distance_inSRGG(network_size_index, average_degree_index, beta_index, ExternalSimutime):
#     Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
#     kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
#     betavec = [2.2, 4, 8, 16, 32, 64, 128]
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
#     if N>100:
#         distance_inlargeSRGG(N, ED, beta, ExternalSimutime)
#     else:
#         # Random select nodepair_num nodes in the largest connected component
#         distance_insmallSRGG(N, ED, beta, rg, ExternalSimutime)
#
# def distance_inSRGG_clu(network_size_index, average_degree_index, beta_index, ExternalSimutime):
#     Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
#     # kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
#     # kvec = [2,2.5,3,3.5,4,4.5,5,5.5,6]
#     kvec = [5.0, 5.6, 6.0, 10, 16, 27, 44, 72, 118, 193]
#     # kvec = np.arange(2, 6.1, 0.2)
#     # kvec = [round(a, 1) for a in kvec]
#     # kvec = np.arange(6.5, 9.6, 0.5)
#     # kvec = [round(a, 1) for a in kvec]
#     # kvec2 = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
#     # kvec = kvec + kvec2
#     # kvec = [5,10,20]
#     # kvec = np.arange(2.5, 5, 0.1)
#     # kvec = [round(a, 1) for a in kvec]
#     # kvec = [8,12,20,34,56,92]
#
#     kvec = [15,16]
#
#     # kvec = [5,20]
#     betavec = [2.2, 4, 8, 16, 32, 64, 128]
#     # betavec = [2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7,3.8,3.9]
#     # betavec = [2.2, 3.0, 4.2, 5.9, 8.3, 11.7, 16.5, 23.2, 32.7, 46.1, 64.9, 91.5, 128.9, 181.7, 256]
#
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
#         distance_inlargeSRGG_clu(N, ED, beta,rg, ExternalSimutime)
#     else:
#         # Random select nodepair_num nodes in the largest connected component
#         distance_insmallSRGG(N, ED, beta, rg, ExternalSimutime)
#
# def distance_inSRGG_withEDCC(network_size_index, average_degree_index, cc_index, ExternalSimutime):
#     Nvec = [10, 100, 200, 500, 1000, 10000]
#     kvec = list(range(2, 20)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
#     cc_vec = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
#     betavec = [2.55, 3.2, 3.99, 5.15, 7.99, 300]
#
#     random.seed(ExternalSimutime)
#     N = Nvec[network_size_index]
#     ED = kvec[average_degree_index]
#     beta = betavec[cc_index]
#     C_G = cc_vec[cc_index]
#     print("input para:", (N, ED, beta,C_G))
#
#     rg = RandomGenerator(-12)
#     rseed = random.randint(0, 100)
#     for i in range(rseed):
#         rg.ran1()
#
#     # for large network, we only generate one network and randomly selected 1,000 node pair.
#     # for small network, we generate 100 networks and selected all the node pair in the LCC
#     if N > 100:
#         distance_inlargeSRGG_clu_cc(N, ED, beta, C_G, ExternalSimutime)
#     else:
#         # Random select nodepair_num nodes in the largest connected component
#         distance_insmallSRGG(N, ED, beta, rg, ExternalSimutime)
#


def compute_edge_Euclidean_length(nodes,nodet,Coorx,Coory):
    xSource = Coorx[nodes]
    ySource = Coory[nodes]
    xEnd = Coorx[nodet]
    yEnd = Coory[nodet]
    edge_length = distR2(xSource, ySource, xEnd, yEnd)
    return edge_length


def distance_insmallSRGG_oneSP_clu(N, ED, beta, rg, ExternalSimutime):
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

    # folder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\"
    folder_name = "/home/qzh/data/"
    if N == 100:
        simu_times = 10
    else:
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
                    flag = 1
                    real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)

        print("real ED:", real_avg)
        real_ave_degree.append(real_avg)
        # ave_clu = nx.average_clustering(G)
        # print("clu:", ave_clu)
        # clustering_coefficient.append(ave_clu)
        # components = list(nx.connected_components(G))
        # largest_component = max(components, key=len)
        # LCC_number = len(largest_component)
        # print("LCC", LCC_number)
        # LCC_num.append(LCC_number)

        # pick up all the node pairs in the LCC and save them in the unique_pairs
        unique_pairs = find_all_connected_node_pairs(G)
        count = 0
        for node_pair in unique_pairs:
            nodei = node_pair[0]
            nodej = node_pair[1]
            # Find the shortest path nodes
            SPNodelist = nx.shortest_path(G, nodei, nodej)
            SPnodenum = len(SPNodelist)-2
            SPnodenum_vec.append(SPnodenum)

            if SPnodenum > 0:
                # hopcount of the SP
                SP_hopcount_fornodepair = nx.shortest_path_length(G, nodei, nodej)
                SP_hopcount.append(SP_hopcount_fornodepair)

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
                hop_for_a_nodepair = []
                for SPnode in SPNodelist[1:len(SPNodelist)-1]:
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

    real_ave_degree_name = folder_name+"real_ave_degree_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(real_ave_degree_name, real_ave_degree)
    # LCC_num_name = folder_name+"LCC_num_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
    #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    # np.savetxt(LCC_num_name, LCC_num, fmt="%i")
    # clustering_coefficient_name = folder_name+"clustering_coefficient_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
    #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    # np.savetxt(clustering_coefficient_name, clustering_coefficient)
    # For each node pair:
    ave_deviation_name = folder_name+"ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(ave_deviation_name, ave_deviation)
    max_deviation_name = folder_name+"max_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(max_deviation_name, max_deviation)
    min_deviation_name = folder_name+"min_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(min_deviation_name, min_deviation)

    ave_baseline_deviation_name = folder_name+"ave_baseline_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(ave_baseline_deviation_name, ave_baseline_deviation)

    length_geodesic_name = folder_name+"length_geodesic_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(length_geodesic_name, length_geodesic)

    SPnodenum_vec_name = folder_name+"SPnodenum_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(SPnodenum_vec_name, SPnodenum_vec, fmt="%i")
    nodepairs_for_eachgraph_vec_name = folder_name+"nodepairs_for_eachgraph_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
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


    max_dev_node_hopcount_name = folder_name+"max_dev_node_hopcount_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(max_dev_node_hopcount_name, max_dev_node_hopcount, fmt="%i")




def distance_inlargeSRGG_oneSP_clu(N, ED, beta, rg, ExternalSimutime):
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
    ave_baseline_deviation =[]
    length_geodesic = []
    length_edge_vec = []
    hopcount_vec = []
    max_dev_node_hopcount = []
    corresponding_sp_max_dev_node_hopcount = []
    SPnodenum_vec =[]
    LCC_vec =[]
    second_vec = []
    delta_vec = []  # delta is the Euclidean geometric distance between two nodes i,k, where i,k is the neighbours of j

    folder_name1 = "/home/qzh/network/"
    folder_name = "/home/qzh/data/"
    try:
        FileNetworkName = folder_name1+"network_N{Nn}ED{EDn}Beta{betan}.txt".format(
            Nn=N, EDn=ED, betan=beta)
        G = loadSRGGandaddnode(N, FileNetworkName)
        # load coordinates with noise
        Coorx = []
        Coory = []

        FileNetworkCoorName = folder_name1+"network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt".format(
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
            SPnodenum = len(SPNodelist)-2
            SPnodenum_vec.append(SPnodenum)
            if SPnodenum>0:
                hopcount_vec.append(nx.shortest_path_length(G, nodei, nodej))

                # compute the length of the edges
                length_edge_for_anodepair = []
                shortest_path_edges = list(zip(SPNodelist[:-1], SPNodelist[1:]))
                for (nodes, nodet) in shortest_path_edges:
                    d_E = compute_edge_Euclidean_length(nodes,nodet,Coorx,Coory)
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
                for SPnode in SPNodelist[1:len(SPNodelist)-1]:
                    xMed = Coorx[SPnode]
                    yMed = Coory[SPnode]
                    dist, _ = dist_to_geodesic_R2(xMed, yMed, xSource, ySource, xEnd, yEnd)
                    deviations_for_a_nodepair.append(dist)

                deviation_vec = deviation_vec+deviations_for_a_nodepair

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
    deviation_vec_name = folder_name+"deviation_shortest_path_nodes_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(deviation_vec_name, deviation_vec)
    # baseline_deviation_vec_name = folder_name+"deviation_baseline_nodes_num_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
    #     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
    # np.savetxt(baseline_deviation_vec_name, baseline_deviation_vec)
    # For each node pair:
    ave_deviation_name = folder_name+"ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
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
    length_geodesic_name = folder_name+"length_geodesic_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(length_geodesic_name, length_geodesic)
    SPnodenum_vec_name = folder_name+"SPnodenum_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(SPnodenum_vec_name, SPnodenum_vec,fmt="%i")
    hopcount_Name = folder_name+"hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
                Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
    np.savetxt(hopcount_Name, hopcount_vec,fmt="%i")
    delta_Name = folder_name+"delta_N{Nn}ED{EDn}beta{betan}Simu{ST}.txt".format(
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


def distance_inSRGG_oneSP(network_size, input_expected_degree, beta, ExternalSimutime):
    """
    Only one shortest path is chosen to test how the deviation changes with different network parameters
    :param network_size_index:
    :param average_degree_index:
    :param beta_index:
    :param ExternalSimutime:
    :return:
    """

    random.seed(ExternalSimutime)
    N = network_size
    # if N ==1000:
    #     kvec = [2, 3, 4, 5, 8, 12, 20, 31, 49, 77, 120, 188, 296, 468, 739, 1166, 1836, 2842] # for N = 1000
    # else:
    #     kvec = [2, 3, 4, 6, 11, 20, 37, 67, 121, 218, 392, 705, 1267, 2275, 4086, 7336, 13169, 23644,29999]  # for N = 1000
    ED = input_expected_degree
    beta = beta
    print("input para:", (N, ED, beta),flush=True)

    rg = RandomGenerator(-12)
    rseed = random.randint(0, 100)
    for i in range(rseed):
        rg.ran1()

    # for large network, we only generate one network and randomly selected 1,000 node pair.
    # for small network, we generate 100 networks and selected all the node pair in the LCC
    if N > 200:
        distance_inlargeSRGG_oneSP_clu(N, ED, beta, rg, ExternalSimutime)
    else:
        # Random select nodepair_num nodes in the largest connected component
        distance_insmallSRGG_oneSP_clu(N, ED, beta, rg, ExternalSimutime)

def compute_deviation():
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
    # Nvec = [10, 100, 1000, 10000]
    # beta_vec = [4]
    # kvec_dict = {
    #     100: [2, 3, 4, 5, 6, 8, 10, 12, 14, 17, 22, 27, 33, 40, 49, 60, 73, 89, 113, 149, 198, 260, 340, 446, 584,
    #           762, 993, 1292, 1690, 2276, 3142, 4339],
    #     1000: [2, 3, 4, 5, 6, 7, 8, 11, 15, 20, 28, 40, 58, 83, 118, 169, 241, 344, 490, 700, 999, 1425, 2033, 2900,
    #            4139, 5909, 8430, 12039, 17177, 24510, 34968, 49887, 71168],
    #     10000: [2.2, 2.8, 3.0, 3.4, 3.8, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276,
    #             3727, 6105,
    #             9999, 16479, 27081, 44767, 73534, 121205, 199999]}

    Nvec = [100, 215, 464, 1000, 2154, 4642]
    beta_vec=[1024]
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

    for N in Nvec:
        if N ==10:
            input_ED_vec = list(range(2, 10)) + [10, 12, 15, 18, 22, 27, 33, 40, 49, 60, 73, 89, 99]  # FOR N =10
        else:
            input_ED_vec = kvec_dict[N]
        for inputED in input_ED_vec:
            for beta in beta_vec:
                tasks.append((N, inputED, beta, 0))

    with mp.Pool(processes=10) as pool:
        pool.starmap(distance_inSRGG_oneSP, tasks)



    # Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """
    Step1 run simulations with different beta and input AVG for one sp case
    """
    compute_deviation()


    """
    SMALL NETWROK CHECK
    """
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
    # # 提取路径中的所有边
    # shortest_path_edges = list(zip(shortest_path[:-1], shortest_path[1:]))
    # for (nodei, nodej) in shortest_path_edges:
    #     print(nodei)
    #     print(nodej)
    #
    # print("最短路径:", shortest_path)
    # print("路径上的所有边:", shortest_path_edges)


