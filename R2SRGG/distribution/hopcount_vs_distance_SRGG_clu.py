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

from R2SRGG import R2SRGG, distR2, dist_to_geodesic_R2, loadSRGGandaddnode, R2SRGG_withgivennodepair
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


def hop_vs_geolength_inlargeSRGG_clu(N, ED, beta, rg, ExternalSimutime, geodesic_distance_AB, x_A,
                                     y_A, x_B, y_B):
    """
    :param N:
    :param ED:
    :param ExternalSimutime:
    :return:
    we investigate the distribution of the hopcount for node pair given geometric distance(Euclidean)
    """
    if N> ED:
        hopcount_vec = []
        # load a network
        # Randomly generate 10 networks
        Network_generate_time = 200

        for network_index in range(Network_generate_time):
            # N = 100 # FOR TEST
            G, Coorx, Coory = R2SRGG_withgivennodepair(N, ED, beta, rg, x_A, y_A, x_B, y_B)
            nodei = N-2
            nodej = N-1
            # Find the shortest path nodes
            if nx.has_path(G, nodei, nodej):
                hopcount_vec.append(nx.shortest_path_length(G, nodei, nodej))
        hopcount_Name = "Givendistancehopcount_sp_N{Nn}ED{EDn}beta{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
            Nn = N,EDn=ED, betan=beta, ST=ExternalSimutime, Geodistance = geodesic_distance_AB)
        np.savetxt(hopcount_Name, hopcount_vec)


def hop_vs_geolength_inSRGG_clu(Geodistance_index, ExternalSimutime):
    random.seed(ExternalSimutime)
    N = 10000
    ED = 10
    beta = 4
    distance_list = [[0.49, 0.5, 0.5, 0.5], [0.35, 0.5, 0.65, 0.5],[0.25, 0.5, 0.75, 0.5]]
    x_A = distance_list[Geodistance_index][0]
    y_A = distance_list[Geodistance_index][1]
    x_B = distance_list[Geodistance_index][2]
    y_B = distance_list[Geodistance_index][3]
    geodesic_distance_AB = x_B - x_A
    geodesic_distance_AB = round(geodesic_distance_AB, 2)

    rg = RandomGenerator(-12)
    rseed = random.randint(0, 100)
    for i in range(rseed):
        rg.ran1()
    print("input para:", (N, ED, beta, geodesic_distance_AB, ExternalSimutime))

    # for large network, we only generate one network and randomly selected 1,000 node pair.
    # for small network, we generate 100 networks and selected all the node pair in the LCC
    if N > 100:
        hop_vs_geolength_inlargeSRGG_clu(N, ED, beta, rg, ExternalSimutime, geodesic_distance_AB, x_A,
                                                    y_A, x_B, y_B)
    else:
        # Random select nodepair_num nodes in the largest connected component
        # distance_insmallSRGG(N, ED, beta, rg, ExternalSimutime)
        pass
if __name__ == '__main__':
    """
    Step 2 try to see with different beta and input AVG 
    """
    # Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    # kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    # kvec = [5, 10, 20]
    # # betavec = [2.1, 4, 8, 16, 32, 64, 128]
    # betavec = [2.2, 2.4, 2.5, 2.6, 2.8, 3, 3.25, 3.5, 3.75, 5, 6, 7, 10, 12]
    #
    # for N_index in [6]:
    #     for ED_index in range(3):
    #         for beta_index in range(14):
    #             distance_inSRGG(N_index, ED_index, beta_index, 0)
    #
    #
    # distance_inSRGG_clu(6, int(ED_index), int(beta_index), int(ExternalSimutime))
    geolength_index = sys.argv[1]
    ExternalSimutime = sys.argv[2]
    hop_vs_geolength_inSRGG_clu(int(geolength_index), int(ExternalSimutime))

