# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 20-8-2024
This script investigates the local optimum of the average distance from the shortest path to the geodesic in SRGG.
We have already seen that with the increase of the average degree, the average distance goes up, then down, then up agian.
We also want to study what will happen if we change the beta, i.e. the parameter that determines clustering coefficient
"""
import itertools

import numpy as np
import networkx as nx
import random
import math

from R2SRGG.R2SRGG import R2SRGG
from SphericalSoftRandomGeomtricGraph import RandomGenerator


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
                print("input para:", (N, ED, beta))
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


def distance_insmallSRGG(N, ED, beta,rg):
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
        length_geodesic = []
        simu_times = 100
        for simu_index in range(simu_times):
            G, Coorx, Coory = R2SRGG(N, ED, beta, rg)
            real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
            print("real ED:", real_avg)
            real_ave_degree.append(real_avg)
            ave_clu = nx.average_clustering(G)
            print("clu:", )
            clustering_coefficient.append(ave_clu)
            components = list(nx.connected_components(G))
            largest_component = max(components, key=len)
            LCC_number = len(largest_component)
            print("LCC", LCC_number)
            LCC_num.append(LCC_number)

            # pick up all the node pairs in the LCC and save them in the unique_pairs
            nodes = list(largest_component)
            unique_pairs = set(tuple(sorted(pair)) for pair in itertools.combinations(nodes, 2))
            count = 0
            for node_pair in unique_pairs:
                count = count+1
                nodei = node_pair[0]
                nodej = node_pair[1]








def distance_inlargeSRGG(N,ED,beta,rg):
    pass


def avedistance_inSRGG(network_size_index, average_degree_index, beta_index, ExternalSimutime):
    Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    betavec = [2.1, 4, 8, 16, 32, 64, 128]

    random.seed(ExternalSimutime)
    rg = RandomGenerator(-12)
    rseed = random.randint(0, 100)
    print(rseed)
    for i in range(rseed):
        rg.ran1()

    simu_times = 100
    N = Nvec[network_size_index]
    ED = kvec[average_degree_index]
    beta = betavec[beta_index]



    # for large network, we only generate one network and randomly selected 1,000 node pair.
    # for small network, we generate 100 networks and selected all the node pair in the LCC
    if N>100:
        distance_inlargeSRGG(N,ED,beta,rg)
    else:
        # Random select nodepair_num nodes in the largest connected component
        distance_insmallSRGG(N, ED, beta, rg)






    # Press the green button in the gutter to run the script.
if __name__ == '__main__':
    generate_r2SRGG()

