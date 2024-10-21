# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 24-10-2024
This .m is simulation for Maksim's mean field explanation.
We generate SRGG and see how the average link distance and radius change following the changement of beta and ED
2. ave geo distance changed with different ed and beta
3. radius(max ave geo distance) changed with different ed and beta
"""
import numpy as np
import networkx as nx
import random
import math
import sys

# from R2SRGG import R2SRGG, distR2, dist_to_geodesic_R2, loadSRGGandaddnode, R2SRGG_withgivennodepair
from R2SRGG.R2SRGG import R2SRGG, distR2, dist_to_geodesic_R2, loadSRGGandaddnode, R2SRGG_withgivennodepair
from SphericalSoftRandomGeomtricGraph import RandomGenerator
from main import all_shortest_path_node, find_k_connected_node_pairs, find_all_connected_node_pairs


def Ave_distance_link_and_radius(N, ED, beta, ExternalSimutime):
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
        # load a network

        # N = 100 # FOR TEST
        # G, Coorx, Coory = R2SRGG(N, ED, beta, rg)

        # # # load a network
        # FileNetworkName = "/home/zqiu1/GSPP/SSRGGpy/R2/distribution/NetworkSRGG/network_N{Nn}ED{EDn}Beta{betan}.txt".format(
        #     Nn=N, EDn=ED, betan=beta)
        # G = loadSRGGandaddnode(N, FileNetworkName)
        # # load coordinates with noise
        # Coorx = []
        # Coory = []
        #
        # FileNetworkCoorName = "/home/zqiu1/GSPP/SSRGGpy/R2/distribution/NetworkSRGG/network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt".format(
        #     Nn=N, EDn=ED, betan=beta)
        # with open(FileNetworkCoorName, "r") as file:
        #     for line in file:
        #         if line.startswith("#"):
        #             continue
        #         data = line.strip().split("\t")  # 使用制表符分割
        #         Coorx.append(float(data[0]))
        #         Coory.append(float(data[1]))

        # # load a network locally
        try:
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
        except:
            rg = RandomGenerator(-12)
            rseed = random.randint(0, 100)
            for i in range(rseed):
                rg.ran1()
            G, Coorx, Coory = R2SRGG(N, ED, beta, rg)
            FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\network_N{Nn}ED{EDn}Beta{betan}.txt".format(
                                Nn=N, EDn=ED, betan=beta)
            nx.write_edgelist(G, FileNetworkName)
            FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt".format(
                Nn=N, EDn=ED, betan=beta)
            with open(FileNetworkCoorName, "w") as file:
                for data1, data2 in zip(Coorx, Coory):
                    file.write(f"{data1}\t{data2}\n")

        # Average_distance_of_the_graph
        for edge in G.edges():
            node_i, node_j = edge
            xSource = Coorx[node_i]
            ySource = Coory[node_i]
            xEnd = Coorx[node_j]
            yEnd = Coory[node_j]
            G[node_i][node_j]['weight'] = distR2(xSource, ySource, xEnd, yEnd)

        max_weights = {}
        for node in G.nodes:
            if nx.degree(G,node)>1:
                # 获取与该节点连接的所有边及其权重
                edges = G.edges(node, data='weight')
                # 获取最大权重
                max_weight = max(weight for _, _, weight in edges)
                max_weights[node] = max_weight


        radius_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ave_distance_link_radius\\radius_N{Nn}ED{EDn}beta{betan}Simu{ST}.txt".format(
            Nn = N,EDn=ED, betan=beta, ST=ExternalSimutime)
        with open(radius_name, "w") as f:
            for key, value in max_weights.items():
                f.write(f"{key}: {value:.4f}\n")
        ave_geodistance_link_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ave_distance_link_radius\\linkweight_N{Nn}ED{EDn}beta{betan}Simu{ST}.txt".format(
            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
        nx.write_edgelist(G,ave_geodistance_link_name,data=['weight'])
        # G  = nx.read_edgelist(ave_geodistance_link_name, nodetype=int, data=(("weight", float),))
        # total_weight = 0
        # edge_count = 0
        # for u, v, data in G.edges(data=True):
        #     total_weight += data['weight']
        #     edge_count += 1
        #
        # # 3. 计算平均权重
        # if edge_count > 0:
        #     average_weight = total_weight / edge_count
        #     print(average_weight)
        # else:
        #     average_weight = 0

    # Press the green button in the gutter to run the script.
if __name__ == '__main__':
    for ED in EDvec:
        for beta in betavec:
            Ave_distance_link_and_radius(10000, ED, beta, 0)