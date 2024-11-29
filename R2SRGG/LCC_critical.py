# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 7-11-2024
"""
import numpy as np
import networkx as nx
import random
import json
import math
import sys

# from R2SRGG import R2SRGG, distR2, dist_to_geodesic_R2, loadSRGGandaddnode, R2SRGG_withgivennodepair,dist_to_geodesic_perpendicular_R2
from R2SRGG import R2SRGG, distR2, dist_to_geodesic_R2, loadSRGGandaddnode, R2SRGG_withgivennodepair, \
    dist_to_geodesic_perpendicular_R2
from SphericalSoftRandomGeomtricGraph import RandomGenerator
from main import all_shortest_path_node, find_k_connected_node_pairs, find_all_connected_node_pairs


def LCC_critical(N, ED,beta,simutime):
    """
    In this function we try to find the critical point of the largest LCC
    :param beta:
    :return:
    """
    random.seed(simutime)
    rg = RandomGenerator(-12)
    rseed = random.randint(0, 1000)
    for i in range(rseed):
        rg.ran1()
    xA = 0.25
    yA = 0.25
    xB = 0.75
    yB = 0.75
    LCC_vec = []
    second_vec = []
    for network_index in range(100):
        # print(network_index)
        G, coorx, coory = R2SRGG_withgivennodepair(N, ED, beta, rg, xA, yA, xB, yB)
        connected_components = sorted(nx.connected_components(G), key=len, reverse=True)
        if len(connected_components) > 1:
            largest_largest_component = connected_components[0]
            largest_largest_size = len(largest_largest_component)
            LCC_vec.append(largest_largest_size)
            # 获取第二大连通分量的节点集合和大小
            second_largest_component = connected_components[1]
            second_largest_size = len(second_largest_component)
            second_vec.append(second_largest_size)
        else:
            second_vec.append(0)
        # print(LCC_vec)
        # print(second_vec)
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\givendistance\\LCC\\"
    LCCname = filefolder_name + "LCC_2LCC_N{Nn}ED{EDn}beta{betan}xA{xA}yA{yA}xB{xB}yB{yB}simu{simu}.txt".format(
        Nn=N, EDn=ED, betan=beta, xA=xA, yA=yA, xB=xB, yB=yB,simu = simutime)
    with open(LCCname, "w") as file:
        file.write("# LCC\tSECLCC\n")  # 使用制表符分隔列
        # 写入数据
        for name, age in zip(LCC_vec, second_vec):
            file.write(f"{name}\t{age}\n")



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """
    An example
    """
    # # LCC_critical(6, 4, 0)
    """
    for cluster
    """
    N = 10000
    beta_vec = [2.2,64,128]
    # input_avg_vec = np.arange(1, 6.1, 0.1)
    # input_avg_vec = np.arange(6.2, 10.1, 0.2)
    kvec = np.arange(2, 6.1, 0.2)
    input_avg_vec = [round(a, 1) for a in kvec]
    # print(input_avg_vec)
    # print(len(input_avg_vec))
    EDindex = sys.argv[1]
    betaindex = sys.argv[2]
    simutime = sys.argv[3]
    ED = input_avg_vec[int(EDindex)]
    beta = beta_vec[int(betaindex)]
    LCC_critical(N, ED,beta,int(simutime))

    # """
    # for small network, run it locally
    # """
    # N = 10000
    # beta = 2.2
    # # input_avg_vec = np.arange(9, 30, 1)
    # input_avg_vec = np.arange(2, 6.1, 0.2)
    # input_avg_vec = [round(a,1) for a in input_avg_vec]
    # a = input_avg_vec[6]
    # # input_avg_vec = np.arange(6.2, 10.1, 0.2)
    # for ED in input_avg_vec:
    #     print(ED)
    #     for simutime in range(10):
    #         print(simutime)
    #         LCC_critical(N, ED, beta, int(simutime))

