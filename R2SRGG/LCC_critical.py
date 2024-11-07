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


def LCC_critical(ED,beta,simutime):
    """
    In this function we try to find the critical point of the largest LCC
    :param beta:
    :return:
    """
    N = 100
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
    beta_vec = [4,8]
    input_avg_vec = np.arange(1, 8.1, 0.1)
    EDindex = sys.argv[1]
    betaindex = sys.argv[2]
    simutime = sys.argv[3]
    ED = input_avg_vec[EDindex]
    beta = beta_vec[betaindex]
    LCC_critical(ED,beta,int(simutime))
