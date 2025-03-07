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
1. number of paths between two nodes changes with different expected degree
2. number of paths between two nodes changes with different hopcount
"""
import itertools
import time

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
import matplotlib.pyplot as plt

def pathnum_vs_hopcount(ED):
    N = 10000
    beta = 4
    rg = RandomGenerator(-12)
    G, Coorx, Coory = R2SRGG(N, ED, beta, rg)
    real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
    print("ED:", ED)
    print("real ED:", real_avg)

    nodepair_num = 10000
    unique_pairs = find_k_connected_node_pairs(G, nodepair_num)
    hopcount_vec = []
    path_num_vec = []
    for node_pair in unique_pairs:
        # print("node_pair:", node_pair)
        nodei = node_pair[0]
        nodej = node_pair[1]
        # Find the number of paths
        a1 = time.time()
        hopcount = nx.shortest_path_length(G, nodei, nodej)
        # print("hop:",hopcount)
        all_paths = nx.all_shortest_paths(G, source=nodei, target=nodej)
        a2 = time.time()
        # print(a2-a1)
        count = 0
        for _ in all_paths:
            count += 1
            if count % 1000 == 0:  # 当 count 是 1000 的整数倍时
                print(count // 1000)  # 输出 count / 1000（整数除法）

        allpaths_num  = count
        # print("allpaths_num:", allpaths_num)
        path_num_vec.append(allpaths_num)
        hopcount_vec.append(hopcount)
        # print(a2-a1)
    SPnodenum_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\pathnums\\SPnum_N{Nn}ED{EDn}Beta{betan}.txt".format(
        Nn=N, EDn=ED, betan=beta)
    np.savetxt(SPnodenum_vec_name, path_num_vec, fmt="%i")
    hopcount_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\pathnums\\hopcount_sp_N{Nn}_ED{EDn}Beta{betan}.txt".format(
        Nn=N, EDn=ED, betan=beta)
    np.savetxt(hopcount_Name, hopcount_vec)


def plot_spnum_vs_hopcount():
    N = 10000
    beta = 4
    ED = 10
    SPnodenum_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\pathnums\\SPnum_N{Nn}ED{EDn}Beta{betan}.txt".format(
        Nn=N, EDn=ED, betan=beta)
    path_num_vec = np.loadtxt(SPnodenum_vec_name,dtype=int)
    hopcount_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\pathnums\\hopcount_sp_N{Nn}_ED{EDn}Beta{betan}.txt".format(
        Nn=N, EDn=ED, betan=beta)
    hopcount_vec = np.loadtxt(hopcount_Name,dtype=int)
    unique_hops = np.unique(hopcount_vec)

    # 计算每个 hopcount 对应的均值和标准差
    mean_values = [np.mean(path_num_vec[hopcount_vec == h]) for h in unique_hops]
    std_values = [np.std(path_num_vec[hopcount_vec == h]) for h in unique_hops]

    # 绘制误差条形图 (Errorbar)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    plt.errorbar(unique_hops, mean_values, yerr=std_values, fmt='o-', capsize=5, capthick=2, markersize=8, linewidth=2)

    num_xticks = 6
    xtick_positions = np.linspace(unique_hops.min(), unique_hops.max(), num_xticks, dtype=int)  # 选择 6 个均匀分布的 hopcount 值

    # 设置 X 轴刻度
    plt.xticks(xtick_positions, labels=[str(h) for h in xtick_positions])  # 保证 X 轴刻度仅显示 unique_hops
    plt.xlabel('Hopcount', fontsize=14, fontweight='bold')
    plt.ylabel('Shortest path number', fontsize=14, fontweight='bold')
    # plt.title('Path Number vs. Hop Count', fontsize=16)
    text = rf"$N = 10^4$" "\n" r"$E[D] = {N}$" "\n" r"$\beta = 4$".format(N = N)
    plt.text(
        0.25, 0.65,  # 文本位置（轴坐标，0.5 表示图中央，1.05 表示轴上方）
        text,
        transform=ax.transAxes,  # 使用轴坐标
        fontsize=20,  # 字体大小
        ha='left',  # 水平居中对齐
        va='bottom'  # 垂直对齐方式
    )

    # 显示网格
    # plt.grid(True, linestyle='--', alpha=0.6)
    # 显示图形
    plt.show()




if __name__ == '__main__':

    for ED in [2,4,8,16,32,64,128]:
        pathnum_vs_hopcount(ED)

    # plot_spnum_vs_hopcount()

    # geolength_index = sys.argv[1]
    # ExternalSimutime = sys.argv[2]
    # pathnum_vs_hopcount(int(geolength_index), int(ExternalSimutime))

