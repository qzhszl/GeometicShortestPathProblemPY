# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 05-8-2025
"""
import random

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from fontTools.tfmLib import PASSTHROUGH
from scipy.optimize import curve_fit

from R2SRGG.R2SRGG import loadSRGGandaddnode
from collections import defaultdict
import math
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import linregress


def bin_and_compute_stats(length_geodesic_vec, data_vec, num_bins=100):
    """
    将 length_geodesic_vec 分为 num_bins 个均匀区间，计算每个区间中对应 data_vec 的均值和方差。

    参数：
        length_geodesic_vec: 1D numpy array，表示地理距离（或任意连续变量）
        data_vec: 1D numpy array，与 length_geodesic_vec 一一对应的数据值
        num_bins: 分箱数量，默认100

    返回：
        bin_centers: 每个区间的中点（长度为 num_bins）
        bin_means: 每个区间中对应 data_vec 的均值（长度为 num_bins）
        bin_vars: 每个区间中对应 data_vec 的方差（长度为 num_bins）
    """
    # 将输入转为 numpy 数组
    length_geodesic_vec = np.asarray(length_geodesic_vec)
    data_vec = np.asarray(data_vec)

    # 创建等距的分箱
    bin_edges = np.linspace(length_geodesic_vec.min(), length_geodesic_vec.max(), num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 给每个元素分配所属的 bin
    bin_indices = np.digitize(length_geodesic_vec, bins=bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)  # 防止越界

    # 初始化结果数组
    bin_means = np.zeros(num_bins)
    bin_vars = np.zeros(num_bins)

    # 逐个 bin 计算均值和方差
    for i in range(num_bins):
        bin_data = data_vec[bin_indices == i]
        if len(bin_data) > 0:
            bin_means[i] = np.mean(bin_data)
            bin_vars[i] = np.std(bin_data)
        else:
            bin_means[i] = np.nan  # 没有数据的 bin 用 NaN 标记
            bin_vars[i] = np.nan

    return bin_centers, bin_means, bin_vars


def power_law(x, a, b):
    return a * x ** b


def load_L(N, ED, beta, ExternalSimutime, folder_name):
    edgelength_vec_name = folder_name + "ave_edgelength_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    ave_edgelength_for_a_para_comb = np.loadtxt(edgelength_vec_name)

    hopcount_Name = folder_name + "hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    hop_vec = np.loadtxt(hopcount_Name, dtype=int)

    # L = np.multiply(ave_edgelength_for_a_para_comb, hop_vec)
    L = [x * y for x, y in zip(ave_edgelength_for_a_para_comb, hop_vec)]

    ave_L = np.mean(L)
    std_L = np.std(L)

    return ave_L, std_L, L


def load_dev(N, ED, beta, ExternalSimutime, folder_name):
    deviation_vec_name = folder_name + "ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    ave_deviation_for_a_para_comb = np.loadtxt(deviation_vec_name)
    ave_L = np.mean(ave_deviation_for_a_para_comb)
    std_L = np.std(ave_deviation_for_a_para_comb)

    return ave_L, std_L, ave_deviation_for_a_para_comb



def load_resort_data(N, beta):
    kvec = list(range(2, 10)) + [10, 12, 15, 18, 22, 27, 33, 40, 49, 60, 73, 89, 99]  # FOR N =10
    exemptionlist = []
    folder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\"
    for N in [N]:
        ave_L_vec = []
        ave_L_dic = {}
        real_ave_degree_vec = []

        for beta in [beta]:
            for ED in kvec:
                for ExternalSimutime in [0]:
                    if N < 200:
                        try:
                            real_ave_degree_name = folder_name+"real_ave_degree_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                            real_ave_degree = np.loadtxt(real_ave_degree_name)
                            real_ave_degree_vec = real_ave_degree_vec + list(real_ave_degree)
                            nodepairs_for_eachgraph_vec_name = folder_name+"nodepairs_for_eachgraph_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                            node_pairs_vec = np.loadtxt(nodepairs_for_eachgraph_vec_name, dtype=int)

                            _,_,L_vec = load_L(N,ED,beta,ExternalSimutime,folder_name)
                            a_index = 0
                            count = 0
                            for nodepair_num_inonegraph in node_pairs_vec:
                                b_index = a_index + nodepair_num_inonegraph - 1
                                ave_L_vec.append(np.mean(L_vec[a_index:b_index]))
                                real_avg = real_ave_degree[count]
                                try:
                                    ave_L_dic[real_avg] = ave_L_dic[real_avg] + list(
                                        L_vec[a_index:b_index])
                                except:
                                    ave_L_dic[real_avg] = list(
                                        L_vec[a_index:b_index])
                                a_index = b_index + 1
                                count = count + 1
                        except FileNotFoundError:
                            exemptionlist.append((N, ED, beta, ExternalSimutime))

    resort_dict = {}
    for key_degree, value_deviation in ave_L_dic.items():
        if round(key_degree) in resort_dict.keys():
            resort_dict[round(key_degree)] = resort_dict[round(key_degree)] + list(value_deviation)
            # a = max(list(value_deviation))
            # b = np.mean(list(value_deviation))
        else:
            resort_dict[round(key_degree)] = list(value_deviation)
            # a = max(list(value_deviation))
            # b = np.mean(list(value_deviation))
    if 0 in resort_dict.keys():
        del resort_dict[0]
    resort_dict = {key: resort_dict[key] for key in sorted(resort_dict.keys())}
    degree_vec_resort = list(resort_dict.keys())
    ave_L_resort = [np.mean(resort_dict[key_d]) for key_d in degree_vec_resort]
    std_L_resort = [np.std(resort_dict[key_d]) for key_d in degree_vec_resort]

    return degree_vec_resort, ave_L_resort, std_L_resort, real_ave_degree_vec, ave_L_vec, ave_L_dic




def load_large_network_results_dev_vs_avg_beta128(N, beta, kvec, realL):
    if realL:
        if beta in [2.1,2.5,3.1]:
            folder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\localmin_hunter\\"
        else:
            # if L = <d_e>h real stretch
            folder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\"
    else:
        # if L = <d_e><h> ave  link length* hopcount
        folder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\hopandedgelength\\"

    exemptionlist = []
    for N in [N]:
        ave_deviation_vec = []
        real_ave_degree_vec = []
        std_deviation_vec = []
        ave_edgelength_vec = []
        std_edgelength_vec = []

        ave_hop_vec = []
        std_hop_vec = []

        ave_hop_vec_no1 = []
        std_hop_vec_no1 = []

        ave_L_vec = []
        std_L_vec = []

        for beta in [beta]:
            for ED in kvec:
                for ExternalSimutime in [0]:
                    try:
                        # FileNetworkName = folder_name+"network_N{Nn}ED{EDn}Beta{betan}.txt".format(
                        #     Nn=N, EDn=ED, betan=beta)
                        # G = loadSRGGandaddnode(N, FileNetworkName)
                        # real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
                        # # print("real ED:", real_avg)
                        if N>200:
                            try:
                                real_avg_name = folder_name + "real_avg_N{Nn}ED{EDn}beta{betan}Simu{ST}.txt".format(
                                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                                real_avg = np.loadtxt(real_avg_name)
                                real_ave_degree_vec.append(real_avg)
                            except:
                                real_ave_degree_name = folder_name + "real_ave_degree_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                                real_avg = np.loadtxt(real_ave_degree_name)
                                real_ave_degree_vec.append(np.mean(real_avg))
                        else:
                            real_ave_degree_name = folder_name + "real_ave_degree_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                            real_avg = np.loadtxt(real_ave_degree_name)
                            real_ave_degree_vec.append(np.mean(real_avg))


                        if realL:
                            #if L = <d_e>h real stretch
                            # deviation_vec_name = folder_name + "ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                            #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                            # ave_deviation_for_a_para_comb = np.loadtxt(deviation_vec_name)
                            # ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
                            # std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))


                            edgelength_vec_name = folder_name + "ave_edgelength_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        else:
                            # if L = <d_e><h> ave  link length* hopcount
                            edgelength_vec_name = folder_name + "ave_edge_length_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)


                        # deviation_vec_name = folder_name + "ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                        #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        # ave_deviation_for_a_para_comb = np.loadtxt(deviation_vec_name)
                        # ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
                        # std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))
                        #
                        # edgelength_vec_name = folder_name + "ave_edgelength_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                        #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)


                        ave_edgelength_for_a_para_comb = np.loadtxt(edgelength_vec_name)
                        ave_edgelength_vec.append(np.mean(ave_edgelength_for_a_para_comb))
                        std_edgelength_vec.append(np.std(ave_edgelength_for_a_para_comb))



                        hopcount_Name = folder_name + "hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
                            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        hop_vec = np.loadtxt(hopcount_Name, dtype=int)

                        ave_hop_vec.append(np.mean(hop_vec))  # include 1
                        std_hop_vec.append(np.std(hop_vec))

                        mask = hop_vec != 1
                        hop_vec_no1 = hop_vec[mask]
                        ave_hop_vec_no1.append(np.mean(hop_vec_no1))
                        std_hop_vec_no1.append(np.std(hop_vec_no1))


                        if realL:
                            #if L = <d_e>h real stretch
                            ave_edgelength_for_a_para_comb_no1 = ave_edgelength_for_a_para_comb[mask]

                            L = [x * y for x, y in zip(ave_edgelength_for_a_para_comb_no1, hop_vec_no1)]
                        else:
                            # if L = <d_e><h> ave  link length* hopcount
                            L = [np.mean(hop_vec)*np.mean(ave_edgelength_for_a_para_comb)]

                        # # L = np.multiply(ave_edgelength_for_a_para_comb, hop_vec)
                        # L = [x * y for x, y in zip(ave_edgelength_for_a_para_comb, hop_vec)]

                        ave_L_vec.append(np.mean(L))
                        std_L_vec.append(np.std(L))

                    except FileNotFoundError:
                        exemptionlist.append((N, ED, beta, ExternalSimutime))
    print(exemptionlist)
    return real_ave_degree_vec, ave_deviation_vec, std_deviation_vec, ave_edgelength_vec, std_edgelength_vec, ave_hop_vec, std_hop_vec, ave_L_vec, std_L_vec
    # return kvec, real_ave_degree_vec, ave_deviation_vec, std_deviation_vec





def load_large_network_results_dev_vs_avg(N, beta, kvec):
    folder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\"
    exemptionlist = []
    for N in [N]:
        ave_deviation_vec = []
        real_ave_degree_vec = []
        std_deviation_vec = []
        ave_edgelength_vec = []
        std_edgelength_vec = []

        ave_hop_vec = []
        std_hop_vec = []
        ave_L_vec = []
        std_L_vec = []

        for beta in [beta]:
            for ED in kvec:
                for ExternalSimutime in [0]:
                    try:
                        # FileNetworkName = folder_name+"network_N{Nn}ED{EDn}Beta{betan}.txt".format(
                        #     Nn=N, EDn=ED, betan=beta)
                        # G = loadSRGGandaddnode(N, FileNetworkName)
                        # real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
                        # # print("real ED:", real_avg)
                        if N>200:
                            real_avg_name = folder_name + "real_avg_N{Nn}ED{EDn}beta{betan}Simu{ST}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                            real_avg = np.loadtxt(real_avg_name)
                            real_ave_degree_vec.append(real_avg)
                        else:
                            real_ave_degree_name = folder_name + "real_ave_degree_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                            real_avg = np.loadtxt(real_ave_degree_name)
                            real_ave_degree_vec.append(np.mean(real_avg))


                        deviation_vec_name = folder_name + "ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        ave_deviation_for_a_para_comb = np.loadtxt(deviation_vec_name)
                        ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
                        std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))

                        edgelength_vec_name = folder_name + "ave_edgelength_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        ave_edgelength_for_a_para_comb = np.loadtxt(edgelength_vec_name)
                        ave_edgelength_vec.append(np.mean(ave_edgelength_for_a_para_comb))
                        std_edgelength_vec.append(np.std(ave_edgelength_for_a_para_comb))

                        hopcount_Name = folder_name + "hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
                            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        hop_vec = np.loadtxt(hopcount_Name, dtype=int)

                        ave_hop_vec.append(np.mean(hop_vec))
                        std_hop_vec.append(np.std(hop_vec))

                        # L = np.multiply(ave_edgelength_for_a_para_comb, hop_vec)
                        L = [x * y for x, y in zip(ave_edgelength_for_a_para_comb, hop_vec)]

                        ave_L_vec.append(np.mean(L))
                        std_L_vec.append(np.std(L))

                    except FileNotFoundError:
                        exemptionlist.append((N, ED, beta, ExternalSimutime))
    print(exemptionlist)
    return real_ave_degree_vec, ave_deviation_vec, std_deviation_vec, ave_edgelength_vec, std_edgelength_vec, ave_hop_vec, std_hop_vec, ave_L_vec, std_L_vec
    # return kvec, real_ave_degree_vec, ave_deviation_vec, std_deviation_vec


def plot_L_with_avg_for_one_network():
    # Figure 4b
    # the x-axis is the input average degree
    N = 10000
    beta = 4
    kvec = [2.2,  3.0, 3.8, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105,
            9999, 16479, 27081, 44767, 73534, 121205, 199999]
    # kvec = [2.2, 2.8, 3.0, 3.4, 3.8, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105,
    #         9999,
    #         16479, 27081, 44767, 73534, 121205, 199999]
    # real_ave_degree_dict = {}
    # ave_deviation_dict = {}
    # std_deviation_dict = {}
    # kvec_dict = {}

    # real_ave_degree_vec, ave_deviation_vec, std_deviation_vec, ave_edgelength_vec, std_edgelength_vec, ave_hop_vec, std_hop_vec, ave_L_vec, std_L_vec = load_large_network_results_dev_vs_avg(
    #     N, beta, kvec)
    real_ave_degree_vec, _, _, _, _, _, _, ave_L_vec, std_L_vec = load_large_network_results_dev_vs_avg(
        N, beta, kvec)

    fig, ax = plt.subplots(figsize=(9, 6))

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']

    plt.errorbar(real_ave_degree_vec, ave_L_vec, yerr=std_L_vec, linestyle="-", linewidth=3, elinewidth=1, capsize=5,
                 marker='o', markersize=16,
                 label=r"$\langle L \rangle$", color=colors[3])

    # text = fr"$N = 10^4$, $\beta = {beta}$"
    # ax.text(
    #     0.3, 0.85,  # 文本位置（轴坐标，0.5 表示图中央，1.05 表示轴上方）
    #     text,
    #     transform=ax.transAxes,  # 使用轴坐标
    #     fontsize=26,  # 字体大小
    #     ha='center',  # 水平居中对齐
    #     va='bottom',  # 垂直对齐方式
    # )

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)

    # plt.yticks([0, 0.1, 0.2, 0.3])
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim([0.07,3])
    plt.xlabel(r'$\langle D \rangle$', fontsize=32)
    plt.ylabel(r'$\langle S \rangle$', fontsize=32)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    # plt.legend(fontsize=26, loc=(0.5, 0.1))
    plt.tick_params(axis='both', which="both", length=6, width=1)
    # picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\LocalOptimumdiffNBeta{betan}.png".format(
    #     betan=beta)
    # plt.savefig(picname, format='png', bbox_inches='tight', dpi=600,transparent=True)
    folder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\"
    picname = folder_name + f"L_vs_avg_N{N}beta{beta}.svg"
    plt.savefig(
        picname,
        format="svg",
        bbox_inches='tight',  # 紧凑边界
        transparent=True  # 背景透明，适合插图叠加
    )
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.show()
    plt.close()


def plot_L_with_avg():
    # Figure 3(c)
    # the x-axis is the input average degree
    Nvec = [100, 1000, 10000]
    # Nvec = [100]
    real_ave_degree_dict = {}
    ave_L = {}
    std_L = {}
    # [2.2, 2.8, 3.0, 3.4, 3.8, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276,
    #  3727, 6105,
    #  9999, 16479, 27081, 44767, 73534, 121205, 199999]
    kvec_dict = {
        100: [2, 3, 4, 5, 6, 8, 10, 12, 14, 17, 22, 27, 33, 40, 49, 60, 73, 89, 113, 149, 198, 260, 340, 446, 584,
              762, 993, 1292, 1690, 2276, 3142, 4339],
        1000: [2, 3, 4, 5, 6, 7, 8, 11, 15, 20, 28, 40, 58, 83, 118, 169, 241, 344, 490, 700, 999, 1425, 2033, 2900,
               4139, 5909, 8430, 12039, 17177, 24510, 34968, 49887, 71168],
        10000: [2.2,  3.0, 3.8, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276,
                3727, 6105,
                9999, 16479, 27081, 44767, 73534, 121205, 199999]}

    for N in Nvec:
        if N < 100:
            for beta in [4]:
                degree_vec_resort, ave_deviation_vec, std_deviation_vec, _, _, _ = load_resort_data(N, beta)
                real_ave_degree_dict[N] = degree_vec_resort
                ave_L[N] = ave_deviation_vec
                std_L[N] = std_deviation_vec
                kvec = degree_vec_resort
                real_ave_degree_vec = degree_vec_resort
                kvec_dict[N] = kvec
                real_ave_degree_dict[N] = real_ave_degree_vec
        else:
            for beta in [4]:
                kvec = kvec_dict[N]
                real_ave_degree_vec, _, _, _, _, _, _, ave_L_vec, std_L_vec = load_large_network_results_dev_vs_avg(
                    N, beta, kvec)
                real_ave_degree_dict[N] = real_ave_degree_vec
                ave_L[N] = ave_L_vec
                std_L[N] = std_L_vec

    # plt.plot(kvec,ave_deviation_vec,"o-")
    # plt.xscale('log')
    # plt.show()
    lengend = [r"$N=10$", r"$N=10^2$", r"$N=10^3$", r"$N=10^4$"]
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    colors = ["#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    # colorvec2 = ['#9FA9C9', '#D36A6A']
    for N_index in range(len(Nvec)):
        N = Nvec[N_index]
        if N == 10:
            # x = real_ave_degree_dict[N]
            # print(len(x))
            # x = x[1:]
            # y = ave_L[N]
            # y = y[1:]
            # error = std_L[N]
            # error = error[1:]
            x = real_ave_degree_dict[N]
            y = ave_L[N]
            error = std_L[N]

        else:
            x = real_ave_degree_dict[N]
            y = ave_L[N]
            error = std_L[N]
        plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=16,
                     label=lengend[N_index], color=colors[N_index])


    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    plt.ylim(0.02, 2)
    # plt.yticks([0, 0.1, 0.2, 0.3])
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'Average degree, $\langle D \rangle$', fontsize=26)
    plt.ylabel(r'Average stretch, $\langle L \rangle$', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    # plt.legend(fontsize=26, loc=(0.5, 0.5))
    plt.tick_params(axis='both', which="both", length=6, width=1)
    # picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\LocalOptimumdiffNBeta{betan}.png".format(
    #     betan=beta)
    # plt.savefig(picname, format='png', bbox_inches='tight', dpi=600,transparent=True)
    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\L_vs_realavg.svg"
    plt.savefig(
        picname,
        format="svg",
        bbox_inches='tight',  # 紧凑边界
        transparent=True  # 背景透明，适合插图叠加
    )
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.show()
    plt.close()


def plot_L_with_avg_beta1024():
    # Figure 3(c)
    # the x-axis is the input average degree
    Nvec = [100, 1000, 10000]
    # Nvec = [100]
    real_ave_degree_dict = {}
    ave_L = {}
    std_L = {}
    # [2.2, 2.8, 3.0, 3.4, 3.8, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276,
    #  3727, 6105,
    #  9999, 16479, 27081, 44767, 73534, 121205, 199999]
    kvec_dict = {
        100: [2, 3, 4, 5, 6, 8, 10, 12, 14, 17, 22, 27, 33, 40, 49, 60, 73, 89, 113, 149, 198, 260, 340, 446, 584,
              762, 993, 1292, 1690, 2276, 3142, 4339],
        1000: [2, 3, 4, 5, 6, 7, 8, 11, 15, 20, 28, 40, 58, 83, 118, 169, 241, 344, 490, 700, 999, 1425, 2033, 2900,
               4139, 5909, 8430, 12039, 17177, 24510, 34968, 49887, 71168],
        10000: [2.2,  3.0, 3.8, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276,
                3727, 6105,
                9999, 16479, 27081, 44767, 73534, 121205, 199999]}

    for N in Nvec:
        if N < 100:
            for beta in [4]:
                degree_vec_resort, ave_deviation_vec, std_deviation_vec, _, _, _ = load_resort_data(N, beta)
                real_ave_degree_dict[N] = degree_vec_resort
                ave_L[N] = ave_deviation_vec
                std_L[N] = std_deviation_vec
                kvec = degree_vec_resort
                real_ave_degree_vec = degree_vec_resort
                kvec_dict[N] = kvec
                real_ave_degree_dict[N] = real_ave_degree_vec
        else:
            for beta in [4]:
                kvec = kvec_dict[N]
                real_ave_degree_vec, _, _, _, _, _, _, ave_L_vec, std_L_vec = load_large_network_results_dev_vs_avg(
                    N, beta, kvec)
                real_ave_degree_dict[N] = real_ave_degree_vec
                ave_L[N] = ave_L_vec
                std_L[N] = std_L_vec

    # plt.plot(kvec,ave_deviation_vec,"o-")
    # plt.xscale('log')
    # plt.show()
    lengend = [r"$N=10$", r"$N=10^2$", r"$N=10^3$", r"$N=10^4$"]
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    colors = ["#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    # colorvec2 = ['#9FA9C9', '#D36A6A']
    for N_index in range(len(Nvec)):
        N = Nvec[N_index]
        if N == 10:
            # x = real_ave_degree_dict[N]
            # print(len(x))
            # x = x[1:]
            # y = ave_L[N]
            # y = y[1:]
            # error = std_L[N]
            # error = error[1:]
            x = real_ave_degree_dict[N]
            y = ave_L[N]
            error = std_L[N]

        else:
            x = real_ave_degree_dict[N]
            y = ave_L[N]
            error = std_L[N]
        plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=16,
                     label=lengend[N_index], color=colors[N_index])


    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    plt.ylim(0.02, 2)
    # plt.yticks([0, 0.1, 0.2, 0.3])
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'Average degree, $\langle D \rangle$', fontsize=26)
    plt.ylabel(r'Average stretch, $\langle L \rangle$', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    # plt.legend(fontsize=26, loc=(0.5, 0.5))
    plt.tick_params(axis='both', which="both", length=6, width=1)
    # picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\LocalOptimumdiffNBeta{betan}.png".format(
    #     betan=beta)
    # plt.savefig(picname, format='png', bbox_inches='tight', dpi=600,transparent=True)
    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\L_vs_realavg.svg"
    plt.savefig(
        picname,
        format="svg",
        bbox_inches='tight',  # 紧凑边界
        transparent=True  # 背景透明，适合插图叠加
    )
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.show()
    plt.close()


def plot_L_vs_N():
    # Figure 4 a(1)
    Nvec = [22, 46, 100, 215, 464, 999, 2154, 4642, 10000]
    ED = 10
    beta = 8
    ave_L_dict = {}
    std_L_dict = {}
    ExternalSimutime = 0
    folder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\"
    for N in Nvec:
        ave_L, std_L,_ = load_L(N, ED, beta, ExternalSimutime, folder_name)
        ave_L_dict[N] = ave_L
        std_L_dict[N] = std_L

    fig, ax = plt.subplots(figsize=(6, 4.5))
    # colors = [[0, 0.4470, 0.7410],
    #           [0.8500, 0.3250, 0.0980],
    #           [0.9290, 0.6940, 0.1250],
    #           [0.4940, 0.1840, 0.5560],
    #           [0.4660, 0.6740, 0.1880]]
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    y = []
    error = []
    for N_index in range(len(Nvec)):
        N = Nvec[N_index]
        y.append(ave_L_dict[N])
        error.append(std_L_dict[N])
    plt.errorbar(Nvec, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=16,
                 color=colors[4])
    print(y)

    text = r"$\mathbb{E}[D] = 10$, $\beta = 8$"
    plt.text(
        0.5, 0.85,  # 文本位置（轴坐标，0.5 表示图中央，1.05 表示轴上方）
        text,
        transform=ax.transAxes,  # 使用轴坐标
        fontsize=30,  # 字体大小
        ha='center',  # 水平居中对齐
        va='bottom'  # 垂直对齐方式
    )
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    plt.xlabel(r'$N$', fontsize=28)
    plt.ylabel(r'$\langle L \rangle$', fontsize=28)
    plt.xscale('log')
    # plt.yscale('log')
    plt.xticks(fontsize=30)
    yticks = [0.4,0.6,0.8,1.0]
    plt.yticks(yticks, fontsize=30)
    # plt.legend(fontsize=26, loc=(0.6, 0.5))
    plt.tick_params(axis='both', which="both", length=6, width=1)

    picname = folder_name + "LVsNlogxlogy.svg".format(
        EDn=ED)
    plt.savefig(
        picname,
        format="svg",
        bbox_inches='tight',  # 紧凑边界
        transparent=True  # 背景透明，适合插图叠加
    )

    # picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\DeviationVsNlogx.pdf".format(
    #     EDn=ED)
    # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.show()


def plot_L_vs_beta(ED):
    # Figure 4
    # the x-axis is the real average degree
    # Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    # betavec = [2.1, 4, 8, 16, 32, 64, 128]
    # betavec = [2.1, 2.2, 2.4, 2.5, 2.6, 2.8, 3, 3.25, 3.5, 3.75, 4, 5, 6, 7, 8, 10, 12, 16, 32, 64, 128]
    # betavec = [2.1, 2.2, 2.4, 2.5, 2.6, 2.8, 3, 3.25, 3.5, 3.75, 4, 5, 6, 8, 16, 32, 64, 128]
    betavec = [2.2, 3.0, 4.2, 5.9, 8.3, 11.7, 16.5, 23.2, 32.7, 46.1, 64.9, 91.5, 128.9, 181.7, 256]

    # betavec  = [3.0, 3.2, 3.4, 3.6,3.8, 3.9, 4.0,4.1,4.2,4.3, 4.4,4.5, 4.6, 4.8, 5.0, 5.2]
    print(len(betavec))
    ExternalSimutime = 0
    Nvec = [100, 1000, 10000]
    # Nvec = [999,9999]

    folder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\"

    clustering_coefficient_dict = {}
    ave_deviation_dict = {}
    std_deviation_dict = {}

    # original file path
    # clustering_coefficient_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\clustering_coefficient_ED{EDn}.txt".format(
    #     EDn=ED)
    for N in Nvec:
        ave_vec = []
        std_vec = []
        N1 = N
        if N>10:
            # if N>100:
            #     betavec = [2.2, 3.0, 4.2, 5.9, 8.3, 11.7, 16.5, 23.2, 32.7, 46.1, 64.9, 91.5, 128.9, 181.7, 256]
            for beta in betavec:
                # if beta==4.2 and N>100:
                #     N = N1 - 1
                # else:
                #     N = N1
                L_ave, L_std, _ = load_L(N, ED, beta, ExternalSimutime, folder_name)
                # L_ave, L_std, _ = load_dev(N, ED, beta, ExternalSimutime, folder_name)
                ave_vec.append(L_ave)
                std_vec.append(L_std)
        ave_deviation_dict[N] = ave_vec
        std_deviation_dict[N] = std_vec

    fig, ax = plt.subplots(figsize=(9, 6))
    # colors = [[0, 0.4470, 0.7410],
    #           [0.8500, 0.3250, 0.0980],
    #           [0.9290, 0.6940, 0.1250],
    #           [0.4940, 0.1840, 0.5560],
    #           [0.4660, 0.6740, 0.1880]]
    colors = ["#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    lengend = [r"$N=10^2$", r"$N=10^3$", r"$N=10^4$"]

    for N_index in range(len(Nvec)):
        N = Nvec[N_index]
        y = ave_deviation_dict[N]
        print(y)
        error = std_deviation_dict[N]
        if N <= ED:
            plt.errorbar([], y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o',
                         label=lengend[N_index], markersize=16, color=colors[N_index])
        else:
            plt.errorbar(betavec, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o',
                         label=lengend[N_index], markersize=16, color=colors[N_index])

        # # 找到峰值后最低点的坐标
        # peak_index = np.argmax(y[0:10])
        # post_peak_y = y[peak_index:]
        # post_peak_min_index = peak_index + np.argmin(post_peak_y)
        # post_peak_min_x = x[post_peak_min_index]
        # post_peak_min_y = y[post_peak_min_index]

        # 标出最低点
        # plt.plot(post_peak_min_x, post_peak_min_y, 'o', color=colors[N_index], markersize=8)

    # plt.xscale('log')
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    plt.xlabel(r'Temperature parameter, $\beta$', fontsize=26)
    plt.ylabel(r'Average stretch, $\langle L \rangle$', fontsize=26)
    plt.xscale('log')
    # plt.yscale('log')
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.legend(fontsize=26, loc=(0.6, 0.6))
    plt.tick_params(axis='both', which="both", length=6, width=1)
    # picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\DeviationVsbetaED{EDn}logy2.pdf".format(
    #     EDn=ED)
    # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    picname = folder_name + "LVsbetaED{EDn}logx.svg".format(
        EDn=ED)
    plt.savefig(
        picname,
        format="svg",
        bbox_inches='tight',  # 紧凑边界
        transparent=True  # 背景透明，适合插图叠加
    )
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.show()





def plot_L_with_avg_for_one_network_beta128():
    # Figure 4b
    # the x-axis is the input average degree
    N = 10000
    realL = True
    beta = 128
    kvec = [2.2, 3.0, 3.8, 5, 6.0, 8.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389,
            2276,
            3727, 6105,
            9999, 16479, 27081, 44767, 73534, 121205, 199999]
    # kvec = [2.2, 2.8, 3.0, 3.4, 3.8, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105,
    #         9999,
    #         16479, 27081, 44767, 73534, 121205, 199999]
    # real_ave_degree_dict = {}
    # ave_deviation_dict = {}
    # std_deviation_dict = {}
    # kvec_dict = {}

    # real_ave_degree_vec, ave_deviation_vec, std_deviation_vec, ave_edgelength_vec, std_edgelength_vec, ave_hop_vec, std_hop_vec, ave_L_vec, std_L_vec = load_large_network_results_dev_vs_avg(
    #     N, beta, kvec)
    real_ave_degree_vec, _, _, _, _, _, _, ave_L_vec, std_L_vec = load_large_network_results_dev_vs_avg_beta128(
        N, beta, kvec, realL)

    fig, ax = plt.subplots(figsize=(9, 6))

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']

    print(real_ave_degree_vec)
    print(ave_L_vec)
    plt.errorbar(real_ave_degree_vec, ave_L_vec, yerr=std_L_vec, linestyle="-", linewidth=3, elinewidth=1, capsize=5,
                 marker='o', markersize=16,
                 label=r"$\langle S \rangle$", color=colors[3])

    # text = fr"$N = 10^4$, $\beta = {beta}$"
    # ax.text(
    #     0.3, 0.85,  # 文本位置（轴坐标，0.5 表示图中央，1.05 表示轴上方）
    #     text,
    #     transform=ax.transAxes,  # 使用轴坐标
    #     fontsize=26,  # 字体大小
    #     ha='center',  # 水平居中对齐
    #     va='bottom',  # 垂直对齐方式
    # )

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)

    # plt.yticks([0, 0.1, 0.2, 0.3])
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim([0.01,3])
    plt.xlabel(r'$\langle D \rangle$', fontsize=32)
    plt.ylabel(r'$\langle S \rangle$', fontsize=32)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    # plt.legend(fontsize=26, loc=(0.5, 0.1))
    plt.tick_params(axis='both', which="both", length=6, width=1)
    # picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\LocalOptimumdiffNBeta{betan}.png".format(
    #     betan=beta)
    # plt.savefig(picname, format='png', bbox_inches='tight', dpi=600,transparent=True)
    folder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\"
    picname = folder_name + f"L_vs_avg_N{N}beta{beta}_N10000only.svg"
    plt.savefig(
        picname,
        format="svg",
        bbox_inches='tight',  # 紧凑边界
        transparent=True  # 背景透明，适合插图叠加
    )
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.show()
    plt.close()


def return_plot_L_with_avg_for_one_beta():
    # the x-axis is the input average degree
    N = 10000
    realL = True
    beta = 3.1

    kvec = [2,4,5,6,7,22,72,236,]


    # if beta ==128:
    #     kvec = [2.2, 3.0, 3.8, 5, 6.0, 8.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389,
    #             2276,
    #             3727, 6105,
    #             9999, 16479, 27081, 44767, 73534, 121205, 199999]
    # else:
    #     kvec = [2,6,12,46,132,375,1067,3040,8657,24657,70224,199999]



    # kvec = [2.2, 2.8, 3.0, 3.4, 3.8, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105,
    #         9999,
    #         16479, 27081, 44767, 73534, 121205, 199999]
    # real_ave_degree_dict = {}
    # ave_deviation_dict = {}
    # std_deviation_dict = {}
    # kvec_dict = {}

    # real_ave_degree_vec, ave_deviation_vec, std_deviation_vec, ave_edgelength_vec, std_edgelength_vec, ave_hop_vec, std_hop_vec, ave_L_vec, std_L_vec = load_large_network_results_dev_vs_avg(
    #     N, beta, kvec)
    real_ave_degree_vec, _, _, _, _, _, _, ave_L_vec, std_L_vec = load_large_network_results_dev_vs_avg_return(
        N, beta, kvec, realL)

    fig, ax = plt.subplots(figsize=(9, 6))

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']

    print(real_ave_degree_vec)
    print(ave_L_vec)
    plt.errorbar(real_ave_degree_vec, ave_L_vec, yerr=std_L_vec, linestyle="-", linewidth=3, elinewidth=1, capsize=5,
                 marker='o', markersize=16,
                 label=r"$\langle S \rangle$", color=colors[3])

    # text = fr"$N = 10^4$, $\beta = {beta}$"
    # ax.text(
    #     0.3, 0.85,  # 文本位置（轴坐标，0.5 表示图中央，1.05 表示轴上方）
    #     text,
    #     transform=ax.transAxes,  # 使用轴坐标
    #     fontsize=26,  # 字体大小
    #     ha='center',  # 水平居中对齐
    #     va='bottom',  # 垂直对齐方式
    # )

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)

    # plt.yticks([0, 0.1, 0.2, 0.3])
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim([0.01,3])
    plt.xlabel(r'$\langle D \rangle$', fontsize=32)
    plt.ylabel(r'$\langle S \rangle$', fontsize=32)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    # plt.legend(fontsize=26, loc=(0.5, 0.1))
    plt.tick_params(axis='both', which="both", length=6, width=1)
    # picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\LocalOptimumdiffNBeta{betan}.png".format(
    #     betan=beta)
    # plt.savefig(picname, format='png', bbox_inches='tight', dpi=600,transparent=True)
    folder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\"
    picname = folder_name + f"L_vs_avg_N{N}beta{beta}_N10000only.svg"
    plt.savefig(
        picname,
        format="svg",
        bbox_inches='tight',  # 紧凑边界
        transparent=True  # 背景透明，适合插图叠加
    )
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.show()
    plt.close()





def plot_L_with_avg_plotfigureonly():
    # the x-axis is the input average degree
    Nvec = [215, 464, 1000, 2154, 4642,10000]
    Nvec = [100,1000,10000, 20000]

    realL = True

    # Nvec = [100]
    real_ave_degree_dict = {}
    ave_L = {}
    std_L = {}


    beta_vec = [128]
    kvec_dict = {
        100: [2, 3, 5, 8, 12, 18, 29, 45, 70, 109, 169, 264, 642, 1000],
        215: [2, 3, 5, 9, 14, 24, 39, 63, 104, 170, 278, 455, 746, 1221, 2000],
        464: [2, 3, 6, 10, 18, 30, 52, 89, 154, 265, 456, 785, 1350, 2324, 4000],
        1000: [2,3, 4,5, 7, 12, 21, 39, 70, 126, 229, 414, 748, 1353, 2446, 4424, 8000],
        2154: [2, 4, 7, 14, 27, 52, 99, 190, 364, 697, 1335, 2558, 4902, 9393, 18000],
        4642: [2, 4, 8, 16, 33, 67, 135, 272, 549, 1107, 2234, 4506, 9091, 18340, 37000],
        10000: [2.2, 3.0, 3.8, 5, 6.0,  8.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389,
                2276,
                3727, 6105,
                9999, 16479, 27081, 44767, 73534, 121205, 199999],
        20000: [2.2, 2.8, 3.4,4.4, 5, 6.0, 8.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389,
                2276,
                3727, 6105,
                9999, 16479, 27081, 44767, 73534, 121205]
    }
    # kvec_dict = {
    #     100: [2, 3, 5, 8, 12, 18, 29, 45, 70, 109, 169, 264, 412, 642, 1000],
    #     215: [2, 3, 5, 9, 14, 24, 39, 63, 104, 170, 278, 455, 746, 1221, 2000],
    #     464: [2, 3, 6, 10, 18, 30, 52, 89, 154, 265, 456, 785, 1350, 2324, 4000],
    #     1000: [2, 4, 7, 12, 21, 39, 70, 126, 229, 414, 748, 1353, 2446, 4424, 8000],
    #     2154: [2, 4, 7, 14, 27, 52, 99, 190, 364, 697, 1335, 2558, 4902, 9393, 18000],
    #     4642: [2, 4, 8, 16, 33, 67, 135, 272, 549, 1107, 2234, 4506, 9091, 18340, 37000],
    #     10000: [2.2, 2.8, 3.0, 3.4, 3.8, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389,
    #             3727]}

    for N in Nvec:
        if N < 100:
            for beta in beta_vec:
                degree_vec_resort, ave_deviation_vec, std_deviation_vec, _, _, _ = load_resort_data(N, beta)
                real_ave_degree_dict[N] = degree_vec_resort
                ave_L[N] = ave_deviation_vec
                std_L[N] = std_deviation_vec
                kvec = degree_vec_resort
                real_ave_degree_vec = degree_vec_resort
                kvec_dict[N] = kvec
                real_ave_degree_dict[N] = real_ave_degree_vec
        else:
            for beta in beta_vec:
                kvec = kvec_dict[N]
                real_ave_degree_vec, _, _, _, _, _, _, ave_L_vec, std_L_vec = load_large_network_results_dev_vs_avg_beta128(
                    N, beta, kvec,realL)
                real_ave_degree_dict[N] = real_ave_degree_vec
                ave_L[N] = ave_L_vec
                std_L[N] = std_L_vec

    # plt.plot(kvec,ave_deviation_vec,"o-")
    # plt.xscale('log')
    # plt.show()

    lengend = [r"$N=10^2$", r"$N=10^3$", r"$N=10^4$", r"$N=2 \times 10^4$"]
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494','#9FA9C9', '#D36A6A']
    colors = ["#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    # colors = c
    # colorvec2 = ['#9FA9C9', '#D36A6A']
    for N_index in range(len(Nvec)):
        N = Nvec[N_index]
        if N == 10:
            # x = real_ave_degree_dict[N]
            # print(len(x))
            # x = x[1:]
            # y = ave_L[N]
            # y = y[1:]
            # error = std_L[N]
            # error = error[1:]
            x = real_ave_degree_dict[N]
            y = ave_L[N]
            error = std_L[N]

        else:
            x = real_ave_degree_dict[N]
            y = ave_L[N]
            error = std_L[N]
        kvec = kvec_dict[N]
        print(kvec)
        print(y)
        plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=16,
                     label=lengend[N_index], color=colors[N_index])

    # k_c = 4.512
    # k_star = [k_c**(2/3) * np.pi * (4/3)**(2/3) * N**(1/3) for N in Nvec]
    # y2 = [analticL(N, k) for (N, k) in zip(Nvec, k_star)]
    # plt.plot(k_star,y2,"-o",markersize = 25,markerfacecolor='none',linewidth=5,label = "analytic local minimum")
    # print(k_star)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # plt.ylim(0.02, 2)
    # plt.yticks([0, 0.1, 0.2, 0.3])
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'Average degree, $\langle D \rangle$', fontsize=26)
    if realL:
        plt.ylabel(r'Average stretch, $\langle S \rangle$', fontsize=26)
    else:
        plt.ylabel(r'Average stretch, $\langle r \rangle \langle h \rangle $', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    # plt.legend(fontsize=26, loc=(0.5, 0.05))
    plt.tick_params(axis='both', which="both", length=6, width=1)
    # picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\LocalOptimumdiffNBeta{betan}.png".format(
    #     betan=beta)
    # plt.savefig(picname, format='png', bbox_inches='tight', dpi=600,transparent=True)
    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\L_vs_realavg_beta128.svg"
    # plt.savefig(
    #     picname,
    #     format="svg",
    #     bbox_inches='tight',  # 紧凑边界
    #     transparent=True  # 背景透明，适合插图叠加
    # )
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.show()
    plt.close()



def plot_L_with_avg_diffbeta():
    # the x-axis is the input average degree
    # The data is loaed from plot_stretch_v_avg&N_capature_localmin.py plot_L_with_avg_loc_largeN()
    N = 10000

    realL = False

    # Nvec = [100]
    real_ave_degree_dict = {}
    ave_L = {}
    std_L = {}
    approx_L =  {}


    beta_vec = [4,128]
    kvec_dict = {
        100: [2, 3, 5, 8, 12, 18, 29, 45, 70, 109, 169, 264, 642, 1000],
        215: [2, 3, 5, 9, 14, 24, 39, 63, 104, 170, 278, 455, 746, 1221, 2000],
        464: [2, 3, 6, 10, 18, 30, 52, 89, 154, 265, 456, 785, 1350, 2324, 4000],
        1000: [2,3, 4,5, 7, 12, 21, 39, 70, 126, 229, 414, 748, 1353, 2446, 4424, 8000],
        2154: [2, 4, 7, 14, 27, 52, 99, 190, 364, 697, 1335, 2558, 4902, 9393, 18000],
        4642: [2, 4, 8, 16, 33, 67, 135, 272, 549, 1107, 2234, 4506, 9091, 18340, 37000],
        10000: [2.2, 3.0, 3.8, 5, 6.0,  8.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389,
                2276,
                3727, 6105,
                9999, 16479, 27081, 44767, 73534, 121205, 199999],
        20000: [2.2, 2.8, 3.4,4.4, 5, 6.0, 8.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389,
                2276,
                3727, 6105,
                9999, 16479, 27081, 44767, 73534, 121205]
    }
    # kvec_dict = {
    #     100: [2, 3, 5, 8, 12, 18, 29, 45, 70, 109, 169, 264, 412, 642, 1000],
    #     215: [2, 3, 5, 9, 14, 24, 39, 63, 104, 170, 278, 455, 746, 1221, 2000],
    #     464: [2, 3, 6, 10, 18, 30, 52, 89, 154, 265, 456, 785, 1350, 2324, 4000],
    #     1000: [2, 4, 7, 12, 21, 39, 70, 126, 229, 414, 748, 1353, 2446, 4424, 8000],
    #     2154: [2, 4, 7, 14, 27, 52, 99, 190, 364, 697, 1335, 2558, 4902, 9393, 18000],
    #     4642: [2, 4, 8, 16, 33, 67, 135, 272, 549, 1107, 2234, 4506, 9091, 18340, 37000],
    #     10000: [2.2, 2.8, 3.0, 3.4, 3.8, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389,
    #             3727]}

    for beta in beta_vec:
        kvec = kvec_dict[N]
        real_ave_degree_vec, _, _, _, _, _, _, ave_L_vec, std_L_vec = load_large_network_results_dev_vs_avg_beta128(
            N, beta, kvec,realL)
        real_ave_degree_dict[beta] = real_ave_degree_vec
        ave_L[beta] = ave_L_vec
        std_L[beta] = std_L_vec

    # plt.plot(kvec,ave_deviation_vec,"o-")
    # plt.xscale('log')
    # plt.show()

    lengend = [r"$\beta=2.1$", r"$\beta=2.5$", r"$\beta=4$", r"$\beta=128$"]
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494','#9FA9C9', '#D36A6A']
    colors = ["#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    # colors = c
    # colorvec2 = ['#9FA9C9', '#D36A6A']


    # real_ave_degree_dict[2.5] = [np.float64(1.3802), np.float64(4.6628), np.float64(13.7504), np.float64(41.3624), np.float64(121.1224),
    #  np.float64(130.4644), np.float64(140.1094), np.float64(150.687), np.float64(162.0182), np.float64(174.1272),
    #  np.float64(187.1632), np.float64(200.8026), np.float64(215.758), np.float64(231.4976), np.float64(248.5472),
    #  np.float64(266.907), np.float64(286.2412), np.float64(306.3448), np.float64(328.5492), np.float64(340.4932),
    #  np.float64(351.8212), np.float64(376.905), np.float64(403.5924), np.float64(432.2924), np.float64(462.6738),
    #  np.float64(494.5304), np.float64(529.2872), np.float64(565.3824), np.float64(604.1762), np.float64(645.247),
    #  np.float64(688.635), np.float64(735.0184), np.float64(783.71), np.float64(835.7892), np.float64(890.686),
    #  np.float64(2103.6792), np.float64(4257.9598), np.float64(6934.8888), np.float64(8898.362), np.float64(9929.8622)]
    # ave_L[2.5]  = [np.float64(2.083753542297957), np.float64(0.8108291291468183), np.float64(0.7086999716180932),
    #  np.float64(0.6743696974975774), np.float64(0.6485367658365878), np.float64(0.6497151015858645),
    #  np.float64(0.6519750995982597), np.float64(0.649283688806195), np.float64(0.6437894976565234),
    #  np.float64(0.6434539118697549), np.float64(0.6386335147524166), np.float64(0.6383315540220368),
    #  np.float64(0.633582391183019), np.float64(0.6331559814304656), np.float64(0.6326149108611301),
    #  np.float64(0.6364490881533421), np.float64(0.6311935559791461), np.float64(0.63162527832276),
    #  np.float64(0.6303570356987208), np.float64(0.6344778917716585), np.float64(0.6324816930851671),
    #  np.float64(0.6335929712525001), np.float64(0.6382516495678294), np.float64(0.6382697612836463),
    #  np.float64(0.6374513277396701), np.float64(0.6412180232617238), np.float64(0.6438241694238583),
    #  np.float64(0.6464903995012711), np.float64(0.6554241486662417), np.float64(0.6518530129876442),
    #  np.float64(0.6570248816812725), np.float64(0.660588289448376), np.float64(0.6649758895766243),
    #  np.float64(0.6612636941366813), np.float64(0.6710528177496031), np.float64(0.7530533967784127),
    #  np.float64(0.8947615820067742), np.float64(0.963460886822925), np.float64(1.1196354619043718),
    #  np.float64(1.2093558418297714)]

    real_ave_degree_dict[2.5] = [np.float64(1.3802), np.float64(4.6628), np.float64(13.7504), np.float64(41.3624),
                                 np.float64(121.1224),
                                 np.float64(130.4644),
                                 np.float64(174.1272),
                                 np.float64(231.4976),
                                 np.float64(306.3448),
                                 np.float64(376.905),
                                 np.float64(494.5304),
                                 np.float64(783.71),
                                 np.float64(835.7892),
                                 np.float64(890.686),
                                 np.float64(2103.6792), np.float64(4257.9598), np.float64(6934.8888),
                                 np.float64(8898.362), np.float64(9929.8622)]
    ave_L[2.5] = [np.float64(2.083753542297957), np.float64(0.8108291291468183), np.float64(0.7086999716180932),
                  np.float64(0.6743696974975774), np.float64(0.6485367658365878), np.float64(0.6497151015858645),
                  np.float64(0.6434539118697549), np.float64(0.6331559814304656), np.float64(0.63162527832276),
                  np.float64(0.6335929712525001), np.float64(0.6412180232617238), np.float64(0.6518530129876442),
                  np.float64(0.6612636941366813), np.float64(0.6710528177496031), np.float64(0.7530533967784127),
                  np.float64(0.8947615820067742), np.float64(0.963460886822925), np.float64(1.1196354619043718),
                  np.float64(1.2093558418297714)]


    real_ave_degree_dict[3.1] = [np.float64(1.533162), np.float64(5.274501999999999), np.float64(16.141509999999997),
     np.float64(50.544774000000004), np.float64(153.98062399999998), np.float64(451.08478800000006),
     np.float64(1224.1397040000002), np.float64(2934.61436), np.float64(5747.6903839999995), np.float64(8460.482898),
     np.float64(9668.840538), np.float64(9943.386174)]
    ave_L[3.1] = [np.float64(0.7073626407706719), np.float64(0.7767907598623258), np.float64(0.6823988009051273),
     np.float64(0.6429823932052088), np.float64(0.628112971110632), np.float64(0.6249892683133915),
     np.float64(0.6728046131127262), np.float64(0.7680096849141987), np.float64(0.9128719689922662),
     np.float64(1.053636194955935), np.float64(1.123536183341839), np.float64(1.1266797833173825)]

    real_ave_degree_dict[2.1] =[np.float64(0.647672), np.float64(0.9536), np.float64(1.2332), np.float64(1.792), np.float64(2.058828),
     np.float64(5.847454000000002), np.float64(16.889622), np.float64(47.695429999999995),
     np.float64(130.94333999999998), np.float64(343.033686), np.float64(842.6122099999999),
     np.float64(1887.5601599999998), np.float64(3706.396326), np.float64(6092.669242000001), np.float64(8187.439658),
     np.float64(9344.583)]
    ave_L[2.1] =[np.float64(0.27143445114259124), np.float64(0.7598339194549935), np.float64(2.675853772043278),
     np.float64(1.4656527772294778), np.float64(1.3136788477251418), np.float64(0.8542345889975687),
     np.float64(0.7386616210765622), np.float64(0.6890922705269413), np.float64(0.6642333052457445),
     np.float64(0.6455108478945255), np.float64(0.6808760138955805), np.float64(0.746380165204815),
     np.float64(0.8396772972302338), np.float64(0.9318348031709808), np.float64(1.024014919492898),
     np.float64(1.3314722164403012)]

    approx_L[2.1] = [np.float64(0.18713185030928067), np.float64(0.7283556244139056), np.float64(2.3453236422507335), np.float64(1.2129600085566379), np.float64(1.0596813847187543), np.float64(0.5570708063885884), np.float64(0.4308826229144635), np.float64(0.3904621312891976), np.float64(0.3930380853589546), np.float64(0.3964590523677577), np.float64(0.4713624573018614), np.float64(0.5520598341571079), np.float64(0.6036795026077996), np.float64(0.605236360802573), np.float64(0.5710837915958688), np.float64(0.5429516015345655)]
    approx_L[2.5] = []
    real_ave_degree_dict[4] = [1.7054, 2.1758, 2.3132, 2.65, 2.9316, 3.4204, 4.6164, 7.6562, 12.1664, 20.397, 32.749, 53.1894, 85.5136, 137.117, 218.353, 345.515, 540.688, 836.2706, 1274.4564, 1903.7746, 2765.9072, 3888.2098, 5252.766, 6702.0802, 8029.6946, 8990.1558, 9553.0216, 9820.4528]
    approx_L[4] = [np.float64(0.047142366971440246), np.float64(0.20861996975776426), np.float64(0.3889985780686527), np.float64(0.6390728663273963), np.float64(0.4679255268407224), np.float64(0.34249510012394296), np.float64(0.2575292756978971), np.float64(0.1950389670926904), np.float64(0.17174246626728473), np.float64(0.16412749193484105), np.float64(0.16554785757124327), np.float64(0.17103442293291765), np.float64(0.18362445384559242), np.float64(0.19996835814461092), np.float64(0.22164503626728538), np.float64(0.24902137730478238), np.float64(0.27450434516084105), np.float64(0.30773424399552624), np.float64(0.35892274446087363), np.float64(0.41679672719049743), np.float64(0.4741504332954842), np.float64(0.5196804807434381), np.float64(0.5532632778358367), np.float64(0.5632517127774106), np.float64(0.5543953726725661), np.float64(0.5443747566911749), np.float64(0.532921861195619), np.float64(0.5270276034416593)]
    ave_L[4] = [np.float64(0.10586833057728005), np.float64(0.32092651829744073), np.float64(0.4211427701655029), np.float64(1.600106730159238), np.float64(1.1207469459545976), np.float64(0.9602755020342794), np.float64(0.8505105399855226), np.float64(0.7521601392454854), np.float64(0.7144736637949518), np.float64(0.6897834045341739), np.float64(0.6611984828894697), np.float64(0.6430243010088226), np.float64(0.6299550989628939), np.float64(0.6236145759731754), np.float64(0.6226740920891127), np.float64(0.6267114073026878), np.float64(0.6282176272232343), np.float64(0.6364987317054913), np.float64(0.6586756556582146), np.float64(0.6946337988528414), np.float64(0.7457111976022143), np.float64(0.826799927138955), np.float64(0.9428326521400886), np.float64(1.0663845876582414), np.float64(1.140146289513965), np.float64(1.1836464727030707), np.float64(1.1939575186864742), np.float64(1.188776107945433)]

    beta_vec = [2.1,2.5,4, 128]
    for beta_index in range(len(beta_vec)):
        beta = beta_vec[beta_index]
        x = real_ave_degree_dict[beta]
        y = ave_L[beta]

        kvec = kvec_dict[N]
        print(kvec)
        print(y)
        if beta in [4, 128]:
            error = std_L[beta]
            plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=16,
                         label=lengend[beta_index], color=colors[beta_index])
        else:
            plt.plot(x, y,linestyle="--", linewidth=3, marker='o',
                         markersize=16,
                         label=lengend[beta_index], color=colors[beta_index])

    # k_c = 4.512
    # k_star = [k_c**(2/3) * np.pi * (4/3)**(2/3) * N**(1/3) for N in Nvec]
    # y2 = [analticL(N, k) for (N, k) in zip(Nvec, k_star)]
    # plt.plot(k_star,y2,"-o",markersize = 25,markerfacecolor='none',linewidth=5,label = "analytic local minimum")
    # print(k_star)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # plt.ylim(0.02, 2)
    # plt.yticks([0, 0.1, 0.2, 0.3])
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'Average degree, $\langle D \rangle$', fontsize=26)
    if realL:
        plt.ylabel(r'Average stretch, $\langle S \rangle$', fontsize=26)
    else:
        plt.ylabel(r'Average stretch, $\langle r \rangle \langle h \rangle $', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=26, loc=(0.5, 0.05))
    plt.tick_params(axis='both', which="both", length=6, width=1)
    # picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\LocalOptimumdiffNBeta{betan}.png".format(
    #     betan=beta)
    # plt.savefig(picname, format='png', bbox_inches='tight', dpi=600,transparent=True)
    # picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\L_vs_realavg_beta128.svg"
    # plt.savefig(
    #     picname,
    #     format="svg",
    #     bbox_inches='tight',  # 紧凑边界
    #     transparent=True  # 背景透明，适合插图叠加
    # )
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.show()
    plt.close()





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Figure 3 (original Figure 4)
    """
    # STEP 1  L versus real average degree  Figure 3(d)
    """
    # plot_L_with_avg()

    """
    # STEP 1.5  L versus real average degree for one network Figure 3(d)
    """
    # plot_L_with_avg_for_one_network()


    """
    # STEP 1_2 real L and <d><h> versus real average degree beta = 128  Figure 3(d)
    """
    # if realL = True,L = real stretch ; else , L = <d><h>

    # plot_L_with_avg_plotfigureonly()
    # plot_L_with_avg_for_one_network_beta128()


    """
    # STEP 1.9 real L and <d><h> versus real average degree diffbeta
    """
    # if realL = True,L = real stretch ; else , L = <d><h>
    plot_L_with_avg_diffbeta()

    """
    # STEP 2 plot L vs N
    """
    # plot_L_vs_N()

    """
    # STEP 3 plot L vs beta
    """
    # plot_L_vs_beta(10)



