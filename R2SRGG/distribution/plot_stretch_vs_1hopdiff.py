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
from matplotlib.pyplot import figure
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


def analticL(N, k_vals):
    pi = np.pi
    # k 的取值范围
    k_vals = np.array(k_vals)
    # 计算 h(k)

    kc = 4.512
    C = 0.52 * np.sqrt(N * pi)
    # print(N, C)
    h_vals = (2 / 3) * np.sqrt(k_vals / (N * pi)) * (1 + 4 / (3 * pi) * np.sqrt(k_vals / (N * pi))) * C * (
                k_vals - kc) ** (-0.5)
    return h_vals


def analtich(N, k_vals):
    pi = np.pi
    # k 的取值范围
    k_vals = np.array(k_vals)
    # 计算 h(k)
    kc = 4.512
    C = 0.52 * np.sqrt(N * pi)
    print(N, C)
    h_vals = C * (k_vals - kc) ** (-0.5)
    return h_vals


def analticdl(N, k_vals):
    pi = np.pi
    # k 的取值范围
    k_vals = np.array(k_vals)
    h_vals = (2 / 3) * np.sqrt(k_vals / (N * pi)) * (1 + 4 / (3 * pi) * np.sqrt(k_vals / (N * pi)))
    return h_vals



def plot_hopcount_L_Lsamu_with_avg_whether_1hopincluded():
    # plot how the hopcount and link length changes with the average degree
    # the x-axis is the input average degree
    Nvec = [10000]

    hop_flag = True

    # Nvec = [100]
    real_ave_degree_dict = {}
    ave_L = {}
    ave_L_no1 = {}
    ave_Lsamu = {}
    ave_Lsamu_no1 = {}
    std_L = {}

    edgelength = {}
    hop = {}
    hop_no1 = {}

    beta_vec = [1024]
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
        for beta in beta_vec:
            kvec = kvec_dict[N]
            real_ave_degree_vec, ave_edgelength_vec, _, ave_hop_vec, _, ave_hop_vec_no1, _, ave_L_vec_no1, _ = load_large_network_results_dev_vs_avg_locmin_1hopdiff(
                N, beta, kvec, True,True)
            _, _, _, _, _, _, _, ave_L_vec, _ = load_large_network_results_dev_vs_avg_locmin_1hopdiff(
                N, beta, kvec, True, False)

            _, _, _, _, _, _, _, ave_L_samu_vec, _ = load_large_network_results_dev_vs_avg_locmin_1hopdiff(
                N, beta, kvec, False, False)

            _, _, _, _, _, _, _, ave_L_samu_vec_no1, _ = load_large_network_results_dev_vs_avg_locmin_1hopdiff(
                N, beta, kvec, False, True)

            real_ave_degree_dict[N] = real_ave_degree_vec
            ave_L[N] = ave_L_vec
            ave_L_no1[N] = ave_L_vec_no1
            ave_Lsamu[N] = ave_L_samu_vec
            ave_Lsamu_no1[N] = ave_L_samu_vec_no1

            edgelength[N] = ave_edgelength_vec
            hop[N] = ave_hop_vec
            hop_no1[N] = ave_hop_vec_no1

    # plt.plot(kvec,ave_deviation_vec,"o-")
    # plt.xscale('log')
    # plt.show()
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494', '#9FA9C9', '#D36A6A']
    # colors = ["#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    # colors = c
    # colorvec2 = ['#9FA9C9', '#D36A6A']
    for N_index in range(len(Nvec)):
        N = Nvec[N_index]
        x = real_ave_degree_dict[N]
        y = edgelength[N]
        y1 = hop[N]
        y2 = hop_no1[N]

        ana_edgelength = [(100*(k-4.512)**(-0.5)-k/(N-1))/(1-k/(N-1)) for k in x]
        ana_hop = [analtich(N, k) for k in x]
        if hop_flag == True:
            plt.plot(x, y1, linestyle="--", linewidth=3, marker='o', markersize=16,
                     label="inculde 1-hop", color=colors[N_index])
            plt.plot(x, y2, linestyle="--", linewidth=3, marker='o', markersize=16,
                     label="exculde 1-hop", color=colors[N_index + 1])

            plt.plot(x, ana_edgelength, linestyle="-", linewidth=3,
                     label="Samu new model", color=colors[N_index + 2])


            # plt.plot(x, [i-j for (i,j) in zip(y2, y1)], linestyle="--", linewidth=3, marker='o', markersize=16,
            #          label="exclude 1-hop, <h>  - include 1-hop, <h>", color=colors[N_index])

        else:
            plt.plot(x, y, linestyle="--", linewidth=3, marker='o', markersize=16,
                     label="inculde 1-hop", color=colors[N_index])

        # if hop_flag == True:
        #     plt.plot(x, ana_hop, "-", linewidth=5, label=r"$0.52\sqrt{(N-1)\pi}(k-k_c)^{\frac{1}{2}}$")
        # else:
        #     plt.plot(x, ana_edgelength, "-", linewidth=5, label=f"analytic link length, N = {N}")

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # plt.ylim(0.02, 2)
    # plt.yticks([0, 0.1, 0.2, 0.3])
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'Average degree, $\langle D \rangle$', fontsize=26)
    if hop_flag:
        plt.ylabel(r'hopcount', fontsize=26)
    else:
        plt.ylabel(r'link length', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=26, loc=(0.4, 0.7))
    plt.tick_params(axis='both', which="both", length=6, width=1)
    # picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\LocalOptimumdiffNBeta{betan}.png".format(
    #     betan=beta)
    # plt.savefig(picname, format='png', bbox_inches='tight', dpi=600,transparent=True)
    # picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\L_vs_realavg.svg"
    # plt.savefig(
    #     picname,
    #     format="svg",
    #     bbox_inches='tight',  # 紧凑边界
    #     transparent=True  # 背景透明，适合插图叠加
    # )
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.show()
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(9, 6))
    for N_index in range(len(Nvec)):
        N = Nvec[N_index]
        x = real_ave_degree_dict[N]
        y1 = ave_L[N]
        y2 = ave_L_no1[N]
        y3 = ave_Lsamu[N]
        y4 = ave_Lsamu_no1[N]

        plt.plot(x, y1, linestyle="--", linewidth=3, marker='o', markersize=16,
                 label="inculde 1-hop, real L", color=colors[N_index])
        plt.plot(x, y2, linestyle="--", linewidth=3, marker='o', markersize=16,
                 label="exculde 1-hop, real L", color=colors[N_index + 1])
        plt.plot(x, y3, linestyle="--", linewidth=3, marker='o', markersize=16,
                 label="inculde 1-hop, <h><r>", color=colors[N_index + 2])
        plt.plot(x, y4, linestyle="--", linewidth=3, marker='o', markersize=16,
                 label="exculde 1-hop, <h><r>", color=colors[N_index + 3])

        ana_L = [analticL(N, k) for k in x]
        plt.plot(x, ana_L, "-", linewidth=5, label=r"analytic: $y = <r><h>$")


    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # plt.ylim(0.02, 2)
    # plt.yticks([0, 0.1, 0.2, 0.3])
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'Average degree, $\langle D \rangle$', fontsize=26)

    plt.ylabel(r'$\langle L \rangle$', fontsize=26)

    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=26, loc=(0.2, 0.1))
    plt.tick_params(axis='both', which="both", length=6, width=1)
    # picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\LocalOptimumdiffNBeta{betan}.png".format(
    #     betan=beta)
    # plt.savefig(picname, format='png', bbox_inches='tight', dpi=600,transparent=True)
    # picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\L_vs_realavg.svg"
    # plt.savefig(
    #     picname,
    #     format="svg",
    #     bbox_inches='tight',  # 紧凑边界
    #     transparent=True  # 背景透明，适合插图叠加
    # )
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.show()
    plt.close()


def load_large_network_results_dev_vs_avg_locmin_1hopdiff(N, beta, kvec, realL,hopno1):
    folder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\1hopdiff\\"

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

                        real_ave_degree_name = folder_name + "real_ave_degree_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        real_avg = np.loadtxt(real_ave_degree_name)
                        real_ave_degree_vec.append(np.mean(real_avg))

                        if realL:
                            # if L = <d_e>h real stretch
                            # deviation_vec_name = folder_name + "ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                            #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                            # ave_deviation_for_a_para_comb = np.loadtxt(deviation_vec_name)
                            # ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
                            # std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))

                            edgelength_vec_name = folder_name + "ave_edgelength_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        else:
                            # if L = <d_e><h> ave  link length* hopcount
                            edgelength_vec_name = folder_name + "ave_graph_edge_length_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)

                        ave_edgelength_for_a_para_comb = np.loadtxt(edgelength_vec_name)
                        ave_edgelength_vec.append(np.mean(ave_edgelength_for_a_para_comb))
                        std_edgelength_vec.append(np.std(ave_edgelength_for_a_para_comb))

                        hopcount_Name = folder_name + "hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
                            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        hop_vec = np.loadtxt(hopcount_Name, dtype=int)

                        ave_hop_vec.append(np.mean(hop_vec))
                        std_hop_vec.append(np.std(hop_vec))


                        mask = hop_vec != 1
                        hop_vec_no1 = hop_vec[mask]
                        ave_hop_vec_no1.append(np.mean(hop_vec_no1))
                        std_hop_vec_no1.append(np.std(hop_vec_no1))



                        if realL:
                            ave_edgelength_for_a_para_comb_no1 = ave_edgelength_for_a_para_comb[mask]
                            # if L = <d_e>h real stretch
                            if hopno1:
                                L = [x * y for x, y in zip(ave_edgelength_for_a_para_comb_no1, hop_vec_no1)]
                            else:
                                L = [x * y for x, y in zip(ave_edgelength_for_a_para_comb, hop_vec)]
                        else:
                            # if L = <d_e><h> ave  link length* hopcount
                            if hopno1:
                                L = [np.mean(hop_vec_no1) * np.mean(ave_edgelength_for_a_para_comb)]
                            else:
                                L = [np.mean(hop_vec) * np.mean(ave_edgelength_for_a_para_comb)]

                        # # L = np.multiply(ave_edgelength_for_a_para_comb, hop_vec)
                        # L = [x * y for x, y in zip(ave_edgelength_for_a_para_comb, hop_vec)]

                        ave_L_vec.append(np.mean(L))
                        std_L_vec.append(np.std(L))

                    except FileNotFoundError:
                        exemptionlist.append((N, ED, beta, ExternalSimutime))
    print(exemptionlist)
    return real_ave_degree_vec, ave_edgelength_vec, std_edgelength_vec, ave_hop_vec, std_hop_vec, ave_hop_vec_no1, std_hop_vec_no1, ave_L_vec, std_L_vec
    # return kvec, real_ave_degree_vec, ave_deviation_vec, std_deviation_vec


def plot_hopcount_L_Lsamu_with_avg():
    # plot how the hopcount and link length changes with the average degree
    # the x-axis is the input average degree
    Nvec = [10000]

    hop_flag = True

    # Nvec = [100]
    real_ave_degree_dict = {}
    ave_L = {}
    ave_L_no1 = {}
    ave_Lsamu = {}
    ave_Lsamu_no1 = {}
    std_L = {}

    edgelength = {}
    hop = {}
    hop_no1 = {}

    beta_vec = [1024]
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
        for beta in beta_vec:
            kvec = kvec_dict[N]
            real_ave_degree_vec, ave_edgelength_vec, _, ave_hop_vec, _, ave_hop_vec_no1, _, ave_L_vec_no1, _ = load_large_network_results_dev_vs_avg_locmin_1hopdiff(
                N, beta, kvec, True,True)
            _, _, _, _, _, _, _, ave_L_vec, _ = load_large_network_results_dev_vs_avg_locmin_1hopdiff(
                N, beta, kvec, True, False)

            _, _, _, _, _, _, _, ave_L_samu_vec, _ = load_large_network_results_dev_vs_avg_locmin_1hopdiff(
                N, beta, kvec, False, False)

            _, _, _, _, _, _, _, ave_L_samu_vec_no1, _ = load_large_network_results_dev_vs_avg_locmin_1hopdiff(
                N, beta, kvec, False, True)

            real_ave_degree_dict[N] = real_ave_degree_vec
            ave_L[N] = ave_L_vec
            ave_L_no1[N] = ave_L_vec_no1
            ave_Lsamu[N] = ave_L_samu_vec
            ave_Lsamu_no1[N] = ave_L_samu_vec_no1

            edgelength[N] = ave_edgelength_vec
            hop[N] = ave_hop_vec
            hop_no1[N] = ave_hop_vec_no1

    # plt.plot(kvec,ave_deviation_vec,"o-")
    # plt.xscale('log')
    # plt.show()
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494', '#9FA9C9', '#D36A6A']
    # colors = ["#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    # colors = c
    # colorvec2 = ['#9FA9C9', '#D36A6A']
    for N_index in range(len(Nvec)):
        N = Nvec[N_index]
        x = real_ave_degree_dict[N]
        y = edgelength[N]
        y1 = hop[N]
        y2 = hop_no1[N]

        ana_edgelength = [(100*(k-4.512)**(-0.5)-k/(N-1))/(1-k/(N-1)) for k in x]
        ana_hop = [analtich(N, k) for k in x]
        if hop_flag == True:

            plt.plot(x, y1, linestyle="--", linewidth=3, marker='o', markersize=16,
                     label="inculde 1-hop", color=colors[N_index])
            plt.plot(x, y2, linestyle="--", linewidth=3, marker='o', markersize=16,
                     label="exculde 1-hop", color=colors[N_index + 1])

            plt.plot(x, ana_edgelength, linestyle="-", linewidth=3,
                     label="Samu new model", color=colors[N_index + 2])


            # plt.plot(x, [i-j for (i,j) in zip(y2, y1)], linestyle="--", linewidth=3, marker='o', markersize=16,
            #          label="exclude 1-hop, <h>  - include 1-hop, <h>", color=colors[N_index])

        else:
            plt.plot(x, y, linestyle="--", linewidth=3, marker='o', markersize=16,
                     label="inculde 1-hop", color=colors[N_index])

        # if hop_flag == True:
        #     plt.plot(x, ana_hop, "-", linewidth=5, label=r"$0.52\sqrt{(N-1)\pi}(k-k_c)^{\frac{1}{2}}$")
        # else:
        #     plt.plot(x, ana_edgelength, "-", linewidth=5, label=f"analytic link length, N = {N}")

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # plt.ylim(0.02, 2)
    # plt.yticks([0, 0.1, 0.2, 0.3])
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'Average degree, $\langle D \rangle$', fontsize=26)
    if hop_flag:
        plt.ylabel(r'hopcount', fontsize=26)
    else:
        plt.ylabel(r'link length', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=26, loc=(0.4, 0.7))
    plt.tick_params(axis='both', which="both", length=6, width=1)
    # picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\LocalOptimumdiffNBeta{betan}.png".format(
    #     betan=beta)
    # plt.savefig(picname, format='png', bbox_inches='tight', dpi=600,transparent=True)
    # picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\L_vs_realavg.svg"
    # plt.savefig(
    #     picname,
    #     format="svg",
    #     bbox_inches='tight',  # 紧凑边界
    #     transparent=True  # 背景透明，适合插图叠加
    # )
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.show()
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(9, 6))
    for N_index in range(len(Nvec)):
        N = Nvec[N_index]
        x = real_ave_degree_dict[N]
        y1 = ave_L[N]
        y2 = ave_L_no1[N]
        y3 = ave_Lsamu[N]
        y4 = ave_Lsamu_no1[N]

        print("data:")
        print(x)
        print(y2)
        print(y3)

        # plt.plot(x, y1, linestyle="--", linewidth=3, marker='o', markersize=16,
        #          label="inculde 1-hop, real L", color=colors[N_index])
        plt.plot(x, y2, linestyle="--", linewidth=3, marker='o', markersize=16,
                 label="real: L", color=colors[N_index + 1])
        plt.plot(x, y3, linestyle="--", linewidth=3, marker='o', markersize=16,
                 label="approx: <r><h>", color=colors[N_index + 2])
        plt.plot(x, y4, linestyle="--", linewidth=3, marker='o', markersize=16,
                 label="exculde 1-hop, <r><h>", color=colors[N_index + 3])

        ana_L = [analticL(N, k) for k in x]
        plt.plot(x, ana_L, "-", linewidth=5, label=r"ana: $y = <r><h>$")

        x3 = np.linspace(5, 20, 10000)
        ana_vec_real = [0.01*analtich(N, k) for k in x3]
        ana_vec_app = [0.007*analtich(N, k) for k in x3]

        # plt.plot(x, [x_value ** 0.5 * (0.12 + 1.28 * np.log(10000) / np.log(x_value)) * np.log(x_value) for x_value in x],
        #          label=f"fit2: analytic formula: Llog(k)~k^0.25")

        plt.plot(x3, ana_vec_real, "--", linewidth=5, label=r"$ana: <S> = c_1(k - k_c)^{-1/2}$")
        plt.plot(x3, ana_vec_app, "--", linewidth=5, label=r"$ana: <S> = c_2(k - k_c)^{-1/2}$")

        x4 = np.linspace(1000, 10000, 10000)
        ana_vec_real_tail = [2.6*analticdl(N, k) for k in x4]
        ana_vec_app_tail = [2.1*analticdl(N, k) for k in x4]
        plt.plot(x4, ana_vec_real_tail, "--", linewidth=5, label=r"$ana: <S> = c_3* \frac{2}{3}\sqrt{\frac{k}{N\pi}}\left( 1 + \frac{4}{3\pi} \sqrt{\frac{k}{N\pi}} \right)$")
        plt.plot(x4, ana_vec_app_tail, "--", linewidth=5, label=r"$ana: <S> = c_4* \frac{2}{3}\sqrt{\frac{k}{N\pi}}\left( 1 + \frac{4}{3\pi} \sqrt{\frac{k}{N\pi}} \right)$")





    plt.text(1.2, 2.6,
             r"$<r><h> = \frac{2}{3} "
             r"\sqrt{\frac{k}{N\pi}} "
             r"\left( 1 + \frac{4}{3\pi} \sqrt{\frac{k}{N\pi}} \right)"
             r"C (k - k_c)^{-1/2}$",
             fontsize=18)    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    plt.ylim(0, 2.6)
    # plt.yticks([0, 0.1, 0.2, 0.3])
    # plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'Average degree, $\langle D \rangle$', fontsize=26)

    plt.ylabel(r'$\langle L \rangle$', fontsize=26)

    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=26, loc=(0.5, 0.45))
    plt.tick_params(axis='both', which="both", length=6, width=1)
    # picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\LocalOptimumdiffNBeta{betan}.png".format(
    #     betan=beta)
    # plt.savefig(picname, format='png', bbox_inches='tight', dpi=600,transparent=True)
    # picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\L_vs_realavg.svg"
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
    # Figure 4 supp

    """
    # STEP 1 <L> and <h> versus real average degree beta = 1024 whether 1hop is included
    """

    # plot_hopcount_L_Lsamu_with_avg_whether_1hopincluded()

    """
   # STEP 2 <L> and <r><h> versus real average degree beta = 1024
   """
    plot_hopcount_L_Lsamu_with_avg()





