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

from stretchL_diffNkbeta_SRGG_ub import generate_ED_log_unifrom
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


def load_large_network_results_dev_vs_avg(N, beta, kvec, realL):
    if realL:
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


                        if realL:
                            #if L = <d_e>h real stretch
                            deviation_vec_name = folder_name + "ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                            ave_deviation_for_a_para_comb = np.loadtxt(deviation_vec_name)
                            ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
                            std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))


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

                        ave_hop_vec.append(np.mean(hop_vec))
                        std_hop_vec.append(np.std(hop_vec))


                        if realL:
                            #if L = <d_e>h real stretch
                            L = [x * y for x, y in zip(ave_edgelength_for_a_para_comb, hop_vec)]
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



def analticL(N, k_vals):
    pi = np.pi
    # k 的取值范围
    k_vals = np.array(k_vals)
    # 计算 h(k)

    kc = 4.512
    C = 0.52*np.sqrt(N*pi)
    # print(N,C)
    h_vals = (2 / 3) * np.sqrt(k_vals / (N * pi)) *(1+4/(3*pi)*np.sqrt(k_vals / (N * pi)))* C * (k_vals - kc) ** (-0.5)
    return h_vals


def analtich(N, k_vals):
    pi = np.pi
    # k 的取值范围
    k_vals = np.array(k_vals)
    # 计算 h(k)
    kc = 4.512
    C = 0.52*np.sqrt(N*pi)
    # print(N,C)
    h_vals = C * (k_vals - kc) ** (-0.5)
    return h_vals


def analticdl(N, k_vals):
    pi = np.pi
    # k 的取值范围
    k_vals = np.array(k_vals)
    h_vals = (2 / 3) * np.sqrt(k_vals / (N * pi)) *(1+4/(3*pi)*np.sqrt(k_vals / (N * pi)))
    return h_vals


def plot_Lavedaveh_with_avg():
    # the x-axis is the input average degree
    Nvec = [215, 464, 1000, 2154, 4642,10000]

    # Nvec = [100]
    real_ave_degree_dict = {}
    ave_L = {}
    std_L = {}


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
                real_ave_degree_vec, _, _, _, _, _, _, ave_L_vec, std_L_vec = load_large_network_results_dev_vs_avg(
                    N, beta, kvec)
                real_ave_degree_dict[N] = real_ave_degree_vec
                ave_L[N] = ave_L_vec
                std_L[N] = std_L_vec

    # plt.plot(kvec,ave_deviation_vec,"o-")
    # plt.xscale('log')
    # plt.show()
    lengend = [r"$N=100$", r"$N=215$", r"$N=464$", r"$N=1000$", r"$N=2154",r"$N=4642",r"$N=10000"]
    lengend = [r"$N=215$", r"$N=464$", r"$N=1000$", r"$N=2154$", r"$N=4642$", r"$N=10000$"]
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494','#9FA9C9', '#D36A6A']
    # colors = ["#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
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
        plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=16,
                     label=lengend[N_index], color=colors[N_index])

    k_c = 4.512
    k_star = [k_c**(2/3) * np.pi * (4/3)**(2/3) * N**(1/3) for N in Nvec]
    y2 = [analticL(N, k) for (N, k) in zip(Nvec, k_star)]
    plt.plot(k_star,y2,"--o",label = "analytic local minimum")

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
    plt.legend(fontsize=26, loc=(0.5, 0.05))
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




def plot_L_with_avg():
    # the x-axis is the input average degree
    Nvec = [215, 464, 1000, 2154, 4642,10000]

    realL = True

    # Nvec = [100]
    real_ave_degree_dict = {}
    ave_L = {}
    std_L = {}


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
                real_ave_degree_vec, _, _, _, _, _, _, ave_L_vec, std_L_vec = load_large_network_results_dev_vs_avg(
                    N, beta, kvec,realL)
                real_ave_degree_dict[N] = real_ave_degree_vec
                ave_L[N] = ave_L_vec
                std_L[N] = std_L_vec

    # plt.plot(kvec,ave_deviation_vec,"o-")
    # plt.xscale('log')
    # plt.show()
    lengend = [r"$N=100$", r"$N=215$", r"$N=464$", r"$N=1000$", r"$N=2154",r"$N=4642",r"$N=10000"]
    lengend = [r"$N=215$", r"$N=464$", r"$N=1000$", r"$N=2154$", r"$N=4642$", r"$N=10000$"]
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494','#9FA9C9', '#D36A6A']
    # colors = ["#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
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
        plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=16,
                     label=lengend[N_index], color=colors[N_index])

    k_c = 4.512
    k_star = [k_c**(2/3) * np.pi * (4/3)**(2/3) * N**(1/3) for N in Nvec]
    y2 = [analticL(N, k) for (N, k) in zip(Nvec, k_star)]
    plt.plot(k_star,y2,"-o",markersize = 25,markerfacecolor='none',linewidth=5,label = "analytic local minimum")
    print(k_star)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    plt.ylim(0.02, 2)
    # plt.yticks([0, 0.1, 0.2, 0.3])
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'Average degree, $\langle D \rangle$', fontsize=26)
    if realL:
        plt.ylabel(r'Average stretch, $\langle L \rangle$', fontsize=26)
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



def plot_L_with_avg2():
    # the x-axis is the input average degree
    # Nvec = [100, 215, 464, 1000, 2154, 4642,10000]
    Nvec = [100, 1000, 10000]
    # Nvec = [100]
    real_ave_degree_dict = {}
    ave_L = {}
    std_L = {}

    kvec_dict = {
        100: [2, 3, 4, 5, 6, 8, 10, 12, 14, 17, 22, 27, 33, 40, 49, 60, 73, 89, 113, 149, 198, 260, 340, 446, 584,
              762, 993, 1292, 1690, 2276, 3142, 4339],
        1000: [2, 3, 4, 5, 6, 7, 8, 11, 15, 20, 28, 40, 58, 83, 118, 169, 241, 344, 490, 700, 999, 1425, 2033, 2900,
               4139, 5909, 8430, 12039, 17177, 24510, 34968, 49887, 71168],
        10000: [2.2,  3.0, 3.8, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276,
                3727, 6105,
                9999, 16479, 27081, 44767, 73534, 121205, 199999]}

    # beta_vec = [1024]
    # kvec_dict = {
    #     100: [2, 3, 5, 8, 12, 18, 29, 45, 70, 109, 169, 264, 412, 642, 1000],
    #     215: [2, 3, 5, 9, 14, 24, 39, 63, 104, 170, 278, 455, 746, 1221, 2000],
    #     464: [2, 3, 6, 10, 18, 30, 52, 89, 154, 265, 456, 785, 1350, 2324, 4000],
    #     1000: [2, 4, 7, 12, 21, 39, 70, 126, 229, 414, 748, 1353, 2446, 4424, 8000],
    #     2154: [2, 4, 7, 14, 27, 52, 99, 190, 364, 697, 1335, 2558, 4902, 9393, 18000],
    #     4642: [2, 4, 8, 16, 33, 67, 135, 272, 549, 1107, 2234, 4506, 9091, 18340, 37000],
    #     10000: [2.2, 2.8, 3.0, 3.4, 3.8, 4.4, 6.0, 7.0, 8.0, 9.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389,
    #             2276,
    #             3727, 6105,
    #             9999, 16479, 27081, 44767, 73534, 121205, 199999]}



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

    k_c = 4.512
    k_star = [k_c**(2/3) * np.pi * (4/3)**(2/3) * N**(1/3) for N in Nvec]
    y2 = [analticL(N, k) for (N, k) in zip(Nvec, k_star)]
    plt.plot(k_star,y2,"--o")

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
    # plt.savefig(
    #     picname,
    #     format="svg",
    #     bbox_inches='tight',  # 紧凑边界
    #     transparent=True  # 背景透明，适合插图叠加
    # )
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.show()
    plt.close()



def plot_hopcount_linklength_with_avg():
    # plot how the hopcount and link length changes with the average degree
    # the x-axis is the input average degree
    Nvec = [215, 1000, 4642,10000]

    realL = True

    hop_flag = True

    # Nvec = [100]
    real_ave_degree_dict = {}
    ave_L = {}
    std_L = {}

    edgelength = {}
    hop = {}


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
                real_ave_degree_vec, _, _, ave_edgelength_vec, _, ave_hop_vec, _, ave_L_vec, std_L_vec = load_large_network_results_dev_vs_avg(
                    N, beta, kvec,realL)

                real_ave_degree_dict[N] = real_ave_degree_vec
                ave_L[N] = ave_L_vec
                std_L[N] = std_L_vec
                edgelength[N] = ave_edgelength_vec
                hop[N] = ave_hop_vec


    # plt.plot(kvec,ave_deviation_vec,"o-")
    # plt.xscale('log')
    # plt.show()
    lengend = [r"$N=100$", r"$N=215$", r"$N=464$", r"$N=1000$", r"$N=2154",r"$N=4642",r"$N=10000"]
    lengend = [r"$N=215$", r"$N=1000$", r"$N=4642$", r"$N=10000$"]
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494','#9FA9C9', '#D36A6A']
    # colors = ["#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
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
            y = edgelength[N]
            y1 = hop[N]
            ana_edgelength = [analticdl(N,k) for k in x]
            ana_hop = [analtich(N,k) for k in x]
        if hop_flag == True:
            plt.plot(x, y1, linestyle="--", linewidth=3, marker='o', markersize=16,
                         label=lengend[N_index], color=colors[N_index])
        else:
            plt.plot(x, y, linestyle="--", linewidth=3, marker='o', markersize=16,
                     label=lengend[N_index], color=colors[N_index])

        if hop_flag == True:
            plt.plot(x, ana_hop, "-", linewidth=5, label=f"analytic hopcount, N = {N}")
        else:
            plt.plot(x, ana_edgelength, "-", linewidth=5, label=f"analytic link length, N = {N}")


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
    plt.legend(fontsize=26, loc=(0.5, 0.5))
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



def load_large_network_results_dev_vs_avg_locmin_hunter(N, beta, kvec, realL):
    folder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\localmin_hunter\\"

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

                        hop_vec_no1 = hop_vec[hop_vec != 1]

                        if realL:
                            if len(ave_edgelength_for_a_para_comb) != len(hop_vec_no1):
                                ave_edgelength_for_a_para_comb_no1 = ave_edgelength_for_a_para_comb[hop_vec != 1]
                            #if L = <d_e>h real stretch
                                L = [x * y for x, y in zip(ave_edgelength_for_a_para_comb_no1, hop_vec_no1)]
                                # L = [x * y for x, y in zip(ave_edgelength_for_a_para_comb, hop_vec)]
                            else:
                                L = [x * y for x, y in zip(ave_edgelength_for_a_para_comb, hop_vec_no1)]

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



def plot_L_with_avg_loc():
    # the x-axis is the input average degree
    # Nvec = [215, 464, 1000, 2154, 4642,10000]
    # Nvec = [10000]
    Nvec = [464, 681, 1000, 1468,2154,3156, 4642,6803,10000]


    # Nvec = [10000]
    # beta = 1024
    # beta = 2.1
    beta = 2.1
    realL = True

    real_ave_degree_dict = {}
    ave_L = {}
    std_L = {}

    real_ave_degree_dict_0 = {}
    ave_L_0 = {}
    std_L_0 = {}

    k_star_dict = {}
    localmin_dict = {}



    if beta == 1024:
        # kvec_dict = {
        #     100: [2, 3, 5, 8, 12, 18, 29, 45, 70, 109, 169, 264, 412, 642, 1000],
        #     215: [2, 3, 5, 9, 14, 24, 39, 63, 104, 170, 278, 455, 746, 1221, 2000],
        #     464: [2, 3, 6, 10, 18, 30, 52, 89, 154, 265, 456, 785, 1350, 2324, 4000],
        #     1000: [2, 4, 7, 12, 21, 39, 70, 126, 229, 414, 748, 1353, 2446, 4424, 8000],
        #     2154: [2, 4, 7, 14, 27, 52, 99, 190, 364, 697, 1335, 2558, 4902, 9393, 18000],
        #     4642: [2, 4, 8, 16, 33, 67, 135, 272, 549, 1107, 2234, 4506, 9091, 18340, 37000],
        #     10000: [2.2, 2.8, 3.0, 3.4, 3.8, 4.4, 6.0, 7.0, 8.0, 9.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389,
        #             2276,
        #             3727, 6105,
        #             9999, 16479, 27081, 44767, 73534, 121205, 199999]}
        kvec_dict = {215: list(range(24, 104 + 1, 2)), 464: list(range(30, 154 + 1, 2)), 1000: list(range(39, 229 + 1, 2)),
                     2154: list(range(52, 364 + 1, 2)), 4642: list(range(67, 272 + 1, 2)), 10000: list(range(118, 316 + 1, 2)),
                     681: list(range(40, 164 + 1, 2)), 1468: list(range(50, 240 + 1, 2)), 3156: list(range(72, 384 + 1, 2)),
                     6803: list(range(87,295 + 1, 2)), 14683: list(range(140, 340 + 1, 2))}
        # kvec_dict = {215: [2, 3, 5, 9, 14] + list(range(24, 104 + 1, 2)) + [170, 278, 455, 746, 1221, 2000],
        #           464: list(range(30, 154 + 1, 2)), 1000: list(range(39, 229 + 1, 2)), 2154: list(
        #         range(52, 364 + 1, 2)), 4642: list(range(67, 272 + 1, 2)), 10000: list(range(118, 316 + 1, 2))}

        kvec_dict_0 = {
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

    elif beta == 2.1:
        kvec_dict = {
            464: generate_ED_log_unifrom(2, 1000000, 12),
            681: generate_ED_log_unifrom(2, 1000000, 12),
            1000: generate_ED_log_unifrom(2, 1000000, 12),
            1468: generate_ED_log_unifrom(2, 1000000, 12),
            2154: generate_ED_log_unifrom(2, 1000000, 12),
            3156: generate_ED_log_unifrom(2, 1000000, 12),
            4642: generate_ED_log_unifrom(2, 1000000, 12),
            6803: generate_ED_log_unifrom(2, 1000000, 12),
            10000: generate_ED_log_unifrom(2, 1000000, 12)}
        # kvec_tem = generate_ED_log_unifrom(523, 3331, 30) # find local optimum
        # kvec_tem1 = generate_ED_log_unifrom(236, 523, 30) # find local optimum extra for N<=1000
        # kvec_tem1_smallN = kvec_tem1[:-1]+kvec_tem
        # # print(kvec_tem1_smallN)
        # kvec_dict = {
        #     464: kvec_tem1_smallN,
        #     681: kvec_tem1_smallN,
        #     1000: kvec_tem1_smallN,
        #     1468: kvec_tem,
        #     2154: kvec_tem,
        #     3156: kvec_tem,
        #     4642: kvec_tem,
        #     6803: kvec_tem,
        #     10000: kvec_tem}
    else:
        # beta ==3.1
        kvec_dict = {
            464: generate_ED_log_unifrom(2, 100000, 12),
            681: generate_ED_log_unifrom(2, 100000, 12),
            1000: generate_ED_log_unifrom(2, 100000, 12),
            1468: generate_ED_log_unifrom(2, 100000, 12),
            2154: generate_ED_log_unifrom(2, 100000, 12),
            3156: generate_ED_log_unifrom(2, 100000, 12),
            4642: generate_ED_log_unifrom(2, 1000000, 12),
            6803: generate_ED_log_unifrom(2, 1000000, 12),
            10000: generate_ED_log_unifrom(2, 1000000, 12)}


    for N in Nvec:
        kvec = kvec_dict[N]
        real_ave_degree_vec, _, _, _, _, _, _, ave_L_vec, std_L_vec = load_large_network_results_dev_vs_avg_locmin_hunter(
            N, beta, kvec,realL)
        real_ave_degree_dict[N] = real_ave_degree_vec
        ave_L[N] = ave_L_vec
        std_L[N] = std_L_vec

        if beta==1024:
            kvec2 = kvec_dict_0[N]
            real_ave_degree_vec_0, _, _, _, _, _, _, ave_L_vec_0, std_L_vec_0 = load_large_network_results_dev_vs_avg(
                N, beta, kvec2, realL)
            real_ave_degree_dict_0[N] = real_ave_degree_vec_0
            ave_L_0[N] = ave_L_vec_0
            std_L_0[N] = std_L_vec_0


    # plt.plot(kvec,ave_deviation_vec,"o-")
    # plt.xscale('log')
    # plt.show()
    # legend_vec = [r"$N=100$", r"$N=215$", r"$N=464$", r"$N=1000$", r"$N=2154",r"$N=4642",r"$N=10000"]
    # legend_vec = [r"$N=215$", r"$N=464$", r"$N=1000$", r"$N=2154$", r"$N=4642$", r"$N=10000$"]
    fig, ax = plt.subplots(figsize=(9, 6))

    # colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494','#9FA9C9', '#D36A6A']
    # colors = ["#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']

    colors = plt.get_cmap('tab10').colors[:len(Nvec)]


    # colorvec2 = ['#9FA9C9', '#D36A6A']
    for N_index in range(len(Nvec)):
        N = Nvec[N_index]

        x = real_ave_degree_dict[N]
        y = ave_L[N]
        # plt.plot(x,y,label="test")
        error = std_L[N]
        print(generate_ED_log_unifrom(2, 1000000, 12))
        print(x)
        print(y)

        if beta==1024:
            # this is for the original data to show all the case for beta=1024. If we only want to see the local minimum,
            # annotate this "if" out
            x_0 = real_ave_degree_dict_0[N]
            y_0 = ave_L_0[N]
            error_0 = std_L_0[N]

            x_final = np.concatenate([x,x_0])
            y_final = np.concatenate([y,y_0])
            error_final = np.concatenate([error, error_0])
            sorted_index = np.argsort(x_final)
            x_final = x_final[sorted_index]
            y_final = y_final[sorted_index]
            error_final = error_final[sorted_index]

            plt.errorbar(x_final, y_final, yerr=error_final, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=16,
                         label=N, color=colors[N_index])


        plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=16,
                     label=N, color=colors[N_index])

        x = np.array(x)
        y = np.array(y)
        mask = ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        mask = x > 6
        x_sub = x[mask]
        y_sub = y[mask]

        min_index = np.argmin(y_sub)
        x_min = x_sub[min_index]
        y_min = y_sub[min_index]

        k_star_dict[N] = x_min
        localmin_dict[N] = y_min
        print(f"N:{N}")
        print(f"最小值点: x = {x_min}, y = {y_min}")

        # plt.scatter(x_min, y_min,
        #             marker='s', s=1000,
        #             facecolors='none', edgecolors='red', linewidths=2,
        #             label='min')



    k_c = 4.512
    k_star = [k_c**(2/3) * np.pi * (4/3)**(2/3) * N**(1/3) for N in Nvec]
    y2 = [analticL(N, k) for (N, k) in zip(Nvec, k_star)]
    # plot the analytic results
    if beta ==1024:
        plt.plot(k_star,y2,"-o",markersize = 25,markerfacecolor='none',linewidth=5,label = "analytic local minimum")
        print(k_star)

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    plt.ylim(0.02, 2)
    # plt.yticks([0, 0.1, 0.2, 0.3])

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'Average degree, $\langle D \rangle$', fontsize=26)
    if realL:
        plt.ylabel(r'Average stretch, $\langle L \rangle$', fontsize=26)
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


    # figure curve fit: load data:
    # Nvec = [215, 464, 1000, 2154, 4642,10000]
    Nvec = [464, 681, 1000, 1468, 2154, 3156, 4642, 6803, 10000]
    y_k_star_vec = []
    y_local_minimum = []
    for N in Nvec:
        y_k_star_vec.append(k_star_dict[N])
        y_local_minimum.append(localmin_dict[N])
    k_c = 4.512
    k_star = [k_c ** (2 / 3) * np.pi * (4 / 3) ** (2 / 3) * N ** (1 / 3) for N in Nvec]
    y2 = [analticL(N, k) for (N, k) in zip(Nvec, k_star)]
    # plt.plot(k_star, y2, "-o", markersize=25, markerfacecolor='none', linewidth=5, label="")


    figure()
    # figure for k^* as a function of N
    plt.plot(Nvec, y_k_star_vec, "-o",markersize=25, linewidth=5, label=r"simulation:$k^*$")

    popt, pcov = curve_fit(power_law, Nvec, y_k_star_vec)
    a_fit, b_fit = popt
    print("拟合参数: a = %.4f, b = %.4f" % (a_fit, b_fit))
    plt.plot(Nvec, power_law(Nvec, a_fit, b_fit), label=f"Fit: y = {a_fit:.2f} * x^{b_fit:.2f}", color='red')

    if beta==1024:
        plt.plot(Nvec, k_star, "--", markersize=25, markerfacecolor='none', linewidth=5, label=r"$k^* = k_c^{\frac{2}{3}}\pi \frac{4}{3}^{\frac{2}{3}} N^{\frac{1}{3}} $")
        if realL:
            plt.plot(Nvec, [0.33*i for i in k_star], "--", markersize=25, markerfacecolor='none', linewidth=5,
                     label=r"$k^* = 0.33 k_c^{\frac{2}{3}}\pi \frac{4}{3}^{\frac{2}{3}} N^{\frac{1}{3}} $")
        else:
            plt.plot(Nvec, [0.6 * i for i in k_star], "--", markersize=25, markerfacecolor='none', linewidth=5,
                     label=r"$k^* = 0.6 k_c^{\frac{2}{3}}\pi \frac{4}{3}^{\frac{2}{3}} N^{\frac{1}{3}} $")

    plt.xlabel(r'Network size, $N$', fontsize=26)
    plt.ylabel(r'Critical Degree, $k^*$', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.yscale('log')
    plt.xscale('log')
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=26, loc=(0.6, 0.05))
    plt.tick_params(axis='both', which="both", length=6, width=1)
    plt.show()
    plt.close()



    figure()
    # local minimum as a funtion of N
    plt.plot(Nvec,y_local_minimum, "-o",markersize=25, linewidth=5, label=r"simulation: local minimum $\langle L \rangle$")
    print(y_local_minimum)

    if beta ==1024:
        plt.plot(Nvec, y2, "--", markersize=25, markerfacecolor='none', linewidth=5,
                 label=r"$y = f_{\langle L \rangle} (k^*) $")
        if realL:
            plt.plot(Nvec, [1.5*i for i in y2], "--", markersize=25, markerfacecolor='none', linewidth=5,
                     label=r"$y = 1.5 f_{\langle L \rangle} (k^*) $")

        else:
            plt.plot(Nvec, [1.07 * i for i in y2], "--", markersize=25, markerfacecolor='none', linewidth=5,
                     label=r"$y = 1.07 f_{\langle L \rangle} (k^*) $")

    popt, pcov = curve_fit(power_law, Nvec, y_local_minimum)
    a_fit, b_fit = popt
    print("拟合参数: a = %.4f, b = %.4f" % (a_fit, b_fit))
    plt.plot(Nvec, power_law(Nvec, a_fit, b_fit), label=f"Fit: y = {a_fit:.2f} * x^{b_fit:.2f}", color='red')

    # x = np.array(Nvec)
    # b_fit = -0.13
    # plt.plot(x, a_fit * x**b_fit, label=f"Fit: y = {a_fit:.2f} * x^{b_fit:.2f}", color='red')


    # latex_expr = r"$f_{\langle L \rangle} = \frac{2}{3}\sqrt{\frac{k}{N\pi}}\left(1+\frac{4}{3\pi}\sqrt{\frac{k}{N\pi}}\right) 0.52\sqrt{N\pi} (k - k_c)^{-\frac{1}{2}}$"
    # plt.text(1000, 0.65, latex_expr, fontsize=26)

    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'Network size, $N$', fontsize=26)
    plt.ylabel(r'Local minimum stretech, $\langle L \rangle_{min}$', fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    if realL:
        plt.legend(fontsize=26, loc=(0.5, 0.1))
    else:
        plt.legend(fontsize=26, loc=(0.5, 0.6))
        # plt.yl/im([min(y2), 0.65])
    plt.tick_params(axis='both', which="both", length=6, width=1)
    plt.show()
    plt.close()


def model_hop_c(x, C):
    return C * (x - 4.512)**(-0.5)

# 定义拟合函数
def fit_and_plot_hop_C(x, y):
    # 用curve_fit拟合参数C
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # 去除NaN和Inf

    x = x[5:-3]
    y = y[5:-3]


    popt, pcov = curve_fit(model_hop_c, x, y)
    C_fit = popt[0]

    print(f"拟合得到的C = {C_fit:.6f}")

    x_fit = np.linspace(min(x), max(x), 200)
    y_fit = model_hop_c(x_fit, C_fit)

    return C_fit,x_fit,y_fit


def hopcount_fit():
    Nvec = [464, 1000, 2154, 4642, 10000]
    # Nvec = [215]
    beta = 1024
    beta = 2.1
    realL = True

    real_ave_degree_dict = {}
    ave_L = {}
    std_L = {}

    real_ave_degree_dict_0 = {}
    ave_L_0 = {}
    std_L_0 = {}

    k_star_dict = {}
    localmin_dict = {}

    hop_dict ={}

    if beta == 1024:
        # kvec_dict = {215: list(range(24, 104 + 1, 2)), 464: list(range(30, 154 + 1, 2)),
        #              1000: list(range(39, 229 + 1, 2)),
        #              2154: list(range(52, 364 + 1, 2)), 4642: list(range(67, 272 + 1, 2)),
        #              10000: list(range(118, 316 + 1, 2)),
        #              681: list(range(40, 164 + 1, 2)), 1468: list(range(50, 240 + 1, 2)),
        #              3156: list(range(72, 384 + 1, 2)),
        #              6803: list(range(87, 295 + 1, 2)), 14683: list(range(140, 340 + 1, 2))}
        # kvec_dict = {215: [2, 3, 5, 9, 14] + list(range(24, 104 + 1, 2)) + [170, 278, 455, 746, 1221, 2000],
        #           464: list(range(30, 154 + 1, 2)), 1000: list(range(39, 229 + 1, 2)), 2154: list(
        #         range(52, 364 + 1, 2)), 4642: list(range(67, 272 + 1, 2)), 10000: list(range(118, 316 + 1, 2))}

        kvec_dict_0 = {
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
    elif beta == 2.1:
        kvec_dict = {
            464: generate_ED_log_unifrom(2, 1000000, 12),
            681: generate_ED_log_unifrom(2, 1000000, 12),
            1000: generate_ED_log_unifrom(2, 1000000, 12),
            1468: generate_ED_log_unifrom(2, 1000000, 12),
            2154: generate_ED_log_unifrom(2, 1000000, 12),
            3156: generate_ED_log_unifrom(2, 1000000, 12),
            4642: generate_ED_log_unifrom(2, 1000000, 12),
            6803: generate_ED_log_unifrom(2, 1000000, 12),
            10000: generate_ED_log_unifrom(2, 1000000, 12)}
    figure()
    c_vec = []
    c_vec_get = [np.float64(23.53467376631608), np.float64(33.68320434287138), np.float64(48.457378123702426),
         np.float64(69.47011764188508), np.float64(100)]

    count = 0
    for N in Nvec:
        # kvec = kvec_dict[N]
        # real_ave_degree_vec, _, _, _, _, _, _, ave_L_vec, std_L_vec = load_large_network_results_dev_vs_avg_locmin_hunter(
        #     N, beta, kvec, realL)
        # real_ave_degree_dict[N] = real_ave_degree_vec
        # ave_L[N] = ave_L_vec
        # std_L[N] = std_L_vec
        if beta ==1024:
            kvec2 = kvec_dict_0[N]
            real_ave_degree_vec_0, _, _, _, _, hop_vec_0, _, ave_L_vec_0, std_L_vec_0 = load_large_network_results_dev_vs_avg(
                N, beta, kvec2, realL)
            hop_dict[N] = hop_vec_0
        else:
            kvec2 = kvec_dict[N]
            real_ave_degree_vec_0, _, _, _, _, hop_vec_0, _, ave_L_vec_0, std_L_vec_0 = load_large_network_results_dev_vs_avg_locmin_hunter(
                N, beta, kvec2, realL)
            hop_dict[N] = hop_vec_0



        plt.plot(real_ave_degree_vec_0,hop_vec_0,"-o",markersize = 25,markerfacecolor='none',linewidth=5,label = f"{N}")
        # C_fit,fitx,fity = fit_and_plot_hop_C(real_ave_degree_vec_0, hop_vec_0)
        # plt.plot(fitx, fity, "--", markersize=25, markerfacecolor='none', linewidth=5,
        #          label=f"y = {C_fit:.4f}(x-4.512)")
        # c_vec.append(C_fit)
        # y2 = [analtich(N,i) for i in real_ave_degree_vec_0]

        # if beta ==1024:
        #     c_get = c_vec_get[count]
        #     plt.plot(real_ave_degree_vec_0,[c_get*(i-4.512)**(-0.5) for i in real_ave_degree_vec_0],label = f"C:{c_get:.2f}")
        # else:
        #     y3 = [np.log(N) / np.log(i) for i in real_ave_degree_vec_0]
        #     plt.plot(real_ave_degree_vec_0, y3, "--", linewidth=5,
        #              label=f"{N}")
        count = count+1

    print(c_vec)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.yscale('log')
    plt.xscale('log')
    # plt.xlabel(r'Network size, $N$', fontsize=26)
    plt.xlabel(r'$k$', fontsize=26)
    plt.ylabel(r' $\langle h \rangle$', fontsize=26)
    plt.legend(fontsize=12, loc=(0.7, 0.2))
    plt.show()



def hopcount_fit_C():
    # C IS COMPUTED FROM HOPCOUNT_FIT
    Nvec = [464, 1000, 2154, 4642, 10000]
    figure()
    c = [np.float64(23.53467376631608), np.float64(33.68320434287138), np.float64(48.457378123702426),
         np.float64(69.47011764188508), np.float64(100)]

    plt.plot(Nvec, c,"--o",markersize = 25,linewidth=5)

    popt, pcov = curve_fit(power_law, Nvec, c)
    a_fit, b_fit = popt
    print("拟合参数: a = %.4f, b = %.4f" % (a_fit, b_fit))
    plt.plot(Nvec, power_law(Nvec, a_fit, b_fit), "-", label=f"Fit: y = {a_fit:.2f} * x^{b_fit:.2f}",linewidth=5)

    plt.yscale('log')
    plt.xscale('log')
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.legend(fontsize=26, loc=(0.5, 0.2))
    # plt.xlabel(r'Network size, $N$', fontsize=26)
    plt.xlabel(r'$N$', fontsize=26)
    plt.ylabel(r'$C$', fontsize=26)
    plt.show()


def analytic_local_min_check():
    # beta = 1024
    Nvec = [464, 681, 1000, 1468, 2154, 3156, 4642, 6803, 10000]
    y_local_minimum = [np.float64(0.591191617025838), np.float64(0.5870473006566197), np.float64(0.5718626910902538),
     np.float64(0.5719635831757929), np.float64(0.5632705176766406), np.float64(0.5585420800829218),
     np.float64(0.553173242642201), np.float64(0.5494322574352855), np.float64(0.5449752296523072)]
    k_c = 4.512
    k_star = [k_c ** (2 / 3) * np.pi * (4 / 3) ** (2 / 3) * N ** (1 / 3) for N in Nvec]
    y2 = [analticL(N, k) for (N, k) in zip(Nvec, k_star)]


    figure()
    plt.plot(Nvec,y_local_minimum, "-o",markersize=25, linewidth=5, label=r"simulation: local minimum $\langle L \rangle$")
    print(y_local_minimum)

    # plt.plot(Nvec, y2, "--", markersize=25, markerfacecolor='none', linewidth=5,
    #          label=r"$y = f_{\langle L \rangle} (k^*) $")
    #
    # plt.plot(Nvec, [1.5*i for i in y2], "--", markersize=25, markerfacecolor='none', linewidth=5,
    #          label=r"$y = 1.5 f_{\langle L \rangle} (k^*) $")

    popt, pcov = curve_fit(power_law, Nvec, y_local_minimum)
    a_fit, b_fit = popt
    print("拟合参数: a = %.4f, b = %.4f" % (a_fit, b_fit))
    plt.plot(Nvec, power_law(Nvec, a_fit, b_fit), label=fr"Fit: $y = {a_fit:.2f} x^{{{b_fit:.2f}}}$", linewidth=5)

    # x = np.array(Nvec)
    # b_fit = -0.13
    # plt.plot(x, a_fit * x**b_fit, label=f"Fit: y = {a_fit:.2f} * x^{b_fit:.2f}", color='red')
    Nvec = [464, 1000, 2154, 4642, 10000]
    c = [np.float64(23.53467376631608), np.float64(33.68320434287138), np.float64(48.457378123702426),
         np.float64(69.47011764188508), np.float64(100)]

    y3=[2*c_value/3/np.sqrt(np.pi)*N**(-0.5)  for (N,c_value) in zip(Nvec,c)]
    plt.plot(Nvec, y3, "--",label=r"$y = \frac{2C}{3\sqrt{\pi}}N^{-1/2}$",linewidth=5)
    plt.plot(Nvec, [1.45*i for i in y3], "--",label=r"$y = 1.45\frac{2C}{3\sqrt{\pi}}N^{-1/2}$", linewidth=5)


    # latex_expr = r"$f_{\langle L \rangle} = \frac{2}{3}\sqrt{\frac{k}{N\pi}}\left(1+\frac{4}{3\pi}\sqrt{\frac{k}{N\pi}}\right) 0.52\sqrt{N\pi} (k - k_c)^{-\frac{1}{2}}$"
    # plt.text(1000, 0.65, latex_expr, fontsize=26)

    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'Network size, $N$', fontsize=26)
    plt.ylabel(r'Local minimum stretech, $\langle L \rangle_{min}$', fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')

    plt.legend(fontsize=26, loc=(0.5, 0.2))
        # plt.yl/im([min(y2), 0.65])
    plt.tick_params(axis='both', which="both", length=6, width=1)
    plt.show()
    plt.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Figure 4 supp
    """
    # STEP 1  real L versus real average degree small beta
    """
    # plot_L_with_avg2()


    """
    # STEP 2 real L and <d><h> versus real average degree beta = 1024
    """
    # if realL = True,L = real stretch ; else , L = <d><h>
    # plot_L_with_avg()

    """
    # STEP 3 <d> and <h> versus real average degree beta = 1024 under differnet N
    """

    # plot_hopcount_linklength_with_avg()

    """
    # STEP 4 <d> and <h> versus real average degree beta = 1024 under differnet N for local minimum
    """
    # if realL = True,L = real stretch ; else , L = <d><h>
    # plot_L_with_avg_loc()
    # print(generate_ED_log_unifrom(2, 199999, 12))



    # Nvec = [464, 681, 1000,1468, 2154,3156, 4642,6803, 10000,14683]
    # # Nvec = [681,1468,3156,6803,14683]
    # k_c = 4.512
    # k_star = [k_c ** (2 / 3) * np.pi * (4 / 3) ** (2 / 3) * N ** (1 / 3) for N in Nvec]
    # print(k_star)

    # print(generate_ED_log_unifrom(523, 3331, 30))

    """
    # STEP 5 <d> and <h> versus real average degree beta = 1024 under differnet N for local minimum
    """
    hopcount_fit()
    # hopcount_fit_C()
    # analytic_local_min_check()



