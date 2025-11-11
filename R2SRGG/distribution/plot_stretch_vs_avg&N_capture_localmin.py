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


def analticL_smallbeta(N, k_vals,A, beta):
    h_vals = k_vals**(beta/2 -1)*(A+np.log(N)/np.log(k_vals))
    return h_vals

def analticL_smallbeta2(N, k_vals,A,B, beta):
    h_vals = k_vals**(beta/2 -1)*(A+B*np.log(N)/np.log(k_vals))
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



def analtich_small_beta(N, k_vals,A):
    return A + np.log(N) / np.log(k_vals)

def analtich_small_beta_logk(N, logkinverse,A):
    return A + np.log(N)*logkinverse

def analtich_mid_beta_logk(N, logkinverse,A,B):
    return A + B*np.log(N)*logkinverse

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
    Nvec = [10000]

    realL = True

    # Nvec = [100]
    real_ave_degree_dict = {}
    ave_L = {}
    std_L = {}


    beta_vec = [4]
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
        print(x)
        print(y)
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
                            # if we include 1-hop sp
                            #     L = [x * y for x, y in zip(ave_edgelength_for_a_para_comb, hop_vec)]
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


        # plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=16,
        #              label=N, color=colors[N_index])
        # plt.errorbar(kvec_dict[N], y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=16,
        #              label=N, color=colors[N_index])
        plt.errorbar([alpha(x,N, beta) for x in kvec_dict[N]], y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=16,
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
    # plt.xlabel(r'Average degree, $\langle D \rangle$', fontsize=26)
    plt.xlabel(r'Expected degree, $E[D]$', fontsize=26)
    plt.xlabel(r'$\alpha$', fontsize=26)

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




def merge_kvec_dicts(dict1, dict2):
    merged = {}
    all_keys = set(dict1.keys()) | set(dict2.keys())  # 所有key的并集

    for k in all_keys:
        v1 = dict1.get(k, [])
        v2 = dict2.get(k, [])
        merged[k] = sorted(set(v1) | set(v2))  # 合并去重并排序
    return merged


def plot_L_with_avg_loc_largeN():
    # the x-axis is the input average degree
    # the y-axis is the L
    # beta can be 2.1 or 2.5 2025/09/24
    # Nvec = [10000, 20000, 40000, 60000, 100000]

    Nvec = [10000, 20000, 40000, 60000, 100000]


    Nvec = [10000]
    # beta = 1024
    beta = 2.5

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

    elif beta in [2.1]:
        kvec_dict = {
            464: generate_ED_log_unifrom(2, 1000000, 12),
            681: generate_ED_log_unifrom(2, 1000000, 12),
            1000: generate_ED_log_unifrom(2, 1000000, 12),
            1468: generate_ED_log_unifrom(2, 1000000, 12),
            2154: generate_ED_log_unifrom(2, 1000000, 12),
            3156: generate_ED_log_unifrom(2, 1000000, 12),
            4642: generate_ED_log_unifrom(2, 1000000, 12),
            6803: generate_ED_log_unifrom(2, 1000000, 12),
            10000: [2,3,4,6, 7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328, 1000000,3296030],
            20000: [2,3,4,6, 7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328, 1000000,3296030],
            40000: [2,3,4,6, 7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328, 1000000,10866500],
            60000: [2,3,4,6, 7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328, 1000000,3296030],
            100000: [2,3,4,6, 7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328]
        }
        kvec_dict_forlocalmin = {
            464: generate_ED_log_unifrom(72, 779, 30),
            681: generate_ED_log_unifrom(72, 779, 30),
            1000: generate_ED_log_unifrom(72, 779, 30),
            1468: generate_ED_log_unifrom(72, 779, 30),
            2154: generate_ED_log_unifrom(236, 2568, 30),
            3156: generate_ED_log_unifrom(236, 2568, 30),
            4642: generate_ED_log_unifrom(134, 2218, 30),
            6803: generate_ED_log_unifrom(134, 2218, 30),
            10000: generate_ED_log_unifrom(523, 3331, 30),
            20000: generate_ED_log_unifrom(2218, 36645, 30),
            40000: generate_ED_log_unifrom(2218, 36645, 30),
            60000: generate_ED_log_unifrom(2218, 9016, 15)+generate_ED_log_unifrom(9016, 148939, 30),
            100000: generate_ED_log_unifrom(5050, 9016, 15)+generate_ED_log_unifrom(9016, 51406, 30)
        }

        # kvec_dict = merge_kvec_dicts(kvec_dict, kvec_dict_forlocalmin)

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
    elif beta in [2.5]:
        kvec_dict = {
            464: generate_ED_log_unifrom(2, 1000000, 12),
            681: generate_ED_log_unifrom(2, 1000000, 12),
            1000: generate_ED_log_unifrom(2, 1000000, 12),
            1468: generate_ED_log_unifrom(2, 1000000, 12),
            2154: generate_ED_log_unifrom(2, 1000000, 12),
            3156: generate_ED_log_unifrom(2, 1000000, 12),
            4642: generate_ED_log_unifrom(2, 1000000, 12),
            6803: generate_ED_log_unifrom(2, 1000000, 12),
            10000: [2, 7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328,3296030],
            20000: [2, 7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328, 1000000],
            40000: [2,7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328, 1000000,10866500],
            60000: [2, 7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328, 1000000],
            100000: [2, 7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328]
        }
        if realL:
            kvec_dict_forlocalmin = {
                10000: generate_ED_log_unifrom(236, 2568, 30),
                20000: generate_ED_log_unifrom(236, 2568, 30),
                40000: generate_ED_log_unifrom(779, 8465, 30),
                60000: generate_ED_log_unifrom(779, 8465, 15),
                100000: generate_ED_log_unifrom(779, 8465, 15)
            }
        else:
            kvec_dict_forlocalmin = {
                464: generate_ED_log_unifrom(72, 779, 30),
                681: generate_ED_log_unifrom(72, 779, 30),
                1000: generate_ED_log_unifrom(72, 779, 30),
                1468: generate_ED_log_unifrom(72, 779, 30),
                2154: generate_ED_log_unifrom(236, 2568, 30),
                3156: generate_ED_log_unifrom(236, 2568, 30),
                4642: generate_ED_log_unifrom(134, 2218, 30),
                6803: generate_ED_log_unifrom(134, 2218, 30),
                10000: generate_ED_log_unifrom(22, 72, 15),
                20000: generate_ED_log_unifrom(22, 72, 15),
                40000: generate_ED_log_unifrom(22, 72, 15),
                60000: generate_ED_log_unifrom(22, 72, 15),
                100000: generate_ED_log_unifrom(22, 72, 15)
            }
        # kvec_dict_forlocalmin = {
        #     464: generate_ED_log_unifrom(72, 779, 30),
        #     681: generate_ED_log_unifrom(72, 779, 30),
        #     1000: generate_ED_log_unifrom(72, 779, 30),
        #     1468: generate_ED_log_unifrom(72, 779, 30),
        #     2154: generate_ED_log_unifrom(236, 2568, 30),
        #     3156: generate_ED_log_unifrom(236, 2568, 30),
        #     4642: generate_ED_log_unifrom(134, 2218, 30),
        #     6803: generate_ED_log_unifrom(134, 2218, 30),
        #     10000: generate_ED_log_unifrom(523, 3331, 30),
        #     20000: generate_ED_log_unifrom(2218, 36645, 30),
        #     40000: generate_ED_log_unifrom(2218, 36645, 30),
        #     60000: generate_ED_log_unifrom(2218, 9016, 15)+generate_ED_log_unifrom(9016, 148939, 30),
        #     100000: generate_ED_log_unifrom(5050, 9016, 15)+generate_ED_log_unifrom(9016, 51406, 30)
        # }

        # kvec_dict = merge_kvec_dicts(kvec_dict, kvec_dict_forlocalmin)

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
        # elif beta ==2.1:
        #     plt.plot(x, [np.log(i) * j for (i, j) in zip(x, y)], "-o", markersize=16,linewidth=5,
        #              label=N, color=colors[N_index])


        print(x)
        print(y)
        plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=16,
                     label=f"N:{N}", color=colors[N_index])
        data = np.column_stack((x, y))
        np.savetxt(f"approximate_stretch_vs_degree_N{N}.txt", data, fmt="%.6f", header="real_avg approximate_stretch",
                   comments='')

        # plt.errorbar(kvec_dict[N], y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=16,
        #              label=N, color=colors[N_index])
        # plt.errorbar([alpha(x,N, beta) for x in kvec_dict[N]], y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=16,
        #              label=N, color=colors[N_index])




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




    # plot the analytic results
    # if beta ==1024:
    #     k_c = 4.512
    #     k_star = [k_c ** (2 / 3) * np.pi * (4 / 3) ** (2 / 3) * N ** (1 / 3) for N in Nvec]
    #     y2 = [analticL(N, k) for (N, k) in zip(Nvec, k_star)]
    #     plt.plot(k_star,y2,"-o",markersize = 25,markerfacecolor='none',linewidth=5,label = "analytic local minimum")
    #     print(k_star)
    #
    # if beta == 2.1:
    #     k_star = [np.exp(2/(beta-2)) for N in Nvec]

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # plt.ylim(0.02, 2)
    # plt.yticks([0, 0.1, 0.2, 0.3])

    plt.yscale('log')
    plt.xscale('log')
    # plt.xlabel(r'Average degree, $\langle D \rangle$', fontsize=26)
    plt.xlabel(r'Expected degree, $E[D]$', fontsize=26)
    # plt.xlabel(r'$\alpha$', fontsize=26)

    if realL:
        plt.ylabel(r'Average stretch, $\langle L \rangle$', fontsize=26)
    else:
        plt.ylabel(r'Average stretch, $\langle r \rangle \langle h \rangle $', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=26, loc=(0.5, 0.6))
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
    Nvec = [10000, 20000, 40000, 60000, 100000]
    y_k_star_vec = []
    y_local_minimum = []
    for N in Nvec:
        y_k_star_vec.append(k_star_dict[N])
        y_local_minimum.append(localmin_dict[N])
    if beta ==1024:
        k_c = 4.512
        k_star = [k_c ** (2 / 3) * np.pi * (4 / 3) ** (2 / 3) * N ** (1 / 3) for N in Nvec]
        y2 = [analticL(N, k) for (N, k) in zip(Nvec, k_star)]
        # plt.plot(k_star, y2, "-o", markersize=25, markerfacecolor='none', linewidth=5, label="")
    elif beta == 2.1:
        k_star = [np.exp(2/(beta-2)) for N in Nvec]
        A_vec = [0.52,0.6,0.62,0.62,0.7]
        y2 = [analticL_smallbeta(N, k_vals,A, beta) for (N, k_vals, A) in zip(Nvec,k_star,A_vec)]
        # plt.plot(k_star, y2, "-o", markersize=25, markerfacecolor='none', linewidth=5, label="")
    elif beta == 2.5:
        k_star = [np.exp(2/(beta-2)) for N in Nvec]
        A_vec = [0.16,0.09,0.12,0.08,0.12]
        B_vec = [1.2475,1.2781,1.2778,1.2838,1.2804]
        y2 = [analticL_smallbeta2(N, k_vals,A,B, beta) for (N, k_vals, A,B) in zip(Nvec,k_star,A_vec,B_vec)]
        # plt.plot(k_star, y2, "-o", markersize=25, markerfacecolor='none', linewidth=5, label="")


    figure()
    # figure for k^* as a function of N
    plt.plot(Nvec, y_k_star_vec, "-o",markersize=25, linewidth=5, label=r"simulation:$k^*$")

    popt, pcov = curve_fit(power_law, Nvec, y_k_star_vec)
    a_fit, b_fit = popt
    print("拟合参数: a = %.4f, b = %.4f" % (a_fit, b_fit))
    plt.plot(Nvec, power_law(Nvec, a_fit, b_fit), label=f"Fit: y = {a_fit:.2f} * x^{b_fit:.2f}", color='red')

    plt.plot(Nvec, k_star, "-o", markersize=25, markerfacecolor='none', linewidth=5, label=r"$y = e^{2/(\beta-2)}$")



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
    plt.legend(fontsize=26, loc=(0.6, 0.2))
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

    # plt.plot(Nvec, [np.e*(0.12+0.25*1.18*np.log(N))  for N in Nvec], "--", markersize=15, linewidth=5,
    #          label=r"analytical formula2")


    plt.plot(Nvec, y2, "--", markersize=15, linewidth=5,
             label=r"analytical formula")
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



def plot_Llogk_with_k_loc_largeN():
    # the x-axis is the input average degree
    # Nvec = [215, 464, 1000, 2154, 4642,10000]
    # Nvec = [10000]
    Nvec = [10000, 20000, 40000, 60000, 100000]


    # Nvec = [10000]
    # beta = 1024
    beta = 2.5

    realL = False

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
            10000: [2, 4, 5, 6, 7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328, 1000000, 3296030],
            20000: [2, 4, 5, 6, 7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328, 1000000, 3296030],
            40000: [2, 4, 5, 6, 7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328, 1000000, 10866500],
            60000: [2, 4, 5, 6, 7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328, 1000000, 3296030],
            100000: [2, 4, 5, 6, 7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328]
        }
        kvec_dict_forlocalmin = {
            464: generate_ED_log_unifrom(72, 779, 30),
            681: generate_ED_log_unifrom(72, 779, 30),
            1000: generate_ED_log_unifrom(72, 779, 30),
            1468: generate_ED_log_unifrom(72, 779, 30),
            2154: generate_ED_log_unifrom(236, 2568, 30),
            3156: generate_ED_log_unifrom(236, 2568, 30),
            4642: generate_ED_log_unifrom(134, 2218, 30),
            6803: generate_ED_log_unifrom(134, 2218, 30),
            10000: generate_ED_log_unifrom(523, 3331, 30),
            20000: generate_ED_log_unifrom(2218, 36645, 30),
            40000: generate_ED_log_unifrom(2218, 36645, 30),
            60000: generate_ED_log_unifrom(2218, 9016, 15) + generate_ED_log_unifrom(9016, 148939, 30),
            100000: generate_ED_log_unifrom(5050, 9016, 15) + generate_ED_log_unifrom(9016, 51406, 30)
        }

        # kvec_dict = merge_kvec_dicts(kvec_dict, kvec_dict_forlocalmin)

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
    elif beta==2.5:
        kvec_dict = {
            464: generate_ED_log_unifrom(2, 1000000, 12),
            681: generate_ED_log_unifrom(2, 1000000, 12),
            1000: generate_ED_log_unifrom(2, 1000000, 12),
            1468: generate_ED_log_unifrom(2, 1000000, 12),
            2154: generate_ED_log_unifrom(2, 1000000, 12),
            3156: generate_ED_log_unifrom(2, 1000000, 12),
            4642: generate_ED_log_unifrom(2, 1000000, 12),
            6803: generate_ED_log_unifrom(2, 1000000, 12),
            10000: [2, 7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328,3296030],
            20000: [2, 7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328, 1000000],
            40000: [2,7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328, 1000000,10866500],
            60000: [2, 7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328, 1000000],
            100000: [2, 7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328]
        }
        if realL:
            kvec_dict_forbeta3 = {
                464: generate_ED_log_unifrom(72, 779, 30),
                681: generate_ED_log_unifrom(72, 779, 30),
                1000: generate_ED_log_unifrom(72, 779, 30),
                1468: generate_ED_log_unifrom(72, 779, 30),
                2154: generate_ED_log_unifrom(236, 2568, 30),
                3156: generate_ED_log_unifrom(236, 2568, 30),
                4642: generate_ED_log_unifrom(134, 2218, 30),
                6803: generate_ED_log_unifrom(134, 2218, 30),
                10000: generate_ED_log_unifrom(236, 2568, 30),
                20000: generate_ED_log_unifrom(236, 2568, 30),
                40000: generate_ED_log_unifrom(779, 8465, 30),
                60000: generate_ED_log_unifrom(779, 8465, 15),
                100000: generate_ED_log_unifrom(779, 8465, 15)
            }
        else:
            kvec_dict_forbeta3 = {
                464: generate_ED_log_unifrom(72, 779, 30),
                681: generate_ED_log_unifrom(72, 779, 30),
                1000: generate_ED_log_unifrom(72, 779, 30),
                1468: generate_ED_log_unifrom(72, 779, 30),
                2154: generate_ED_log_unifrom(236, 2568, 30),
                3156: generate_ED_log_unifrom(236, 2568, 30),
                4642: generate_ED_log_unifrom(134, 2218, 30),
                6803: generate_ED_log_unifrom(134, 2218, 30),
                10000: generate_ED_log_unifrom(22, 72, 15),
                20000: generate_ED_log_unifrom(22, 72, 15),
                40000: generate_ED_log_unifrom(22, 72, 15),
                60000: generate_ED_log_unifrom(22, 72, 15),
                100000: generate_ED_log_unifrom(22, 72, 15)
            }
        kvec_dict = merge_kvec_dicts(kvec_dict, kvec_dict_forbeta3)
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
        elif beta ==2.1:
            Llogk = [np.log(i) * j for (i, j) in zip(x, y)]
            plt.plot(x[1:], Llogk[1:], "-o", markersize=16,linewidth=5,
                     label=N, color=colors[N_index])
            # print(x)
            # print([np.log(i) * j for (i, j) in zip(x, y)])
        elif beta ==2.5:
            Llogk = [np.log(i) * j for (i, j) in zip(x, y)]
            plt.plot(x[1:], Llogk[1:], "-o", markersize=16,linewidth=5,
                     label=N, color=colors[N_index])



        # plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=16,
        #              label=N, color=colors[N_index])
        # plt.errorbar(kvec_dict[N], y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=16,
        #              label=N, color=colors[N_index])
        # plt.errorbar([alpha(x,N, beta) for x in kvec_dict[N]], y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=16,
        #              label=N, color=colors[N_index])




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
    if beta ==2.1:
        popt, pcov = curve_fit(power_law, x[7:-5], Llogk[7:-5])
        a_fit, b_fit = popt
        print("拟合参数: a = %.4f, b = %.4f" % (a_fit, b_fit))
        plt.plot(x[5:], power_law(x[5:], a_fit, b_fit), label=f"Fit: y = {a_fit:.2f} * x^{b_fit:.2f}", color='red')
    elif beta ==2.5:
        popt, pcov = curve_fit(power_law, x[7:-5], Llogk[7:-5])
        a_fit, b_fit = popt
        print("拟合参数: a = %.4f, b = %.4f" % (a_fit, b_fit))
        plt.plot(x, power_law(x, a_fit, b_fit), label=f"Fit: y = {a_fit:.2f} * x^{b_fit:.2f}", color='red')

        # MODEL FIT
        plt.plot(x, [x_value**0.5*(0.12+1.28*np.log(10000)/np.log(x_value))*np.log(x_value)  for x_value in x], label=f"fit2: analytic formula: Llog(k)~k^0.25")



    k_c = 4.512
    k_star = [k_c**(2/3) * np.pi * (4/3)**(2/3) * N**(1/3) for N in Nvec]
    y2 = [analticL(N, k) for (N, k) in zip(Nvec, k_star)]
    # plot the analytic results
    if beta ==1024:
        plt.plot(k_star,y2,"-o",markersize = 25,markerfacecolor='none',linewidth=5,label = "analytic local minimum")
        print(k_star)

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # plt.ylim(0.02, 2)
    # plt.yticks([0, 0.1, 0.2, 0.3])

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'Average degree, $\langle k \rangle$', fontsize=26)
    # plt.xlabel(r'Expected degree, $E[D]$', fontsize=26)
    # plt.xlabel(r'$\alpha$', fontsize=26)

    if realL:
        plt.ylabel(r'$\langle L \rangle log (\langle k \rangle)$', fontsize=26)
    else:
        plt.ylabel(r'$\langle r \rangle \langle h \rangle log (\langle k \rangle)$', fontsize=26)

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


    # figure curve fit: load data:
    # Nvec = [215, 464, 1000, 2154, 4642,10000]
    Nvec = [10000, 20000, 40000, 60000, 100000]
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

def alpha(avg, N, beta):
    R = 2.0  # manually tuned value
    alpha = (2 * N / avg * R * R) * (np.pi / (np.sin(2 * np.pi / beta) * beta))
    alpha = np.sqrt(alpha)
    return alpha





def hopcount_fit():
    Nvec = [464, 1000, 2154, 4642, 10000]
    Nvec = [10000, 20000, 40000, 60000, 100000]
    # Nvec = [215]
    beta = 1024
    beta = 3.1
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
            10000: [2, 7,8,10,12,14, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328, 1000000]+[3296030],
            20000: [2, 7,8,10,12,14, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328, 1000000]+[3296030,10866500, 35826700],
            40000: [2, 7,8,10,12,14, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328, 1000000]+[3296030,10866500, 35826700],
            60000: [2, 7,8,10,12,14, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328, 1000000]+[3296030,10866500, 35826700],
            100000: [2, 7,8,10,12,14, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328, 1000000],
            }
    elif beta == 2.5:
        kvec_dict = {
            464: generate_ED_log_unifrom(2, 1000000, 12),
            681: generate_ED_log_unifrom(2, 1000000, 12),
            1000: generate_ED_log_unifrom(2, 1000000, 12),
            1468: generate_ED_log_unifrom(2, 1000000, 12),
            2154: generate_ED_log_unifrom(2, 1000000, 12),
            3156: generate_ED_log_unifrom(2, 1000000, 12),
            4642: generate_ED_log_unifrom(2, 1000000, 12),
            6803: generate_ED_log_unifrom(2, 1000000, 12),
            10000: [2, 7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328, 3296030],
            20000: [2, 7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328, 1000000],
            40000: [2, 7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328, 1000000, 10866500],
            60000: [2, 7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328, 1000000],
            100000: [2, 7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328]
        }
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


        plt.plot(real_ave_degree_vec_0,hop_vec_0,"--o",markersize = 20,markerfacecolor='none',linewidth=5,label = f"{N}")
        # data = np.column_stack((real_ave_degree_vec_0, hop_vec_0))
        # np.savetxt(f"hopcount_vs_degree_N{N}.txt", data, fmt="%.6f", header="real_avg hopcount",
        #            comments='')

        # plt.plot([1/np.log(x) for x in real_ave_degree_vec_0[1:]], hop_vec_0[1:], "--o", markersize=20, markerfacecolor='none', linewidth=5,
        #          label=f"{N}")
        # print("x:",[1/np.log(x) for x in real_ave_degree_vec_0[1:]])
        # plt.plot(kvec2, hop_vec_0, "-o", markersize=25, markerfacecolor='none', linewidth=5,
        #          label=f"{N}")
        # plt.plot([alpha(x, N, beta) for x in kvec2], hop_vec_0, "-o", markersize=25, markerfacecolor='none', linewidth=5,
        #          label=f"{N}")


        if beta ==2.1:
            # curve fit using log(n)/log(k)+A
            def model_fixed(x, A):
                return analtich_small_beta_logk(N, x, A)
            # 曲线拟合
            popt, pcov = curve_fit(model_fixed, [1/np.log(x) for x in real_ave_degree_vec_0[1:-5]], hop_vec_0[1:-5])
            A_fit = popt[0]
            print(A_fit)

            x_fit = np.linspace(min([1/np.log(x) for x in real_ave_degree_vec_0[1:]]), max([1/np.log(x) for x in real_ave_degree_vec_0[1:]]), 20000)
            y_fit = model_fixed(x_fit, A_fit)
            plt.plot(x_fit, y_fit, '-', label=fr"fit: $\langle L \rangle = {A_fit:.4f} + log({N})/log(k)$")
        elif beta ==2.5:
            # curve fit using log(n)/log(k)+A
            def model_fixed(x, A,B):
                return analtich_mid_beta_logk(N, x, A,B)

            # 曲线拟合
            popt, pcov = curve_fit(model_fixed, [1 / np.log(x) for x in real_ave_degree_vec_0[1:-5]], hop_vec_0[1:-5])
            A_fit,B_fit = popt
            print("拟合结果: A =", A_fit, "B =", B_fit)

            x_fit = np.linspace(min([1 / np.log(x) for x in real_ave_degree_vec_0[1:]]),
                                max([1 / np.log(x) for x in real_ave_degree_vec_0[1:]]), 20000)
            y_fit = model_fixed(x_fit, A_fit,B_fit)
            plt.plot(x_fit, y_fit, '-', label=fr"fit: $\langle L \rangle = {A_fit:.4f} + {B_fit:.4f}log({N})/log(k)$")



        # curve fit using log(n)/log(k) when beta ==2.1 or c(k_Kc)^{-0.5} when beta ==1024

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
    plt.xlabel(r'$1/log(k)$', fontsize=26)
    # plt.xlabel(r'Expected degree $E[D]$', fontsize=26)
    # plt.xlabel(r'$\alpha$', fontsize=26)
    plt.ylabel(r' $\langle h \rangle$', fontsize=26)
    plt.legend(fontsize=12, loc=(0.7, 0.3))
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


def plot_realL_approxL_together():
    real_ave_degree_dict = {}
    ave_L = {}
    approx_L = {}

    beta = 2.5

    real_ave_degree_dict[2.5] = [np.float64(1.3802), np.float64(4.6628), np.float64(13.7504), np.float64(41.3624), np.float64(121.1224), np.float64(340.4932), np.float64(890.686), np.float64(2103.6792), np.float64(4257.9598), np.float64(6934.8888), np.float64(8898.362), np.float64(9929.8622)]

    approx_L[2.5] = [np.float64(1.3923903541527687), np.float64(0.380112602051347), np.float64(0.2971211105167318), np.float64(0.2889925589370189), np.float64(0.3106141680232836), np.float64(0.3357510492606415), np.float64(0.42221412497733235), np.float64(0.5215702329593628), np.float64(0.5885804829600149), np.float64(0.5911024549298902), np.float64(0.5555009671460683), np.float64(0.526097563249207)]

    ave_L[2.5] = [np.float64(2.083753542297957), np.float64(0.8108291291468183), np.float64(0.7086999716180932), np.float64(0.6743696974975774), np.float64(0.6485367658365878), np.float64(0.6344778917716585), np.float64(0.6710528177496031), np.float64(0.7530533967784127), np.float64(0.8947615820067742), np.float64(0.963460886822925), np.float64(1.1196354619043718), np.float64(1.2093558418297714)]


    real_ave_degree_dict[2.1] = [np.float64(0.647672), np.float64(0.9536), np.float64(1.2332), np.float64(1.792),
                                 np.float64(2.058828),
                                 np.float64(5.847454000000002), np.float64(16.889622), np.float64(47.695429999999995),
                                 np.float64(130.94333999999998), np.float64(343.033686), np.float64(842.6122099999999),
                                 np.float64(1887.5601599999998), np.float64(3706.396326), np.float64(6092.669242000001),
                                 np.float64(8187.439658),
                                 np.float64(9344.583)]
    ave_L[2.1] = [np.float64(0.27143445114259124), np.float64(0.7598339194549935), np.float64(2.675853772043278),
                  np.float64(1.4656527772294778), np.float64(1.3136788477251418), np.float64(0.8542345889975687),
                  np.float64(0.7386616210765622), np.float64(0.6890922705269413), np.float64(0.6642333052457445),
                  np.float64(0.6455108478945255), np.float64(0.6808760138955805), np.float64(0.746380165204815),
                  np.float64(0.8396772972302338), np.float64(0.9318348031709808), np.float64(1.024014919492898),
                  np.float64(1.3314722164403012)]

    approx_L[2.1] = [np.float64(0.18713185030928067), np.float64(0.7283556244139056), np.float64(2.3453236422507335),
                     np.float64(1.2129600085566379), np.float64(1.0596813847187543), np.float64(0.5570708063885884),
                     np.float64(0.4308826229144635), np.float64(0.3904621312891976), np.float64(0.3930380853589546),
                     np.float64(0.3964590523677577), np.float64(0.4713624573018614), np.float64(0.5520598341571079),
                     np.float64(0.6036795026077996), np.float64(0.605236360802573), np.float64(0.5710837915958688),
                     np.float64(0.5429516015345655)]
    real_ave_degree_dict[4] = [1.7054, 2.1758, 2.3132, 2.65, 2.9316, 3.4204, 4.6164, 7.6562, 12.1664, 20.397, 32.749,
                               53.1894, 85.5136, 137.117, 218.353, 345.515, 540.688, 836.2706, 1274.4564, 1903.7746,
                               2765.9072, 3888.2098, 5252.766, 6702.0802, 8029.6946, 8990.1558, 9553.0216, 9820.4528]
    approx_L[4] = [np.float64(0.047142366971440246), np.float64(0.20861996975776426), np.float64(0.3889985780686527),
                   np.float64(0.6390728663273963), np.float64(0.4679255268407224), np.float64(0.34249510012394296),
                   np.float64(0.2575292756978971), np.float64(0.1950389670926904), np.float64(0.17174246626728473),
                   np.float64(0.16412749193484105), np.float64(0.16554785757124327), np.float64(0.17103442293291765),
                   np.float64(0.18362445384559242), np.float64(0.19996835814461092), np.float64(0.22164503626728538),
                   np.float64(0.24902137730478238), np.float64(0.27450434516084105), np.float64(0.30773424399552624),
                   np.float64(0.35892274446087363), np.float64(0.41679672719049743), np.float64(0.4741504332954842),
                   np.float64(0.5196804807434381), np.float64(0.5532632778358367), np.float64(0.5632517127774106),
                   np.float64(0.5543953726725661), np.float64(0.5443747566911749), np.float64(0.532921861195619),
                   np.float64(0.5270276034416593)]
    ave_L[4] = [np.float64(0.10586833057728005), np.float64(0.32092651829744073), np.float64(0.4211427701655029),
                np.float64(1.600106730159238), np.float64(1.1207469459545976), np.float64(0.9602755020342794),
                np.float64(0.8505105399855226), np.float64(0.7521601392454854), np.float64(0.7144736637949518),
                np.float64(0.6897834045341739), np.float64(0.6611984828894697), np.float64(0.6430243010088226),
                np.float64(0.6299550989628939), np.float64(0.6236145759731754), np.float64(0.6226740920891127),
                np.float64(0.6267114073026878), np.float64(0.6282176272232343), np.float64(0.6364987317054913),
                np.float64(0.6586756556582146), np.float64(0.6946337988528414), np.float64(0.7457111976022143),
                np.float64(0.826799927138955), np.float64(0.9428326521400886), np.float64(1.0663845876582414),
                np.float64(1.140146289513965), np.float64(1.1836464727030707), np.float64(1.1939575186864742),
                np.float64(1.188776107945433)]




    figure()
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494', '#9FA9C9', '#D36A6A']
    plt.plot(real_ave_degree_dict[beta], ave_L[beta], linestyle="--", linewidth=3, marker='o', markersize=16,
             label=r"real $<S>$", color=colors[1])


    plt.plot(real_ave_degree_dict[beta],approx_L[beta] , linestyle="--", linewidth=3, marker='o', markersize=16,
             label=r"approx $<S> = <r><h>$", color = colors[2])


    if beta ==2.5:
        x2 = np.linspace(min(real_ave_degree_dict[beta]), max(real_ave_degree_dict[beta]), 100000)
        ana_vec = [0.033 * x_value ** 0.25 * (0.12 + 1.28 * np.log(10000) / np.log(x_value)) for x_value in x2]

        # plt.plot(x, [x_value ** 0.5 * (0.12 + 1.28 * np.log(10000) / np.log(x_value)) * np.log(x_value) for x_value in x],
        #          label=f"fit2: analytic formula: Llog(k)~k^0.25")

        plt.plot(x2, ana_vec, "--", linewidth=5, label=r"$ana: <S> = c* k^{0.25}(A+B* log(N)/log(<D>))$")


    plt.xscale('log')
    plt.xlabel(r'Average degree, $\langle D \rangle$', fontsize=26)
    # plt.xlabel(r'Expected degree, $E[D]$', fontsize=26)
    # plt.xlabel(r'$\alpha$', fontsize=26)

    plt.ylabel(r'Average stretch, $\langle S \rangle$', fontsize=26)

    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=26, loc=(0.4, 0.7))
    # plt.text(0.5, 1.6, r"$N = 10^4,\beta = 2.5,c = 0.025, A = 0.12, B = 1.28$", fontsize=26)
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





def plot_realL_approxL_together_diff():
    real_ave_degree_dict = {}
    ave_L = {}
    approx_L = {}

    beta_vec = [2.1,2.5,4,1024]
    real_ave_degree_dict[2.5] = [np.float64(1.3802), np.float64(4.6628), np.float64(13.7504), np.float64(41.3624), np.float64(121.1224), np.float64(340.4932), np.float64(890.686), np.float64(2103.6792), np.float64(4257.9598), np.float64(6934.8888), np.float64(8898.362), np.float64(9929.8622)]
    approx_L[2.5] = [np.float64(1.3923903541527687), np.float64(0.380112602051347), np.float64(0.2971211105167318), np.float64(0.2889925589370189), np.float64(0.3106141680232836), np.float64(0.3357510492606415), np.float64(0.42221412497733235), np.float64(0.5215702329593628), np.float64(0.5885804829600149), np.float64(0.5911024549298902), np.float64(0.5555009671460683), np.float64(0.526097563249207)]
    ave_L[2.5] = [np.float64(2.083753542297957), np.float64(0.8108291291468183), np.float64(0.7086999716180932), np.float64(0.6743696974975774), np.float64(0.6485367658365878), np.float64(0.6344778917716585), np.float64(0.6710528177496031), np.float64(0.7530533967784127), np.float64(0.8947615820067742), np.float64(0.963460886822925), np.float64(1.1196354619043718), np.float64(1.2093558418297714)]
    real_ave_degree_dict[2.1] = [np.float64(0.647672), np.float64(0.9536), np.float64(1.2332), np.float64(1.792),
                                 np.float64(2.058828),
                                 np.float64(5.847454000000002), np.float64(16.889622), np.float64(47.695429999999995),
                                 np.float64(130.94333999999998), np.float64(343.033686), np.float64(842.6122099999999),
                                 np.float64(1887.5601599999998), np.float64(3706.396326), np.float64(6092.669242000001),
                                 np.float64(8187.439658),
                                 np.float64(9344.583)]
    ave_L[2.1] = [np.float64(0.27143445114259124), np.float64(0.7598339194549935), np.float64(2.675853772043278),
                  np.float64(1.4656527772294778), np.float64(1.3136788477251418), np.float64(0.8542345889975687),
                  np.float64(0.7386616210765622), np.float64(0.6890922705269413), np.float64(0.6642333052457445),
                  np.float64(0.6455108478945255), np.float64(0.6808760138955805), np.float64(0.746380165204815),
                  np.float64(0.8396772972302338), np.float64(0.9318348031709808), np.float64(1.024014919492898),
                  np.float64(1.3314722164403012)]
    approx_L[2.1] = [np.float64(0.18713185030928067), np.float64(0.7283556244139056), np.float64(2.3453236422507335),
                     np.float64(1.2129600085566379), np.float64(1.0596813847187543), np.float64(0.5570708063885884),
                     np.float64(0.4308826229144635), np.float64(0.3904621312891976), np.float64(0.3930380853589546),
                     np.float64(0.3964590523677577), np.float64(0.4713624573018614), np.float64(0.5520598341571079),
                     np.float64(0.6036795026077996), np.float64(0.605236360802573), np.float64(0.5710837915958688),
                     np.float64(0.5429516015345655)]
    real_ave_degree_dict[4] = [1.7054, 2.1758, 2.3132, 2.65, 2.9316, 3.4204, 4.6164, 7.6562, 12.1664, 20.397, 32.749,
                               53.1894, 85.5136, 137.117, 218.353, 345.515, 540.688, 836.2706, 1274.4564, 1903.7746,
                               2765.9072, 3888.2098, 5252.766, 6702.0802, 8029.6946, 8990.1558, 9553.0216, 9820.4528]
    approx_L[4] = [np.float64(0.047142366971440246), np.float64(0.20861996975776426), np.float64(0.3889985780686527),
                   np.float64(0.6390728663273963), np.float64(0.4679255268407224), np.float64(0.34249510012394296),
                   np.float64(0.2575292756978971), np.float64(0.1950389670926904), np.float64(0.17174246626728473),
                   np.float64(0.16412749193484105), np.float64(0.16554785757124327), np.float64(0.17103442293291765),
                   np.float64(0.18362445384559242), np.float64(0.19996835814461092), np.float64(0.22164503626728538),
                   np.float64(0.24902137730478238), np.float64(0.27450434516084105), np.float64(0.30773424399552624),
                   np.float64(0.35892274446087363), np.float64(0.41679672719049743), np.float64(0.4741504332954842),
                   np.float64(0.5196804807434381), np.float64(0.5532632778358367), np.float64(0.5632517127774106),
                   np.float64(0.5543953726725661), np.float64(0.5443747566911749), np.float64(0.532921861195619),
                   np.float64(0.5270276034416593)]
    ave_L[4] = [np.float64(0.10586833057728005), np.float64(0.32092651829744073), np.float64(0.4211427701655029),
                np.float64(1.600106730159238), np.float64(1.1207469459545976), np.float64(0.9602755020342794),
                np.float64(0.8505105399855226), np.float64(0.7521601392454854), np.float64(0.7144736637949518),
                np.float64(0.6897834045341739), np.float64(0.6611984828894697), np.float64(0.6430243010088226),
                np.float64(0.6299550989628939), np.float64(0.6236145759731754), np.float64(0.6226740920891127),
                np.float64(0.6267114073026878), np.float64(0.6282176272232343), np.float64(0.6364987317054913),
                np.float64(0.6586756556582146), np.float64(0.6946337988528414), np.float64(0.7457111976022143),
                np.float64(0.826799927138955), np.float64(0.9428326521400886), np.float64(1.0663845876582414),
                np.float64(1.140146289513965), np.float64(1.1836464727030707), np.float64(1.1939575186864742),
                np.float64(1.188776107945433)]

    real_ave_degree_dict[1024] = [np.float64(1.7116), np.float64(2.1844), np.float64(2.3392), np.float64(2.6532), np.float64(2.9732),
     np.float64(3.4334), np.float64(4.6354), np.float64(5.3908), np.float64(6.1402), np.float64(6.9),
     np.float64(7.6764), np.float64(12.3402), np.float64(20.6852), np.float64(33.4042), np.float64(54.2158),
     np.float64(88.3954), np.float64(142.3096), np.float64(229.154), np.float64(366.9758), np.float64(583.1674),
     np.float64(919.8802), np.float64(1433.2412), np.float64(2195.174), np.float64(3294.6078), np.float64(4799.9524),
     np.float64(6698.3978), np.float64(8672.6344), np.float64(9864.5238), np.float64(9998.864)]
    ave_L[1024] = [np.float64(0.01987554665708353), np.float64(0.03287787661219253), np.float64(0.035876445191642306),
     np.float64(0.042606405081301875), np.float64(0.05784025783644552), np.float64(0.08975560238485644),
     np.float64(0.7001941719058954), np.float64(0.8306130331451235), np.float64(0.6940836963924846),
     np.float64(0.6455466085838456), np.float64(0.6203823932708646), np.float64(0.5864119891790662),
     np.float64(0.5637336078513), np.float64(0.5598800266217759), np.float64(0.5463299728864828),
     np.float64(0.547043323891516), np.float64(0.5517868153198979), np.float64(0.5575264146255045),
     np.float64(0.5633000938215947), np.float64(0.5765767486273359), np.float64(0.5945258693209504),
     np.float64(0.6216593917416774), np.float64(0.662109100382452), np.float64(0.7164883365773754),
     np.float64(0.7952310833520863), np.float64(0.9316378099919851), np.float64(1.1189564706029702),
     np.float64(1.3498288747942115), np.float64(1.3614779279461366)]
    approx_L[1024] = [np.float64(0.012364489654434993), np.float64(0.022783388620295532), np.float64(0.025777878255248993),
     np.float64(0.03152838515302665), np.float64(0.04413341338566713), np.float64(0.07119757212682272),
     np.float64(0.5890709304342409), np.float64(0.6770327026765943), np.float64(0.5623908780358712),
     np.float64(0.5169195441787936), np.float64(0.4925402506161351), np.float64(0.44685510976643766),
     np.float64(0.41555558280686494), np.float64(0.4024550999177894), np.float64(0.3869105579878664),
     np.float64(0.3839149620928747), np.float64(0.3845259298642634), np.float64(0.3856300418541556),
     np.float64(0.3877712867458492), np.float64(0.39596798341774775), np.float64(0.40427218175748175),
     np.float64(0.4182040863808713), np.float64(0.4381904656746554), np.float64(0.45377704335633373),
     np.float64(0.47530842844887394), np.float64(0.5122203177077956), np.float64(0.5240494099880836),
     np.float64(0.5219177462438847), np.float64(0.5239580735399473)]


    figure()
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494', '#9FA9C9', '#D36A6A']

    for beta in beta_vec:
        y = [i/j for (i,j) in zip(ave_L[beta], approx_L[beta])]
        plt.plot(real_ave_degree_dict[beta], y, linestyle="-", linewidth=3, markersize=16,
                 label=rf"$\beta:${beta}")



    plt.xscale('log')
    plt.xlabel(r'Average degree, $\langle D \rangle$', fontsize=26)
    # plt.xlabel(r'Expected degree, $E[D]$', fontsize=26)
    # plt.xlabel(r'$\alpha$', fontsize=26)

    plt.ylabel(r'real $<S>$ / approx $<r><h>$', fontsize=26)

    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=26, loc=(0.4, 0.7))
    # plt.text(0.5, 1.6, r"$N = 10^4,\beta = 2.5,c = 0.025, A = 0.12, B = 1.28$", fontsize=26)
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








def plot_realLFlaseLananlyticL_together():
    figure()
    N = 10000

    real_avg_vec = [np.float64(1.46954), np.float64(5.03142), np.float64(15.24824), np.float64(47.79766), np.float64(147.43898),
     np.float64(447.46456), np.float64(1312.27718), np.float64(3668.05386), np.float64(9550.78412),
     np.float64(22354.68586), np.float64(44602.77848)]
    approx_vec = [np.float64(0.8182731846623904), np.float64(0.254000769163314), np.float64(0.2015803223922931),
     np.float64(0.1937751811010468), np.float64(0.20360747037771829), np.float64(0.23695464716334422),
     np.float64(0.25732389727935495), np.float64(0.3288515302072136), np.float64(0.4295282447630304),
     np.float64(0.5283123673344998), np.float64(0.5901199694952706)]

    realL_vec = [np.float64(1.3863705186983992), np.float64(0.7547550428302459), np.float64(0.6763889039859593),
     np.float64(0.6472066091029783), np.float64(0.6206690419300328), np.float64(0.617937406241449),
     np.float64(0.605152825990719), np.float64(0.6199646665115169), np.float64(0.6738063983102505),
     np.float64(0.7542639664055537), np.float64(0.8809445530712725)]

    x2  = np.linspace(min(real_avg_vec),max(real_avg_vec),100000)
    ana_vec = [0.025*x_value ** 0.25 * (0.12 + 1.28 * np.log(10000) / np.log(x_value)) for x_value in x2]

    # plt.plot(x, [x_value ** 0.5 * (0.12 + 1.28 * np.log(10000) / np.log(x_value)) * np.log(x_value) for x_value in x],
    #          label=f"fit2: analytic formula: Llog(k)~k^0.25")

    plt.plot(real_avg_vec, approx_vec, "-o", markersize=25, markerfacecolor='none', linewidth=5, label=r"$approx <L> = <r><h>$")
    plt.plot(real_avg_vec, realL_vec, "-o", markersize=25, markerfacecolor='none', linewidth=5,
             label=r"$real <L>$")
    plt.plot(x2, ana_vec, "--", linewidth=5, label=r"$ana: y = c* k^{0.25}(A+B* log(N)/log(k))$")

    # plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'Average degree, $\langle k \rangle$', fontsize=26)
    # plt.xlabel(r'Expected degree, $E[D]$', fontsize=26)
    # plt.xlabel(r'$\alpha$', fontsize=26)

    plt.ylabel(r'Average stretch, $\langle L \rangle$', fontsize=26)

    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=26, loc=(0.4, 0.7))
    plt.text(0.5,1.6,r"$N = 10^4,\beta = 2.5,c = 0.025, A = 0.12, B = 1.28$", fontsize=26)
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


def plot_Llogk_realLFlaseLananlyticL_together():
    figure()
    N = 10000

    real_avg_vec = [np.float64(1.46954), np.float64(5.03142), np.float64(15.24824), np.float64(47.79766), np.float64(147.43898),
     np.float64(447.46456), np.float64(1312.27718), np.float64(3668.05386), np.float64(9550.78412),
     np.float64(22354.68586), np.float64(44602.77848)]
    approx_vec = [np.float64(0.8182731846623904), np.float64(0.254000769163314), np.float64(0.2015803223922931),
     np.float64(0.1937751811010468), np.float64(0.20360747037771829), np.float64(0.23695464716334422),
     np.float64(0.25732389727935495), np.float64(0.3288515302072136), np.float64(0.4295282447630304),
     np.float64(0.5283123673344998), np.float64(0.5901199694952706)]

    realL_vec = [np.float64(1.3863705186983992), np.float64(0.7547550428302459), np.float64(0.6763889039859593),
     np.float64(0.6472066091029783), np.float64(0.6206690419300328), np.float64(0.617937406241449),
     np.float64(0.605152825990719), np.float64(0.6199646665115169), np.float64(0.6738063983102505),
     np.float64(0.7542639664055537), np.float64(0.8809445530712725)]

    x2  = np.linspace(min(real_avg_vec),max(real_avg_vec),100000)
    ana_vec = [0.025*x_value ** 0.25 * (0.12 + 1.28 * np.log(10000) / np.log(x_value))* np.log(x_value) for x_value in x2]

    # plt.plot(x, [x_value ** 0.5 * (0.12 + 1.28 * np.log(10000) / np.log(x_value)) * np.log(x_value) for x_value in x],
    #          label=f"fit2: analytic formula: Llog(k)~k^0.25")



    plt.plot(real_avg_vec, [np.log(i) * j for (i, j) in zip(real_avg_vec, approx_vec)], "-o", markersize=25, markerfacecolor='none', linewidth=5, label=r"$approx <L> = <r><h>$")
    plt.plot(real_avg_vec, [np.log(i) * j for (i, j) in zip(real_avg_vec, realL_vec)], "-o", markersize=25, markerfacecolor='none', linewidth=5,
             label=r"$real <L>$")
    plt.plot(x2, ana_vec, "--", linewidth=5, label=r"$ana: y = c* k^{0.25}(A+B* log(N)/log(k))*log(k)$")

    plt.plot(x2, [0.025*x_value ** 0.25 * (1.28 * np.log(10000)) for x_value in x2], "--", linewidth=5, label=r"$ana: y = c* k^{0.25}B* log(N)$")

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'Average degree, $\langle k \rangle$', fontsize=26)
    # plt.xlabel(r'Expected degree, $E[D]$', fontsize=26)
    # plt.xlabel(r'$\alpha$', fontsize=26)

    plt.ylabel(r'$\langle L \rangle log(k)$', fontsize=26)

    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=26, loc=(0.4, 0))
    plt.text(0.5,12,r"$N = 10^4,\beta = 1.5,c = 0.025, B = 1.28$", fontsize=26)
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
    # STEP 1  real L versus real average degree small beta
    """
    # plot_L_with_avg2()


    """
    # STEP 2 real L and <d><h> versus real average degree beta = 1024
    """
    # if realL = True,L = real stretch ; else , L = <r><h>
    # plot_L_with_avg()


    """
    # STEP 3 <d> and <h> versus real average degree beta = 1024 under differnet N
    """

    # plot_hopcount_linklength_with_avg()

    """
    # STEP 4 <L> versus real average degree beta = 1024/2.1,2.5 under differnet N for local minimum
    """
    # if realL = True,L = real stretch ; else , L = <d><h>
    # plot_L_with_avg_loc()
    # plot_L_with_avg_loc_largeN()
    # plot_Llogk_with_k_loc_largeN()

    """
    # STEP 4.5 <L> and <r><h> versus real average degree beta = 1024 under differnet N for local minimum
    """

    # plot_realLFlaseLananlyticL_together()
    # plot_Llogk_realLFlaseLananlyticL_together()

    # plot_realL_approxL_together()

    plot_realL_approxL_together_diff()

    """
    # STEP 5 <d> and <h> versus real average degree beta = 1024 under differnet N for local minimum
    """
    # hopcount_fit()
    # hopcount_fit_C()
    # analytic_local_min_check()


    # print(generate_ED_log_unifrom(2, 1000000, 12))

    # print([alpha(i, 100000000, 2.1) for i in generate_ED_log_unifrom(2,100000,12)])
    # x  =
    # print()







