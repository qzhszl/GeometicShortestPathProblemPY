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
import pandas as pd


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



def load_large_network_results_dev_vs_avg_approxLrealdiff(N, beta, kvec, realL,exclude_hop1_flag):
    folder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\approxLrealdiff\\test\\"
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
                            if exclude_hop1_flag == True:
                                L = [np.mean(hop_vec_no1) * np.mean(ave_edgelength_for_a_para_comb)]
                            else:
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


def load_large_network_results_dev_vs_avg_approxLlargedlrealdiff(N, beta, kvec, realL):
    folder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\approxLrealdiff\\test\\"

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
                            edgelength_vec_name = folder_name + "max_graph_edge_length_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
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



def plot_realL_approxL_maxrh_together():
    # produce strech vs <r><h> vs <max(r)><h> with curve fit.

    real_ave_degree_dict = {}
    ave_L = {}
    approx_L = {}
    approx_L_maxdl = {}

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

    N = 10000

    if beta in [3.1,4,8,128]:
        realL = True
        kvec = [2.2, 2.8, 3.4, 4.4,  6.0, 8.0,  10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999, 16479, 27081, 44767, 73534, 121205, 199999]
        real_ave_degree_vec, _, _, _, _, _, _, ave_L_vec, _ = load_large_network_results_dev_vs_avg_approxLrealdiff(
            N, beta, kvec,realL,False)
        real_ave_degree_dict[beta] = real_ave_degree_vec
        ave_L[beta] = ave_L_vec
        realL = False
        real_ave_degree_vec, _, _, _, _, _, _, ave_L_vec2, _ = load_large_network_results_dev_vs_avg_approxLrealdiff(
            N, beta, kvec, realL,False)
        approx_L[beta] = ave_L_vec2


    figure()
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494', '#9FA9C9', '#D36A6A']
    plt.plot(real_ave_degree_dict[beta], ave_L[beta], linestyle="--", linewidth=3, marker='o', markersize=16,
             label=r"real $<S>$", color=colors[1])


    plt.plot(real_ave_degree_dict[beta],approx_L[beta] , linestyle="--", linewidth=3, marker='o', markersize=16,
             label=r"approx $<S> = <r><h>$", color = colors[2])


    kvec = [2.2, 2.8, 3.4, 4.4, 6.0, 8.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999,
            16479, 27081, 44767, 73534, 121205, 199999]
    realL = False
    real_ave_degree_vec, _, _, _, _, _, _, ave_L_vec2, _ = load_large_network_results_dev_vs_avg_approxLlargedlrealdiff(
        N, beta, kvec, realL)
    approx_L_maxdl[beta] = ave_L_vec2


    plt.plot(real_ave_degree_vec, approx_L_maxdl[beta], linestyle="--", linewidth=3, marker='o', markersize=16,
             label=r"approx $<S> = <r><h>$", color=colors[3])


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
    plt.ylim(0, 2.7)
    # plt.xlim(2, 12000)
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



def plot_realL_approxL_maxrh_hop_withN():
    """
    how the realL, <r><h>, <max(r)><h> for given E[D] change with N
    :return:
    """
    ave_L = []
    approx_L = []
    approx_L_maxdl = []

    hop_count = []

    beta = 2.1
    N_VEC = [5000, 10000, 15000, 20000, 50000, 100000]

    for N in N_VEC:
        realL = True
        kvec = [10]
        real_ave_degree_vec, _, _, _, _, _, _, ave_L_vec, _ = load_large_network_results_dev_vs_avg_approxLrealdiff(
            N, beta, kvec,realL,True)
        ave_L =  ave_L+ave_L_vec
        realL = False
        real_ave_degree_vec, _, _, _, _, _, _, ave_L_vec2, _ = load_large_network_results_dev_vs_avg_approxLrealdiff(
            N, beta, kvec, realL,True)
        approx_L = approx_L+ave_L_vec2

        _, _, _, _, _, hop_vec, _, ave_L_vec2, _ = load_large_network_results_dev_vs_avg_approxLlargedlrealdiff(
            N, beta, kvec, realL)
        approx_L_maxdl = approx_L_maxdl+ave_L_vec2
        hop_count = hop_count+hop_vec


    figure()
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494', '#9FA9C9', '#D36A6A']
    plt.plot(N_VEC, ave_L, linestyle="--", linewidth=3, marker='o', markersize=16,
             label=r"real $<S>$", color=colors[1])

    plt.plot(N_VEC,approx_L , linestyle="--", linewidth=3, marker='o', markersize=16,
             label=r"approx $<S> = <r><h>$", color = colors[2])

    plt.plot(N_VEC, approx_L_maxdl, linestyle="--", linewidth=3, marker='o', markersize=16,
             label=r"approx $<S> = max(r) <h>$", color=colors[3])


    plt.xscale('log')
    plt.xlabel(r'Average degree, $\langle N \rangle$', fontsize=26)
    # plt.xlabel(r'Expected degree, $E[D]$', fontsize=26)
    # plt.xlabel(r'$\alpha$', fontsize=26)
    # plt.ylim(0, 2.7)
    # plt.xlim(2, 12000)
    plt.ylabel(r'Average stretch, $\langle S \rangle$', fontsize=26)

    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=26, loc=(0.4, 0.5))
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

    ax1 = plt.gca()  # 左边 y 轴
    ax2 = ax1.twinx()  # 创建右边 y 轴
    print(hop_count)
    ax2.plot(N_VEC, hop_count,
             linestyle="-", linewidth=3, marker='s', markersize=14,
             label="hop count", color=colors[4])

    ax2.set_ylabel("hop count")


    plt.show()
    plt.close()





def plot_realL_approxL_together_withcurvefit_datasaved():
    """
    Figure 6 : local minimum of stretch
    produce strech vs <r><h> vs <max(r)><h> with curve fit(beta == 2.5)
    if we do want to choose <max(r)><h>, we can comment something out
    there is also a function for exporting data
    :return:
    """
    real_ave_degree_dict = {}
    ave_L = {}
    approx_L = {}
    approx_L_maxdl = {}

    hop_count = {}

    beta = 2.5
    N = 10000

    realL = True


    # kvec = [2.1, 2.4, 2.8,  3.6,4.4, 5,5.5, 6.0, 8.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105,
    #         9999, 16479, 27081, 44767, 73534, 121205, 199999,328000,539744] # for 2.1
    kvec = [2.1, 2.4, 2.8, 3.6, 4.4, 5, 5.5, 6.0, 8.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727,
            6105,
            9999, 16479, 27081, 44767, 73534, 121205, 199999]  # for 2.1

    # kvec = [2.1, 2.4, 2.8, 3.4, 3.6, 3.8, 4, 4.4, 5, 5.5, 6.0, 8.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389,
    #         2276, 3727, 6105,
    #         9999, 16479, 27081, 44767, 73534, 121205, 199999]  # for 2.1
    # kvec = [2.1,2.4, 2.8, 3.4, 4.4,5,  6.0, 8.0,  10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999, 16479, 27081, 44767, 73534, 121205, 199999]
    # kvec = [2.2, 2.8, 3.4, 4.4, 6.0, 8.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999,
    #         16479, 27081, 44767, 73534, 121205, 199999] # for 2.5


    real_ave_degree_vec, _, _, _, _, _, _, ave_L_vec, _ = load_large_network_results_dev_vs_avg_approxLrealdiff(
        N, beta, kvec,realL,False)
    real_ave_degree_dict[beta] = real_ave_degree_vec
    ave_L[beta] = ave_L_vec
    realL = False
    real_ave_degree_vec, _, _, _, _, _, _, ave_L_vec2, _ = load_large_network_results_dev_vs_avg_approxLrealdiff(
        N, beta, kvec, realL,False)
    approx_L[beta] = ave_L_vec2

    _, _, _, _, _, hop_vec, _, ave_L_vec2, _ = load_large_network_results_dev_vs_avg_approxLlargedlrealdiff(
        N, beta, kvec, realL)
    approx_L_maxdl[beta] = ave_L_vec2
    hop_count[beta] = hop_vec


    figure()
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494', '#9FA9C9', '#D36A6A']
    plt.plot(real_ave_degree_dict[beta], ave_L[beta], linestyle="--", linewidth=3, marker='o', markersize=16,
             label=r"real $<S>$", color=colors[1])
    x = real_ave_degree_dict[beta]
    y = ave_L[beta]

    print(x)
    print(y)


    # # data exported:
    # column_names = [
    #     'avg',
    #     'real stretch',
    # ]
    # # 创建 DataFrame
    # data = {name: var for name, var in zip(column_names, [x, y])}
    # df = pd.DataFrame(data)
    #
    # filename_txt = f'realstretchVSk_beta{beta}.txt'
    #
    # # 步骤 1: 使用 pandas 将数据保存到文件，使用逗号分隔
    # df.to_csv(
    #     filename_txt,
    #     sep=',',  # 使用逗号作为分隔符
    #     index=False,  # 不写入行索引
    #     header=True,  # 写入列名/表头
    #     encoding='utf-8'
    # )



    plt.plot(real_ave_degree_dict[beta],approx_L[beta] , linestyle="--", linewidth=3, marker='o', markersize=16,
             label=r"approx $<S> = <r><h>$", color = colors[2])

    plt.plot(real_ave_degree_dict[beta], approx_L_maxdl[beta], linestyle="--", linewidth=3, marker='o', markersize=16,
             label=r"approx $<S> = max(r) <h>$", color=colors[3])


    if beta ==2.5:
        x2 = np.linspace(min(real_ave_degree_dict[beta]), max(real_ave_degree_dict[beta]), 100000)
        ana_vec = [0.033 * x_value ** 0.25 * (0.12 + 1.28 * np.log(10000) / np.log(x_value)) for x_value in x2]


        x3 = np.linspace(min(real_ave_degree_dict[beta]), 20, 10000)
        ana_vec_real = [0.07 * (0.12 + 1.28 * np.log(10000) / np.log(x_value)) for x_value in x3]
        ana_vec_app = [0.04 * (0.12 + 1.28 * np.log(10000) / np.log(x_value)) for x_value in x3]


        # plt.plot(x, [x_value ** 0.5 * (0.12 + 1.28 * np.log(10000) / np.log(x_value)) * np.log(x_value) for x_value in x],
        #          label=f"fit2: analytic formula: Llog(k)~k^0.25")

        plt.plot(x2, ana_vec, "--", linewidth=5, label=r"$ana: <S> = c* k^{0.25}(A+B* log(N)/log(k))$")

        plt.plot(x3, ana_vec_real, "--", linewidth=5, label=r"$ana: <S> = c_1(A+B* log(N)/log(k))$")
        plt.plot(x3, ana_vec_app, "--", linewidth=5, label=r"$ana: <S> = c_2(A+B* log(N)/log(k))$")

        x4 = np.linspace(100, 10000, 10000)
        ana_vec_real_tail = [0.033 * x_value ** 0.25 * 2.4 for x_value in x4]
        ana_vec_app_tail = [0.033 * x_value ** 0.25 * 3.3 for x_value in
                             x4]

        plt.plot(x4, ana_vec_real_tail, "--", linewidth=5, label=r"$ana: <S> = c_3* k^{0.25}$")
        plt.plot(x4, ana_vec_app_tail, "--", linewidth=5, label=r"$ana: <S> = c_4* k^{0.25}$")



    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Average degree, $\langle D \rangle$', fontsize=26)
    # plt.xlabel(r'Expected degree, $E[D]$', fontsize=26)
    # plt.xlabel(r'$\alpha$', fontsize=26)
    # plt.ylim(0, 2.7)
    # plt.xlim(2, 12000)
    plt.ylabel(r'Average stretch, $\langle S \rangle$', fontsize=26)

    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=26, loc=(0.4, 0.5))
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

    ax1 = plt.gca()  # 左边 y 轴
    ax2 = ax1.twinx()  # 创建右边 y 轴

    ax2.plot(real_ave_degree_dict[beta], hop_count[beta],
             linestyle="-", linewidth=3, marker='s', markersize=14,
             label="hop count", color=colors[4])

    ax2.set_ylabel("hop count")


    plt.show()
    plt.close()


def plot_realL_approxL_together_fit_curve_tail_asconstant():
    """
    Figure 6 inset: local minimum of stretch

    produce strech vs <r><h> with curve fit (beta == 2.5 and 1024)
    the y-axis is <S>*k^{-tau}
    :return:
    """

    real_ave_degree_dict = {}
    ave_L = {}
    approx_L = {}
    approx_L_maxdl = {}

    hop_count = {}

    beta = 2.5

    N = 10000
    exclude_hop_flag = True
    realL = True
    kvec = [2.1,2.2,2.3,2.4,2.5,2.6,2.7, 2.8, 3.4, 4.4,  6.0, 8.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999, 16479, 27081, 44767, 73534, 121205, 199999]
    real_ave_degree_vec, _, _, _, _, _, _, ave_L_vec, _ = load_large_network_results_dev_vs_avg_approxLrealdiff(
        N, beta, kvec,realL,exclude_hop_flag)
    real_ave_degree_dict[beta] = real_ave_degree_vec
    ave_L[beta] = ave_L_vec
    realL = False
    real_ave_degree_vec, _, _, _, _, _, _, ave_L_vec2, _ = load_large_network_results_dev_vs_avg_approxLrealdiff(
        N, beta, kvec, realL,exclude_hop_flag)
    approx_L[beta] = ave_L_vec2

    _, _, _, _, _, hop_vec, _, ave_L_vec2, _ = load_large_network_results_dev_vs_avg_approxLlargedlrealdiff(
        N, beta, kvec, realL)
    approx_L_maxdl[beta] = ave_L_vec2
    hop_count[beta] = hop_vec


    figure()
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494', '#9FA9C9', '#D36A6A']

    x = real_ave_degree_dict[beta]

    y1 = ave_L[beta]
    a = 0.25
    b = 0.25
    y1 = [i*j**(-a) for (i,j) in zip(y1,x)]
    y2 = approx_L[beta]
    y2 = [i*j**(-b) for (i,j) in zip(y2,x)]

    # x = [np.log(i) for i in x]
    plt.plot(x, y1, linestyle="--", linewidth=3, marker='s', markersize=16,
             label=fr"real $\langle S \rangle k^{{-{a}}}$", color=colors[1])
    if exclude_hop_flag:
        plt.plot(x, y2, linestyle="--", linewidth=3, marker='s', markersize=16,
                 label=fr"exculde 1-hop, approx $<r><h>k^{{-{b}}}$", color = colors[2])
    else:
        plt.plot(x, y2, linestyle="--", linewidth=3, marker='s', markersize=16,
                 label=fr"include 1-hop, approx $<r><h>k^{{-{b}}}$", color=colors[2])

    # y3 = [1/i for i in x]
    # plt.plot(x, y3, linestyle="-", linewidth=3,
    #          label=r"y = c/log(x)")
    #
    # y4 = [0.5 / i for i in x]
    # plt.plot(x, y4, linestyle="-", linewidth=3,
    #          label=r"y = c2/log(x)")

    # plt.plot(real_ave_degree_dict[beta], approx_L_maxdl[beta], linestyle="--", linewidth=3, marker='o', markersize=16,
    #          label=r"approx $<S> = max(r) <h>$", color=colors[3])


    # if beta ==2.5:
    #     x2 = np.linspace(min(real_ave_degree_dict[beta]), max(real_ave_degree_dict[beta]), 100000)
    #     ana_vec = [0.033 * x_value ** 0.25 * (0.12 + 1.28 * np.log(10000) / np.log(x_value)) for x_value in x2]
    #
    #
    #     x3 = np.linspace(min(real_ave_degree_dict[beta]), 20, 10000)
    #     ana_vec_real = [0.07 * (0.12 + 1.28 * np.log(10000) / np.log(x_value)) for x_value in x3]
    #     ana_vec_app = [0.04 * (0.12 + 1.28 * np.log(10000) / np.log(x_value)) for x_value in x3]
    #
    #
    #     # plt.plot(x, [x_value ** 0.5 * (0.12 + 1.28 * np.log(10000) / np.log(x_value)) * np.log(x_value) for x_value in x],
    #     #          label=f"fit2: analytic formula: Llog(k)~k^0.25")
    #
    #     plt.plot(x2, ana_vec, "--", linewidth=5, label=r"$ana: <S> = c* k^{0.25}(A+B* log(N)/log(k))$")
    #
    #     plt.plot(x3, ana_vec_real, "--", linewidth=5, label=r"$ana: <S> = c_1(A+B* log(N)/log(k))$")
    #     plt.plot(x3, ana_vec_app, "--", linewidth=5, label=r"$ana: <S> = c_2(A+B* log(N)/log(k))$")
    #
    #     x4 = np.linspace(100, 10000, 10000)
    #     ana_vec_real_tail = [0.033 * x_value ** 0.25 * 2.4 for x_value in x4]
    #     ana_vec_app_tail = [0.033 * x_value ** 0.25 * 3.3 for x_value in
    #                          x4]
    #
    #     plt.plot(x4, ana_vec_real_tail, "--", linewidth=5, label=r"$ana: <S> = c_3* k^{0.25}$")
    #     plt.plot(x4, ana_vec_app_tail, "--", linewidth=5, label=r"$ana: <S> = c_4* k^{0.25}$")

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'$<k>$', fontsize=26)
    # plt.xlabel(r'Expected degree, $E[D]$', fontsize=26)
    # plt.xlabel(r'$\alpha$', fontsize=26)
    # plt.ylim(0, 2.7)
    # plt.xlim(2, 12000)
    plt.ylabel(r'$\langle S \rangle k^{-\tau}$', fontsize=26)

    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=26, loc=(0.4, 0.8))
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


def plot_realL_approxL_together_test3():
    """
    Figure 6 inset2: local minimum of stretch

    produce strech vs <r><h> with curve fit (beta == 2.5 and 1024)
    the y-axis is <S>*k^{-tau} and the x-axis is 1/log(k)
    :return:
    """

    real_ave_degree_dict = {}
    ave_L = {}
    approx_L = {}
    approx_L_maxdl = {}

    hop_count = {}

    beta = 2.5

    N = 10000
    exclude_hop_flag = False
    realL = True
    # 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 3.4, 4.4, 6.0, 8.0,
    if beta == 2.5:
        kvec = [2.1, 2.2,2.4, 2.8, 3.4, 4.4, 5, 6.0, 8.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848,
                1389, 2276, 3727, 6105, 9999, 16479, 27081, 44767, 73534, 121205, 199999]
    elif beta ==2.1:
        kvec = [3.5,3.6,3.8, 4,4.4, 5, 6.0, 8.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848,
                1389, 2276, 3727, 6105, 9999, 16479, 27081, 44767, 73534, 121205, 199999]
    real_ave_degree_vec, _, _, _, _, _, _, ave_L_vec, _ = load_large_network_results_dev_vs_avg_approxLrealdiff(
        N, beta, kvec, realL, exclude_hop_flag)
    real_ave_degree_dict[beta] = real_ave_degree_vec
    ave_L[beta] = ave_L_vec
    realL = False
    real_ave_degree_vec, _, _, _, _, _, _, ave_L_vec2, _ = load_large_network_results_dev_vs_avg_approxLrealdiff(
        N, beta, kvec, realL, exclude_hop_flag)
    approx_L[beta] = ave_L_vec2

    # _, _, _, _, _, hop_vec, _, ave_L_vec2, _ = load_large_network_results_dev_vs_avg_approxLlargedlrealdiff(
    #     N, beta, kvec, realL)
    # approx_L_maxdl[beta] = ave_L_vec2
    # hop_count[beta] = hop_vec

    figure()
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494', '#9FA9C9', '#D36A6A']

    x = real_ave_degree_dict[beta]

    y1 = ave_L[beta]
    a = 0.05
    b = 0.05
    y1 = [i * j ** (-a) for (i, j) in zip(y1, x)]
    y2 = approx_L[beta]
    y2 = [i * j ** (-b) for (i, j) in zip(y2, x)]

    y3 = [0.2 / np.log(i)+1.4 for i in x]
    y4 = [0.2 / np.log(i)+1 for i in x]
    x = [1/np.log(i) for i in x]
    # x = [np.log(i) for i in x]
    plt.plot(x, y1, linestyle="--", linewidth=3, marker='s', markersize=16,
             label=fr"real $\langle S \rangle k^{{-{a}}}$", color=colors[1])
    if exclude_hop_flag:
        plt.plot(x, y2, linestyle="--", linewidth=3, marker='s', markersize=16,
                 label=fr"exculde 1-hop, approx $<r><h>k^{{-{b}}}$", color=colors[2])
    else:
        plt.plot(x, y2, linestyle="--", linewidth=3, marker='s', markersize=16,
                 label=fr"include 1-hop, approx $<r><h>k^{{-{b}}}$", color=colors[2])


    plt.plot(x, y3, linestyle="-", linewidth=3,
             label=r"y = 0.4/log(x)")

    plt.plot(x, y4, linestyle="-", linewidth=3,
             label=r"y = 0.4/log(x)+0.4")
    #
    # y4 = [0.5 / i for i in x]
    # plt.plot(x, y4, linestyle="-", linewidth=3,
    #          label=r"y = c2/log(x)")

    # plt.plot(real_ave_degree_dict[beta], approx_L_maxdl[beta], linestyle="--", linewidth=3, marker='o', markersize=16,
    #          label=r"approx $<S> = max(r) <h>$", color=colors[3])

    # if beta ==2.5:
    #     x2 = np.linspace(min(real_ave_degree_dict[beta]), max(real_ave_degree_dict[beta]), 100000)
    #     ana_vec = [0.033 * x_value ** 0.25 * (0.12 + 1.28 * np.log(10000) / np.log(x_value)) for x_value in x2]
    #
    #
    #     x3 = np.linspace(min(real_ave_degree_dict[beta]), 20, 10000)
    #     ana_vec_real = [0.07 * (0.12 + 1.28 * np.log(10000) / np.log(x_value)) for x_value in x3]
    #     ana_vec_app = [0.04 * (0.12 + 1.28 * np.log(10000) / np.log(x_value)) for x_value in x3]
    #
    #
    #     # plt.plot(x, [x_value ** 0.5 * (0.12 + 1.28 * np.log(10000) / np.log(x_value)) * np.log(x_value) for x_value in x],
    #     #          label=f"fit2: analytic formula: Llog(k)~k^0.25")
    #
    #     plt.plot(x2, ana_vec, "--", linewidth=5, label=r"$ana: <S> = c* k^{0.25}(A+B* log(N)/log(k))$")
    #
    #     plt.plot(x3, ana_vec_real, "--", linewidth=5, label=r"$ana: <S> = c_1(A+B* log(N)/log(k))$")
    #     plt.plot(x3, ana_vec_app, "--", linewidth=5, label=r"$ana: <S> = c_2(A+B* log(N)/log(k))$")
    #
    #     x4 = np.linspace(100, 10000, 10000)
    #     ana_vec_real_tail = [0.033 * x_value ** 0.25 * 2.4 for x_value in x4]
    #     ana_vec_app_tail = [0.033 * x_value ** 0.25 * 3.3 for x_value in
    #                          x4]
    #
    #     plt.plot(x4, ana_vec_real_tail, "--", linewidth=5, label=r"$ana: <S> = c_3* k^{0.25}$")
    #     plt.plot(x4, ana_vec_app_tail, "--", linewidth=5, label=r"$ana: <S> = c_4* k^{0.25}$")

    # plt.yscale('log')
    # plt.xscale('log')
    plt.xlabel(r'$1/log(k)$', fontsize=26)
    # plt.xlabel(r'Expected degree, $E[D]$', fontsize=26)
    # plt.xlabel(r'$\alpha$', fontsize=26)
    # plt.ylim(0, 2.7)
    # plt.xlim(2, 12000)
    plt.ylabel(r'$\langle S \rangle k^{-\tau}$', fontsize=26)

    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=26, loc=(0.1, 0.8))
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

    for beta in [3.1,8]:
        N = 10000
        realL = True
        kvec = [2.2, 2.8, 3.4, 4.4, 6.0, 8.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999,
                16479, 27081, 44767, 73534, 121205, 199999]
        real_ave_degree_vec, _, _, _, _, _, _, ave_L_vec, _ = load_large_network_results_dev_vs_avg_approxLrealdiff(
            N, beta, kvec, realL,True)
        real_ave_degree_dict[beta] = real_ave_degree_vec
        ave_L[beta] = ave_L_vec
        realL = False
        real_ave_degree_vec, _, _, _, _, _, _, ave_L_vec2, _ = load_large_network_results_dev_vs_avg_approxLrealdiff(
            N, beta, kvec, realL,True)
        approx_L[beta] = ave_L_vec2


    figure()
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494', '#9FA9C9', '#D36A6A']
    beta_vec = [2.1, 2.5,3.1, 4,8, 1024]
    for beta in beta_vec:
        # y = [i/j for (i,j) in zip(ave_L[beta], approx_L[beta])]
        y = [i - j for (i, j) in zip(ave_L[beta], approx_L[beta])]
        plt.plot(real_ave_degree_dict[beta], y, linestyle="-", linewidth=3, markersize=16,
                 label=rf"$\beta:${beta}")



    plt.xscale('log')
    plt.xlabel(r'Average degree, $\langle D \rangle$', fontsize=26)
    # plt.xlabel(r'Expected degree, $E[D]$', fontsize=26)
    # plt.xlabel(r'$\alpha$', fontsize=26)

    plt.ylabel(r'real $<S>$ - approx $<r><h>$', fontsize=26)

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
    """
    A quick work to show how different the real stretch vs <r><h>, as a function of expected degree
    :return:
    """
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
    """
    A quick work to show how different the real stretch vs <r><h>, as a function of expected degree,
    The y axis is L* Logk,
    :return:
    """
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
    """
    # STEP 1 <L> and <r><h> versus real average degree beta = 1024 under differnet N for local minimum
    """
    # function~1 figure: A quick work to show how different the real stretch vs , as a function of expected degree
    # plot_realLFlaseLananlyticL_together()
    # __________________________________________________________________________________________________________________

    # function~2 figure: A quick work to show how different the real stretch vs <r><h>, as a function of expected degree,
    #     The y axis is L* Logk,
    #___________________________________________________________________________________________________________________
    # plot_Llogk_realLFlaseLananlyticL_together()



    # function~3 figure:produce strech vs <r><h> vs <max(r)><h> with curve fit.
    # ___________________________________________________________________________________________________________________
    plot_realL_approxL_maxrh_together()



    # function~4 figure: Figure 6: local minimum of stretch
    # ___________________________________________________________________________________________________________________
    # plot_realL_approxL_together_withcurvefit_datasaved()



    # function~5 figure:
    # Figure 6 inset: local minimum of stretch
    #     produce strech vs <r><h> with curve fit (beta == 2.5 and 1024)
    #     the y-axis is <S>*k^{-tau}
    # ___________________________________________________________________________________________________________________
    # plot_realL_approxL_together_fit_curve_tail_asconstant()



    # function~6 figure:Figure 6 inset2: local minimum of stretch
    #
    #     produce strech vs <r><h> with curve fit (beta == 2.5 and 1024)
    #     the y-axis is <S>*k^{-tau} and the x-axis is 1/log(k)
    # ___________________________________________________________________________________________________________________
    # plot_realL_approxL_together_test3()


    # function~8 figure:  how the realL, <r><h>, <max(r)><h> for given E[D] change with N
    # ___________________________________________________________________________________________________________________
    # plot_realL_approxL_maxrh_hop_withN()


    # function~9 figure: See how the difference between <s>and <r><h> changes with beta
    # ___________________________________________________________________________________________________________________
    # plot_realL_approxL_together_diff()







