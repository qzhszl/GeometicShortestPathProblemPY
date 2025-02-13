# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 22-8-2024
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from R2SRGG.R2SRGG import loadSRGGandaddnode
from collections import defaultdict
import math


def load_small_network_results_beta(N, ED):
    # return betavec = [2.1, 4, 8, 16, 32, 64, 128] and corresponding
    # betavec = [2.1, 4, 8, 16, 32, 64, 128]
    # betavec = [2.1, 2.2, 2.4, 2.5, 2.6, 2.8, 3, 3.25, 3.5, 3.75, 4, 5, 6, 8, 10, 12, 16, 32, 64, 128]
    betavec = [2.1, 2.2, 2.4, 2.5, 2.6, 2.8, 3, 3.25, 3.5, 3.75, 4, 5, 6, 8, 16, 32, 64, 128]
    betavec = [2.2, 3.0, 4.2, 5.9, 8.3, 11.7, 16.5, 23.2, 32.7, 46.1, 64.9, 91.5, 128.9, 181.7, 256]
    betavec = [8]
    exemptionlist = []
    ave_deviation_vec = []
    std_deviation_vec = []
    ave_cc_vec = []
    ExternalSimutime = 0
    if ED < N:
        for beta in betavec:
            try:
                clustering_coefficient_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\clustering_coefficient_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                clustering_coefficient = np.loadtxt(clustering_coefficient_name)
                ave_cc_vec.append(np.mean(clustering_coefficient))
                ave_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                ave_deviation_for_a_para_comb = np.loadtxt(ave_deviation_name)
                ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
                std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))
            except FileNotFoundError:
                exemptionlist.append((N, ED, beta, ExternalSimutime))
                print(exemptionlist)
    return ave_cc_vec, ave_deviation_vec, std_deviation_vec


def load_large_network_results_beta(N, ED):
    # betavec = [2.1, 4, 8, 16, 32, 64, 128]
    # betavec = [2.1, 2.2, 2.4, 2.5, 2.6, 2.8, 3, 3.25, 3.5, 3.75, 4, 5, 6,8, 10, 12,16,32,64,128]
    betavec = [2.1, 2.2, 2.4, 2.5, 2.6, 2.8, 3, 3.25, 3.5, 3.75, 4, 5, 6, 8, 16, 32, 64, 128]
    betavec = [2.2, 3.0, 4.2, 5.9, 8.3, 11.7, 16.5, 23.2, 32.7, 46.1, 64.9, 91.5, 128.9, 181.7, 256]
    betavec = [8]
    exemptionlist = []
    ave_deviation_vec = []
    clustering_coefficient_vec = []
    std_deviation_vec = []
    ExternalSimutime = 0
    if ED < N:
        for beta in betavec:
            try:
                FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\network_N{Nn}ED{EDn}Beta{betan}.txt".format(
                    Nn=N, EDn=ED, betan=beta)
                G = loadSRGGandaddnode(N, FileNetworkName)
                clustering_coefficient = nx.average_clustering(G)
                # print("real ED:", real_avg)
                clustering_coefficient_vec.append(clustering_coefficient)

                deviation_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                ave_deviation_for_a_para_comb = np.loadtxt(deviation_vec_name)
                ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
                std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))
            except FileNotFoundError:
                exemptionlist.append((N, ED, beta, ExternalSimutime))
                print(exemptionlist)
    combined = sorted(zip(clustering_coefficient_vec, ave_deviation_vec, std_deviation_vec))
    # 拆分成独立的列表
    a_sorted, b_sorted, c_sorted = zip(*combined)
    # 将元组转换为列表
    clustering_coefficient_vec = list(a_sorted)
    ave_deviation_vec = list(b_sorted)
    std_deviation_vec = list(c_sorted)
    return clustering_coefficient_vec, ave_deviation_vec, std_deviation_vec


def load_10000nodenetwork_results_beta(ED):
    # betavec = [2.1, 4, 8, 16, 32, 64, 128]
    betavec = [2.1, 2.2, 2.4, 2.5, 2.6, 2.8, 3, 3.25, 3.5, 3.75, 4, 5, 6, 8, 16, 32, 64, 128]
    # betavec = [2.1, 2.2, 2.4, 2.5, 2.6, 2.8, 3, 3.25, 3.5, 3.75, 4, 5, 6, 8, 10, 12, 16, 32, 64, 128]
    # betavec = [2.1, 2.2, 2.4, 2.5, 2.6, 2.8, 3, 3.25, 3.5, 3.75, 4, 5, 6, 8, 16, 32, 64, 128]
    betavec = [2.2, 3.0, 4.2, 5.9, 8.3, 11.7, 16.5, 23.2, 32.7, 46.1, 64.9, 91.5, 128.9, 181.7, 256]
    N = 10000
    exemptionlist = []

    ave_deviation_vec = []
    std_deviation_vec = []
    clustering_coefficient_vec = []
    for beta in betavec:
        ave_deviation_for_a_para_comb = []
        # original path D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\
        # FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\network_N{Nn}ED{EDn}Beta{betan}.txt".format(
        #     Nn=N, EDn=ED, betan=beta)
        # G = loadSRGGandaddnode(N, FileNetworkName)
        # clustering_coefficient = nx.average_clustering(G)
        # # print("real ED:", real_avg)
        # clustering_coefficient_vec.append(clustering_coefficient)

        for ExternalSimutime in range(20):
            try:
                deviation_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                ave_deviation_for_a_para_comb_10times = np.loadtxt(deviation_vec_name)
                ave_deviation_for_a_para_comb.extend(ave_deviation_for_a_para_comb_10times)
            except FileNotFoundError:
                exemptionlist.append((N, ED, beta, ExternalSimutime))

        ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
        std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))
    clustering_coefficient_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\clustering_coefficient_ED{EDn}.txt".format(
        EDn=ED)
    np.savetxt(clustering_coefficient_Name, clustering_coefficient_vec)
    ave_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\ave_deviation_ED{EDn}.txt".format(
        EDn=ED)
    np.savetxt(ave_deviation_Name, ave_deviation_vec)
    std_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\std_deviation_ED{EDn}.txt".format(
        EDn=ED)
    np.savetxt(std_deviation_Name, std_deviation_vec)
    print(exemptionlist)
    return clustering_coefficient_vec, ave_deviation_vec, std_deviation_vec


def load_resort_data_beta(N, ED):
    # betavec = [2.1, 4, 8, 16, 32, 64, 128]
    betavec = [2.1, 2.2, 2.4, 2.5, 2.6, 2.8, 3, 3.25, 3.5, 3.75, 4, 5, 6, 8, 16, 32, 64, 128]
    betavec = [2.1, 2.2, 2.4, 2.5, 2.6, 2.8, 3, 3.25, 3.5, 3.75, 4, 5, 6, 7, 8, 10, 12, 16, 32, 64, 128]
    # betavec = [2.2, 2.4, 2.5, 2.6, 2.8, 3, 3.25, 3.5, 3.75, 5, 6, 7, 10, 12]
    exemptionlist = []

    ave_deviation_vec = []
    ave_deviation_dic = {}
    clustering_coefficient_vec = []
    for beta in betavec:
        for ExternalSimutime in [0]:
            if N < 200:
                try:
                    clustering_coefficient_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\clustering_coefficient_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                    clustering_coefficient = np.loadtxt(clustering_coefficient_name)
                    clustering_coefficient_vec = clustering_coefficient_vec + list(clustering_coefficient)
                    nodepairs_for_eachgraph_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\nodepairs_for_eachgraph_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                    node_pairs_vec = np.loadtxt(nodepairs_for_eachgraph_vec_name, dtype=int)

                    ave_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                    ave_deviation_for_a_para_comb = np.loadtxt(ave_deviation_name)

                    a_index = 0
                    count = 0
                    for nodepair_num_inonegraph in node_pairs_vec:
                        b_index = a_index + nodepair_num_inonegraph - 1
                        ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb[a_index:b_index]))
                        real_cc = clustering_coefficient[count]
                        try:
                            ave_deviation_dic[real_cc] = ave_deviation_dic[real_cc] + list(
                                ave_deviation_for_a_para_comb[a_index:b_index])
                        except:
                            ave_deviation_dic[real_cc] = list(ave_deviation_for_a_para_comb[a_index:b_index])
                        a_index = b_index + 1
                        count = count + 1
                except FileNotFoundError:
                    exemptionlist.append((N, ED, beta, ExternalSimutime))

    resort_dict = {}
    for key_cc, value_deviation in ave_deviation_dic.items():
        if round_onetenthquator(key_cc) in resort_dict.keys():
            resort_dict[round_onetenthquator(key_cc)] = resort_dict[round_onetenthquator(key_cc)] + list(
                value_deviation)
            # a = max(list(value_deviation))
            # b = np.mean(list(value_deviation))
        else:
            resort_dict[round_onetenthquator(key_cc)] = list(value_deviation)
            # a = max(list(value_deviation))
            # b = np.mean(list(value_deviation))
    if 0 in resort_dict.keys():
        del resort_dict[0]
    resort_dict = {key: resort_dict[key] for key in sorted(resort_dict.keys())}
    degree_vec_resort = list(resort_dict.keys())
    ave_deviation_resort = [np.mean(resort_dict[key_d]) for key_d in degree_vec_resort]
    std_deviation_resort = [np.std(resort_dict[key_d]) for key_d in degree_vec_resort]

    return degree_vec_resort, ave_deviation_resort, std_deviation_resort, clustering_coefficient_vec, ave_deviation_vec, ave_deviation_dic


def load_clean_data_small_graph(N):
    """
    In this.m we save the real avg,cc and corresponding deviation list in an array [avg,cc] and a dict{1,[1,2,3,4,4]}
    :param N: network size
    :return:
    """
    # betavec = [2.1, 4, 8, 16, 32, 64, 128]
    # betavec = [2.2, 2.4, 2.5, 2.6, 2.8, 3, 3.25, 3.5, 3.75, 5, 6, 7, 10, 12]
    # betavec = [2.1, 2.2, 2.4, 2.5, 2.6, 2.8, 3, 3.25, 3.5, 3.75, 4, 5, 6, 8, 16, 32, 64, 128]
    betavec = [2.1, 2.2, 2.4, 2.5, 2.6, 2.8, 3, 3.25, 3.5, 3.75, 4, 5, 6, 7, 8, 10, 12, 16, 32, 64, 128]
    kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    exemptionlist = []

    datamat = {}
    ave_deviation_vec = []
    ave_deviation_dic = {}
    clustering_coefficient_vec = []
    ave_avg_vec = []
    dict_index = 0
    for ED in kvec:
        if ED < N:
            for beta in betavec:
                for ExternalSimutime in [0]:
                    if N < 200:
                        try:
                            clustering_coefficient_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\clustering_coefficient_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                            clustering_coefficient = np.loadtxt(clustering_coefficient_name)
                            clustering_coefficient_vec = clustering_coefficient_vec + list(clustering_coefficient)

                            nodepairs_for_eachgraph_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\nodepairs_for_eachgraph_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                            node_pairs_vec = np.loadtxt(nodepairs_for_eachgraph_vec_name, dtype=int)

                            ave_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                            ave_deviation_for_a_para_comb = np.loadtxt(ave_deviation_name)

                            real_ave_degree_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\real_ave_degree_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                            real_ave_degree = np.loadtxt(real_ave_degree_name)
                            ave_avg_vec = ave_avg_vec + list(real_ave_degree)
                            a_index = 0
                            count = 0
                            for nodepair_num_inonegraph in node_pairs_vec:
                                dict_index = dict_index + 1
                                b_index = a_index + nodepair_num_inonegraph - 1
                                # ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb[a_index:b_index]))
                                ave_deviation_dic[dict_index] = list(ave_deviation_for_a_para_comb[a_index:b_index])
                                a_index = b_index + 1
                                count = count + 1

                        except FileNotFoundError:
                            exemptionlist.append((N, ED, beta, ExternalSimutime))

    print(len(ave_avg_vec))
    print(len(clustering_coefficient_vec))
    print(len(ave_deviation_dic))

    dataCCED = np.array(ave_avg_vec).reshape(-1, 1)
    dataCCED = np.column_stack((dataCCED, clustering_coefficient_vec))
    # filtered_data = data[(data[:, 0] == 1) & (data[:, 1] == 2)]

    # resort_dict = {}
    # for key_cc, value_deviation in ave_deviation_dic.items():
    #     if round_onetenthquator(key_cc) in resort_dict.keys():
    #         resort_dict[round_onetenthquator(key_cc)] = resort_dict[round_onetenthquator(key_cc)] + list(value_deviation)
    #         # a = max(list(value_deviation))
    #         # b = np.mean(list(value_deviation))
    #     else:
    #         resort_dict[round_onetenthquator(key_cc)] = list(value_deviation)
    #         # a = max(list(value_deviation))
    #         # b = np.mean(list(value_deviation))
    # if 0 in resort_dict.keys():
    #     del resort_dict[0]
    # resort_dict = {key: resort_dict[key] for key in sorted(resort_dict.keys())}
    # degree_vec_resort = list(resort_dict.keys())
    # ave_deviation_resort = [np.mean(resort_dict[key_d]) for key_d in degree_vec_resort]
    # std_deviation_resort = [np.std(resort_dict[key_d]) for key_d in degree_vec_resort]

    # return degree_vec_resort, ave_deviation_resort, std_deviation_resort, clustering_coefficient_vec, ave_deviation_vec, ave_deviation_dic


def load_clean_data_large_graph(N):
    """
    In this.m we save the real avg,cc and corresponding deviation list in an array [avg,cc] and a dict{1,[1,2,3,4,4]}
    :param N: network size
    :return:
    """
    # betavec = [2.1, 4, 8, 16, 32, 64, 128]
    # betavec = [2.2, 2.4, 2.5, 2.6, 2.8, 3, 3.25, 3.5, 3.75, 5, 6, 7, 10, 12]
    # betavec = [2.1, 2.2, 2.4, 2.5, 2.6, 2.8, 3, 3.25, 3.5, 3.75, 4, 5, 6, 8, 16, 32, 64, 128]
    betavec = [2.1, 2.2, 2.4, 2.5, 2.6, 2.8, 3, 3.25, 3.5, 3.75, 4, 5, 6, 7, 8, 10, 12, 16, 32, 64, 128]
    kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    exemptionlist = []

    datamat = {}
    ave_deviation_vec = []
    ave_deviation_dic = {}
    clustering_coefficient_vec = []
    ave_avg_vec = []
    dict_index = 0
    for ED in kvec:
        if ED < N:
            for beta in betavec:
                for ExternalSimutime in [0]:
                    if N < 200:
                        try:
                            clustering_coefficient_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\clustering_coefficient_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                            clustering_coefficient = np.loadtxt(clustering_coefficient_name)
                            clustering_coefficient_vec = clustering_coefficient_vec + list(clustering_coefficient)

                            nodepairs_for_eachgraph_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\nodepairs_for_eachgraph_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                            node_pairs_vec = np.loadtxt(nodepairs_for_eachgraph_vec_name, dtype=int)

                            ave_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                            ave_deviation_for_a_para_comb = np.loadtxt(ave_deviation_name)

                            real_ave_degree_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\real_ave_degree_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                            real_ave_degree = np.loadtxt(real_ave_degree_name)
                            ave_avg_vec = ave_avg_vec + list(real_ave_degree)
                            a_index = 0
                            count = 0
                            for nodepair_num_inonegraph in node_pairs_vec:
                                dict_index = dict_index + 1
                                b_index = a_index + nodepair_num_inonegraph - 1
                                # ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb[a_index:b_index]))
                                ave_deviation_dic[dict_index] = list(ave_deviation_for_a_para_comb[a_index:b_index])
                                a_index = b_index + 1
                                count = count + 1

                        except FileNotFoundError:
                            exemptionlist.append((N, ED, beta, ExternalSimutime))

    print(len(ave_avg_vec))
    print(len(clustering_coefficient_vec))
    print(len(ave_deviation_dic))

    dataCCED = np.array(ave_avg_vec).reshape(-1, 1)
    dataCCED = np.column_stack((dataCCED, clustering_coefficient_vec))
    # filtered_data = data[(data[:, 0] == 1) & (data[:, 1] == 2)]

    # resort_dict = {}
    # for key_cc, value_deviation in ave_deviation_dic.items():
    #     if round_onetenthquator(key_cc) in resort_dict.keys():
    #         resort_dict[round_onetenthquator(key_cc)] = resort_dict[round_onetenthquator(key_cc)] + list(value_deviation)
    #         # a = max(list(value_deviation))
    #         # b = np.mean(list(value_deviation))
    #     else:
    #         resort_dict[round_onetenthquator(key_cc)] = list(value_deviation)
    #         # a = max(list(value_deviation))
    #         # b = np.mean(list(value_deviation))
    # if 0 in resort_dict.keys():
    #     del resort_dict[0]
    # resort_dict = {key: resort_dict[key] for key in sorted(resort_dict.keys())}
    # degree_vec_resort = list(resort_dict.keys())
    # ave_deviation_resort = [np.mean(resort_dict[key_d]) for key_d in degree_vec_resort]
    # std_deviation_resort = [np.std(resort_dict[key_d]) for key_d in degree_vec_resort]

    # return degree_vec_resort, ave_deviation_resort, std_deviation_resort, clustering_coefficient_vec, ave_deviation_vec, ave_deviation_dic


def round_quarter(x):
    x_tail = x - math.floor(x)
    if x_tail < 0.25:
        return math.floor(x)
    elif x_tail < 0.75:
        return math.floor(x) + 0.5
    else:
        return math.ceil(x)


def round_onetenthquator(x):
    x = x * 10
    y = round_quarter(x)
    return y / 10


def plot_local_optimum_with_N(ED, beta):
    # the x-axis is the real average degree

    Nvec = [10, 22, 46, 100, 215, 464, 1000, 2154, 4642, 10000]
    clustering_coefficient_dict = {}
    ave_deviation_dict = {}
    std_deviation_dict = {}

    # original file path
    # clustering_coefficient_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\clustering_coefficient_ED{EDn}.txt".format(
    #     EDn=ED)
    for N in Nvec:
        if N < 400:
            for ED in [ED]:
                clustering_coefficient_vec, ave_deviation_vec, std_deviation_vec = load_small_network_results_beta(N,
                                                                                                                   ED)
                clustering_coefficient_dict[N] = clustering_coefficient_vec
                ave_deviation_dict[N] = ave_deviation_vec
                std_deviation_dict[N] = std_deviation_vec
        elif N < 10000:
            for ED in [ED]:
                clustering_coefficient_vec, ave_deviation_vec, std_deviation_vec = load_large_network_results_beta(N,
                                                                                                                   ED)
                clustering_coefficient_dict[N] = clustering_coefficient_vec
                ave_deviation_dict[N] = ave_deviation_vec
                std_deviation_dict[N] = std_deviation_vec
        else:
            for ED in [ED]:
                clustering_coefficient_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\clustering_coefficient_ED{EDn}.txt".format(
                    EDn=ED)

                ave_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\ave_deviation_ED{EDn}.txt".format(
                    EDn=ED)

                std_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\std_deviation_ED{EDn}.txt".format(
                    EDn=ED)
                clustering_coefficient_vec = np.loadtxt(clustering_coefficient_Name)
                ave_deviation_vec = np.loadtxt(ave_deviation_Name)
                std_deviation_vec = np.loadtxt(std_deviation_Name)

                clustering_coefficient_dict[N] = clustering_coefficient_vec
                ave_deviation_dict[N] = ave_deviation_vec
                std_deviation_dict[N] = std_deviation_vec

    fig, ax = plt.subplots(figsize=(9, 6))
    # colors = [[0, 0.4470, 0.7410],
    #           [0.8500, 0.3250, 0.0980],
    #           [0.9290, 0.6940, 0.1250],
    #           [0.4940, 0.1840, 0.5560],
    #           [0.4660, 0.6740, 0.1880]]
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    y = []
    error =[]
    for N_index in range(len(Nvec)):
        N = Nvec[N_index]
        y.append(ave_deviation_dict[N])
        error.append(std_deviation_dict[N])
        plt.errorbar(Nvec, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=16, color=colors[N_index])

    # plt.xscale('log')
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    plt.xlabel(r'$N$', fontsize=26)
    plt.ylabel('Average Deviation', fontsize=26)
    plt.xscale('log')
    # plt.yscale('log')
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.legend(fontsize=26, loc=(0.6, 0.5))
    plt.tick_params(axis='both', which="both", length=6, width=1)
    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\DeviationVsbetaED{EDn}logy2.pdf".format(
        EDn=ED)
    plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """
    Plot deviation versus different beta
    """
    # first time plot deviation with different beta
    # load_10000nodenetwork_results_beta(10)
    # plot_local_optimum_with_beta(5)

    # second time plot deviation with different beta
    plot_local_optimum_with_N(10, 8)
