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

def load_small_network_results_maxminave(N, ED, beta):
    exemptionlist =[]
    for N in [N]:
        ave_deviation_vec = []
        max_deviation_vec = []
        min_deviation_vec = []
        ran_deviation_vec = []
        for ExternalSimutime in [0]:
            try:
                clustering_coefficient_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\clustering_coefficient_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                clustering_coefficient = np.loadtxt(clustering_coefficient_name)
                print(np.mean(clustering_coefficient))

                real_ave_degree_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\real_ave_degree_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                real_ave_degree = np.loadtxt(real_ave_degree_name)
                print(np.mean(real_ave_degree))

                deviation_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                ave_deviation_for_a_para_comb_10times = np.loadtxt(deviation_vec_name)
                ave_deviation_vec.extend(ave_deviation_for_a_para_comb_10times)

                max_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\max_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                max_deviation_for_a_para_comb_10times = np.loadtxt(max_deviation_name)
                max_deviation_vec.extend(max_deviation_for_a_para_comb_10times)

                min_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\min_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                min_deviation_for_a_para_comb_10times = np.loadtxt(min_deviation_name)
                min_deviation_vec.extend(min_deviation_for_a_para_comb_10times)

                ave_baseline_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\ave_baseline_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                ave_baseline_deviation_for_a_para_comb_10times = np.loadtxt(ave_baseline_deviation_name)
                ran_deviation_vec.extend(ave_baseline_deviation_for_a_para_comb_10times)
            except FileNotFoundError:
                exemptionlist.append((N, ED, beta, ExternalSimutime))
                print(exemptionlist)
    return ave_deviation_vec, max_deviation_vec, min_deviation_vec, ran_deviation_vec, exemptionlist


def load_resort_data_smallN_maxminave(N, ED, beta):
    k_key = ED
    kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    exemptionlist =[]
    for N in [N]:
        ave_deviation_vec = []
        max_deviation_vec = []
        min_deviation_vec = []
        ran_deviation_vec = []
        ave_deviation_dic ={}
        max_deviation_dic = {}
        min_deviation_dic = {}
        ran_deviation_dic = {}


        real_ave_degree_vec = []

        for beta in [beta]:
            for ED in kvec:
                if ED<N:
                    for ExternalSimutime in [0]:
                        if N< 200:
                            try:
                                real_ave_degree_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\real_ave_degree_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                                real_ave_degree = np.loadtxt(real_ave_degree_name)
                                real_ave_degree_vec=real_ave_degree_vec+list(real_ave_degree)

                                nodepairs_for_eachgraph_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\nodepairs_for_eachgraph_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                                node_pairs_vec = np.loadtxt(nodepairs_for_eachgraph_vec_name, dtype=int)

                                ave_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                                ave_deviation_for_a_para_comb = np.loadtxt(ave_deviation_name)

                                max_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\max_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                                max_deviation_for_a_para_comb = np.loadtxt(max_deviation_name)


                                min_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\min_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                                min_deviation_for_a_para_comb = np.loadtxt(min_deviation_name)


                                ave_baseline_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\ave_baseline_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                                ran_deviation_for_a_para_comb = np.loadtxt(ave_baseline_deviation_name)

                                a_index = 0
                                count = 0
                                for nodepair_num_inonegraph in node_pairs_vec:
                                    b_index = a_index+nodepair_num_inonegraph-1
                                    ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb[a_index:b_index]))
                                    real_avg = real_ave_degree[count]
                                    try:
                                        ave_deviation_dic[real_avg] = ave_deviation_dic[real_avg] + list(ave_deviation_for_a_para_comb[a_index:b_index])
                                        max_deviation_dic[real_avg] = max_deviation_dic[real_avg] + list(
                                            max_deviation_for_a_para_comb[a_index:b_index])
                                        min_deviation_dic[real_avg] = min_deviation_dic[real_avg] + list(
                                            min_deviation_for_a_para_comb[a_index:b_index])
                                        ran_deviation_dic[real_avg] = ran_deviation_dic[real_avg] + list(
                                            ran_deviation_for_a_para_comb[a_index:b_index])
                                    except:
                                        ave_deviation_dic[real_avg] = list(ave_deviation_for_a_para_comb[a_index:b_index])
                                        max_deviation_dic[real_avg] = list(
                                            max_deviation_for_a_para_comb[a_index:b_index])
                                        min_deviation_dic[real_avg] = list(
                                            min_deviation_for_a_para_comb[a_index:b_index])
                                        ran_deviation_dic[real_avg] = list(
                                            ran_deviation_for_a_para_comb[a_index:b_index])
                                    a_index = b_index+1
                                    count = count+1
                            except FileNotFoundError:
                                exemptionlist.append((N, ED, beta, ExternalSimutime))

    resort_dict_ave = {}
    resort_dict_max = {}
    resort_dict_min = {}
    resort_dict_ran = {}

    for key_degree, value_deviation in ave_deviation_dic.items():
        if round_quarter(key_degree) in resort_dict_ave.keys():
            resort_dict_ave[round_quarter(key_degree)] = resort_dict_ave[round_quarter(key_degree)] + list(value_deviation)
            resort_dict_max[round_quarter(key_degree)] = resort_dict_max[round_quarter(key_degree)] + list(
                max_deviation_dic[key_degree])
            resort_dict_min[round_quarter(key_degree)] = resort_dict_min[round_quarter(key_degree)] + list(
                min_deviation_dic[key_degree])
            resort_dict_ran[round_quarter(key_degree)] = resort_dict_ran[round_quarter(key_degree)] + list(
                ran_deviation_dic[key_degree])
            # a = max(list(value_deviation))
            # b = np.mean(list(value_deviation))
        else:
            resort_dict_ave[round_quarter(key_degree)] = list(value_deviation)
            resort_dict_max[round_quarter(key_degree)] = list(max_deviation_dic[key_degree])
            resort_dict_min[round_quarter(key_degree)] = list(min_deviation_dic[key_degree])
            resort_dict_ran[round_quarter(key_degree)] = list(ran_deviation_dic[key_degree])
            # a = max(list(value_deviation))
            # b = np.mean(list(value_deviation))
    # if 0 in resort_dict_ave.keys():
    #     del resort_dict_ave[0]
    # if 0.5 in resort_dict_ave.keys():
    #     del resort_dict_ave[0.5]
    # resort_dict_ave = {key: resort_dict_ave[key] for key in sorted(resort_dict_ave.keys())}
    # degree_vec_resort = list(resort_dict_ave.keys())
    # ave_deviation_resort = [np.mean(resort_dict_ave[key_d]) for key_d in degree_vec_resort]
    # std_deviation_resort = [np.std(resort_dict_ave[key_d]) for key_d in degree_vec_resort]
    return list(resort_dict_ave[k_key]), list(resort_dict_max[k_key]), list(resort_dict_min[k_key]), list(resort_dict_ran[k_key]),exemptionlist



def load_large_network_results_maxminave(N, ED, beta):
    exemptionlist = []
    for N in [N]:
        ave_deviation_vec = []
        max_deviation_vec = []
        min_deviation_vec = []
        ran_deviation_vec = []
        for ExternalSimutime in [0]:
            try:
                clustering_coefficient_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\clustering_coefficient_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                clustering_coefficient = np.loadtxt(clustering_coefficient_name)
                print(np.mean(clustering_coefficient))

                real_ave_degree_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\real_ave_degree_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                real_ave_degree = np.loadtxt(real_ave_degree_name)
                print(np.mean(real_ave_degree))

                deviation_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                ave_deviation_for_a_para_comb_10times = np.loadtxt(deviation_vec_name)
                ave_deviation_vec.extend(ave_deviation_for_a_para_comb_10times)

                max_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\max_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                max_deviation_for_a_para_comb_10times = np.loadtxt(max_deviation_name)
                max_deviation_vec.extend(max_deviation_for_a_para_comb_10times)

                min_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\min_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                min_deviation_for_a_para_comb_10times = np.loadtxt(min_deviation_name)
                min_deviation_vec.extend(min_deviation_for_a_para_comb_10times)

                ave_baseline_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\ave_baseline_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                ave_baseline_deviation_for_a_para_comb_10times = np.loadtxt(ave_baseline_deviation_name)
                ran_deviation_vec.extend(ave_baseline_deviation_for_a_para_comb_10times)
            except FileNotFoundError:
                exemptionlist.append((N, ED, beta, ExternalSimutime))
                print(exemptionlist)
    return ave_deviation_vec, max_deviation_vec, min_deviation_vec, ran_deviation_vec, exemptionlist


def load_10000nodenetwork_maxminave(ED, beta):
    exemptionlist =[]
    for N in [10000]:
        ave_deviation_vec = []
        max_deviation_vec = []
        min_deviation_vec = []
        ran_deviation_vec = []

        # foldername = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\1000realization\\"
        foldername = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\1000000realization\\"

        # FileNetworkName = foler_name+"network_N{Nn}ED{EDn}Beta{betan}.txt".format(
        #     Nn=N, EDn=ED, betan=beta)
        # G = loadSRGGandaddnode(N, FileNetworkName)
        # real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
        # print("real ED:", real_avg)

        for ExternalSimutime in range(20):
            try:
                deviation_vec_name = foldername+ "ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                ave_deviation_for_a_para_comb_10times = np.loadtxt(deviation_vec_name)
                ave_deviation_vec.extend(ave_deviation_for_a_para_comb_10times)

                max_deviation_name = foldername+"max_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                max_deviation_for_a_para_comb_10times = np.loadtxt(max_deviation_name)
                max_deviation_vec.extend(max_deviation_for_a_para_comb_10times)

                min_deviation_name = foldername+"min_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                min_deviation_for_a_para_comb_10times = np.loadtxt(min_deviation_name)
                min_deviation_vec.extend(min_deviation_for_a_para_comb_10times)

                ave_baseline_deviation_name = foldername+"ave_baseline_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                ave_baseline_deviation_for_a_para_comb_10times = np.loadtxt(ave_baseline_deviation_name)
                ran_deviation_vec.extend(ave_baseline_deviation_for_a_para_comb_10times)
            except FileNotFoundError:
                exemptionlist.append((N, ED, beta, ExternalSimutime))
                # print(exemptionlist)
    return ave_deviation_vec, max_deviation_vec, min_deviation_vec, ran_deviation_vec, exemptionlist


def load_resort_data(N ,beta):
    kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    exemptionlist =[]
    for N in [N]:
        ave_deviation_vec = []
        ave_deviation_dic ={}
        real_ave_degree_vec = []

        for beta in [beta]:
            for ED in kvec:
                if ED<N:
                    for ExternalSimutime in [0]:
                        if N< 200:
                            try:
                                real_ave_degree_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\real_ave_degree_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                                real_ave_degree = np.loadtxt(real_ave_degree_name)
                                real_ave_degree_vec=real_ave_degree_vec+list(real_ave_degree)
                                nodepairs_for_eachgraph_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\nodepairs_for_eachgraph_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                                node_pairs_vec = np.loadtxt(nodepairs_for_eachgraph_vec_name, dtype=int)

                                ave_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                                ave_deviation_for_a_para_comb = np.loadtxt(ave_deviation_name)

                                a_index = 0
                                count = 0
                                for nodepair_num_inonegraph in node_pairs_vec:
                                    b_index = a_index+nodepair_num_inonegraph-1
                                    ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb[a_index:b_index]))
                                    real_avg = real_ave_degree[count]
                                    try:
                                        ave_deviation_dic[real_avg] = ave_deviation_dic[real_avg] + list(ave_deviation_for_a_para_comb[a_index:b_index])
                                    except:
                                        ave_deviation_dic[real_avg] = list(ave_deviation_for_a_para_comb[a_index:b_index])
                                    a_index = b_index+1
                                    count = count+1
                            except FileNotFoundError:
                                exemptionlist.append((N, ED, beta, ExternalSimutime))

    resort_dict = {}
    for key_degree, value_deviation in ave_deviation_dic.items():
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
    ave_deviation_resort = [np.mean(resort_dict[key_d]) for key_d in degree_vec_resort]
    std_deviation_resort = [np.std(resort_dict[key_d]) for key_d in degree_vec_resort]

    return degree_vec_resort, ave_deviation_resort, std_deviation_resort, real_ave_degree_vec, ave_deviation_vec, ave_deviation_dic


def round_quarter(x):
    x_tail = x- math.floor(x)
    if x_tail<0.25:
        return math.floor(x)
    elif x_tail<0.75:
        return math.floor(x)+0.5
    else:
        return math.ceil(x)


def plot_distribution(N, ED, beta):
    """
    Compared maximum, minimum, average deviation with randomly selected nodes
    :return:
    """
    # Nvec = [20,50,100,1000]
    # # Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    # beta = 8
    if N < 200:
        ave_deviation_vec, max_deviation_vec, min_deviation_vec, ran_deviation_vec, _ = load_resort_data_smallN_maxminave(
            N, ED, beta)
    elif N < 10000:
        ave_deviation_vec, max_deviation_vec, min_deviation_vec, ran_deviation_vec, _ = load_large_network_results_maxminave(
            N, ED, beta)
    else:
        ave_deviation_vec, max_deviation_vec, min_deviation_vec, ran_deviation_vec, _ = load_10000nodenetwork_maxminave(
            ED, beta)

    # cuttail = [9,19,34,24]
    # peakcut = [9,5,5,5]

    data1 = ave_deviation_vec
    # data1 = [0,0,0]
    data2 = max_deviation_vec
    data3 = min_deviation_vec
    data4 = ran_deviation_vec

    # fig, ax = plt.subplots(figsize=(6, 4.5))
    fig, ax = plt.subplots(figsize=(8, 4.5))

    datasets = [data1,data2,data3,data4]
    colors = [[0, 0.4470, 0.7410],
              [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250],
              [0.4940, 0.1840, 0.5560]]
    labels = ["Ave","Max","Min","Ran"]
    for data, color, label in zip(datasets, colors, labels):
        hvalue, bin_vec = np.histogram(data, bins=60, density=True)
        print(bin_vec[1:len(bin_vec)])
        plt.plot(bin_vec[1:len(bin_vec)], hvalue, color=color, label=label, linewidth=5)


    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlim([0,1])
    # plt.yticks([0,5,10,15,20,25])
    # plt.yticks([0, 10, 20, 30, 40, 50])

    plt.xlabel(r'x',fontsize = 32)
    plt.ylabel(r'$f_{d(q,\gamma(i,j))}(x)$',fontsize = 32)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=28,handlelength=1, handletextpad=0.5, frameon=False)
    plt.tick_params(axis='both', which="both",length=6, width=1)
    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\DistributionN{Nn}ED{EDn}Beta{betan}.pdf".format(Nn = N, EDn = ED, betan=beta)
    plt.savefig(picname,format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    # plt.close()


def plot_distribution_10000node(N, ED, beta):
    """
    Compared maximum, minimum, average deviation with randomly selected nodes
    :return:
    """
    # Nvec = [20,50,100,1000]
    # # Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    # beta = 8
    if N < 200:
        ave_deviation_vec, max_deviation_vec, min_deviation_vec, ran_deviation_vec, _ = load_resort_data_smallN_maxminave(
            N, ED, beta)
    elif N < 10000:
        ave_deviation_vec, max_deviation_vec, min_deviation_vec, ran_deviation_vec, _ = load_large_network_results_maxminave(
            N, ED, beta)
    else:
        ave_deviation_vec, max_deviation_vec, min_deviation_vec, ran_deviation_vec, _ = load_10000nodenetwork_maxminave(
            ED, beta)

    # cuttail = [9,19,34,24]
    # peakcut = [9,5,5,5]

    data1 = ave_deviation_vec
    # data1 = [0,0,0]
    data2 = max_deviation_vec
    data3 = min_deviation_vec
    data4 = ran_deviation_vec

    fig, ax = plt.subplots(figsize=(8, 4.5))

    datasets = [data1,data2,data3,data4]
    colors = [[0, 0.4470, 0.7410],
              [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250],
              [0.4940, 0.1840, 0.5560]]
    labels = ["Ave","Max","Min","Ran"]
    for data, color, label in zip(datasets, colors, labels):
        hvalue, bin_vec = np.histogram(data, bins=60, density=True)
        print(bin_vec[1:len(bin_vec)])
        plt.plot(bin_vec[1:len(bin_vec)], hvalue, color=color, label=label, linewidth=5)

    text = r"$N = 10^4$, $\beta = {beta}$, $E[D] = {ED}$".format(beta=beta, ED=ED)
    ax.text(
        0.4, 0.85,  # 文本位置（轴坐标，0.5 表示图中央，1.05 表示轴上方）
        text,
        transform=ax.transAxes,  # 使用轴坐标
        fontsize=26,  # 字体大小
        ha='center',  # 水平居中对齐
        va='bottom'  # 垂直对齐方式
    )


    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.spines['left'].set_position(('data', 0))
    # ax.spines['bottom'].set_position(('data', 0))
    # plt.xscale('log')
    plt.yscale('log')

    plt.xlim([0,1.4])

    ymin = 0.0001  # 设置最低点
    current_ylim = ax.get_ylim()  # 获取当前的 y 轴范围
    ax.set_ylim(ymin, current_ylim[1])  # 保持最大值不变
    # plt.yticks([0,5,10,15,20,25])
    # plt.yticks([0, 10, 20, 30, 40, 50])

    plt.xlabel(r'x',fontsize = 32)
    plt.ylabel(r'$f_{d(q,\gamma(i,j))}(x)$',fontsize = 32)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)

    plt.legend(fontsize=26, handlelength=1, handletextpad=0.5, frameon=False,loc='right')
    plt.tick_params(axis='both', which="both",length=6, width=1)
    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\DistributionN{Nn}ED{EDn}Beta{betan}logy.pdf".format(Nn = N, EDn = ED, betan=beta)
    plt.savefig(picname,format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    # plt.close()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # plot_distribution(100,5,4)
    plot_distribution_10000node(10000, 5.0, 4)



