# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 22-8-2024
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import curve_fit

from R2SRGG.R2SRGG import loadSRGGandaddnode
from collections import defaultdict
import math
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def load_small_network_results(N, beta):
    # Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    # betavec = [2.1, 4, 8, 16, 32, 64, 128]

    exemptionlist = []
    for N in [N]:
        ave_deviation_vec = []
        std_deviation_vec = []
        real_ave_degree_vec = []
        for beta in [beta]:
            for ED in kvec:
                if ED < N:
                    for ExternalSimutime in [0]:
                        try:
                            real_ave_degree_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\real_ave_degree_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                            real_ave_degree = np.loadtxt(real_ave_degree_name)
                            real_ave_degree_vec.append(np.mean(real_ave_degree))
                            ave_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                            ave_deviation_for_a_para_comb = np.loadtxt(ave_deviation_name)
                            ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
                            std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))
                        except FileNotFoundError:
                            exemptionlist.append((N, ED, beta, ExternalSimutime))
                            print(exemptionlist)
    return real_ave_degree_vec, ave_deviation_vec, std_deviation_vec


def load_large_network_results(N, beta):
    # Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    # kvec = list(range(2, 15)) + [20, 28, 40, 58, 83, 118, 169, 241, 344, 490, 700., 999]

    kvec = [2, 3] + list(range(4, 16)) + [20, 28, 40, 58, 83, 118, 169, 241, 344, 490, 700, 999]
    # betavec = [2.1, 4, 8, 16, 32, 64, 128]

    exemptionlist = []
    for N in [N]:
        ave_deviation_vec = []
        real_ave_degree_vec = []
        std_deviation_vec = []
        for beta in [beta]:
            for ED in kvec:
                for ExternalSimutime in [0]:
                    try:
                        FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\network_N{Nn}ED{EDn}Beta{betan}.txt".format(
                            Nn=N, EDn=ED, betan=beta)
                        G = loadSRGGandaddnode(N, FileNetworkName)
                        real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
                        # print("real ED:", real_avg)
                        real_ave_degree_vec.append(real_avg)

                        deviation_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        ave_deviation_for_a_para_comb = np.loadtxt(deviation_vec_name)
                        ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
                        std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))
                    except FileNotFoundError:
                        exemptionlist.append((N, ED, beta, ExternalSimutime))
                        print(exemptionlist)
    return real_ave_degree_vec, ave_deviation_vec, std_deviation_vec


def load_10000nodenetwork_results(beta):
    # load data for beta = beta, the deviation of each k in kvec

    # Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    # betavec = [2.1, 4, 8, 16, 32, 64, 128]

    exemptionlist = []
    for N in [10000]:
        ave_deviation_vec = []
        std_deviation_vec = []
        real_ave_degree_vec = []
        for beta in [beta]:
            for ED in kvec:
                ave_deviation_for_a_para_comb = []
                FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\network_N{Nn}ED{EDn}Beta{betan}.txt".format(
                    Nn=N, EDn=ED, betan=beta)
                G = loadSRGGandaddnode(N, FileNetworkName)
                real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
                # print("real ED:", real_avg)
                real_ave_degree_vec.append(real_avg)

                for ExternalSimutime in range(100):
                    try:
                        deviation_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\1000realization\\ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        ave_deviation_for_a_para_comb_10times = np.loadtxt(deviation_vec_name)
                        ave_deviation_for_a_para_comb.extend(ave_deviation_for_a_para_comb_10times)
                    except FileNotFoundError:
                        exemptionlist.append((N, ED, beta, ExternalSimutime))

                ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
                std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))
    print(exemptionlist)
    real_ave_degree_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\1000realization\\real_ave_degree_Beta{betan}.txt".format(
        betan=beta)
    np.savetxt(real_ave_degree_Name, real_ave_degree_vec)
    ave_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\1000realization\\ave_deviation_Beta{betan}.txt".format(
        betan=beta)
    np.savetxt(ave_deviation_Name, ave_deviation_vec)
    std_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\1000realization\\std_deviation_Beta{betan}.txt".format(
        betan=beta)
    np.savetxt(std_deviation_Name, std_deviation_vec)
    return real_ave_degree_vec, ave_deviation_vec, std_deviation_vec, exemptionlist


def plot_local_optimum():
    # the x-axis is the real average degree
    # Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    Nvec = [10, 20, 50, 100]
    real_ave_degree_dict = {}
    ave_deviation_dict = {}
    std_deviation_dict = {}

    for N in Nvec:
        if N < 200:
            for beta in [4]:
                real_ave_degree_vec, ave_deviation_vec, std_deviation_vec = load_small_network_results(N, beta)
                real_ave_degree_dict[N] = real_ave_degree_vec
                ave_deviation_dict[N] = ave_deviation_vec
                std_deviation_dict[N] = std_deviation_vec
        elif N < 10000:
            for beta in [4]:
                real_ave_degree_vec, ave_deviation_vec, std_deviation_vec = load_large_network_results(N, beta)
                real_ave_degree_dict[N] = real_ave_degree_vec
                ave_deviation_dict[N] = ave_deviation_vec
                std_deviation_dict[N] = std_deviation_vec
        else:
            for beta in [4]:
                real_ave_degree_vec, ave_deviation_vec, std_deviation_vec = load_10000nodenetwork_results(beta)
                real_ave_degree_dict[N] = real_ave_degree_vec
                ave_deviation_dict[N] = ave_deviation_vec
                std_deviation_dict[N] = std_deviation_vec

    plt.figure(figsize=(10, 8))
    colors = [[0.3059, 0.4745, 0.6549],
              [0.9490, 0.5569, 0.1686],
              [0.8824, 0.3412, 0.3490],
              [0.4627, 0.7176, 0.6980],
              [0.3490, 0.6314, 0.3098],
              [0.9294, 0.7882, 0.2824],
              [0.6902, 0.4784, 0.6314],
              [1.0000, 0.6157, 0.6549],
              [0.6118, 0.4588, 0.3725],
              [0.7294, 0.6902, 0.6745]]

    for N_index in range(len(Nvec)):
        N = Nvec[N_index]
        x = real_ave_degree_dict[N]
        y = ave_deviation_dict[N]
        error = std_deviation_dict[N]
        plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=2, elinewidth=1, capsize=2, alpha=0.5, marker='o',
                     markerfacecolor="none", label=f'N={N}', color=colors[N_index])

        # 找到峰值后最低点的坐标
        peak_index = np.argmax(y[0:10])
        post_peak_y = y[peak_index:]
        post_peak_min_index = peak_index + np.argmin(post_peak_y)
        post_peak_min_x = x[post_peak_min_index]
        post_peak_min_y = y[post_peak_min_index]

        # 标出最低点
        plt.plot(post_peak_min_x, post_peak_min_y, 'o', color=colors[N_index], markersize=8)

    plt.xscale('log')
    plt.xlabel('E[D]')
    plt.ylabel('Distance form shortest path nodes to the geodesic')
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend()
    plt.show()


def load_resort_data(N, beta):
    kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    exemptionlist = []
    for N in [N]:
        ave_deviation_vec = []
        ave_deviation_dic = {}
        real_ave_degree_vec = []

        for beta in [beta]:
            for ED in kvec:
                if ED < N:
                    for ExternalSimutime in [0]:
                        if N < 200:
                            try:
                                real_ave_degree_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\real_ave_degree_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                                real_ave_degree = np.loadtxt(real_ave_degree_name)
                                real_ave_degree_vec = real_ave_degree_vec + list(real_ave_degree)
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
                                    real_avg = real_ave_degree[count]
                                    try:
                                        ave_deviation_dic[real_avg] = ave_deviation_dic[real_avg] + list(
                                            ave_deviation_for_a_para_comb[a_index:b_index])
                                    except:
                                        ave_deviation_dic[real_avg] = list(
                                            ave_deviation_for_a_para_comb[a_index:b_index])
                                    a_index = b_index + 1
                                    count = count + 1
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
    x_tail = x - math.floor(x)
    if x_tail < 0.25:
        return math.floor(x)
    elif x_tail < 0.75:
        return math.floor(x) + 0.5
    else:
        return math.ceil(x)


def load_resort_data_smallN(N, beta):
    kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    exemptionlist = []
    for N in [N]:
        ave_deviation_vec = []
        ave_deviation_dic = {}
        real_ave_degree_vec = []

        for beta in [beta]:
            for ED in kvec:
                if ED < N:
                    for ExternalSimutime in [0]:
                        if N < 200:
                            try:
                                real_ave_degree_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\real_ave_degree_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                                real_ave_degree = np.loadtxt(real_ave_degree_name)
                                real_ave_degree_vec = real_ave_degree_vec + list(real_ave_degree)
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
                                    real_avg = real_ave_degree[count]
                                    try:
                                        ave_deviation_dic[real_avg] = ave_deviation_dic[real_avg] + list(
                                            ave_deviation_for_a_para_comb[a_index:b_index])
                                    except:
                                        ave_deviation_dic[real_avg] = list(
                                            ave_deviation_for_a_para_comb[a_index:b_index])
                                    a_index = b_index + 1
                                    count = count + 1
                            except FileNotFoundError:
                                exemptionlist.append((N, ED, beta, ExternalSimutime))

    resort_dict = {}

    for key_degree, value_deviation in ave_deviation_dic.items():
        if round_quarter(key_degree) in resort_dict.keys():
            resort_dict[round_quarter(key_degree)] = resort_dict[round_quarter(key_degree)] + list(value_deviation)
            # a = max(list(value_deviation))
            # b = np.mean(list(value_deviation))
        else:
            resort_dict[round_quarter(key_degree)] = list(value_deviation)
            # a = max(list(value_deviation))
            # b = np.mean(list(value_deviation))
    if 0 in resort_dict.keys():
        del resort_dict[0]
    if 0.5 in resort_dict.keys():
        del resort_dict[0.5]
    resort_dict = {key: resort_dict[key] for key in sorted(resort_dict.keys())}
    degree_vec_resort = list(resort_dict.keys())
    ave_deviation_resort = [np.mean(resort_dict[key_d]) for key_d in degree_vec_resort]
    std_deviation_resort = [np.std(resort_dict[key_d]) for key_d in degree_vec_resort]

    return degree_vec_resort, ave_deviation_resort, std_deviation_resort, real_ave_degree_vec, ave_deviation_vec, ave_deviation_dic


def plot_local_optimum_with_realED():
    # the x-axis is the real degree

    Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    betavec = [2.1, 4, 8, 16, 32, 64, 128]
    exemptionlist = []
    for N in [100]:
        ave_deviation_vec = []
        real_ave_degree_vec = []
        for beta in [4]:
            for ED in kvec:
                if ED < N:
                    for ExternalSimutime in [0]:
                        try:
                            real_ave_degree_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\real_ave_degree_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                            real_ave_degree = np.loadtxt(real_ave_degree_name)
                            real_ave_degree_vec = real_ave_degree_vec + list(real_ave_degree)

                            nodepairs_for_eachgraph_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\nodepairs_for_eachgraph_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                            node_pairs_vec = np.loadtxt(nodepairs_for_eachgraph_vec_name, dtype=int)

                            ave_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                            ave_deviation_for_a_para_comb = np.loadtxt(ave_deviation_name)

                            a_index = 0
                            for nodepair_num_inonegraph in node_pairs_vec:
                                b_index = a_index + nodepair_num_inonegraph - 1
                                ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb[a_index:b_index]))
                                a_index = b_index + 1

                        except FileNotFoundError:
                            exemptionlist.append((N, ED, beta, ExternalSimutime))

    x = real_ave_degree_vec
    y = ave_deviation_vec
    nan_indices = [i for i, value in enumerate(y) if not np.isnan(value)]
    x = [x[a] for a in nan_indices]
    y = [y[a] for a in nan_indices]
    # 步骤1: 将具有相同x坐标的点分组
    grouped_data = defaultdict(list)
    for i in range(len(x)):
        grouped_data[x[i]].append(y[i])

    # 步骤2: 计算每组的y均值
    x_values = []
    y_means = []
    y_stds = []

    for x_value in sorted(grouped_data):
        y_mean = sum(grouped_data[x_value]) / len(grouped_data[x_value])
        y_std = np.std(grouped_data[x_value])
        x_values.append(x_value)
        y_means.append(y_mean)
        y_stds.append(y_std)

    # 步骤3: 绘制x和y均值的图表
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_means, marker='o', linestyle='-', color='b', label='Mean of y')
    plt.xlabel('x')
    plt.ylabel('Mean of y')
    plt.title('Mean of y for each x')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_local_optimum_with_realED2(beta):
    """
    Compared with the previous one(***ED), this function resort all the degree and round all the degree
    the x-axis is the combined degree, 1.92 and 2.11 will be regarded as 2
    this is only for network size <200
    :return:
    """
    real_ave_degree_dict = {}
    ave_deviation_dict = {}
    std_deviation_dict = {}
    Nvec = [10, 100, 1000, 10000]
    # Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    # beta = 8
    for N in Nvec:
        if N < 200:
            degree_vec_resort, ave_deviation_resort, std_deviation_resort, _, _, _ = load_resort_data(N, beta)
            real_ave_degree_dict[N] = degree_vec_resort
            ave_deviation_dict[N] = ave_deviation_resort
            std_deviation_dict[N] = std_deviation_resort
        # elif N < 200:
        #     for beta in [4]:
        #         real_ave_degree_vec, ave_deviation_vec, std_deviation_vec = load_small_network_results(N,beta)
        #         real_ave_degree_dict[N] = real_ave_degree_vec
        #         ave_deviation_dict[N] = ave_deviation_vec
        #         std_deviation_dict[N] = std_deviation_vec
        elif N < 10000:
            for beta in [beta]:
                real_ave_degree_vec, ave_deviation_vec, std_deviation_vec = load_large_network_results(N, beta)
                real_ave_degree_dict[N] = real_ave_degree_vec
                ave_deviation_dict[N] = ave_deviation_vec
                std_deviation_dict[N] = std_deviation_vec
        else:
            x1 = np.arange(2, 6.1, 0.2)
            y1 = [0.00770638, 0.01068419, 0.0144987, 0.02114211, 0.03095507, 0.05568157,
                  0.08224888, 0.08943058, 0.08294274, 0.07499516, 0.07045126, 0.06704344,
                  0.06514699, 0.0639876, 0.06208567, 0.06061299, 0.05922611, 0.05922872,
                  0.05914097, 0.06084116, 0.06136418]
            x2 = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
            y2 = [0.05891465, 0.05533494, 0.05265912, 0.05150453, 0.05037399, 0.0513435,
                  0.05523626, 0.0590772, 0.06754352, 0.07553414, 0.08651017, 0.10125545,
                  0.11933796, 0.14024103, 0.1657074]
            z1 = [0.0066603,  0.00962583, 0.01291115, 0.01930005, 0.02704265, 0.04229857,
                  0.06186865, 0.06237527, 0.05695633, 0.04744045, 0.04617877, 0.04206581,
                  0.04079666, 0.04185382, 0.04055155, 0.03993016, 0.03869862, 0.03918321,
                  0.03763912, 0.04249137, 0.04233556]
            z2 = [0.04165133, 0.04115107, 0.03823124, 0.03346443, 0.03030473, 0.03030013,
                  0.03079327, 0.02692964, 0.03021144, 0.03654027, 0.03683304, 0.02206111,
                  0.01436064, 0.01017871, 0.00959986]
            x = np.concatenate((x1, x2))
            y = np.concatenate((y1, y2))
            z = np.concatenate((z1, z2))
            filter_index = [1, 4, 5, 7, 9, 12, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                            30, 31, 32, 33, 34, 35]
            x = [x[a] for a in filter_index]
            y = [y[a] for a in filter_index]
            print(x)
            print(y)
            print(z)
            z = [z[a] for a in filter_index]

            real_ave_degree_dict[N] = x
            ave_deviation_dict[N] = y
            std_deviation_dict[N] = z

    lengend = [r"$N=10$", r"$N=10^2$", r"$N=10^3$", r"$N=10^4$"]
    fig, ax = plt.subplots(figsize=(9, 6))

    # colors = [[0.3059, 0.4745, 0.6549],
    #           [0.9490, 0.5569, 0.1686],
    #           [0.8824, 0.3412, 0.3490],
    #           [0.4627, 0.7176, 0.6980],
    #           [0.9294, 0.7882, 0.2824],
    #           [0.6902, 0.4784, 0.6314],
    #           [1.0000, 0.6157, 0.6549],
    #           [0.6118, 0.4588, 0.3725],
    #           [0.7294, 0.6902, 0.6745]]
    # colors = [[0, 0.4470, 0.7410],
    #           [0.8500, 0.3250, 0.0980],
    #           [0.9290, 0.6940, 0.1250],
    #           [0.4940, 0.1840, 0.5560],
    #           [0.4660, 0.6740, 0.1880]]
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    # colorvec2 = ['#9FA9C9', '#D36A6A']
    cuttail = [5, 34, 23, 22]
    # peakcut = [9,5,5,5,5]
    for N_index in range(len(Nvec)):
        N = Nvec[N_index]
        if N == 100:
            x = real_ave_degree_dict[N]
            print(len(x))
            x = x[0:cuttail[N_index]]
            print(x)
            y = ave_deviation_dict[N]
            y = y[0:cuttail[N_index]]
            error = std_deviation_dict[N]
            error = error[0:cuttail[N_index]]

            filter_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 22, 24, 26, 28,
                            30]
            x = [x[a] for a in filter_index]
            y = [y[a] for a in filter_index]
            print(x)
            print(y)
            error = [error[a] for a in filter_index]
        elif N == 1000:
            # x = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
            x = [2, 3] + list(range(4, 16)) + [20, 28, 40, 58, 83, 118, 169, 241, 344, 490, 700, 999]
            print(len(x))
            x = x[0:cuttail[N_index]]
            y = ave_deviation_dict[N]
            y = y[0:cuttail[N_index]]
            print(x)
            print(y)
            error = std_deviation_dict[N]
            error = error[0:cuttail[N_index]]
        elif N == 10000:
            x = real_ave_degree_dict[N]
            print(len(x))
            x = x[0:cuttail[N_index]]
            y = ave_deviation_dict[N]
            y = y[0:cuttail[N_index]]
            print(x)
            print(y)
            error = std_deviation_dict[N]
            error = error[0:cuttail[N_index]]
        else:
            x = real_ave_degree_dict[N]
            print(len(x))
            x = x[1:cuttail[N_index]]
            y = ave_deviation_dict[N]
            y = y[1:cuttail[N_index]]
            error = std_deviation_dict[N]
            error = error[1:cuttail[N_index]]
        plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=16,
                     label=lengend[N_index], color=colors[N_index])



    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # plt.ylim(0, 0.30)
    # plt.yticks([0, 0.1, 0.2, 0.3])

    plt.xscale('log')
    plt.xlabel('Expected degree, E[D]', fontsize=26)
    plt.ylabel('Average deviation', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=26, loc=(0.5, 0.5))
    plt.tick_params(axis='both', which="both", length=6, width=1)
    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\LocalOptimumdiffNBeta{betan}.png".format(
        betan=beta)
    # plt.savefig(picname, format='png', bbox_inches='tight', dpi=600,transparent=True)
    plt.show()
    plt.close()




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """
    # STEP 4 plot local optimum: deviation versus expected degree
    """
    # kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    # print(len(kvec))

    plot_local_optimum_with_realED2(4)
    # plot_local_optimum_with_realED2(8)






