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

    kvec = [2, 3, 3.5] + list(range(4, 16)) + [20, 28, 40, 58, 83, 118, 169, 241, 344, 490, 700, 999]
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
            x = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
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
    plt.savefig(picname, format='png', bbox_inches='tight', dpi=600,transparent=True)
    plt.show()
    plt.close()


def analyse_local_optimum_with_diffED_tail():
    """
    The results are based on the plot_local_optimum_with_realED2()
    We checked the data 3 points:
    1. whether the slope of the tail follows the <k>^(1/2)
    :return:
    """
    kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    Nvec = [100, 10000]
    fig, ax = plt.subplots(figsize=(9, 6))
    beta = 8
    colors = [[0, 0.4470, 0.7410],
              [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250],
              [0.4940, 0.1840, 0.5560],
              [0.4660, 0.6740, 0.1880]]
    x_dic = {100: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 21, 23, 25, 27, 29, 31],
             10000: kvec[0:23]}
    y_dic = {
        100: [0.04486272955903646, 0.08271803920667539, 0.10217807919903567, 0.0973051691703707, 0.0906554276818128,
              0.08694971128688211, 0.08445621830353241, 0.08439564522559129, 0.08365839172483538, 0.08651702815304013,
              0.08939024678409463, 0.08840592225439187, 0.09229530706050895, 0.09348592598817931, 0.09774683807580578,
              0.10072145944111077, 0.10357256402624641, 0.10533681658273401, 0.11403048765531756, 0.11433702505430877,
              0.11844978424412508, 0.12184460683340481],
        10000: [0.00288125, 0.00538421, 0.01645766, 0.06271673, 0.05693332, 0.03579269,
                0.03517003, 0.02955597, 0.02756434, 0.02833479, 0.02518618, 0.02663983,
                0.02803844, 0.02847542, 0.02949756, 0.0303218, 0.0341523, 0.03737956,
                0.03192923, 0.03703796, 0.03490155, 0.0366545, 0.03711506]}
    legend = [r"$N=10^2$", r"$N=10^4$"]
    for N_index in range(2):
        N = Nvec[N_index]
        x = x_dic[N]
        y = y_dic[N]
        plt.plot(x, y, linestyle="-", linewidth=3, marker='o', markersize=16,
                 label=legend[N_index], color=colors[N_index])

    # plt.ylim(0, 0.30)
    # plt.yticks([0, 0.1, 0.2, 0.3])
    x1 = x_dic[100]
    listslice_index = x1.index(13)
    x1 = x1[listslice_index:]
    y1 = y_dic[100]
    y1 = y1[listslice_index:]
    params, covariance = curve_fit(power_law, x1, y1)

    # 获取拟合的参数
    a_fit, k_fit = params
    print(f"拟合结果: a = {a_fit}, k = {k_fit}")

    # 绘制原始数据和拟合曲线

    plt.plot(x1, power_law(x1, *params), linewidth=5, label=f'fit curve: $y={a_fit:.4f}x^{{{k_fit:.2f}}}$', color='red')
    # plt.plot(x1, [xv**(1/2) for xv in x1], linewidth=5, label=f'fit curve: $y={a_fit:.4f}x^{{{k_fit:.2f}}}$', color='red')

    x2 = x_dic[10000]
    listslice_index2 = x2.index(15.3092)
    x2 = x2[listslice_index2:]
    y2 = y_dic[10000]
    y2 = y2[listslice_index2:]
    params2, covariance2 = curve_fit(power_law, x2, y2)

    # 获取拟合的参数
    a_fit2, k_fit2 = params2
    print(f"拟合结果: a = {a_fit2}, k = {k_fit2}")

    # 绘制原始数据和拟合曲线

    plt.plot(x2, power_law(x2, *params2), linewidth=5, label=f'fit curve: $y={a_fit2:.4f}x^{{{k_fit2:.2f}}}$',
             color='green')
    # plt.plot(x1, [xv**(1/2) for xv in x1], linewidth=5, label=f'fit curve: $y={a_fit:.4f}x^{{{k_fit:.2f}}}$', color='red')

    plt.xscale('log')
    plt.yscale("log")
    plt.xlabel('Expected degree, E[D]', fontsize=26)
    plt.ylabel('Average deviation', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=20)
    plt.tick_params(axis='both', which="both", length=6, width=1)
    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\LocalOptimumBeta{betan}2.pdf".format(
        betan=beta)
    # plt.savefig(picname,format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()


def load_LCC_second_LCC_data(beta):
    """
    function for loading data for analysis first peak about Lcc AND second Lcc
    :param beta:
    :return:
    """
    xA = 0.25
    yA = 0.25
    xB = 0.75
    yB = 0.75
    input_avg_vec = np.arange(1, 6.1, 0.1)
    input_avg_vec2 = np.arange(6.2, 10.1, 0.2)
    input_avg_vec = list(input_avg_vec) + list(input_avg_vec2)
    input_avg_vec = np.arange(1, 6.1, 0.2)
    N = 10000
    filefolder_name_lcc = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\givendistance\\LCC\\"
    LCC_vec = []
    LCC_std_vec = []
    second_LCC_vec = []
    second_LCC_std_vec = []
    for ED in input_avg_vec:
        ED = round(ED, 1)
        LCC_oneED = []
        second_LCC_oneED = []
        for simutime in range(10):
            LCC_onesimu = []
            second_LCC_onesimu = []
            LCCname = filefolder_name_lcc + "LCC_2LCC_N{Nn}ED{EDn}beta{betan}xA{xA}yA{yA}xB{xB}yB{yB}simu{simu}.txt".format(
                Nn=N, EDn=ED, betan=beta, xA=xA, yA=yA, xB=xB, yB=yB, simu=simutime)
            try:
                with open(LCCname, "r") as file:
                    for line in file:
                        if line.startswith("#"):
                            continue
                        else:
                            data = line.strip().split("\t")
                            LCC_onesimu.append(int(data[0]))
                            second_LCC_onesimu.append(int(data[1]))
                LCC_oneED = LCC_oneED + LCC_onesimu
                second_LCC_oneED = second_LCC_oneED + second_LCC_onesimu
            except:
                print("Not data",ED,simutime)
        LCC_vec.append(np.mean(LCC_oneED))
        LCC_std_vec.append(np.std(LCC_oneED))
        second_LCC_vec.append(np.mean(second_LCC_oneED))
        second_LCC_std_vec.append(np.std(second_LCC_oneED))
    return LCC_vec,LCC_std_vec,second_LCC_vec,second_LCC_std_vec

def load_LCC_second_LCC_data_small_network(beta):
    """
    function for loading data for analysis first peak about Lcc AND second Lcc
    :param beta:
    :return:
    """
    xA = 0.25
    yA = 0.25
    xB = 0.75
    yB = 0.75
    input_avg_vec = np.arange(1, 4.2, 0.1)
    input_avg_vec2 = np.arange(4.1, 8.1, 0.1)
    input_avg_vec3 = np.arange(9, 30, 1)
    input_avg_vec = list(input_avg_vec) + list(input_avg_vec2)[1:] + list(input_avg_vec3)
    N = 100
    filefolder_name_lcc = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\givendistance\\LCC\\"
    LCC_vec = []
    LCC_std_vec = []
    second_LCC_vec = []
    second_LCC_std_vec = []
    for ED in input_avg_vec:
        LCC_oneED = []
        second_LCC_oneED = []
        for simutime in range(10):
            LCC_onesimu = []
            second_LCC_onesimu = []
            LCCname = filefolder_name_lcc + "LCC_2LCC_N{Nn}ED{EDn}beta{betan}xA{xA}yA{yA}xB{xB}yB{yB}simu{simu}.txt".format(
                Nn=N, EDn=ED, betan=beta, xA=xA, yA=yA, xB=xB, yB=yB, simu=simutime)
            with open(LCCname, "r") as file:
                for line in file:
                    if line.startswith("#"):
                        continue
                    else:
                        data = line.strip().split("\t")
                        LCC_onesimu.append(int(data[0]))
                        second_LCC_onesimu.append(int(data[1]))
            LCC_oneED = LCC_oneED + LCC_onesimu
            second_LCC_oneED = second_LCC_oneED + second_LCC_onesimu
        LCC_vec.append(np.mean(LCC_oneED))
        LCC_std_vec.append(np.std(LCC_oneED))
        second_LCC_vec.append(np.mean(second_LCC_oneED))
        second_LCC_std_vec.append(np.std(second_LCC_oneED))
    return LCC_vec,LCC_std_vec,second_LCC_vec,second_LCC_std_vec


def analyse_local_optimum_with_diffED_firstpeak():
    """
    The results are based on the plot_local_optimum_with_realED2()
    We checked the data 3 points:
    1. whether the peak happens where the LCC appear
    :return:
    """

    N = 10000
    beta = 4
    kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    fileflodername = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\"

    LCC = []
    for ED in kvec:
        filename = fileflodername + "network_N{Nn}ED{EDn}Beta{betan}.txt".format(
            Nn=N, EDn=ED, betan=beta)
        G = loadSRGGandaddnode(N, filename)
        components = list(nx.connected_components(G))
        largest_component = max(components, key=len)
        LCC_number = len(largest_component)
        LCC.append(LCC_number)
    print(LCC)
    LCC_vec, LCC_std_vec, second_LCC_vec, second_LCC_std_vec = load_LCC_second_LCC_data(beta)


    Nvec = [100, 10000]
    colors = [[0, 0.4470, 0.7410],
              [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250],
              [0.4940, 0.1840, 0.5560],
              [0.4660, 0.6740, 0.1880]]
    x_dic = {100: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 21, 23, 25, 27, 29, 31],
             10000: kvec[0:24]}

    y_dic = {
        100: [0.0941342400308666, 0.11562131645104153, 0.11103150326856681, 0.10797367304090624, 0.10492340822031251,
              0.10456671266895967, 0.10563094356702428, 0.10567521602153218, 0.1059839015128367, 0.11208596350920187,
              0.11000001392289922, 0.11554853189542773, 0.11599318476431886, 0.11874523084390971, 0.122524449779534,
              0.12601873173480152, 0.12697846547307626, 0.1350446922244735, 0.1379754851333052, 0.14169685440174795,
              0.1466336763433363, 0.1509530394683909],
        10000: [0.00848294, 0.06449598, 0.06819583, 0.05713195, 0.05341567, 0.06048276,
                0.05291581, 0.05562735, 0.0572445, 0.05908386, 0.05646044, 0.05822908,
                0.05466406, 0.05566516, 0.05342556, 0.0520834, 0.05376526, 0.05172981,
                0.05271027, 0.05189549, 0.05250138, 0.05120223, 0.05315753]}

    ave_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\1000realization\\ave_deviation_Beta{betan}.txt".format(
        betan=beta)
    y_dic[10000] =np.loadtxt(ave_deviation_Name)


    print(len(x_dic[10000]))
    legend = [r"$N=10^2$", r"$N=10^4$"]

    fig, ax1 = plt.subplots(figsize=(12, 8))
    for N_index in range(1,2):
        N = Nvec[N_index]
        x = x_dic[N]
        y = y_dic[N]
        ax1.plot(x, y, linestyle="-", linewidth=3, marker='o', markersize=16,
                 label=legend[N_index], color=colors[N_index])

    # x1 = x_dic[100]
    # listslice_index = x1.index(13)
    # x1 = x1[listslice_index:]
    # y1 = y_dic[100]
    # y1 = y1[listslice_index:]
    # params, covariance = curve_fit(power_law, x1, y1)
    # a_fit, k_fit = params
    # ax1.plot(x1, power_law(x1, *params), linewidth=5, label=f'fit curve: $y={a_fit:.4f}x^{{{k_fit:.2f}}}$', color='red')

    x2 = x_dic[10000]
    listslice_index2 = x2.index(35)
    x2 = x2[listslice_index2:]
    y2 = y_dic[10000]
    y2 = y2[listslice_index2:]
    params2, covariance2 = curve_fit(power_law, x2, y2)

    a_fit2, k_fit2 = params2
    ax1.plot(x2, power_law(x2, *params2), linewidth=5, label=f'fit curve: $y={a_fit2:.4f}x^{{{k_fit2:.2f}}}$',
             color='green')

    plt.xscale('log')
    ax1.set_yscale("log")
    plt.xlabel('input avg', fontsize=26)
    ax1.set_ylabel('average deviation', fontsize=26)
    ax1.legend(fontsize=20)
    ax1.tick_params(axis='x', labelsize=26)
    ax1.tick_params(axis='y', labelsize=26)

    # ax2 = ax1.twinx()
    # ax2.plot(kvec, LCC, linestyle="-", linewidth=3, marker='o', color="blue", label='LCC')
    # ax2.set_ylabel('Number of nodes', color='b', fontsize=26)
    # ax2.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    input_avg_vec = np.arange(1, 6.1, 0.1)
    input_avg_vec2 = np.arange(6.2, 10.1, 0.2)
    input_avg_vec = list(input_avg_vec)+ list(input_avg_vec2)
    # ax2.errorbar(input_avg_vec, LCC_vec, LCC_std_vec, linestyle="-", linewidth=3, marker='o', color="blue", label='LCC')
    # ax2.errorbar(input_avg_vec, LCC_vec, LCC_std_vec, linestyle="-", linewidth=3, marker='o', color="blue", label='LCC')
    #
    # ax2.errorbar(input_avg_vec, second_LCC_vec, second_LCC_std_vec, linestyle="-", linewidth=3, marker='o', color="green",
    #              label='Second LCC')

    ax2.plot(input_avg_vec, LCC_vec, linestyle="-", linewidth=3, marker='o', color="blue", label='LCC')

    ax2.plot(input_avg_vec, second_LCC_vec, linestyle="-", linewidth=3, marker='o',
                 color="purple",
                 label='Second LCC')
    ax2.set_ylabel('Number of nodes', color='b', fontsize=26)
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.legend(fontsize=20, loc = 'center right')
    ax2.tick_params(axis='x', labelsize=26)
    ax2.tick_params(axis='y', labelsize=26)

    # plt.title('Errorbar Curves with Minimum Points after Peak')


    plt.tick_params(axis='both', which="both", length=6, width=1)
    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\LocalOptimum_comp_with_LCC_Beta{betan}.pdf".format(
        betan=beta)
    plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()


def analyse_local_optimum_with_diffED_firstpeak_small_network():
    """
    The results are based on the plot_local_optimum_with_realED2()
    We checked the data 3 points:
    1. whether the peak happens where the LCC appear
    :return:
    """

    N = 10000
    beta = 4
    kvec = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    LCC_vec, LCC_std_vec, second_LCC_vec, second_LCC_std_vec = load_LCC_second_LCC_data_small_network(beta)

    Nvec = [100, 10000]
    colors = [[0, 0.4470, 0.7410],
              [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250],
              [0.4940, 0.1840, 0.5560],
              [0.4660, 0.6740, 0.1880]]
    x_dic = {100: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 21, 23, 25, 27, 29, 31]}

    y_dic = {
        100: [0.0941342400308666, 0.11562131645104153, 0.11103150326856681, 0.10797367304090624, 0.10492340822031251,
              0.10456671266895967, 0.10563094356702428, 0.10567521602153218, 0.1059839015128367, 0.11208596350920187,
              0.11000001392289922, 0.11554853189542773, 0.11599318476431886, 0.11874523084390971, 0.122524449779534,
              0.12601873173480152, 0.12697846547307626, 0.1350446922244735, 0.1379754851333052, 0.14169685440174795,
              0.1466336763433363, 0.1509530394683909]}

    legend = [r"$N=10^2$", r"$N=10^4$"]

    fig, ax1 = plt.subplots(figsize=(12, 8))
    for N_index in range(1):
        N = Nvec[N_index]
        x = x_dic[N]
        y = y_dic[N]
        ax1.plot(x, y, linestyle="-", linewidth=3, marker='o', markersize=16,
                 label=legend[N_index], color=colors[N_index])

    x1 = x_dic[100]
    listslice_index = x1.index(12)
    x1 = x1[listslice_index:]
    y1 = y_dic[100]
    y1 = y1[listslice_index:]
    params, covariance = curve_fit(power_law, x1, y1)
    a_fit, k_fit = params
    ax1.plot(x1, power_law(x1, *params), linewidth=5, label=f'fit curve: $y={a_fit:.4f}x^{{{k_fit:.2f}}}$', color='red')

    plt.xscale('log')
    ax1.set_yscale("log")
    plt.xlabel('input avg', fontsize=26)
    ax1.set_ylabel('average deviation', fontsize=26)
    ax1.legend(fontsize=20)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)

    # ax2 = ax1.twinx()
    # ax2.plot(kvec, LCC, linestyle="-", linewidth=3, marker='o', color="blue", label='LCC')
    # ax2.set_ylabel('Number of nodes', color='b', fontsize=26)
    # ax2.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    input_avg_vec = np.arange(1, 4.2, 0.1)
    input_avg_vec2 = np.arange(4.1, 8.1, 0.1)
    input_avg_vec3 = np.arange(9, 30, 1)
    input_avg_vec = list(input_avg_vec) + list(input_avg_vec2)[1:]+list(input_avg_vec3)
    # ax2.errorbar(input_avg_vec, LCC_vec, LCC_std_vec, linestyle="-", linewidth=3, marker='o', color="blue", label='LCC')
    # ax2.errorbar(input_avg_vec, LCC_vec, LCC_std_vec, linestyle="-", linewidth=3, marker='o', color="blue", label='LCC')
    #
    # ax2.errorbar(input_avg_vec, second_LCC_vec, second_LCC_std_vec, linestyle="-", linewidth=3, marker='o', color="green",
    #              label='Second LCC')

    ax2.plot(input_avg_vec, LCC_vec, linestyle="-", linewidth=3, marker='o', color="blue", label='LCC')

    ax2.plot(input_avg_vec, second_LCC_vec, linestyle="-", linewidth=3, marker='o',
                 color="purple",
                 label='Second LCC')
    ax2.set_ylabel('Number of nodes', color='b', fontsize=26)
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.legend(fontsize=20, loc = 'center right')
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)

    plt.tick_params(axis='both', which="both", length=6, width=1)
    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\LocalOptimum_comp_with_LCC_Beta{betan}100ndoe.pdf".format(
        betan=beta)
    plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()

def power_law(x, a, k):
    return a * x ** k


def plot_local_optimum_with_realED_diffCG():
    """
    the x-axis is the expected degree, the y-axis is the average deviation, different line is different c_G
    inset is the min(average deviation) vs c_G
    the x-axis is the combined degree, 1.92 and 2.11 will be regarded as 2
    :return:
    """
    real_ave_degree_dict = {}
    ave_deviation_dict = {}
    std_deviation_dict = {}
    betavec = [2.1, 4, 8, 16, 128]
    # betavec = [2.1, 2.2, 2.4, 2.5, 2.6, 2.8, 3, 3.25, 3.5, 3.75, 4, 5, 6, 7, 8, 10, 12, 16, 32, 64, 128]
    # betavec = [2.1, 2.2, 2.4, 2.5, 2.6, 2.8, 3, 3.25, 3.5, 3.75, 4, 5, 6, 8, 16, 32, 64, 128]

    N = 10000
    count = 0
    for beta in betavec:
        real_ave_degree_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\real_ave_degree_Beta{betan}.txt".format(
            betan=beta)
        real_ave_degree_vec = np.loadtxt(real_ave_degree_Name)
        ave_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\ave_deviation_Beta{betan}.txt".format(
            betan=beta)
        ave_deviation_vec = np.loadtxt(ave_deviation_Name)
        std_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\std_deviation_Beta{betan}.txt".format(
            betan=beta)
        std_deviation_vec = np.loadtxt(std_deviation_Name)
        # real_ave_degree_vec, ave_deviation_vec, std_deviation_vec = load_10000nodenetwork_results(beta)
        real_ave_degree_dict[count] = real_ave_degree_vec
        ave_deviation_dict[count] = ave_deviation_vec
        std_deviation_dict[count] = std_deviation_vec
        count = count + 1

    lengend = [r"$C_G=0.03$", r"$C_G=0.31$", r"$C_G=0.51$", r"$C_G=0.56$", r"$C_G=0.59$"]
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = [[0, 0.4470, 0.7410],
              [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250],
              [0.4940, 0.1840, 0.5560],
              [0.4660, 0.6740, 0.1880]]
    # cuttail = [5,34,23,23]
    peakcut = [5, 6, 6, 6, 6]
    c_g_vec = [0.03, 0.31, 0.51, 0.56, 0.59]
    LO_ED = []
    LO_Dev = []
    for count in range(len(betavec)):
        beta = betavec[count]
        x = real_ave_degree_dict[count]
        # print(len(x))
        # x = x[0:cuttail[N_index]]
        y = ave_deviation_dict[count]
        # y = y[0:cuttail[N_index]]
        error = std_deviation_dict[count]
        # error = error[0:cuttail[N_index]]
        plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=16,
                     label=lengend[count], color=colors[count])

        # # 找到峰值后最低点的坐标
        peak_index = np.argmax(y[0:peakcut[count]])
        post_peak_y = y[peak_index:]
        post_peak_min_index = peak_index + np.argmin(post_peak_y)
        post_peak_min_x = x[post_peak_min_index]
        LO_ED.append(post_peak_min_x)
        post_peak_min_y = y[post_peak_min_index]
        LO_Dev.append(post_peak_min_y)

        # 标出最低点
        # plt.plot(post_peak_min_x, post_peak_min_y, 'o', color=colors[count], markersize=25, markerfacecolor="none")
    # inset pic

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # plt.ylim(0,0.30)
    # plt.yticks([0,0.1,0.2,0.3])

    plt.xscale('log')
    plt.xlabel('Expected degree, E[D]', fontsize=26)
    plt.ylabel('Average deviation', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=20, loc="upper left")
    plt.tick_params(axis='both', which="both", length=6, width=1)

    # inset_ax = inset_axes(ax, width="40%", height="30%")
    inset_ax = fig.add_axes([0.58, 0.55, 0.3, 0.3])
    inset_ax.plot(c_g_vec, LO_Dev, linewidth=3, marker='o', markersize=10, color="b")
    inset_ax.set_xlabel("$C_G$", fontsize=18)
    inset_ax.set_ylabel(r"Local $\min(\overline{d}(q,\gamma(i,j)))$", fontsize=18)
    inset_ax.tick_params(axis='y', labelsize=18)
    inset_ax.tick_params(axis='x', labelsize=18)
    inset_ax.set_xlim(0, 0.6)
    inset_ax.text(0.8, 0.85, r'$N = 10^4$', transform=inset_ax.transAxes,
                  fontsize=20, verticalalignment='center', horizontalalignment='center')
    # inset_ax2 = inset_ax.twinx()
    # inset_ax2.plot(c_g_vec, LO_ED, 'r-', label='log(x+1)')
    # inset_ax2.set_ylabel(r"Local minimum $E[D]$", color='r',fontsize=18)
    # inset_ax2.tick_params(axis='y', labelcolor='r')

    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\LocalOptimumdiffc_G.pdf".format(
        betan=beta)
    plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()


def load_10000nodenetwork_results_clean(beta):
    """
    the function is the same as load_10000nodenetwork_results but the results are based on clean data
    :param beta:
    :return:
    """
    # Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    kvec = list(range(2, 20)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    # betavec = [2.1, 4, 8, 16, 32, 64, 128]

    exemptionlist = []
    for N in [10000]:
        ave_deviation_vec = []
        std_deviation_vec = []
        real_ave_degree_vec = []
        for beta in [beta]:
            for ED in kvec:
                ave_deviation_for_a_para_comb = []
                # FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\network_N{Nn}ED{EDn}Beta{betan}.txt".format(
                #     Nn=N, EDn=ED, betan=beta)
                # G = loadSRGGandaddnode(N, FileNetworkName)
                # real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
                # # print("real ED:", real_avg)
                # real_ave_degree_vec.append(real_avg)

                for ExternalSimutime in range(10):
                    try:
                        deviation_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\cleanresult\\ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        ave_deviation_for_a_para_comb_10times = np.loadtxt(deviation_vec_name)
                        ave_deviation_for_a_para_comb.extend(ave_deviation_for_a_para_comb_10times)
                    except FileNotFoundError:
                        exemptionlist.append((N, ED, beta, ExternalSimutime))

                ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
                std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))
    print(exemptionlist)
    # real_ave_degree_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\real_ave_degree_Beta{betan}.txt".format(betan=beta)
    # np.savetxt(real_ave_degree_Name, real_ave_degree_vec)
    ave_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\cleanresult\\ave_deviation_Beta{betan}.txt".format(
        betan=beta)
    np.savetxt(ave_deviation_Name, ave_deviation_vec)
    std_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\cleanresult\\std_deviation_Beta{betan}.txt".format(
        betan=beta)
    np.savetxt(std_deviation_Name, std_deviation_vec)
    return ave_deviation_vec, std_deviation_vec, exemptionlist


def plot_local_optimum_with_realED_diffCG_clean():
    """
    the x-axis is the expected degree, the y-axis is the average deviation, different line is different c_G
    inset is the min(average deviation) vs c_G
    the x-axis is real (approximate) degree
    when use this function, use before
    :return:
    """
    real_ave_degree_dict = {}
    ave_deviation_dict = {}
    std_deviation_dict = {}
    C_G_index = [0.1, 0.2, 0.3, 0.4, 0.5]
    betavec = [2.55, 3.2, 3.99, 5.15, 7.99, 300]
    # betavec = [2.55, 3.2]
    # betavec = [2.1, 2.2, 2.4, 2.5, 2.6, 2.8, 3, 3.25, 3.5, 3.75, 4, 5, 6, 7, 8, 10, 12, 16, 32, 64, 128]
    # betavec = [2.1, 2.2, 2.4, 2.5, 2.6, 2.8, 3, 3.25, 3.5, 3.75, 4, 5, 6, 8, 16, 32, 64, 128]

    N = 10000
    count = 0
    for beta in betavec:
        ave_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\cleanresult\\ave_deviation_Beta{betan}.txt".format(
            betan=beta)
        ave_deviation_vec = np.loadtxt(ave_deviation_Name)
        std_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\cleanresult\\std_deviation_Beta{betan}.txt".format(
            betan=beta)
        std_deviation_vec = np.loadtxt(std_deviation_Name)
        # real_ave_degree_vec, ave_deviation_vec, std_deviation_vec = load_10000nodenetwork_results(beta)

        ave_deviation_dict[count] = ave_deviation_vec
        std_deviation_dict[count] = std_deviation_vec
        count = count + 1

    lengend = [r"$C_G=0.1$", r"$C_G=0.2$", r"$C_G=0.3$", r"$C_G=0.4$", r"$C_G=0.5$", r"$C_G=0.6$"]
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = [[0, 0.4470, 0.7410],
              [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250],
              [0.4940, 0.1840, 0.5560],
              [0.4660, 0.6740, 0.1880],
              [0.3010, 0.7450, 0.9330]]
    # cuttail = [5,34,23,23]
    peakcut = [5, 6, 6, 6, 6, 6]
    c_g_vec = [0.03, 0.31, 0.51, 0.56, 0.59]
    LO_ED = []
    LO_Dev = []
    for count in range(len(betavec)):
        beta = betavec[count]
        x = list(range(2, 20)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
        # print(len(x))
        # x = x[0:cuttail[N_index]]
        y = ave_deviation_dict[count]
        # y = y[0:cuttail[N_index]]
        error = std_deviation_dict[count]
        # error = error[0:cuttail[N_index]]
        plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=16,
                     label=lengend[count], color=colors[count])

        # # 找到峰值后最低点的坐标
        peak_index = np.argmax(y[0:peakcut[count]])
        post_peak_y = y[peak_index:]
        post_peak_min_index = peak_index + np.argmin(post_peak_y)
        post_peak_min_x = x[post_peak_min_index]
        LO_ED.append(post_peak_min_x)
        post_peak_min_y = y[post_peak_min_index]
        LO_Dev.append(post_peak_min_y)

        # 标出最低点
        # plt.plot(post_peak_min_x, post_peak_min_y, 'o', color=colors[count], markersize=25, markerfacecolor="none")
    # inset pic

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # plt.ylim(0,0.30)
    # plt.yticks([0,0.1,0.2,0.3])

    plt.xscale('log')
    plt.xlabel('Expected degree, E[D]', fontsize=26)
    plt.ylabel('Average deviation', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=20, loc="upper left")
    plt.tick_params(axis='both', which="both", length=6, width=1)

    # inset_ax = inset_axes(ax, width="40%", height="30%")
    # inset_ax = fig.add_axes([0.58, 0.55, 0.3, 0.3])
    # inset_ax.plot(c_g_vec, LO_Dev,linewidth=3, marker='o', markersize=10, color = "b")
    # inset_ax.set_xlabel("$C_G$",fontsize=18)
    # inset_ax.set_ylabel(r"Local $\min(\overline{d}(q,\gamma(i,j)))$",fontsize=18)
    # inset_ax.tick_params(axis='y', labelsize=18)
    # inset_ax.tick_params(axis='x', labelsize=18)
    # inset_ax.set_xlim(0, 0.6)
    # inset_ax.text(0.8, 0.85, r'$N = 10^4$', transform=inset_ax.transAxes,
    #               fontsize=20, verticalalignment='center', horizontalalignment='center')
    # inset_ax2 = inset_ax.twinx()
    # inset_ax2.plot(c_g_vec, LO_ED, 'r-', label='log(x+1)')
    # inset_ax2.set_ylabel(r"Local minimum $E[D]$", color='r',fontsize=18)
    # inset_ax2.tick_params(axis='y', labelcolor='r')

    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\LocalOptimumdiffc_G_clean.pdf".format(
        betan=beta)
    # plt.savefig(picname,format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()


def plot_local_optimum_function(beta):
    """
        THIS ONE and the later one plot HOW the local optimum ED changed with different CC
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
            for beta in [beta]:
                real_ave_degree_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\real_ave_degree_Beta{betan}.txt".format(
                    betan=beta)
                real_ave_degree_vec = np.loadtxt(real_ave_degree_Name)
                ave_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\ave_deviation_Beta{betan}.txt".format(
                    betan=beta)
                ave_deviation_vec = np.loadtxt(ave_deviation_Name)
                std_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\std_deviation_Beta{betan}.txt".format(
                    betan=beta)
                std_deviation_vec = np.loadtxt(std_deviation_Name)
                # real_ave_degree_vec, ave_deviation_vec, std_deviation_vec = load_10000nodenetwork_results(beta)
                real_ave_degree_dict[N] = real_ave_degree_vec
                ave_deviation_dict[N] = ave_deviation_vec
                std_deviation_dict[N] = std_deviation_vec

    local_optimum = []

    lengend = [r"$N=10$", r"$N=10^2$", r"$N=10^3$", r"$N=10^4$"]
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = [[0, 0.4470, 0.7410],
              [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250],
              [0.4940, 0.1840, 0.5560],
              [0.4660, 0.6740, 0.1880]]
    cuttail = [5, 34, 23, 23]
    peakcut = [1, 5, 5, 5, 5]
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

            # filter_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 22, 24, 26, 28,
            #                 30]
            # x = [x[a] for a in filter_index]
            # y = [y[a] for a in filter_index]
            # error = [error[a] for a in filter_index]
        elif N > 100:
            x = real_ave_degree_dict[N]
            print(len(x))
            x = x[0:cuttail[N_index]]
            y = ave_deviation_dict[N]
            y = y[0:cuttail[N_index]]
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

        # 找到峰值后最低点的坐标
        peak_index = np.argmax(y[0:peakcut[N_index]])
        post_peak_y = y[peak_index:]
        post_peak_min_index = peak_index + np.argmin(post_peak_y)
        post_peak_min_x = x[post_peak_min_index]
        local_optimum.append(post_peak_min_x)
        post_peak_min_y = y[post_peak_min_index]

        # 标出最低点
        plt.plot(post_peak_min_x, post_peak_min_y, 'o', color=colors[N_index], markersize=30, markerfacecolor="none")

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.ylim(0, 0.30)
    plt.yticks([0, 0.1, 0.2, 0.3])

    plt.xscale('log')
    plt.xlabel('Clustering coefficient, E[D]', fontsize=26)
    plt.ylabel('Average deviation', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=20, loc=(0.72, 0.58))
    plt.tick_params(axis='both', which="both", length=6, width=1)
    # picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\LocalOptimumFunction.pdf".format(
    #     betan=beta)
    # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    # plt.close()
    return local_optimum


def plot_local_optimum_function2():
    cc = [0.05, 0.3, 0.5, 0.6]
    beta = [2.1,4,8,300]

    A = [5, 5, 5]
    B = [9, 7, 10, 11]
    C = [12.05, 8.398, 10.332, 18.728]
    D = [8.9272, 18.893, 9.2908, 15.3964]

    lengend = [r"$N=10$", r"$N=10^2$", r"$N=10^3$", r"$N=10^4$"]
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = [[0, 0.4470, 0.7410],
              [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250],
              [0.4940, 0.1840, 0.5560],
              [0.4660, 0.6740, 0.1880]]

    plt.plot(beta[1:4], A, linewidth=3, marker='o', markersize=16, label=r"$N=10$")
    plt.plot(beta, B, linewidth=3, marker='o', markersize=16, label=r"$N=10^2$")
    plt.plot(beta, C, linewidth=3, marker='o', markersize=16, label=r"$N=10^3$")
    plt.plot(beta, D, linewidth=3, marker='o', markersize=16, label=r"$N=10^4$")

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # plt.ylim(0, 0.30)
    # plt.yticks([0, 0.1, 0.2, 0.3])

    plt.xscale('log')
    plt.xlabel(r'$beta$', fontsize=26)
    plt.ylabel('local optimum E[D]', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=20)
    plt.tick_params(axis='both', which="both", length=6, width=1)
    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\LocalOptimumFunctionbeta.pdf"
    plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()



def loadhopcount(beta):
    """
    retrun a list include the mean hopcount of the SP for different ED with specific beta
    :param beta:
    :return:
    """
    # Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    # betavec = [2.1, 4, 8, 16, 32, 64, 128]

    exemptionlist = []
    for N in [10000]:
        hopcount_vec = []
        std_hocount_vec = []
        for beta in [beta]:
            for ED in kvec:
                ave_deviation_for_a_para_comb = []
                for ExternalSimutime in range(100):
                    try:
                        hopcount_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\1000realization\\hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
                            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        ave_deviation_for_a_para_comb_10times = np.loadtxt(hopcount_vec_name)
                        ave_deviation_for_a_para_comb.extend(ave_deviation_for_a_para_comb_10times)
                    except FileNotFoundError:
                        exemptionlist.append((N, ED, beta, ExternalSimutime))

                hopcount_vec.append(np.mean(ave_deviation_for_a_para_comb))
                std_hocount_vec.append(np.std(ave_deviation_for_a_para_comb))
    print(exemptionlist)
    ave_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\1000realization\\ave_hopcount_Beta{betan}.txt".format(
        betan=beta)
    np.savetxt(ave_deviation_Name, hopcount_vec)
    std_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\1000realization\\std_hopcount_Beta{betan}.txt".format(
        betan=beta)
    np.savetxt(std_deviation_Name, std_hocount_vec)
    return hopcount_vec, std_hocount_vec, exemptionlist


def plot_local_optimum_with_hopcount_dataloaded(beta):
    """
    We plot for the average hopcount, sp node num and the local optimum, with the results obtained by loadhopcount(beta)
    :return:
    """
    real_ave_degree_dict = {}
    ave_deviation_dict = {}
    std_deviation_dict = {}
    Nvec = [10000]
    # Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
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
            for beta in [beta]:
                ave_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\1000realization\\ave_deviation_Beta{betan}.txt".format(
                    betan=beta)
                ave_deviation_vec = np.loadtxt(ave_deviation_Name)
                std_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\1000realization\\std_deviation_Beta{betan}.txt".format(
                    betan=beta)
                std_deviation_vec = np.loadtxt(std_deviation_Name)
                # real_ave_degree_vec, ave_deviation_vec, std_deviation_vec = load_10000nodenetwork_results(beta)

                ave_deviation_dict[N] = ave_deviation_vec
                std_deviation_dict[N] = std_deviation_vec

    lengend = ["ave deviation", "hopcount"]
    fig, ax1 = plt.subplots(figsize=(9, 6))
    # colors = [[0.3059, 0.4745, 0.6549],
    #           [0.9490, 0.5569, 0.1686],
    #           [0.8824, 0.3412, 0.3490],
    #           [0.4627, 0.7176, 0.6980],
    #           [0.9294, 0.7882, 0.2824],
    #           [0.6902, 0.4784, 0.6314],
    #           [1.0000, 0.6157, 0.6549],
    #           [0.6118, 0.4588, 0.3725],
    #           [0.7294, 0.6902, 0.6745]]
    colors = [[0, 0.4470, 0.7410],
              [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250],
              [0.4940, 0.1840, 0.5560],
              [0.4660, 0.6740, 0.1880]]
    cuttail = [24]
    peakcut = [5]
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
            error = [error[a] for a in filter_index]
        elif N > 100:
            x = kvec
            print(len(x))
            x = x[0:cuttail[N_index]]
            y = ave_deviation_dict[N]
            y = y[0:cuttail[N_index]]
            error = std_deviation_dict[N]
            error = error[0:cuttail[N_index]]
        else:
            print(len(x))
            x = x[1:cuttail[N_index]]
            y = ave_deviation_dict[N]
            y = y[1:cuttail[N_index]]
            error = std_deviation_dict[N]
            error = error[1:cuttail[N_index]]
        line1 = ax1.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o',
                             markersize=16, label=lengend[0], color=colors[N_index])

        # # 找到峰值后最低点的坐标
        # peak_index = np.argmax(y[0:peakcut[N_index]])
        # post_peak_y = y[peak_index:]
        # post_peak_min_index = peak_index + np.argmin(post_peak_y)
        # post_peak_min_x = x[post_peak_min_index]
        # post_peak_min_y = y[post_peak_min_index]
        #
        # # 标出最低点
        # ax1.plot(post_peak_min_x, post_peak_min_y, 'o', color=colors[N_index], markersize=30, markerfacecolor="none")

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    plt.yticks([0, 0.1, 0.2, 0.3])
    ax1.tick_params(axis='y', labelcolor=colors[N_index], labelsize=26)
    ax1.set_ylabel('Average deviation', fontsize=26)
    ax1.set_xlabel(r'Input avg', fontsize=26)
    ax1.tick_params(axis='x', labelsize=26)

    # hopcount
    hop_ave_vec = []
    hop_std_vec = []
    kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    for beta in [beta]:
        ave_hop_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\1000realization\\ave_hopcount_Beta{betan}.txt".format(
            betan=beta)

        std_hop_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\1000realization\\std_hopcount_Beta{betan}.txt".format(
            betan=beta)
        hop_ave_vec = np.loadtxt(ave_hop_Name)
        hop_std_vec = np.loadtxt(std_hop_Name)
    y2 = hop_ave_vec
    y2 = y2[0:cuttail[N_index]]
    error2 = hop_std_vec
    error2 = error2[0:cuttail[N_index]]

    ax2 = ax1.twinx()
    line2 = ax2.errorbar(x, y2, yerr=error2, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o',
                         markersize=16,
                         label=lengend[1], color=colors[1])

    ax2.set_ylabel('Hopcount', color=colors[1], fontsize=26)
    ax2.tick_params(axis='y', labelcolor=colors[1], labelsize=26)

    plt.xscale('log')
    plt.xlabel('Expected degree, E[D]', fontsize=26)

    # plt.xticks(fontsize=26)
    # plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    # ax1.legend(loc='upper left',fontsize=20)
    # ax2.legend(loc='upper right',fontsize=20)
    # lines = [line1, line2]
    # plt.legend(lines, lengend, loc='upper left',fontsize=20)
    # plt.tick_params(axis='both', which="both",length=6, width=1)
    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\LocalOptimumwithhop10000nodeBeta{betan}.pdf".format(
        betan=beta)
    plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()



def plot_local_optimum_with_hopcount(beta):
    """
    We plot for the average hopcount, sp node num and the local optimum
    :return:
    """
    real_ave_degree_dict = {}
    ave_deviation_dict = {}
    std_deviation_dict = {}
    Nvec = [10000]
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
            for beta in [beta]:
                real_ave_degree_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\1000realization\\real_ave_degree_Beta{betan}.txt".format(
                    betan=beta)
                real_ave_degree_vec = np.loadtxt(real_ave_degree_Name)
                ave_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\1000realization\\ave_deviation_Beta{betan}.txt".format(
                    betan=beta)
                ave_deviation_vec = np.loadtxt(ave_deviation_Name)
                std_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\1000realization\\std_deviation_Beta{betan}.txt".format(
                    betan=beta)
                std_deviation_vec = np.loadtxt(std_deviation_Name)
                # real_ave_degree_vec, ave_deviation_vec, std_deviation_vec = load_10000nodenetwork_results(beta)
                real_ave_degree_dict[N] = real_ave_degree_vec
                ave_deviation_dict[N] = ave_deviation_vec
                std_deviation_dict[N] = std_deviation_vec

    lengend = ["ave deviation", "hopcount"]
    fig, ax1 = plt.subplots(figsize=(9, 6))
    # colors = [[0.3059, 0.4745, 0.6549],
    #           [0.9490, 0.5569, 0.1686],
    #           [0.8824, 0.3412, 0.3490],
    #           [0.4627, 0.7176, 0.6980],
    #           [0.9294, 0.7882, 0.2824],
    #           [0.6902, 0.4784, 0.6314],
    #           [1.0000, 0.6157, 0.6549],
    #           [0.6118, 0.4588, 0.3725],
    #           [0.7294, 0.6902, 0.6745]]
    colors = [[0, 0.4470, 0.7410],
              [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250],
              [0.4940, 0.1840, 0.5560],
              [0.4660, 0.6740, 0.1880]]
    cuttail = [23]
    peakcut = [5]
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
            error = [error[a] for a in filter_index]
        elif N > 100:
            x = real_ave_degree_dict[N]
            print(len(x))
            x = x[0:cuttail[N_index]]
            y = ave_deviation_dict[N]
            y = y[0:cuttail[N_index]]
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
        line1 = ax1.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o',
                             markersize=16, label=lengend[0], color=colors[N_index])

        # 找到峰值后最低点的坐标
        peak_index = np.argmax(y[0:peakcut[N_index]])
        post_peak_y = y[peak_index:]
        post_peak_min_index = peak_index + np.argmin(post_peak_y)
        post_peak_min_x = x[post_peak_min_index]
        post_peak_min_y = y[post_peak_min_index]

        # 标出最低点
        ax1.plot(post_peak_min_x, post_peak_min_y, 'o', color=colors[N_index], markersize=30, markerfacecolor="none")

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    plt.yticks([0, 0.1, 0.2, 0.3])
    ax1.tick_params(axis='y', labelcolor=colors[N_index], labelsize=26)
    ax1.set_ylabel('Average deviation', fontsize=26)
    ax1.set_xlabel(r'Expected degree, $E[D]$', fontsize=26)
    ax1.tick_params(axis='x', labelsize=26)

    # hopcount
    hop_ave_vec = []
    hop_std_vec = []
    kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    for ED in kvec:
        hopcount_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\1000realization\\hopcount_sp_ED{EDn}Beta{betan}.txt".format(
            EDn=ED, betan=beta)
        hopcount_for_one_graph = np.loadtxt(hopcount_Name, dtype=int)
        hop_ave_vec.append(np.mean(hopcount_for_one_graph))
        hop_std_vec.append(np.std(hopcount_for_one_graph))
    y2 = hop_ave_vec
    y2 = y2[0:cuttail[N_index]]
    error2 = hop_std_vec
    error2 = error2[0:cuttail[N_index]]

    ax2 = ax1.twinx()
    line2 = ax2.errorbar(x, y2, yerr=error2, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o',
                         markersize=16,
                         label=lengend[1], color=colors[1])

    ax2.set_ylabel('Hopcount', color=colors[1], fontsize=26)
    ax2.tick_params(axis='y', labelcolor=colors[1], labelsize=26)

    plt.xscale('log')
    plt.xlabel('Expected degree, E[D]', fontsize=26)

    # plt.xticks(fontsize=26)
    # plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    # ax1.legend(loc='upper left',fontsize=20)
    # ax2.legend(loc='upper right',fontsize=20)
    # lines = [line1, line2]
    # plt.legend(lines, lengend, loc='upper left',fontsize=20)
    # plt.tick_params(axis='both', which="both",length=6, width=1)
    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\LocalOptimumwithhop10000nodeBeta{betan}.pdf".format(
        betan=beta)
    # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()


def plot_local_optimum_with_spnodenum(beta):
    """
    We plot for the average hopcount, sp node num and the local optimum
    :return:
    """
    real_ave_degree_dict = {}
    ave_deviation_dict = {}
    std_deviation_dict = {}
    Nvec = [10000]
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
            for beta in [beta]:
                real_ave_degree_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\real_ave_degree_Beta{betan}.txt".format(
                    betan=beta)
                real_ave_degree_vec = np.loadtxt(real_ave_degree_Name)
                ave_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\ave_deviation_Beta{betan}.txt".format(
                    betan=beta)
                ave_deviation_vec = np.loadtxt(ave_deviation_Name)
                std_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\std_deviation_Beta{betan}.txt".format(
                    betan=beta)
                std_deviation_vec = np.loadtxt(std_deviation_Name)
                # real_ave_degree_vec, ave_deviation_vec, std_deviation_vec = load_10000nodenetwork_results(beta)
                real_ave_degree_dict[N] = real_ave_degree_vec
                ave_deviation_dict[N] = ave_deviation_vec
                std_deviation_dict[N] = std_deviation_vec

    lengend = ["ave deviation", "hopcount"]
    fig, ax1 = plt.subplots(figsize=(9, 6))
    # colors = [[0.3059, 0.4745, 0.6549],
    #           [0.9490, 0.5569, 0.1686],
    #           [0.8824, 0.3412, 0.3490],
    #           [0.4627, 0.7176, 0.6980],
    #           [0.9294, 0.7882, 0.2824],
    #           [0.6902, 0.4784, 0.6314],
    #           [1.0000, 0.6157, 0.6549],
    #           [0.6118, 0.4588, 0.3725],
    #           [0.7294, 0.6902, 0.6745]]
    colors = [[0, 0.4470, 0.7410],
              [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250],
              [0.4940, 0.1840, 0.5560],
              [0.4660, 0.6740, 0.1880]]
    cuttail = [23]
    peakcut = [5]
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
            error = [error[a] for a in filter_index]
        elif N > 100:
            x = real_ave_degree_dict[N]
            print(len(x))
            x = x[0:cuttail[N_index]]
            y = ave_deviation_dict[N]
            y = y[0:cuttail[N_index]]
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
        line1 = ax1.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o',
                             markersize=16, label=lengend[0], color=colors[N_index])

        # 找到峰值后最低点的坐标
        peak_index = np.argmax(y[0:peakcut[N_index]])
        post_peak_y = y[peak_index:]
        post_peak_min_index = peak_index + np.argmin(post_peak_y)
        post_peak_min_x = x[post_peak_min_index]
        post_peak_min_y = y[post_peak_min_index]

        # 标出最低点
        ax1.plot(post_peak_min_x, post_peak_min_y, 'o', color=colors[N_index], markersize=30, markerfacecolor="none")

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    # plt.yticks([0,0.1,0.2,0.3])
    ax1.set_xlabel(r'Expected degree, $E[D]$', fontsize=26)
    ax1.tick_params(axis='y', labelcolor=colors[N_index], labelsize=26)
    ax1.set_ylabel('Average deviation', fontsize=26)
    ax1.tick_params(axis='x', labelsize=26)
    # hopcount
    hop_ave_vec = []
    hop_std_vec = []
    kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    for ED in kvec:
        spnodenum_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\SPnodenum_ED{EDn}Beta{betan}.txt".format(
            EDn=ED, betan=beta)
        hopcount_for_one_graph = np.loadtxt(spnodenum_Name, dtype=int)
        hop_ave_vec.append(np.mean(hopcount_for_one_graph))
        hop_std_vec.append(np.std(hopcount_for_one_graph))
    y2 = hop_ave_vec
    y2 = y2[0:cuttail[N_index]]
    error2 = hop_std_vec
    error2 = error2[0:cuttail[N_index]]

    ax2 = ax1.twinx()
    line2 = ax2.errorbar(x, y2, yerr=error2, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o',
                         markersize=16,
                         label=lengend[1], color=colors[1])

    ax2.set_ylabel('Shortest paths nodes', color=colors[1], fontsize=26)
    ax2.tick_params(axis='y', labelcolor=colors[1], labelsize=26)

    # plt.xlabel('Expected degree, E[D]',fontsize = 26)

    # plt.xticks(fontsize=26)
    # plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    # ax1.legend(loc='upper left',fontsize=20)
    # ax2.legend(loc='upper right',fontsize=20)
    # lines = [line1, line2]
    # plt.legend(lines, lengend, loc='upper left',fontsize=20)
    # plt.tick_params(axis='both', which="both",length=6, width=1)
    plt.xscale('log')
    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\LocalOptimumwithspnodenum10000nodeBeta{betan}.pdf".format(
        betan=beta)
    plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()


def scattter_deviationvsdeviation_nearlocaloptimum(beta):
    kvec = [8, 9, 10, 11, 12]
    kvec = [10]
    hop_vec = []
    spnodenum_vec = []
    ave_deviation_vec = []

    for beta in [beta]:
        for ED in kvec:
            hopcount_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\hopcount_sp_ED{EDn}Beta{betan}.txt".format(
                EDn=ED, betan=beta)
            hopcount_for_one_graph = np.loadtxt(hopcount_Name, dtype=int)
            hop_vec = hop_vec + list(hopcount_for_one_graph)

            spnodenum_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\SPnodenum_ED{EDn}Beta{betan}.txt".format(
                EDn=ED, betan=beta)
            spnum_for_one_graph = np.loadtxt(spnodenum_Name, dtype=int)
            spnodenum_vec = spnodenum_vec + list(spnum_for_one_graph)

            ave_deviation_for_one_graph = []
            for ExternalSimutime in range(10):
                N = 10000
                deviation_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                ave_deviation_for_a_para_comb_10times = np.loadtxt(deviation_vec_name)
                ave_deviation_for_one_graph.extend(list(ave_deviation_for_a_para_comb_10times))
            ave_deviation_vec = ave_deviation_vec + ave_deviation_for_one_graph
            a = 1

    # lengend = [r"$N=10$", r"$N=10^2$", r"$N=10^3$", r"$N=10^4$"]
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = [[0, 0.4470, 0.7410],
              [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250],
              [0.4940, 0.1840, 0.5560],
              [0.4660, 0.6740, 0.1880]]

    plt.scatter(ave_deviation_vec, hop_vec, marker='o', s=30, c=colors[0], label=r"$N=10^4$")
    # plt.scatter(ave_deviation_vec, spnodenum_vec, marker='o', c=colors[1],markersize=16, label=r"$N=10^2$")

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # plt.ylim(0, 0.30)
    # plt.yticks([0, 0.1, 0.2, 0.3])

    # plt.xscale('log')
    plt.xlabel('Average deviation', fontsize=26)
    plt.ylabel('hopcount', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=20)
    plt.tick_params(axis='both', which="both", length=6, width=1)
    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\scattter_deviationvsdeviation_nearlocaloptimum10000nodeBeta{betan}.pdf".format(
        betan=beta)
    plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """
    # STEP 4 plot local optimum: deviation versus expected degree
    """
    # kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    # print(len(kvec))

    plot_local_optimum_with_realED2(4)
    # plot_local_optimum_with_realED2(8)






