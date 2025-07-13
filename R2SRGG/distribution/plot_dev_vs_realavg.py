# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 09-5-2025
"""
import random

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from fontTools.tfmLib import PASSTHROUGH
from scipy.optimize import curve_fit

from R2SRGG.R2SRGG import loadSRGGandaddnode, R2SRGG
from collections import defaultdict
import math
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from SphericalSoftRandomGeomtricGraph import RandomGenerator


def load_small_network_results(N, beta):
    if N == 10:
        kvec = list(range(2, 10)) + [10, 12, 15, 18, 22, 27, 33, 40, 49, 60, 73, 89, 99]
    elif N == 100:
        kvec = [2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 22, 27, 33, 40, 49, 60, 73, 89]
        kvec = [2, 3, 4, 5, 6, 8, 10, 12, 14, 17, 22, 27, 33, 40, 49, 60, 73, 89, 113, 149, 198, 260, 340, 446, 584,
                762, 993, 1292, 1690, 2276, 3142, 4339]
        # kvec = [2, 3, 4, 5, 6, 8, 10, 12, 14, 17, 22, 27, 33, 40, 49, 60, 73, 89, 113, 149, 198, 260, 340, 446, 584,
        #         762, 993, 1292, 2276, 4339]

    exemptionlist = []
    for N in [N]:
        ave_deviation_vec = []
        std_deviation_vec = []
        real_ave_degree_vec = []
        for beta in [beta]:
            for ED in kvec:
                if N == 20:
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
                else:
                    try:
                        ave_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\ave_deviation_N{Nn}ED{EDn}Beta{betan}.txt".format(
                            Nn=N, EDn=ED, betan=beta)
                        ave_deviation_for_a_para_comb = np.loadtxt(ave_deviation_name)
                        ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
                        std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))
                    except FileNotFoundError:
                        exemptionlist.append((N, ED, beta))
    print(exemptionlist)
    print(real_ave_degree_vec)
    print(ave_deviation_vec)
    return kvec, real_ave_degree_vec, ave_deviation_vec, std_deviation_vec


def load_large_network_results(N, beta):
    if N == 1000:
        kvec = [2, 3] + list(range(4, 16)) + [20, 28, 40, 58, 83, 118, 169, 241, 344, 490, 700, 999]
        kvec = [2, 3, 4, 5, 6, 7, 8, 11, 15, 20, 28, 40, 58, 83, 118, 169, 241, 344, 490, 700, 999, 1425, 2033, 2900,
                4139, 5909, 8430, 12039, 17177, 24510, 34968, 49887, 71168]  # FOR BETA = 4
        # kvec = [2, 3, 4, 5, 6, 7, 8, 11, 15, 20, 25, 40, 60, 80, 118, 169, 241, 344, 490, 700, 999] # FOR BETA = 128

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
                        try:
                            FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\inputavgbeta\\network_N{Nn}ED{EDn}Beta{betan}.txt".format(
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
                        except:
                            exemptionlist.append((N, ED, beta, ExternalSimutime))
    print(exemptionlist)
    print(real_ave_degree_vec)
    return kvec, real_ave_degree_vec, ave_deviation_vec, std_deviation_vec


def load_10000nodenetwork_results(beta):
    # for beta = 4
    kvec = [2.2, 2.8, 3.0, 3.4, 3.8, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105,
            9999,
            16479, 27081, 44767, 73534, 121205, 199999]
    real_ave_degree_vec = [1.7024, 2.1224, 2.2988, 2.6058, 2.941, 3.3956, 4.6198, 7.6544, 12.1272, 20.414358564143587,
                           32.9682,
                           53.2058, 85.6794, 137.1644, 218.4686, 345.3296, 541.029, 836.6424, 1278.4108, 1902.8332,
                           2783.4186,
                           3911.416, 5253, 6700, 8029, 8990, 9552, 9820]
    # kvec = [2.2, 2.8, 3.0, 3.4, 3.8, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105,
    #         9999]
    # real_ave_degree_vec = [1.7024, 2.1224, 2.2988, 2.6058, 2.941, 3.3956, 4.6198, 7.6544, 12.1272, 20.414358564143587,
    #                        32.9682,
    #                        53.2058, 85.6794, 137.1644, 218.4686, 345.3296, 541.029, 836.6424, 1278.4108, 1902.8332,
    #                        2783.4186,
    #                        3911.416]

    # for beta = 8
    kvec = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999, 16479, 27081, 44767, 73534,
            121205, 199999, 316226, 499999]
    real_ave_degree_vec = [1.7024, 2.1224, 2.2988, 2.6058, 2.941, 3.3956, 4.6198, 7.6544, 12.1272, 20.414358564143587,
                           32.9682,
                           53.2058, 85.6794, 137.1644, 218.4686, 345.3296, 541.029, 836.6424, 1278.4108, 1902.8332,
                           2783.4186,
                           3911.416, 5253, 6700, 8029, 8990, 9552, 9820]
    kvec = [6105, 9999, 16479, 27081, 44767, 73534,
            121205, 199999, 316226]
    real_ave_degree_vec = [3189.2608, 4591.8122, 6326.808, 8066.9882, 9344.1624, 9860.3746, 9978.1308, 9996.094,
                           9998.5504]

    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\"
    # print(beta)
    exemptionlist = []
    for N in [10000]:
        ave_deviation_vec = []
        std_deviation_vec = []
        for beta in [beta]:
            for ED in kvec:
                ave_deviation_for_a_para_comb = np.array([])
                for ExternalSimutime in range(20):
                    try:
                        deviation_vec_name = filefolder_name + "ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        ave_deviation_for_a_para_comb_10times = np.loadtxt(deviation_vec_name)
                        ave_deviation_for_a_para_comb = np.hstack(
                            (ave_deviation_for_a_para_comb, ave_deviation_for_a_para_comb_10times))
                    except FileNotFoundError:
                        exemptionlist.append((N, ED, beta, ExternalSimutime))

                # print(len(ave_deviation_for_a_para_comb))
                ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
                std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))
    print("DATA LOST:", exemptionlist)
    # ave_deviation_Name = filefolder_name + "ave_deviation_N{Nn}_beta{betan}.txt".format(
    #     Nn=N, betan=beta)
    # np.savetxt(ave_deviation_Name, ave_deviation_vec)
    # std_deviation_Name = filefolder_name + "std_deviation_N{Nn}_beta{betan}.txt".format(Nn=N,
    #                                                                                     betan=beta)
    # np.savetxt(std_deviation_Name, std_deviation_vec)

    # print(list(map(float, ave_deviation_vec)))
    # print(list(map(float,std_deviation_vec)))
    return kvec, real_ave_degree_vec, ave_deviation_vec, std_deviation_vec


def load_resort_data(N, beta):
    kvec = list(range(2, 10)) + [10, 12, 15, 18, 22, 27, 33, 40, 49, 60, 73, 89, 99]  # FOR N =10
    exemptionlist = []
    for N in [N]:
        ave_deviation_vec = []
        ave_deviation_dic = {}
        real_ave_degree_vec = []

        for beta in [beta]:
            for ED in kvec:
                for ExternalSimutime in [0]:
                    if N < 200:
                        try:
                            real_ave_degree_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\real_ave_degree_N{Nn}ED{EDn}Beta{betan}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                            real_ave_degree = np.loadtxt(real_ave_degree_name)
                            real_ave_degree_vec = real_ave_degree_vec + list(real_ave_degree)
                            nodepairs_for_eachgraph_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\nodepairs_for_eachgraph_N{Nn}ED{EDn}Beta{betan}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                            node_pairs_vec = np.loadtxt(nodepairs_for_eachgraph_vec_name, dtype=int)

                            ave_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\ave_deviation_N{Nn}ED{EDn}Beta{betan}.txt".format(
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


def load_real_ave_degree(N, beta):
    if N == 100:
        kvec = [2, 3, 4, 5, 6, 8, 10, 12, 14, 17, 22, 27, 33, 40, 49, 60, 73, 89, 113, 149, 198, 260, 340, 446, 584,
                762, 993, 1292, 1690, 2276, 3142, 4339]
    elif N == 1000:
        pass

    exemptionlist = []
    for N in [N]:
        real_ave_degree_vec = []
        for beta in [beta]:
            for ED in kvec:
                if N == 100:
                    for ExternalSimutime in [0]:
                        try:
                            real_ave_degree_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\real_ave_degree_N{Nn}ED{EDn}Beta{betan}.txt".format(
                                Nn=N, EDn=ED, betan=beta)
                            real_ave_degree = np.loadtxt(real_ave_degree_name)
                            real_ave_degree_vec.append(np.mean(real_ave_degree))
                        except FileNotFoundError:
                            try:
                                real_ave_degree_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\real_ave_degree_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                                real_ave_degree = np.loadtxt(real_ave_degree_name)
                                real_ave_degree_vec.append(np.mean(real_ave_degree))
                            except FileNotFoundError:
                                exemptionlist.append((N, ED, beta))
                                rg = RandomGenerator(-12)
                                rseed = random.randint(0, 100)
                                for i in range(rseed):
                                    rg.ran1()
                                real_ave_degree = []
                                for simu_index in range(100):
                                    G, Coorx, Coory = R2SRGG(N, ED, beta, rg)
                                    real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
                                    # print("real ED:", real_avg)
                                    real_ave_degree.append(real_avg)
                                real_ave_degree_vec.append(np.mean(real_ave_degree))

    print(exemptionlist)
    real_ave_degree_vec = [float(i) for i in real_ave_degree_vec]
    return kvec, real_ave_degree_vec


def plot_local_optimum():
    # the x-axis is the input average degree
    Nvec = [10, 100, 1000, 10000]
    # Nvec = [1000]
    real_ave_degree_dict = {}
    ave_deviation_dict = {}
    std_deviation_dict = {}
    kvec_dict = {}

    for N in Nvec:
        if N == 10:
            for beta in [4]:
                degree_vec_resort, ave_deviation_vec, std_deviation_vec, _, _, _ = load_resort_data(N, beta)
                real_ave_degree_dict[N] = degree_vec_resort
                ave_deviation_dict[N] = ave_deviation_vec
                std_deviation_dict[N] = std_deviation_vec
                kvec = degree_vec_resort
                real_ave_degree_vec = degree_vec_resort
                kvec_dict[N] = kvec
                real_ave_degree_dict[N] = real_ave_degree_vec
        elif N < 200:
            for beta in [4]:
                kvec, real_ave_degree_vec, ave_deviation_vec, std_deviation_vec = load_small_network_results(N, beta)
                real_ave_degree_dict[N] = [1.3878000000000004, 2.0258000000000003, 2.6606, 3.2968, 3.9018, 5.0796,
                                           6.2132000000000005, 7.311599999999999, 8.342, 9.8808, 12.2334, 14.4614,
                                           17.141800000000003, 19.8532, 23.2854, 26.9778, 31.156, 35.631400000000006,
                                           41.68019999999999, 49.1888, 57.876999999999995, 65.243, 72.5054,
                                           79.60860000000001, 85.1024, 89.85379999999999, 92.81960000000001, 95.0446,
                                           96.645, 97.6358, 98.25179999999999, 98.60859999999998]  # for beta = 4
                # real_ave_degree_dict[N] = [1.3878000000000004, 2.0258000000000003, 2.6606, 3.2968, 3.9018, 5.0796,
                #                            6.2132000000000005, 7.311599999999999, 8.342, 9.8808, 12.2334, 14.4614,
                #                            17.141800000000003, 19.8532, 23.2854, 26.9778, 31.156, 35.631400000000006,
                #                            41.68019999999999, 49.1888, 57.876999999999995, 65.243, 72.5054,
                #                            79.60860000000001, 85.1024, 89.85379999999999, 92.81960000000001, 95.0446,
                #                         97.6358, 98.60859999999998]  # for beta = 4
                ave_deviation_dict[N] = ave_deviation_vec
                std_deviation_dict[N] = std_deviation_vec
                kvec_dict[N] = kvec
        elif N == 1000:
            for beta in [4]:
                kvec, real_ave_degree_vec, ave_deviation_vec, std_deviation_vec = load_large_network_results(N, beta)
                real_ave_degree_dict[N] = real_ave_degree_vec
                kvec_dict[N] = kvec
                ave_deviation_dict[N] = ave_deviation_vec
                std_deviation_dict[N] = std_deviation_vec
        else:
            for beta in [4]:
                kvec, real_ave_degree_vec, ave_deviation_vec, std_deviation_vec = load_10000nodenetwork_results(beta)
                real_ave_degree_dict[N] = real_ave_degree_vec
                kvec_dict[N] = kvec
                ave_deviation_dict[N] = ave_deviation_vec
                std_deviation_dict[N] = std_deviation_vec

    # plt.plot(kvec,ave_deviation_vec,"o-")
    # plt.xscale('log')
    # plt.show()
    lengend = [r"$N=10$", r"$N=10^2$", r"$N=10^3$", r"$N=10^4$"]
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    # colorvec2 = ['#9FA9C9', '#D36A6A']
    for N_index in range(len(Nvec)):
        N = Nvec[N_index]
        if N == 10:
            x = real_ave_degree_dict[N]
            print(len(x))
            x = x[1:]
            y = ave_deviation_dict[N]
            y = y[1:]
            error = std_deviation_dict[N]
            error = error[1:]
        else:
            x = real_ave_degree_dict[N]
            y = ave_deviation_dict[N]
            error = std_deviation_dict[N]
        plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=16,
                     label=lengend[N_index], color=colors[N_index])

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # plt.ylim(0, 0.30)
    # plt.yticks([0, 0.1, 0.2, 0.3])
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'Average degree, $\langle D \rangle$', fontsize=26)
    plt.ylabel(r'Average deviation, $\langle d \rangle$', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    # plt.legend(fontsize=26, loc=(0.5, 0.5))
    plt.tick_params(axis='both', which="both", length=6, width=1)
    # picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\LocalOptimumdiffNBeta{betan}.png".format(
    #     betan=beta)
    # plt.savefig(picname, format='png', bbox_inches='tight', dpi=600,transparent=True)
    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\dev_vs_realavg.svg"
    plt.savefig(
        picname,
        format="svg",
        bbox_inches='tight',  # 紧凑边界
        transparent=True  # 背景透明，适合插图叠加
    )
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.show()
    plt.close()


def plot_local_optimum_forappendix(beta):
    # the x-axis is the input average degree
    Nvec = [10, 100, 1000, 10000]
    # Nvec = [10]
    real_ave_degree_dict = {}
    ave_deviation_dict = {}
    std_deviation_dict = {}
    kvec_dict = {}

    for N in Nvec:
        if N == 10:
            for beta in [beta]:
                degree_vec_resort, ave_deviation_vec, std_deviation_vec, _, _, _ = load_resort_data(N, beta)
                real_ave_degree_dict[N] = degree_vec_resort
                ave_deviation_dict[N] = ave_deviation_vec
                std_deviation_dict[N] = std_deviation_vec
                kvec = degree_vec_resort
                real_ave_degree_vec = degree_vec_resort
                kvec_dict[N] = kvec
                real_ave_degree_dict[N] = real_ave_degree_vec
        elif N < 200:
            for beta in [beta]:
                kvec, real_ave_degree_vec, ave_deviation_vec, std_deviation_vec = load_small_network_results(N, beta)
                real_ave_degree_dict[N] = real_ave_degree_vec
                ave_deviation_dict[N] = ave_deviation_vec
                std_deviation_dict[N] = std_deviation_vec
                kvec_dict[N] = kvec
        elif N == 1000:
            for beta in [beta]:
                kvec, real_ave_degree_vec, ave_deviation_vec, std_deviation_vec = load_large_network_results(N, beta)
                real_ave_degree_dict[N] = real_ave_degree_vec
                kvec_dict[N] = kvec
                ave_deviation_dict[N] = ave_deviation_vec
                std_deviation_dict[N] = std_deviation_vec
        else:
            for beta in [beta]:
                kvec, real_ave_degree_vec, ave_deviation_vec, std_deviation_vec = load_10000nodenetwork_results(beta)
                real_ave_degree_dict[N] = real_ave_degree_vec
                kvec_dict[N] = kvec
                ave_deviation_dict[N] = ave_deviation_vec
                std_deviation_dict[N] = std_deviation_vec

    # plt.plot(kvec,ave_deviation_vec,"o-")
    # plt.xscale('log')
    # plt.show()
    lengend = [r"$N=10$", r"$N=10^2$", r"$N=10^3$", r"$N=10^4$"]
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    # colorvec2 = ['#9FA9C9', '#D36A6A']
    cuttail = [5, 34, 23, 22]
    # peakcut = [9,5,5,5,5]
    for N_index in range(len(Nvec)):
        N = Nvec[N_index]
        if N == 10:
            x = real_ave_degree_dict[N]
            print(len(x))
            x = x[1:]
            y = ave_deviation_dict[N]
            y = y[1:]
            error = std_deviation_dict[N]
            error = error[1:]
        else:
            x = kvec_dict[N]
            y = ave_deviation_dict[N]
            error = std_deviation_dict[N]
        plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=16,
                     label=lengend[N_index], color=colors[N_index])

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # plt.ylim(0, 0.30)
    # plt.yticks([0, 0.1, 0.2, 0.3])

    plt.xscale('log')
    plt.xlabel(r'Expected degree, $E[D]$', fontsize=26)
    plt.ylabel(r'Average deviation, $<d>$', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    # plt.legend(fontsize=26, loc=(0.5, 0.5))
    plt.tick_params(axis='both', which="both", length=6, width=1)
    # picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\LocalOptimumdiffNBeta{betan}.png".format(
    #     betan=beta)
    # plt.savefig(picname, format='png', bbox_inches='tight', dpi=600,transparent=True)
    picname = f"D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\dev_vs_avg_beta{beta}.svg"
    plt.savefig(
        picname,
        format="svg",
        bbox_inches='tight',  # 紧凑边界
        transparent=True  # 背景透明，适合插图叠加
    )
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.show()
    plt.close()


def plot_local_optimum_foronenetwork(N, beta):
    # plot the dev vs avg of one network para(beta, N)
    # the x-axis is the input average degree
    real_ave_degree_dict = {}
    ave_deviation_dict = {}
    std_deviation_dict = {}
    kvec_dict = {}

    for N in [N]:
        if N == 10:
            for beta in [beta]:
                degree_vec_resort, ave_deviation_vec, std_deviation_vec, _, _, _ = load_resort_data(N, beta)
                real_ave_degree_dict[N] = degree_vec_resort
                ave_deviation_dict[N] = ave_deviation_vec
                std_deviation_dict[N] = std_deviation_vec
                kvec = degree_vec_resort
                real_ave_degree_vec = degree_vec_resort
                kvec_dict[N] = kvec
                real_ave_degree_dict[N] = real_ave_degree_vec
        elif N < 200:
            for beta in [beta]:
                kvec, real_ave_degree_vec, ave_deviation_vec, std_deviation_vec = load_small_network_results(N, beta)
                real_ave_degree_dict[N] = real_ave_degree_vec
                ave_deviation_dict[N] = ave_deviation_vec
                std_deviation_dict[N] = std_deviation_vec
                kvec_dict[N] = kvec
        elif N == 1000:
            for beta in [beta]:
                kvec, real_ave_degree_vec, ave_deviation_vec, std_deviation_vec = load_large_network_results(N, beta)
                real_ave_degree_dict[N] = real_ave_degree_vec
                kvec_dict[N] = kvec
                ave_deviation_dict[N] = ave_deviation_vec
                std_deviation_dict[N] = std_deviation_vec
        else:
            for beta in [beta]:
                kvec, real_ave_degree_vec, ave_deviation_vec, std_deviation_vec = load_10000nodenetwork_results(beta)
                real_ave_degree_dict[N] = real_ave_degree_vec
                kvec_dict[N] = kvec
                ave_deviation_dict[N] = ave_deviation_vec
                std_deviation_dict[N] = std_deviation_vec

    # plt.plot(kvec,ave_deviation_vec,"o-")
    # plt.xscale('log')
    # plt.show()
    lengend = [r"$N=10$", r"$N=10^2$", r"$N=10^3$", r"$N=10^4$"]
    fig, ax = plt.subplots(figsize=(6, 4))

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    N_index = 3
    if N == 10:
        x = real_ave_degree_dict[N]
        print(len(x))
        x = x[1:]
        y = ave_deviation_dict[N]
        y = y[1:]
        error = std_deviation_dict[N]
        error = error[1:]
    else:
        x = real_ave_degree_dict[N]
        y = ave_deviation_dict[N]
        error = std_deviation_dict[N]
    plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=16,
                 label=lengend[N_index], color=colors[N_index])

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    plt.ylim(8*10**(-3), 4*10**(-1))
    # plt.yticks([0, 0.1, 0.2, 0.3])

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\langle D \rangle$', fontsize=28)
    plt.ylabel(r'$\langle d \rangle$', fontsize=28)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    # plt.legend(fontsize=26, loc=(0.5, 0.5))
    plt.tick_params(axis='both', which="both", length=6, width=1)
    # picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\LocalOptimumdiffNBeta{betan}.png".format(
    #     betan=beta)
    # plt.savefig(picname, format='png', bbox_inches='tight', dpi=600,transparent=True)
    picname = f"D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\dev_vs_realavg_N{N}_beta{beta}.svg"
    plt.savefig(
        picname,
        format="svg",
        bbox_inches='tight',  # 紧凑边界
        transparent=True  # 背景透明，适合插图叠加
    )
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.show()
    plt.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """
    # STEP 0 load real ave degree 
    """
    # kvec,realkvec=load_real_ave_degree(100, 4)
    # print(kvec)
    # print(realkvec)

    """
    # STEP 1 plot local optimum: deviation versus real ave degree
    """
    # plot_local_optimum()

    """
    # STEP 1.5 plot local optimum: deviation versus real ave degree: same function as step 1 but different beta
    """
    # plot_local_optimum_forappendix(128)

    """
    # STEP 2 plot local optimum: deviation versus real ave degree for one N
    """
    plot_local_optimum_foronenetwork(10000, 8)
