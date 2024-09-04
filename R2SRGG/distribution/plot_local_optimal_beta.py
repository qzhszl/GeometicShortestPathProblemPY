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
    betavec = [2.1, 4, 8, 16, 32, 64, 128]
    exemptionlist =[]
    ave_deviation_vec = []
    std_deviation_vec =[]
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
    return ave_cc_vec, ave_deviation_vec,std_deviation_vec


def load_large_network_results_beta(N, ED):
    # betavec = [2.1, 4, 8, 16, 32, 64, 128]
    betavec = [2.1, 2.2, 2.4, 2.5, 2.6, 2.8, 3, 3.25, 3.5, 3.75, 4, 5, 6, 7,8, 10, 12,16,32,64,128]
    exemptionlist =[]
    ave_deviation_vec = []
    clustering_coefficient_vec =[]
    std_deviation_vec = []
    ExternalSimutime =0
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
    betavec = [2.1, 4, 8, 16, 32, 64, 128]
    N=10000
    exemptionlist =[]

    ave_deviation_vec = []
    std_deviation_vec = []
    clustering_coefficient_vec =[]
    for beta in betavec:
        ave_deviation_for_a_para_comb=[]
        FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\network_N{Nn}ED{EDn}Beta{betan}.txt".format(
            Nn=N, EDn=ED, betan=beta)
        G = loadSRGGandaddnode(N, FileNetworkName)
        clustering_coefficient = nx.average_clustering(G)
        # print("real ED:", real_avg)
        clustering_coefficient_vec.append(clustering_coefficient)

        for ExternalSimutime in range(10):
            try:
                deviation_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                ave_deviation_for_a_para_comb_10times = np.loadtxt(deviation_vec_name)
                ave_deviation_for_a_para_comb.extend(ave_deviation_for_a_para_comb_10times)
            except FileNotFoundError:
                exemptionlist.append((N, ED, beta, ExternalSimutime))
                print(exemptionlist)
        ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
        std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))
    clustering_coefficient_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\clustering_coefficient_ED{EDn}.txt".format(EDn=ED)
    np.savetxt(clustering_coefficient_Name, clustering_coefficient_vec)
    ave_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\ave_deviation_ED{EDn}.txt".format(EDn=ED)
    np.savetxt(ave_deviation_Name, ave_deviation_vec)
    std_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\std_deviation_ED{EDn}.txt".format(EDn=ED)
    np.savetxt(std_deviation_Name, std_deviation_vec)
    return clustering_coefficient_vec, ave_deviation_vec,std_deviation_vec


def load_resort_data_beta(N, ED):
    betavec = [2.1, 4, 8, 16, 32, 64, 128]
    # betavec = [2.2, 2.4, 2.5, 2.6, 2.8, 3, 3.25, 3.5, 3.75, 5, 6, 7, 10, 12]
    exemptionlist =[]

    ave_deviation_vec = []
    ave_deviation_dic = {}
    clustering_coefficient_vec = []
    for beta in betavec:
        for ExternalSimutime in [0]:
            if N< 200:
                try:
                    clustering_coefficient_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\clustering_coefficient_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                    clustering_coefficient = np.loadtxt(clustering_coefficient_name)
                    clustering_coefficient_vec=clustering_coefficient_vec+list(clustering_coefficient)
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
                        real_cc = clustering_coefficient[count]
                        try:
                            ave_deviation_dic[real_cc] = ave_deviation_dic[real_cc] + list(ave_deviation_for_a_para_comb[a_index:b_index])
                        except:
                            ave_deviation_dic[real_cc] = list(ave_deviation_for_a_para_comb[a_index:b_index])
                        a_index = b_index+1
                        count = count+1
                except FileNotFoundError:
                    exemptionlist.append((N, ED, beta, ExternalSimutime))

    resort_dict = {}
    for key_cc, value_deviation in ave_deviation_dic.items():
        if round_onetenthquator(key_cc) in resort_dict.keys():
            resort_dict[round_onetenthquator(key_cc)] = resort_dict[round_onetenthquator(key_cc)] + list(value_deviation)
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


def round_quarter(x):
    x_tail = x- math.floor(x)
    if x_tail<0.25:
        return math.floor(x)
    elif x_tail<0.75:
        return math.floor(x)+0.5
    else:
        return math.ceil(x)


def round_onetenthquator(x):
    x = x*10
    y = round_quarter(x)
    return y/10


def plot_local_optimum_with_beta(ED):
    # the x-axis is the real average degree
    # Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    betavec = [2.1, 4, 8, 16, 32, 64, 128]
    Nvec = [20, 50, 100,1000]
    clustering_coefficient_dict = {}
    ave_deviation_dict = {}
    std_deviation_dict = {}

    for N in Nvec:
        if N < 200:
            for ED in [ED]:
                clustering_coefficient_vec, ave_deviation_vec, std_deviation_vec = load_small_network_results_beta(N,ED)
                clustering_coefficient_dict[N] = clustering_coefficient_vec
                ave_deviation_dict[N] = ave_deviation_vec
                std_deviation_dict[N] = std_deviation_vec
        elif N < 10000:
            for ED in [ED]:
                clustering_coefficient_vec, ave_deviation_vec, std_deviation_vec = load_large_network_results_beta(N, ED)
                clustering_coefficient_dict[N] = clustering_coefficient_vec
                ave_deviation_dict[N] = ave_deviation_vec
                std_deviation_dict[N] = std_deviation_vec
        else:
            for ED in [ED]:
                clustering_coefficient_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\clustering_coefficient_ED{EDn}.txt".format(
                    EDn=ED)

                ave_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\ave_deviation_ED{EDn}.txt".format(
                    EDn=ED)

                std_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\std_deviation_ED{EDn}.txt".format(
                    EDn=ED)
                clustering_coefficient_vec = np.loadtxt(clustering_coefficient_Name)
                ave_deviation_vec= np.loadtxt(ave_deviation_Name)
                std_deviation_vec= np.loadtxt(std_deviation_Name)

                clustering_coefficient_dict[N] = clustering_coefficient_vec
                ave_deviation_dict[N] = ave_deviation_vec
                std_deviation_dict[N] = std_deviation_vec

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = [[0, 0.4470, 0.7410],
              [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250],
              [0.4940, 0.1840, 0.5560],
              [0.4660, 0.6740, 0.1880]]

    for N_index in range(len(Nvec)):
        N = Nvec[N_index]
        x = clustering_coefficient_dict[N]
        y = ave_deviation_dict[N]
        error = std_deviation_dict[N]
        plt.errorbar(betavec, y, yerr=error, linestyle="--", linewidth=3, elinewidth= 1, capsize=5,marker='o',label=f'N={N}',markersize=16,color=colors[N_index])

        # # 找到峰值后最低点的坐标
        # peak_index = np.argmax(y[0:10])
        # post_peak_y = y[peak_index:]
        # post_peak_min_index = peak_index + np.argmin(post_peak_y)
        # post_peak_min_x = x[post_peak_min_index]
        # post_peak_min_y = y[post_peak_min_index]

        # 标出最低点
        # plt.plot(post_peak_min_x, post_peak_min_y, 'o', color=colors[N_index], markersize=8)

    # plt.xscale('log')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel(r'$\beta$', fontsize=26)
    plt.ylabel('Average Deviation', fontsize=26)
    plt.ylabel('Distance form shortest path nodes to the geodesic')
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.legend(fontsize=20, loc=(0.72, 0.58))
    plt.tick_params(axis='both', which="both", length=6, width=1)
    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\LocalOptimumwithbeta_ED{EDn}.pdf".format(
        EDn=ED)
    plt.legend(fontsize=20, loc=(0.7, 0.58))
    plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    # plt.title('Errorbar Curves with Minimum Points after Peak')

    plt.show()


def plot_local_optimum_with_cc(ED):
    """
    return:
    """
    clustering_coefficient_dict = {}
    ave_deviation_dict = {}
    std_deviation_dict = {}
    Nvec = [20, 50, 100, 1000,10000]
    # Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]

    for N in Nvec:
        if N < 200:
            degree_vec_resort, ave_deviation_resort, std_deviation_resort, _, _, _ = load_resort_data_beta(N, ED)
            clustering_coefficient_dict[N] = degree_vec_resort
            ave_deviation_dict[N] = ave_deviation_resort
            std_deviation_dict[N] = std_deviation_resort
        # elif N < 200:
        #     for beta in [4]:
        #         clustering_coefficient_vec, ave_deviation_vec, std_deviation_vec = load_small_network_results(N,beta)
        #         clustering_coefficient_dict[N] = clustering_coefficient_vec
        #         ave_deviation_dict[N] = ave_deviation_vec
        #         std_deviation_dict[N] = std_deviation_vec
        elif N < 10000:
            for ED in [ED]:
                clustering_coefficient_vec, ave_deviation_vec, std_deviation_vec = load_large_network_results_beta(N, ED)
                clustering_coefficient_dict[N] = clustering_coefficient_vec
                ave_deviation_dict[N] = ave_deviation_vec
                std_deviation_dict[N] = std_deviation_vec
        else:
            for ED in [ED]:
                clustering_coefficient_vec, ave_deviation_vec, std_deviation_vec = load_10000nodenetwork_results_beta(ED)
                clustering_coefficient_dict[N] = clustering_coefficient_vec
                ave_deviation_dict[N] = ave_deviation_vec
                std_deviation_dict[N] = std_deviation_vec
    lengend = [r"$N=20$", r"$N=50$", r"$N=10^2$", r"$N=10^3$", r"$N=10^4$"]
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = [[0, 0.4470, 0.7410],
              [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250],
              [0.4940, 0.1840, 0.5560],
              [0.4660, 0.6740, 0.1880]]
    cuttail = [12,10,9,19,7]
    # peakcut = [9,5,5]
    for N_index in range(len(Nvec)):
        N = Nvec[N_index]
        x = clustering_coefficient_dict[N]
        print(len(x))
        x = x[0:cuttail[N_index]]
        y = ave_deviation_dict[N]
        y = y[0:cuttail[N_index]]
        error = std_deviation_dict[N]
        error = error[0:cuttail[N_index]]
        plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o',markersize=16, label=lengend[N_index], color=colors[N_index])

        # 找到峰值后最低点的坐标
        # peak_index = np.argmax(y[0:peakcut[N_index]])
        # post_peak_y = y[peak_index:]
        # post_peak_min_index = peak_index + np.argmin(post_peak_y)
        # post_peak_min_x = x[post_peak_min_index]
        # post_peak_min_y = y[post_peak_min_index]
        #
        # # 标出最低点
        # plt.plot(post_peak_min_x, post_peak_min_y, 'o', color=colors[N_index], markersize=16)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.xlim(0, 0.65)
    plt.xticks([0, 0.1, 0.2, 0.3,0.4,0.5,0.6])

    plt.ylim(0, 0.3)
    plt.yticks([0, 0.1, 0.2, 0.3])
    # plt.xscale('log')
    plt.xlabel(r'Clustering coefficient, $C_G$',fontsize = 26)
    plt.ylabel('Average deviation',fontsize = 26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=20, loc=(0.72,0.58))
    plt.tick_params(axis='both', which="both",length=6, width=1)
    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\LocalOptimumED{EDn}.pdf".format(
        EDn=ED)
    plt.savefig(picname,format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    # plt.close()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # i = 1
    # exemptionlist = np.loadtxt("notrun.txt")
    # notrun_pair = exemptionlist[1]
    # print(notrun_pair)
    # Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    # kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    # betavec = [2.1, 4, 8, 16, 32, 64, 128]
    # ED_index = kvec.index(notrun_pair[1])
    # beta_index = betavec.index(notrun_pair[2])
    # print(ED_index)
    # print(beta_index)

    # plot_local_optimum_with_beta(5)
    for ED in [20]:
        # betavec = [2.2, 2.4, 2.5, 2.6, 2.8, 3, 3.25, 3.5, 3.75, 5, 6, 7, 10, 12]
        # load_10000nodenetwork_results_beta(ED)
        plot_local_optimum_with_cc(ED)



