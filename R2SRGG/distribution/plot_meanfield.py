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
from scipy.optimize import curve_fit
import math
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def load_small_network_results(N, beta):
    # Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    # betavec = [2.1, 4, 8, 16, 32, 64, 128]

    exemptionlist =[]
    for N in [N]:
        ave_deviation_vec = []
        std_deviation_vec =[]
        real_ave_degree_vec =[]
        for beta in [beta]:
            for ED in kvec:
                if ED<N:
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
    return real_ave_degree_vec,ave_deviation_vec,std_deviation_vec


def load_large_network_results(N, beta):
    # Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    # betavec = [2.1, 4, 8, 16, 32, 64, 128]

    exemptionlist =[]
    for N in [N]:
        ave_deviation_vec = []
        real_ave_degree_vec =[]
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
    return real_ave_degree_vec,ave_deviation_vec, std_deviation_vec


def load_10000nodenetwork_results(beta):
    # Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    # betavec = [2.1, 4, 8, 16, 32, 64, 128]

    exemptionlist =[]
    for N in [10000]:
        ave_deviation_vec = []
        std_deviation_vec = []
        real_ave_degree_vec =[]
        for beta in [beta]:
            for ED in kvec:
                ave_deviation_for_a_para_comb=[]
                FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\network_N{Nn}ED{EDn}Beta{betan}.txt".format(
                    Nn=N, EDn=ED, betan=beta)
                G = loadSRGGandaddnode(N, FileNetworkName)
                real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
                # print("real ED:", real_avg)
                real_ave_degree_vec.append(real_avg)

                for ExternalSimutime in range(10):
                    try:
                        deviation_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        ave_deviation_for_a_para_comb_10times = np.loadtxt(deviation_vec_name)
                        ave_deviation_for_a_para_comb.extend(ave_deviation_for_a_para_comb_10times)
                    except FileNotFoundError:
                        exemptionlist.append((N, ED, beta, ExternalSimutime))

                ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
                std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))
    print(exemptionlist)
    real_ave_degree_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\real_ave_degree_Beta{betan}.txt".format(betan=beta)
    np.savetxt(real_ave_degree_Name, real_ave_degree_vec)
    ave_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\ave_deviation_Beta{betan}.txt".format(betan=beta)
    np.savetxt(ave_deviation_Name, ave_deviation_vec)
    std_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\std_deviation_Beta{betan}.txt".format(betan=beta)
    np.savetxt(std_deviation_Name, std_deviation_vec)
    return real_ave_degree_vec, ave_deviation_vec,std_deviation_vec, exemptionlist


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
                if ED<N:
                    for ExternalSimutime in [0]:
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
                            for nodepair_num_inonegraph in node_pairs_vec:
                                b_index = a_index+nodepair_num_inonegraph-1
                                ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb[a_index:b_index]))
                                a_index = b_index+1

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
    Nvec = [10,100,1000,10000]
    # Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    # beta = 8
    for N in Nvec:
        if N < 200:
            degree_vec_resort, ave_deviation_resort, std_deviation_resort, _, _, _=load_resort_data(N,beta)
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

    lengend = [r"$N=10$",r"$N=10^2$",r"$N=10^3$",r"$N=10^4$"]
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
    colors = [[0, 0.4470, 0.7410],
              [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250],
              [0.4940, 0.1840, 0.5560],
              [0.4660, 0.6740, 0.1880]]
    cuttail = [5,34,23,23]
    # peakcut = [9,5,5,5,5]
    for N_index in range(len(Nvec)):
        N = Nvec[N_index]
        if N==100:
            x = real_ave_degree_dict[N]
            print(len(x))
            x = x[0:cuttail[N_index]]
            print(x)
            y = ave_deviation_dict[N]
            y = y[0:cuttail[N_index]]
            error = std_deviation_dict[N]
            error = error[0:cuttail[N_index]]

            filter_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16,  18,  20,  22,  24,  26,  28,
             30]
            x = [x[a] for a in filter_index]
            y = [y[a] for a in filter_index]
            error = [error[a] for a in filter_index]
        elif N>100:
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
        plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o',markersize=16, label=lengend[N_index], color=colors[N_index])

        # # 找到峰值后最低点的坐标
        # peak_index = np.argmax(y[0:peakcut[N_index]])
        # post_peak_y = y[peak_index:]
        # post_peak_min_index = peak_index + np.argmin(post_peak_y)
        # post_peak_min_x = x[post_peak_min_index]
        # post_peak_min_y = y[post_peak_min_index]

        # 标出最低点
        # plt.plot(post_peak_min_x, post_peak_min_y, 'o', color=colors[N_index], markersize=16, markerfacecolor="none")

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    plt.ylim(0,0.30)
    plt.yticks([0,0.1,0.2,0.3])

    plt.xscale('log')
    plt.xlabel('Expected degree, E[D]',fontsize = 26)
    plt.ylabel('Average deviation',fontsize = 26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=20,loc=(0.68,0.58))
    plt.tick_params(axis='both', which="both",length=6, width=1)
    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\LocalOptimumBeta{betan}2.pdf".format(
        betan=beta)
    plt.savefig(picname,format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()


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
        count = count+1

    lengend = [r"$C_G=0.03$",r"$C_G=0.31$",r"$C_G=0.51$",r"$C_G=0.56$",r"$C_G=0.59$"]
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = [[0, 0.4470, 0.7410],
              [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250],
              [0.4940, 0.1840, 0.5560],
              [0.4660, 0.6740, 0.1880]]
    # cuttail = [5,34,23,23]
    peakcut = [5,6,6,6,6]
    c_g_vec = [0.03,0.31,0.51,0.56,0.59]
    LO_ED = []
    LO_Dev =[]
    for count in range(len(betavec)):
        beta = betavec[count]
        x = real_ave_degree_dict[count]
        # print(len(x))
        # x = x[0:cuttail[N_index]]
        y = ave_deviation_dict[count]
        # y = y[0:cuttail[N_index]]
        error = std_deviation_dict[count]
        # error = error[0:cuttail[N_index]]
        plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o',markersize=16, label=lengend[count], color=colors[count])

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
    plt.xlabel('Expected degree, E[D]',fontsize = 26)
    plt.ylabel('Average deviation',fontsize = 26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=20,loc="upper left")
    plt.tick_params(axis='both', which="both",length=6, width=1)

    # inset_ax = inset_axes(ax, width="40%", height="30%")
    inset_ax = fig.add_axes([0.58, 0.55, 0.3, 0.3])
    inset_ax.plot(c_g_vec, LO_Dev,linewidth=3, marker='o', markersize=10, color = "b")
    inset_ax.set_xlabel("$C_G$",fontsize=18)
    inset_ax.set_ylabel(r"Local $\min(\overline{d}(q,\gamma(i,j)))$",fontsize=18)
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
    plt.savefig(picname,format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()


def load_10000nodenetwork_commonneighbour_results_clean(C_G):
    """
    the function is the same as load_10000 node network_results but the results are based on clean data
    :param C_G: clustering coefficient of the graph
    :return:
    """
    # Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    kvec = list(range(2, 20)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    # betavec = [2.1, 4, 8, 16, 32, 64, 128]

    exemptionlist =[]
    for N in [10000]:
        ave_deviation_vec = []
        std_deviation_vec = []
        real_ave_degree_vec =[]
        for cc in [C_G]:
            for target_ED in kvec:
                ave_deviation_for_a_para_comb=[]

                for ExternalSimutime in range(10):
                    try:
                        deviation_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\neighbour_distance\\Givendistancedeviation_neighbour_nodes_N{Nn}ED{EDn}CG{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
                            Nn=N, EDn=target_ED, betan=cc, ST=ExternalSimutime, Geodistance=0.01)
                        ave_deviation_for_a_para_comb_10times = np.loadtxt(deviation_vec_name)
                        ave_deviation_for_a_para_comb.extend(ave_deviation_for_a_para_comb_10times)
                    except FileNotFoundError:
                        exemptionlist.append((N, target_ED, cc, ExternalSimutime))

                ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
                std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))
    print(exemptionlist)
    # real_ave_degree_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\real_ave_degree_Beta{betan}.txt".format(betan=beta)
    # np.savetxt(real_ave_degree_Name, real_ave_degree_vec)
    ave_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\neighbour_distance\\ave_deviation_C_G{cc}.txt".format(cc=cc)
    np.savetxt(ave_deviation_Name, ave_deviation_vec)
    std_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\neighbour_distance\\std_deviation_C_G{cc}.txt".format(cc=cc)
    np.savetxt(std_deviation_Name, std_deviation_vec)
    return ave_deviation_vec,std_deviation_vec, exemptionlist


def plot_common_neighbour_deviation_vs_realED_with_diffCG_clean():
    """
    the x-axis is the expected degree, the y-axis is the average deviation of common neighbours, different line is different c_G
    inset is the min(average deviation) vs c_G
    N  = 10000 NODES
    the x-axis is real (approximate) degree
    when use this function, use load_10000nodenetwork_results_clean(beta) before
    :return:
    """
    real_ave_degree_dict = {}
    ave_deviation_dict = {}
    std_deviation_dict = {}
    C_G_vec = [0.1,0.2,0.3,0.4,0.5,0.6]
    kvec = list(range(2, 20)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]

    realEDdic = {
        0.1: [2, 3, 4, 5, 4.1, 4.8, 5.4, 6, 6.7, 7.3, 8, 8.6, 9.2, 10, 10.5, 11.1, 11.7, 12.3, 13, 16, 19, 22, 25, 30,
              36, 41.8, 47.3, 57.8],
        0.2: [2, 3, 4, 5, 4.6, 5.3, 6, 6.8, 7.6, 8.3, 9, 9.7, 10.5, 11.2, 11.9, 12.7, 13.4, 14, 14.8, 18.4, 21.9, 25.5,
              29, 35.7, 43, 49.6, 56.5, 70],
        0.3: [2, 3, 4, 5, 4.6, 5.3, 6.1, 6.9, 7.7, 8.4, 9.3, 9.9, 10.7, 11.4, 12.2, 12.8, 13.7, 14.3, 15.2, 18.9, 22.6,
              26.3, 30, 37.3, 44.5, 51.8, 58.6, 72.9],
        0.4: [2, 3, 4, 5, 4.6, 5.4, 6.1, 7, 7.7, 8.5, 9.3, 10, 10.8, 11.5, 12.2, 13, 13.8, 14.5, 15.3, 17.1, 22.8, 26.5,
              30.2, 37.7, 44.9, 52.3, 59.5, 74],
        0.5: [0, 3, 4, 5, 4.6424, 5.4226, 6.1336, 6.8984, 7.6758, 8.4566, 9.276, 9.9374, 10.787, 11.5666, 12.249, 13.0488, 13.8072, 14.5978, 15.3866, 19.0646, 22.8862, 26.5628, 30.3552, 37.9332, 45.2712, 52.843, 60.217, 74.9706],
        0.6: [0, 0, 4, 5, 4.666, 5.4168, 6.1932, 6.9122, 7.6814, 8.4992, 9.2658, 9.981, 10.7524, 11.5144, 12.2716, 13.1054, 13.88, 14.5636, 15.3324, 19.1594, 22.9678, 26.773, 30.4384, 38.0788, 45.5374, 52.7334, 60.1636, 74.9164]
    }
    indexvec = list(range(0, 4)) + list(range(7, 28))
    ed_vec = [realEDdic[0.1][i] for i in indexvec]
    print(ed_vec)

    indexdic = {0.1:list(range(0, 4)) + list(range(7, 28)),
                0.2:list(range(0, 4)) + list(range(6, 28)),
                0.3:list(range(0, 4)) + list(range(6, 28)),
                0.4:list(range(0, 4)) + list(range(5, 28)),
                0.5:list(range(1, 4)) + list(range(5, 28)),
                0.6:list(range(2, 4)) + list(range(5, 28))}

    N = 10000
    count = 0
    for cc in C_G_vec:
        ave_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\neighbour_distance\\ave_deviation_C_G{cc}.txt".format(
            cc=cc)
        std_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\neighbour_distance\\std_deviation_C_G{cc}.txt".format(
            cc=cc)
        ave_deviation_vec = np.loadtxt(ave_deviation_Name)
        std_deviation_vec = np.loadtxt(std_deviation_Name)
        # real_ave_degree_vec, ave_deviation_vec, std_deviation_vec = load_10000nodenetwork_results(beta)

        ave_deviation_dict[count] = ave_deviation_vec
        std_deviation_dict[count] = std_deviation_vec
        count = count+1

    lengend = [r"$C_G=0.1$",r"$C_G=0.2$",r"$C_G=0.3$",r"$C_G=0.4$",r"$C_G=0.5$",r"$C_G=0.6$"]
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = [[0, 0.4470, 0.7410],
              [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250],
              [0.4940, 0.1840, 0.5560],
              [0.4660, 0.6740, 0.1880],
              [0.3010, 0.7450, 0.9330]]
    # cuttail = [5,34,23,23]
    peakcut = [5,6,6,6,6,6]
    c_g_vec = [0.03,0.31,0.51,0.56,0.59]
    LO_ED = []
    LO_Dev =[]
    # for count in range(len(C_G_vec)):
    for count in [1]:
        beta = C_G_vec[count]
        x = realEDdic[beta]
        indexvec = indexdic[beta]
        x = [x[i] for i in indexvec]

        # print(len(x))
        # x = x[0:cuttail[N_index]]
        y = ave_deviation_dict[count]
        y = [y[i] for i in indexvec]
        # y = y[0:cuttail[N_index]]
        error = std_deviation_dict[count]
        error = [error[i] for i in indexvec]
        # error = error[0:cuttail[N_index]]
        plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o',markersize=16, label=lengend[count], color=colors[count])

        # # # 找到峰值后最低点的坐标
        # peak_index = np.argmax(y[0:peakcut[count]])
        # post_peak_y = y[peak_index:]
        # post_peak_min_index = peak_index + np.argmin(post_peak_y)
        # post_peak_min_x = x[post_peak_min_index]
        # LO_ED.append(post_peak_min_x)
        # post_peak_min_y = y[post_peak_min_index]
        # LO_Dev.append(post_peak_min_y)

        # 标出最低点
        # plt.plot(post_peak_min_x, post_peak_min_y, 'o', color=colors[count], markersize=25, markerfacecolor="none")

    # 拟合幂律曲线
    params, covariance = curve_fit(power_law, x, y)

    # 获取拟合的参数
    a_fit, k_fit = params
    print(f"拟合结果: a = {a_fit}, k = {k_fit}")

    # 绘制原始数据和拟合曲线

    plt.plot(x, power_law(x, *params),linewidth=5, label=f'fit curve: $y={a_fit:.4f}x^{{{k_fit:.2f}}}$', color='red')

    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('Expected degree, E[D]',fontsize = 26)
    plt.ylabel('Average deviation',fontsize = 26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=20)
    plt.tick_params(axis='both', which="both",length=6, width=1)

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

    # picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\CommonNeighbourDeviationvsEDwithdiffc_G_cleanloglog.pdf".format(
    #     betan=beta)

    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\CommonNeighbourDeviationvsEDwithdiffc_G{betan}_curvefitloglog.pdf".format(
        betan=beta)
    plt.savefig(picname,format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()


# 定义幂律函数
def power_law(x, a, k):
    return a * x ** k


def plot_common_neighbour_deviation_vs_beta_temporary():
    """
    the x-axis is the beta, the y-axis is the average deviation of common neighbours, different line is different ED
    N  = 10000 NODES
    the x-axis is real (approximate) degree
    when use this function, use load_10000nodenetwork_results_clean(beta) before
    :return:
    """
    real_ave_degree_dict = {}
    ave_deviation_dict = {}
    std_deviation_dict = {}
    C_G_vec = [0.1,0.2,0.3,0.4,0.5,0.6]
    kvec = list(range(2, 20)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    betavec = [2.55, 3.2, 3.99, 5.15, 7.99, 300]


    N = 10000
    count = 0
    for cc in C_G_vec:
        ave_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\neighbour_distance\\ave_deviation_C_G{cc}.txt".format(
            cc=cc)
        std_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\neighbour_distance\\std_deviation_C_G{cc}.txt".format(
            cc=cc)
        ave_deviation_vec = np.loadtxt(ave_deviation_Name)
        std_deviation_vec = np.loadtxt(std_deviation_Name)
        # real_ave_degree_vec, ave_deviation_vec, std_deviation_vec = load_10000nodenetwork_results(beta)

        ave_deviation_dict[count] = ave_deviation_vec
        std_deviation_dict[count] = std_deviation_vec
        count = count+1


    legend = [r"$ED=6$",r"$ED=10$",r"$ED=15$",r"$ED=20$",r"$ED=50$",r"$ED=100$"]
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = [[0, 0.4470, 0.7410],
              [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250],
              [0.4940, 0.1840, 0.5560],
              [0.4660, 0.6740, 0.1880],
              [0.3010, 0.7450, 0.9330]]


    ED_index_vec = [kvec.index(ED) for ED in [6,10,15,20,50,100]]

    for count in range(len(ED_index_vec)):
        x = betavec
        ED_index = ED_index_vec[count]
        y = [ave_deviation_dict[cc_index][ED_index] for cc_index in range(len(C_G_vec))]
        error = [std_deviation_dict[cc_index][ED_index] for cc_index in range(len(C_G_vec))]
        # error = error[0:cuttail[N_index]]
        plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o',markersize=16, label=legend[count], color=colors[count])

        # # # 找到峰值后最低点的坐标
        # peak_index = np.argmax(y[0:peakcut[count]])
        # post_peak_y = y[peak_index:]
        # post_peak_min_index = peak_index + np.argmin(post_peak_y)
        # post_peak_min_x = x[post_peak_min_index]
        # LO_ED.append(post_peak_min_x)
        # post_peak_min_y = y[post_peak_min_index]
        # LO_Dev.append(post_peak_min_y)

        # 标出最低点
        # plt.plot(post_peak_min_x, post_peak_min_y, 'o', color=colors[count], markersize=25, markerfacecolor="none")
    # inset pic

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # plt.ylim(0,0.30)
    # plt.yticks([0,0.1,0.2,0.3])

    plt.xscale('log')
    plt.xlabel(r'$\beta$',fontsize = 26)
    plt.ylabel('Average deviation',fontsize = 26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=20,loc="upper left")
    plt.tick_params(axis='both', which="both",length=6, width=1)

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

    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\CommonNeighbourDeviationvsbeta.pdf"
    plt.savefig(picname,format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()


def load_10000nodenetwork_commonneighbour_results_beta(ED):
    """
    diff beta, same ED
    :param ED average degree
    :return:
    """
    # Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    betavec = [2.2,2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4, 8, 16, 32, 64, 128]

    distance_list = [[0.49, 0.5, 0.5, 0.5], [0.25, 0.25, 0.3, 0.3], [0.25, 0.25, 0.3, 0.3], [0.25, 0.25, 0.5, 0.5],
                     [0.25, 0.25, 0.75, 0.75]]
    Geodistance_index = 0
    x_A = distance_list[Geodistance_index][0]
    y_A = distance_list[Geodistance_index][1]
    x_B = distance_list[Geodistance_index][2]
    y_B = distance_list[Geodistance_index][3]
    geodesic_distance_AB = round(x_B - x_A, 4)

    exemptionlist =[]
    for N in [10000]:
        ave_deviation_vec = []
        std_deviation_vec = []
        for ED in [ED]:
            for beta in betavec:
                ave_deviation_for_a_para_comb=[]

                for ExternalSimutime in range(10):
                    try:
                        deviation_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\neighbour_distance\\Givendistancedeviation_neighbour_nodes_N{Nn}ED{EDn}beta{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
                            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime, Geodistance=geodesic_distance_AB)
                        ave_deviation_for_a_para_comb_10times = []
                        with open(deviation_vec_name, "r") as file:
                            for line in file:
                                if line.startswith("#"):
                                    continue
                                data = line.strip().split("\t")  # 使用制表符分割
                                ave_deviation_for_a_para_comb_10times.append(float(data[0]))
                        ave_deviation_for_a_para_comb.extend(ave_deviation_for_a_para_comb_10times)
                    except FileNotFoundError:
                        exemptionlist.append((N, ED, beta, ExternalSimutime))

                ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
                std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))
    print(exemptionlist)
    # real_ave_degree_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\real_ave_degree_Beta{betan}.txt".format(betan=beta)
    # np.savetxt(real_ave_degree_Name, real_ave_degree_vec)
    ave_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\neighbour_distance\\ave_deviation_ED{ED}.txt".format(ED=ED)
    np.savetxt(ave_deviation_Name, ave_deviation_vec)
    std_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\neighbour_distance\\std_deviation_ED{ED}.txt".format(ED=ED)
    np.savetxt(std_deviation_Name, std_deviation_vec)
    return ave_deviation_vec,std_deviation_vec, exemptionlist


def plot_common_neighbour_deviation_vs_beta():
    """
    the x-axis is the beta, the y-axis is the average deviation of common neighbours, different line is different ED

    N  = 10000 NODES
    the x-axis is real (approximate) degree
    when use this function, use load_10000nodenetwork_results_clean(beta) before
    :return:
    """
    real_ave_degree_dict = {}
    ave_deviation_dict = {}
    std_deviation_dict = {}
    C_G_vec = [0.1,0.2,0.3,0.4,0.5,0.6]

    kvec = [2, 5, 10, 20, 100]
    # betavec = [2.2, 4, 8, 16, 32, 64, 128]
    betavec = [2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4, 8, 16, 32, 64, 128]

    distance_list = [[0.49, 0.5, 0.5, 0.5], [0.25, 0.25, 0.3, 0.3], [0.25, 0.25, 0.3, 0.3], [0.25, 0.25, 0.5, 0.5],
                     [0.25, 0.25, 0.75, 0.75]]
    Geodistance_index = 0
    x_A = distance_list[Geodistance_index][0]
    y_A = distance_list[Geodistance_index][1]
    x_B = distance_list[Geodistance_index][2]
    y_B = distance_list[Geodistance_index][3]
    # geodesic_distance_AB = round(x_B - x_A, 4)

    N = 10000
    count = 0
    for ED in kvec:
        ave_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\neighbour_distance\\ave_deviation_ED{ED}.txt".format(
            ED = ED)
        std_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\neighbour_distance\\std_deviation_ED{ED}.txt".format(
            ED = ED)
        ave_deviation_vec = np.loadtxt(ave_deviation_Name)
        std_deviation_vec = np.loadtxt(std_deviation_Name)
        # real_ave_degree_vec, ave_deviation_vec, std_deviation_vec = load_10000nodenetwork_results(beta)

        ave_deviation_dict[count] = ave_deviation_vec
        std_deviation_dict[count] = std_deviation_vec
        count = count+1

    legend = [r"$E[D]=2$",r"$E[D]=5$",r"$E[D]=10$",r"$E[D]=20$",r"$E[D]=100$"]
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = [[0, 0.4470, 0.7410],
              [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250],
              [0.4940, 0.1840, 0.5560],
              [0.4660, 0.6740, 0.1880],
              [0.3010, 0.7450, 0.9330]]
    for count in range(len(kvec)):
    # for count in [5]:
        x = betavec
        # print(len(x))
        # x = x[0:cuttail[N_index]]
        y = ave_deviation_dict[count]
        # y = y[0:cuttail[N_index]]
        error = std_deviation_dict[count]
        # error = error[0:cuttail[N_index]]
        plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o',markersize=16, label=legend[count], color=colors[count])

    # 拟合幂律曲线
    # params, covariance = curve_fit(power_law, x, y)
    #
    # # 获取拟合的参数
    # a_fit, k_fit = params
    # print(f"拟合结果: a = {a_fit}, k = {k_fit}")
    #
    # # 绘制原始数据和拟合曲线
    #
    # plt.plot(x, power_law(x, *params),linewidth=5, label=f'fit curve: $y={a_fit:.4f}x^{{{k_fit:.2f}}}$', color='red')

    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel(r'$\beta$',fontsize = 26)
    plt.ylabel('Average deviation',fontsize = 26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=20)
    plt.tick_params(axis='both', which="both",length=6, width=1)

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

    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\CommonNeighbourDeviationvsbetawithdiffED_cleanloglog.pdf"

    # picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\CommonNeighbourDeviationvsEDwithdiffED{ED}_curvefitloglog.pdf".format(
    #     betan=ED)
    plt.savefig(picname,format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # STEP10
    # plot_local_optimum_with_realED_diffCG()
    """
     # STEP 11 TEST CLEAN DATA ED and CC
    """
    # kvec = list(range(2, 20)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    # cc_vec = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    # betavec = [2.55, 3.2, 3.99, 5.15, 7.99, 300]
    #
    # distance_list = [[0.25, 0.25, 0.3, 0.3], [0.25, 0.25, 0.5, 0.5], [0.25, 0.25, 0.75, 0.75]]
    # N = 10000
    # Geodistance_index = 0
    # x_A = distance_list[Geodistance_index][0]
    # y_A = distance_list[Geodistance_index][1]
    # x_B = distance_list[Geodistance_index][2]
    # y_B = distance_list[Geodistance_index][3]
    # geodesic_distance_AB = x_B - x_A
    # for input_ED in kvec:
    #     for C_G in cc_vec:
    #         FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\cleanwithEDCC\\GivenDistance\\Givendistancenetwork_N{Nn}ED{EDn}CC{betan}Geodistance{Geodistance}.txt".format(
    #         Nn=N, EDn=input_ED, betan=C_G, Geodistance=geodesic_distance_AB)
    #         G = loadSRGGandaddnode(10000, FileNetworkName)
    #         clustering_coefficient = nx.average_clustering(G)
    #         print(clustering_coefficient)
    #         real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
    #         print("real ED:", real_avg)


    """
    assure all the graph has reasonable ED and CC
    """
    # smallED_matrix = [[2.6,2.6,2.6,2.6,0,0,0],
    #                   [4.3,4,4,4,4,0],
    #                   [5.6,5.4,5.2,5.2,5.2,5.2,5.2],
    #                   [7.5,6.7,6.5,6.5,6.5,6.5]]
    # smallbeta_matrix = [[3.1,4.5,6,300,0,0],
    #                   [2.7,	3.5,4.7,7.6,300,0],
    #                   [2.7,	3.4,4.3,5.7,11,	300],
    #                   [2.55,3.2,4,5.5,8.5,300]]
    # print(smallbeta_matrix[3][5])

    """
    plot average NEIGHBOUR deviation vs real_avg_degree for clean data
    first time need to run the two line below:
    """

    # for C_G in [0.1,0.2,0.3,0.4,0.5,0.6]:
    #     load_10000nodenetwork_commonneighbour_results_clean(C_G)

    # plot_common_neighbour_deviation_vs_realED_with_diffCG_clean()

    """
    plot average NEIGHBOUR deviation vs beta
    """
    # plot_common_neighbour_deviation_vs_beta()


    """
    plot deviation vs defferent beta
    """
    # kvec = [2, 5, 10, 20, 100]
    # for ED in kvec:
    #     load_10000nodenetwork_commonneighbour_results_beta(ED)
    plot_common_neighbour_deviation_vs_beta()

