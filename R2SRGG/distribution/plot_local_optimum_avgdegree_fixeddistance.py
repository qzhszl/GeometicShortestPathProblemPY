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


def load_10000nodenetwork_results_fixdistance(cc,geodesic_distance_AB):
    """
    the function is the same as load_10000nodenetwork_results but the results are based on clean data
    :param beta:
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
        for cc in [cc]:
            for ED in kvec:
                ave_deviation_for_a_para_comb=[]
                for ExternalSimutime in range(10):
                    try:
                        deviation_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\GivenGeodistance\\Givendistancedeviation_shortest_path_nodes_N{Nn}ED{EDn}CG{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
                            Nn=N, EDn=ED, betan=cc, ST=ExternalSimutime, Geodistance=geodesic_distance_AB)
                        ave_deviation_for_a_para_comb_10times = np.loadtxt(deviation_vec_name)
                        ave_deviation_for_a_para_comb.extend(ave_deviation_for_a_para_comb_10times)
                    except FileNotFoundError:
                        exemptionlist.append((N, ED, cc, ExternalSimutime))

                ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
                std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))
    print(exemptionlist)
    # real_ave_degree_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\real_ave_degree_Beta{betan}.txt".format(betan=beta)
    # np.savetxt(real_ave_degree_Name, real_ave_degree_vec)
    ave_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\GivenGeodistance\\ave_deviation_C_G{betan}Dis{geodesic_distance_AB}.txt".format(betan=cc,geodesic_distance_AB = geodesic_distance_AB)
    np.savetxt(ave_deviation_Name, ave_deviation_vec)
    std_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\GivenGeodistance\\std_deviation_C_G{betan}Dis{geodesic_distance_AB}.txt".format(betan=cc,geodesic_distance_AB = geodesic_distance_AB)
    np.savetxt(std_deviation_Name, std_deviation_vec)
    return ave_deviation_vec,std_deviation_vec, exemptionlist


def plot_local_optimum_with_realED_diffCG_fix_distance_clean(Geodistance_index):
    """
    the x-axis is the expected degree, the y-axis is the average deviation, different line is different c_G
    inset is the min(average deviation) vs c_G
    the x-axis is real (approximate) degree
    when use this function, use before
    :return:
    """

    distance_list = [[0.25, 0.25, 0.3, 0.3], [0.25, 0.25, 0.5, 0.5], [0.25, 0.25, 0.75, 0.75]]
    x_A = distance_list[Geodistance_index][0]
    y_A = distance_list[Geodistance_index][1]
    x_B = distance_list[Geodistance_index][2]
    y_B = distance_list[Geodistance_index][3]
    geodesic_distance_AB = x_B - x_A


    real_ave_degree_dict = {}
    ave_deviation_dict = {}
    std_deviation_dict = {}
    C_G_vec = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    realEDdic = {
        0.1: [2, 3, 4, 5, 4.1, 4.8, 5.4, 6, 6.7, 7.3, 8, 8.6, 9.2, 10, 10.5, 11.1, 11.7, 12.3, 13, 16, 19, 22, 25, 30,
              36, 41.8, 47.3, 57.8],
        0.2: [2, 3, 4, 5, 4.6, 5.3, 6, 6.8, 7.6, 8.3, 9, 9.7, 10.5, 11.2, 11.9, 12.7, 13.4, 14, 14.8, 18.4, 21.9, 25.5,
              29, 35.7, 43, 49.6, 56.5, 70],
        0.3: [2, 3, 4, 5, 4.6, 5.3, 6.1, 6.9, 7.7, 8.4, 9.3, 9.9, 10.7, 11.4, 12.2, 12.8, 13.7, 14.3, 15.2, 18.9, 22.6,
              26.3, 30, 37.3, 44.5, 51.8, 58.6, 72.9],
        0.4: [2, 3, 4, 5, 4.6, 5.4, 6.1, 7, 7.7, 8.5, 9.3, 10, 10.8, 11.5, 12.2, 13, 13.8, 14.5, 15.3, 17.1, 22.8, 26.5,
              30.2, 37.7, 44.9, 52.3, 59.5, 74],
        0.5: [0, 3, 4, 5, 4.6424, 5.4226, 6.1336, 6.8984, 7.6758, 8.4566, 9.276, 9.9374, 10.787, 11.5666, 12.249,
              13.0488, 13.8072, 14.5978, 15.3866, 19.0646, 22.8862, 26.5628, 30.3552, 37.9332, 45.2712, 52.843, 60.217,
              74.9706],
        0.6: [0, 0, 4, 5, 4.666, 5.4168, 6.1932, 6.9122, 7.6814, 8.4992, 9.2658, 9.981, 10.7524, 11.5144, 12.2716,
              13.1054, 13.88, 14.5636, 15.3324, 19.1594, 22.9678, 26.773, 30.4384, 38.0788, 45.5374, 52.7334, 60.1636,
              74.9164]
    }
    indexvec = list(range(0, 4)) + list(range(7, 28))
    ed_vec = [realEDdic[0.1][i] for i in indexvec]
    print(ed_vec)

    indexdic = {0.1: list(range(0, 4)) + list(range(7, 28)),
                0.2: list(range(0, 4)) + list(range(6, 28)),
                0.3: list(range(0, 4)) + list(range(6, 28)),
                0.4: list(range(0, 4)) + list(range(5, 28)),
                0.5: list(range(1, 4)) + list(range(5, 28)),
                0.6: list(range(2, 4)) + list(range(5, 28))}
    N = 10000
    count = 0
    for C_G in C_G_vec:
        ave_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\GivenGeodistance\\ave_deviation_C_G{betan}Dis{geodesic_distance_AB}.txt".format(betan=C_G,geodesic_distance_AB = geodesic_distance_AB)
        ave_deviation_vec = np.loadtxt(ave_deviation_Name)
        std_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\GivenGeodistance\\std_deviation_C_G{betan}Dis{geodesic_distance_AB}.txt".format(betan=C_G,geodesic_distance_AB = geodesic_distance_AB)
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
    for count in range(len(C_G_vec)):
        beta = C_G_vec[count]
        x = list(range(2, 20)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
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


    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\LocalOptimumdiffc_G_distance{dis}.pdf".format(
        dis=geodesic_distance_AB)
    plt.savefig(picname,format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """
    plot_local_optimum_with_realED_diffCG_fix_distance_clean for different geodistance index: 
    
    """
    # # load data: the first time use it
    # #_______________________________________________________________________
    # distance_list = [[0.25, 0.25, 0.3, 0.3], [0.25, 0.25, 0.5, 0.5], [0.25, 0.25, 0.75, 0.75]]
    # for Geodistance_index in range(3):
    #     x_A = distance_list[Geodistance_index][0]
    #     y_A = distance_list[Geodistance_index][1]
    #     x_B = distance_list[Geodistance_index][2]
    #     y_B = distance_list[Geodistance_index][3]
    #     geodesic_distance_AB = x_B - x_A
    #     for cc in [0.1,0.2,0.3,0.4,0.5,0.6]:
    #         load_10000nodenetwork_results_fixdistance(cc,geodesic_distance_AB)
    # # _______________________________________________________________________

    plot_local_optimum_with_realED_diffCG_fix_distance_clean(2)


