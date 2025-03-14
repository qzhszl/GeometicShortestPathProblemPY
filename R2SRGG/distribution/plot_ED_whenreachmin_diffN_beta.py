# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 03-13-2025
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import curve_fit

from R2SRGG.R2SRGG import loadSRGGandaddnode
from collections import defaultdict
import math
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

"""
In this figure, we would like to inverstigate how the expected degree ED where dev reaches minimum changes with different N and 
beta. 
"""

def load_network_results(N, beta):
    if N ==100:
        kvec = [10, 12, 15, 18, 22, 27, 33, 40, 49, 60, 73, 89] # for k = 100
    elif N == 1000:
        kvec = [2, 3, 4, 5, 7, 10, 14, 20, 27, 38, 53, 73, 101, 140, 195, 270, 375, 519]  # for N = 1000
    elif N == 10000:
        kvec = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    else:
        kvec = [2, 4, 6, 11, 20, 34, 61, 108, 190, 336, 595, 1051, 1857, 3282, 5800, 10250, 18116, 32016, 56582,
                99999]  # for N = 10^5
    exemptionlist = []
    for N in [N]:
        ave_deviation_vec = []
        std_deviation_vec = []
        # real_ave_degree_vec = []
        for ED in kvec:
            if N<500:
                try:
                    ave_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\ave_deviation_N{Nn}ED{EDn}Beta{betan}.txt".format(
                        Nn=N, EDn=ED, betan=beta)
                    ave_deviation_for_a_para_comb = np.loadtxt(ave_deviation_name)
                    ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
                    std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))
                except FileNotFoundError:
                    exemptionlist.append((N, ED, beta))
            else:
                if N == 10000:
                    ave_deviation_for_a_para_comb = []
                    if beta ==4:
                        simutime  =100
                    else:
                        simutime = 20
                    for ExternalSimutime in range(simutime):
                        try:
                            deviation_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                            ave_deviation_for_a_para_comb_10times = np.loadtxt(deviation_vec_name)
                            ave_deviation_for_a_para_comb.extend(ave_deviation_for_a_para_comb_10times)
                        except FileNotFoundError:
                            exemptionlist.append((N, ED, beta, ExternalSimutime))

                    ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
                    std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))
                else:
                    ave_deviation_for_a_para_comb = []
                    for ExternalSimutime in range(5):
                        try:
                            deviation_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\OneSP\\ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                            ave_deviation_for_a_para_comb_10times = np.loadtxt(deviation_vec_name)
                            ave_deviation_for_a_para_comb.extend(ave_deviation_for_a_para_comb_10times)
                        except FileNotFoundError:
                            exemptionlist.append((N, ED, beta, ExternalSimutime))
                    ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
                    std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))
    print(exemptionlist)
    return ave_deviation_vec, std_deviation_vec


def plot_dev_vs_ED_diffN(beta):
    """
    THIS ONE and the later one plot HOW the local optimum ED changed with different CC
    :return:
    """
    ave_deviation_dict = {}
    std_deviation_dict = {}
    Nvec = [100, 1000, 10000,100000]
    for N in Nvec[0:4]:
        ave_deviation_vec, std_deviation_vec = load_network_results(N, beta)
        ave_deviation_dict[N] = ave_deviation_vec
        std_deviation_dict[N] = std_deviation_vec

    local_optimum = []

    legend = [r"$N=10^2$", r"$N=10^3$", r"$N=10^4$",r"$N=10^5$"]
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = [[0, 0.4470, 0.7410],
              [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250],
              [0.4940, 0.1840, 0.5560],
              [0.4660, 0.6740, 0.1880]]
    peakcut =[1,5,1,3]

    for N_index in range(len(Nvec)):
        N = Nvec[N_index]
        if N == 100:
            kvec = [10, 12, 15, 18, 22, 27, 33, 40, 49, 60, 73, 89]  # for k = 100
        elif N == 1000:
            kvec = [2, 3, 4, 5, 7, 10, 14, 20, 27, 38, 53, 73, 101, 140, 195, 270, 375, 519]  # for N = 1000
        elif N == 10000:
            kvec = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
        else:
            kvec = [2, 4, 6, 11, 20, 34, 61, 108, 190, 336, 595, 1051, 1857, 3282, 5800, 10250, 18116, 32016, 56582,
                    99999]  # for N = 10^5
        x = kvec
        y = ave_deviation_dict[N]
        x = [x_val for x_val, y_val in zip(x, y) if y_val > 0.01]
        y = [y_val for y_val in y if y_val > 0.01]
        error = std_deviation_dict[N]
        error = [x_val for x_val, y_val in zip(error, y) if y_val > 0.01]

        plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=16,
                     label=legend[N_index], color=colors[N_index])
        # 找到峰值后最低点的坐标
        peak_index = np.argmax(y[0:peakcut[N_index]])
        post_peak_y = y[peak_index:]
        post_peak_min_index = peak_index + np.argmin(post_peak_y)
        post_peak_min_x = x[post_peak_min_index]
        local_optimum.append(post_peak_min_x)
        post_peak_min_y = y[post_peak_min_index]

        # 标出最低点
        plt.plot(post_peak_min_x, post_peak_min_y, 'o', color=colors[N_index], markersize=30, markerfacecolor="none")

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # plt.ylim(0, 0.30)
    # plt.yticks([0, 0.1, 0.2, 0.3])

    plt.xscale('log')
    plt.xlabel('E[D]', fontsize=26)
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
    plt.close()
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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # STEP 1
    betavec = [2.2, 4, 8, 16, 32, 64, 128]
    # betavec = [2.2, 4, 8]
    beta = 4
    local_optimum = plot_dev_vs_ED_diffN(beta)
    print(local_optimum)

