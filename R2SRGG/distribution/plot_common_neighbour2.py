# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 13-11-2024
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from R2SRGG.R2SRGG import loadSRGGandaddnode, distR2
from collections import defaultdict
from scipy.optimize import curve_fit
import math
import json
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from collections import Counter
import pandas as pd


def load_10000nodenetwork_results_perpendicular(beta):
    # Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    kvec = [2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    # kvec = [2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848,1389]
    # betavec = [2.1, 4, 8, 16, 32, 64, 128]
    filefolder_name2 = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\neighbour_distance\\commonneighbourmodel\\"
    exemptionlist =[]
    for N in [10000]:
        ave_deviation_vec = []
        std_deviation_vec = []
        real_ave_degree_vec =[]
        for beta in [beta]:
            for ED in kvec:
                try:
                    deviations_name = filefolder_name2 + "common_neigthbour_deviationlist_N{Nn}ED{EDn}beta{betan}xA{xA}yA{yA}xB{xB}yB{yB}Simu{simu}.json".format(
                        Nn=N, EDn=ED, betan=beta, xA=-0.005, yA=0, xB=0.005, yB=0, simu=0)
                    with open(deviations_name, 'r') as file:
                        deviations_dict = {int(k): v for k, v in json.load(file).items()}

                    ave_deviation_for_a_para_comb = []
                    for neighbour_dev in [dev for dev in deviations_dict.values()]:
                        ave_deviation_for_a_para_comb = ave_deviation_for_a_para_comb+neighbour_dev
                except FileNotFoundError:
                    exemptionlist.append((N, ED, beta))

                ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
                std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))
    print(exemptionlist)
    real_ave_degree_Name = filefolder_name2 + "real_ave_degree_Beta{betan}.txt".format(betan=beta)
    np.savetxt(real_ave_degree_Name, real_ave_degree_vec)
    ave_deviation_Name = filefolder_name2 + "ave_deviation_Beta{betan}.txt".format(betan=beta)
    np.savetxt(ave_deviation_Name, ave_deviation_vec)
    std_deviation_Name = filefolder_name2 + "std_deviation_Beta{betan}.txt".format(betan=beta)
    np.savetxt(std_deviation_Name, std_deviation_vec)
    return real_ave_degree_vec, ave_deviation_vec,std_deviation_vec, exemptionlist

def plot_common_neighbour_deviation_vs_inputED_with_beta(beta):
    """
    the x-axis is the input expected degree(avg), the y-axis is the average deviation of common neighbours, different line is different beta
    N  = 10000 NODES
    when use this function, use load_10000nodenetwork_results_clean(beta) before
    :return:
    """

    N = 10000
    kvec = [2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    beta_vec = [beta]
    Geodistance_index = 0
    distance_list = [[0.491, 0.5, 0.509, 0.5], [0.25, 0.25, 0.3, 0.3], [0.25, 0.25, 0.3, 0.3], [0.25, 0.25, 0.5, 0.5],
                     [0.25, 0.25, 0.75, 0.75]]
    x_A = distance_list[Geodistance_index][0]
    y_A = distance_list[Geodistance_index][1]
    x_B = distance_list[Geodistance_index][2]
    y_B = distance_list[Geodistance_index][3]
    geodesic_distance_AB = round(x_B - x_A, 2)
    ave_deviation_dict = {}
    std_deviation_dict = {}
    count = 0
    for beta in beta_vec:
        ave_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\neighbour_distance\\commonneighbourmodel\\ave_deviation_beta{beta}.txt".format(
            beta=beta)
        std_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\neighbour_distance\\commonneighbourmodel\\std_deviation_beta{beta}.txt".format(
            beta=beta)
        ave_deviation_vec = np.loadtxt(ave_deviation_Name)
        std_deviation_vec = np.loadtxt(std_deviation_Name)
        # real_ave_degree_vec, ave_deviation_vec, std_deviation_vec = load_10000nodenetwork_results(beta)

        ave_deviation_dict[count] = ave_deviation_vec
        std_deviation_dict[count] = std_deviation_vec
        count = count+1

    lengend = [r"$\beta=2^2$"]
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = [[0, 0.4470, 0.7410],
              [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250],
              [0.4940, 0.1840, 0.5560],
              [0.4660, 0.6740, 0.1880],
              [0.3010, 0.7450, 0.9330]]

    # for count in range(len(beta_vec)):
    for count in [0]:
        beta = beta_vec[count]
        x = kvec
        y = ave_deviation_dict[count]
        error = std_deviation_dict[count]
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
    ky = []
    for k in x:
        ky.append(0.007 * k ** 0.4994)
    plt.plot(x, power_law(x, *params),linewidth=5, label=f'fit curve: $y={a_fit:.6f}x^{{{k_fit:.4f}}}$', color='red')
    plt.plot(x, ky, linewidth=5, label=f'analytic dev: $y=0.001799x^{{0.4994}}$', color='green')

    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('Input avg',fontsize = 26)
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

    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\neighbour_distance\\commonneighbourmodel\\CommonNeighbourDeviationvsEDwithbeta{betan}_curvefitloglog.pdf".format(
        betan=beta)
    plt.savefig(picname,format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()

def power_law(x, a, k):
    return a * x ** k


def load_10000nodenetwork_results_perpendicular_withED(ED):
    # Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    betavec = [2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64, 128, 256, 512]
    # kvec = [2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848,1389]
    # betavec = [2.1, 4, 8, 16, 32, 64, 128]
    filefolder_name2 = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\neighbour_distance\\commonneighbourmodel\\"
    exemptionlist =[]
    for N in [10000]:
        ave_deviation_vec = []
        std_deviation_vec = []
        for beta in betavec:
            for ED in [ED]:
                try:
                    deviations_name = filefolder_name2 + "common_neigthbour_deviationlist_N{Nn}ED{EDn}beta{betan}xA{xA}yA{yA}xB{xB}yB{yB}Simu{simu}.json".format(
                        Nn=N, EDn=ED, betan=beta, xA=-0.005, yA=0, xB=0.005, yB=0, simu=0)
                    with open(deviations_name, 'r') as file:
                        deviations_dict = {int(k): v for k, v in json.load(file).items()}

                    ave_deviation_for_a_para_comb = []
                    for neighbour_dev in [dev for dev in deviations_dict.values()]:
                        ave_deviation_for_a_para_comb = ave_deviation_for_a_para_comb+neighbour_dev
                except FileNotFoundError:
                    exemptionlist.append((N, ED, beta))

                ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
                std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))
    print(exemptionlist)
    ave_deviation_Name = filefolder_name2 + "ave_deviation_ED{ED}.txt".format(ED=ED)
    np.savetxt(ave_deviation_Name, ave_deviation_vec)
    std_deviation_Name = filefolder_name2 + "std_deviation_ED{ED}.txt".format(ED=ED)
    np.savetxt(std_deviation_Name, std_deviation_vec)
    return ave_deviation_vec,std_deviation_vec, exemptionlist

def plot_common_neighbour_deviation_vs_beta_with_ED(ED):
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

    betavec = [2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64, 128, 256, 512]

    distance_list = [[0.491, 0.5, 0.509, 0.5], [0.25, 0.25, 0.3, 0.3], [0.25, 0.25, 0.3, 0.3], [0.25, 0.25, 0.5, 0.5],
                     [0.25, 0.25, 0.75, 0.75]]

    N = 10000
    count = 0
    for ED in [ED]:
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

    # legend = [r"$E[D]=2$",r"$E[D]=5$",r"$E[D]=10$",r"$E[D]=20$",r"$E[D]=100$"]
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = [[0, 0.4470, 0.7410],
              [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250],
              [0.4940, 0.1840, 0.5560],
              [0.4660, 0.6740, 0.1880],
              [0.3010, 0.7450, 0.9330]]
    for count in range(len(betavec)):
    # for count in [5]:
        x = betavec
        # print(len(x))
        # x = x[0:cuttail[N_index]]
        y = ave_deviation_dict[count]
        # y = y[0:cuttail[N_index]]
        error = std_deviation_dict[count]
        # error = error[0:cuttail[N_index]]
        plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o',markersize=16, label=legend[count], color=colors[count])

    x = betavec
    y = [0.0157788349590166,
         0.0187287447381409,
         0.0201260746652333,
         0.0209308580521077,
         0.0214569115894703,
         0.0218339894101099,
         0.0221234572787197,
         0.0223572760944025,
         0.0225534217514617,
         0.0227226734117023,
         0.0233403946530454,
         0.0237612260103278,
         0.0240802622456526,
         0.0243344474884966,
         0.0245430536807898,
         0.0247178034369095,
         0.0253671940890477,
         0.0259649205336020,
         0.0262150607642736,
         0.0262871130529518,
         0.0263053868898680,
         0.0263099642618328]
    plt.plot(x, y,linewidth=5, label=f'fit curve: avg = 10$', color='red')

    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel(r'$\beta$',fontsize = 26)
    plt.ylabel('Average deviation',fontsize = 26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=20)
    plt.tick_params(axis='both', which="both",length=6, width=1)


    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\neighbour_distance\\commonneighbourmodel\\CommonNeighbourDeviationvsbetawithdiffED_cleanloglog.pdf"

    # picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\CommonNeighbourDeviationvsEDwithdiffED{ED}_curvefitloglog.pdf".format(
    #     betan=ED)
    plt.savefig(picname,format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # load_10000nodenetwork_results_perpendicular(4)
    # plot_common_neighbour_deviation_vs_inputED_with_beta(4)


    load_10000nodenetwork_results_perpendicular_withED(10)
    plot_common_neighbour_deviation_vs_beta_with_ED(4)
