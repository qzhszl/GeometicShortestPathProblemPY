# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 2024/11/17
This file is for the minimum of the shortest path
we first prove the degree is because of the relative deviation decrease for the shortest path(about k and s)
"""
import json
import random
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import networkx as nx
from R2SRGG.R2SRGG import loadSRGGandaddnode, distR2
from SphericalSoftRandomGeomtricGraph import RandomGenerator
from main import find_k_connected_node_pairs
import pandas as pd


def filter_data_from_hop_geo_dev(N, ED, beta):
    """
    :param N: network size of SRGG
    :param ED: expected degree of the SRGG
    :param beta: temperature parameter of the SRGG
    :return: a dict: key is hopcount, value is list for relative deviation = ave deviation of the shortest paths nodes for a node pair / geodesic of the node pair
    """
    exemptionlist = []
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\" # FOR all sp nodes

    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\OneSP\\"  # FOR ONE sp

    hopcount_for_a_para_comb = np.array([])
    geodistance_for_a_para_comb = np.array([])
    ave_deviation_for_a_para_comb = np.array([])

    ExternalSimutimeNum = 5
    for ExternalSimutime in range(ExternalSimutimeNum):
        try:
            # load data for hopcount for all node pairs
            hopcount_vec_name = filefolder_name + "hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
            hopcount_for_a_para_comb_10times = np.loadtxt(hopcount_vec_name)
            hopcount_for_a_para_comb = np.hstack(
                (hopcount_for_a_para_comb, hopcount_for_a_para_comb_10times))

            # load data for geo distance for all node pairs
            geodistance_vec_name = filefolder_name + "length_geodesic_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
            geodistance_for_a_para_comb_10times = np.loadtxt(geodistance_vec_name)
            geodistance_for_a_para_comb = np.hstack(
                (geodistance_for_a_para_comb, geodistance_for_a_para_comb_10times))
            # load data for ave_deviation for all node pairs
            deviation_vec_name = filefolder_name + "ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
            ave_deviation_for_a_para_comb_10times = np.loadtxt(deviation_vec_name)

            ave_deviation_for_a_para_comb = np.hstack(
                (ave_deviation_for_a_para_comb, ave_deviation_for_a_para_comb_10times))
        except FileNotFoundError:
            exemptionlist.append((N, ED, beta, ExternalSimutime))

    hopcount_for_a_para_comb = hopcount_for_a_para_comb[hopcount_for_a_para_comb>1]
    counter = Counter(hopcount_for_a_para_comb)
    # print(counter)

    relative_dev = ave_deviation_for_a_para_comb/geodistance_for_a_para_comb
    hop_dev_dict = {}
    hop_relativedev_dict = {}
    for key in np.unique(hopcount_for_a_para_comb):
        hop_dev_dict[key] = ave_deviation_for_a_para_comb[hopcount_for_a_para_comb == key].tolist()
        hop_relativedev_dict[key] = relative_dev[hopcount_for_a_para_comb == key].tolist()
    return hop_dev_dict, hop_relativedev_dict


def plot_deviation_andrelativedev_vsdiffhop_withdiffk():
    """
    plot the relative deviation of the shortest path vs hopcount with different ED
    :return:
    """
    ave_dev_dict = {}
    std_dev_dict = {}

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']

    kvec = [6.0,8,10, 12,16, 20,27,34,44,56,72,92,118]   # for all shortest path
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\" # for all shortest path

    kvec = [11,21,30,39,50,91]  # for ONE SP
    kvec = [11, 21, 56, 107, 518, 1389] # for ONE SP beta = 4
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\OneSP\\"  # for ONE SP
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494', "#A2C7A4", "#9DB0C2", "#E3B6A4"]
    beta = 4
    exemptionlist = []
    N = 10000
    # fig, ax = plt.subplots(figsize=(12, 8))
    colorindex = 0

    ave_dev_dict ={}
    std_deviation_dict ={}
    hop_dict = {}
    count = 0
    for ED in kvec:

        print("ED:", ED)
        ave_deviation_vec = []
        std_deviation_vec = []
        ave_relative_deviation_vec = []
        std_relative_deviation_vec = []
        hop_vec =[]

        hop_dev_dict, hop_relativedev_dict = filter_data_from_hop_geo_dev(N, ED, beta)
        for hop, dev_data in hop_relativedev_dict.items():
            # print(f"{hop}DATA NUM:", len(dev_data))
            hop_vec.append(hop)
            ave_relative_deviation_vec.append(np.mean(dev_data))
            std_relative_deviation_vec.append(np.std(dev_data))


        for hop, dev_data in hop_dev_dict.items():
            # print(f"{hop}DATA NUM:", len(dev_data))
            ave_deviation_vec.append(np.mean(dev_data))
            std_deviation_vec.append(np.std(dev_data))

        # plt.errorbar(hop_vec, ave_relative_deviation_vec, std_relative_deviation_vec,
        #              label=f'ED: {ED}', linestyle="--", linewidth=3, elinewidth=1,
        #              capsize=5, marker='o', markersize=16,alpha=0.8)

        # a = 5
        # b = -10
        #
        #
        # Avec = [0.3,0.2,0.17,0.155,0.145,0.12]
        # tau_vec = [-0.5 for i in range(len(Avec))]
        # a_fit = Avec[count]
        # k_fit = tau_vec[count]
        # params = np.array([a_fit, k_fit])
        #
        #
        # # params, covariance = curve_fit(power_law, hop_vec[a:b],ave_relative_deviation_vec[a:b])
        # # # 获取拟合的参数
        # # a_fit, k_fit = params
        # # Avec.append(a_fit)
        # # tau_vec.append(k_fit)
        # print(f"拟合结果: a = {a_fit}, k = {k_fit}")
        # plt.plot(hop_vec[a:b], power_law(hop_vec[a:b], *params), linewidth=10, label=f'fit curve: $y={a_fit:.4f}x^{{{k_fit:.4f}}}$',
        #                  color='red')


        ave_dev_dict[ED] = ave_relative_deviation_vec
        std_deviation_dict[ED] = std_relative_deviation_vec
        hop_dict[ED] = hop_vec

        colorindex += 1
        count += 1

    # ave_df = pd.DataFrame(ave_dev_dict, index=kvec)
    # std_df = pd.DataFrame(std_dev_dict, index=kvec)
    # ave_df.index.name = None
    # std_df.index.name = None
    # ave_df.index = ave_df.index.astype(int)
    # std_df.index = std_df.index.astype(int)
    # print(ave_df)
    # ave_df = ave_df.dropna(how='all')
    # std_df = std_df.dropna(how = "all")
    # ave_df.to_csv(filefolder_name + f"RelativeDeviation_Average_beta{beta}.csv",index = True,encoding='utf-8')
    # std_df.to_csv(filefolder_name + f"RelativeDeviation_Std_beta{beta}.csv",index = True,encoding='utf-8')

    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('hopcount', fontsize=32)
    # plt.ylabel('Relative deviation', fontsize=32)
    # plt.xticks(fontsize=26)
    # plt.yticks(fontsize=26)
    # # plt.legend(fontsize=26, loc="best",ncol=2)
    # plt.tick_params(axis='both', which="both", length=6, width=1)
    #
    # picname = filefolder_name + "Relative_dev_vs_hop_beta{beta}.pdf".format(beta=beta)
    # # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    # plt.show()
    # plt.close()

    fig, ax = plt.subplots(figsize=(12, 8))
    count = 0
    for key, value in ave_dev_dict.items():
        ED = key
        plt.errorbar(hop_dict[ED], value, std_deviation_dict[ED],
                     label=f'ED: {ED}', linestyle="--", linewidth=3, elinewidth=1,
                     capsize=5, marker='o', markersize=16, color=colors[count])
        a = 10
        b = -20


        # Avec = [0.0035,0.0036,0.0039,0.0042,0.0044,0.0055]
        # tau_vec = [0.5 for i in range(len(Avec))]
        # a_fit = Avec[count]
        # k_fit = tau_vec[count]
        # params = np.array([a_fit, k_fit])
        #
        # hop_vec = hop_dict[ED]
        #
        # plt.plot(hop_vec[a:b], power_law(hop_vec[a:b], *params), linewidth=10, label=f'fit curve: $y={a_fit:.4f}x^{{{k_fit:.4f}}}$',
        #                  color='red')
        count +=1



    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Hopcount, $h$', fontsize=28)
    plt.ylabel(r'Relative deviation', fontsize=28)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=30, loc="best", ncol=2)
    plt.tick_params(axis='both', which="both", length=6, width=1)

    picname = filefolder_name + "Relative_Dev_vs_hop_beta{beta}.pdf".format(beta=beta)
    plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()

    # fig, ax1 = plt.subplots(figsize=(8, 6))
    # avg = kvec
    # avg = [int(i) for i in avg]
    # # 第一个y轴（左侧）
    # color1 = 'tab:blue'
    # ax1.set_xlabel('E[D]', fontsize=20)
    # ax1.set_ylabel(r'$\tau$', color=color1, fontsize=20)
    # ax1.plot(avg, np.abs(tau_vec), 'o-', color=color1, label=r'$\tau$')
    # ax1.tick_params(axis='y', labelcolor=color1)
    #
    # # 第二个y轴（右侧）
    # ax2 = ax1.twinx()
    # color2 = 'tab:red'
    # ax2.set_ylabel('A', color=color2, fontsize=20)
    # ax2.plot(avg, Avec, 's--', color=color2, label='A')
    # ax2.tick_params(axis='y', labelcolor=color2)
    #
    # # 设置对数坐标轴
    # ax2.set_xscale('log')
    # ax1.set_yscale('log')
    # ax2.set_yscale('log')
    # a =1
    # params, covariance = curve_fit(power_law, avg[a:], Avec[a:])
    # # 获取拟合的参数
    # a_fit, k_fit = params
    # print(f"拟合结果: a = {a_fit}, k = {k_fit}")
    # ax2.plot(avg[a:], power_law(avg[a:], *params), linewidth=5, label=f'fit curve: $y={a_fit:.4f}x^{{{k_fit:.4f}}}$',
    #          color='green')
    #
    #
    #
    # # 添加辅助text信息
    # plt.text(0.35, 0.6, r'$f(h) = A(h)\cdot <k>^{-\tau}$',
    #          transform=ax1.transAxes,
    #          fontsize=20,
    #          bbox=dict(facecolor='white', alpha=0.5))
    #
    # # 添加legend (需要合并两个轴的legend)
    # lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=20)
    #
    # # 标题和网格
    # # plt.title('Dual Y-axis Log-Log Plot')
    # # ax1.grid(True, which='both', ls='--', alpha=0.5)
    #
    # plt.tight_layout()
    # plt.show()


# def plot_deviation_andrelativedev_vsdiffhop_withdiffk():
#     """
#     plot the relative deviation of the shortest path vs hopcount with different ED
#     :return:
#     """
#     ave_dev_dict = {}
#     std_dev_dict = {}
#
#     colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
#
#     kvec = [6.0,8,10, 12,16, 20,27,34,44,56,72,92,118]   # for all shortest path
#     filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\" # for all shortest path
#
#     kvec = [11,21,30,39,50,91]  # for ONE SP
#     filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\OneSP\\"  # for ONE SP
#
#     beta = 128
#     exemptionlist = []
#     N = 10000
#     # fig, ax = plt.subplots(figsize=(12, 8))
#     colorindex = 0
#
#     ave_dev_dict ={}
#     std_deviation_dict ={}
#     hop_dict = {}
#     count = 0
#     for ED in kvec:
#
#         print("ED:", ED)
#         ave_deviation_vec = []
#         std_deviation_vec = []
#         ave_relative_deviation_vec = []
#         std_relative_deviation_vec = []
#         hop_vec =[]
#
#         hop_dev_dict, hop_relativedev_dict = filter_data_from_hop_geo_dev(N, ED, beta)
#         for hop, dev_data in hop_relativedev_dict.items():
#             # print(f"{hop}DATA NUM:", len(dev_data))
#             hop_vec.append(hop)
#             ave_relative_deviation_vec.append(np.mean(dev_data))
#             std_relative_deviation_vec.append(np.std(dev_data))
#
#
#         for hop, dev_data in hop_dev_dict.items():
#             # print(f"{hop}DATA NUM:", len(dev_data))
#             ave_deviation_vec.append(np.mean(dev_data))
#             std_deviation_vec.append(np.std(dev_data))
#
#         # plt.errorbar(hop_vec, ave_relative_deviation_vec, std_relative_deviation_vec,
#         #              label=f'ED: {ED}', linestyle="--", linewidth=3, elinewidth=1,
#         #              capsize=5, marker='o', markersize=16,alpha=0.8)
#
#         # a = 5
#         # b = -10
#         #
#         #
#         # Avec = [0.3,0.2,0.17,0.155,0.145,0.12]
#         # tau_vec = [-0.5 for i in range(len(Avec))]
#         # a_fit = Avec[count]
#         # k_fit = tau_vec[count]
#         # params = np.array([a_fit, k_fit])
#         #
#         #
#         # # params, covariance = curve_fit(power_law, hop_vec[a:b],ave_relative_deviation_vec[a:b])
#         # # # 获取拟合的参数
#         # # a_fit, k_fit = params
#         # # Avec.append(a_fit)
#         # # tau_vec.append(k_fit)
#         # print(f"拟合结果: a = {a_fit}, k = {k_fit}")
#         # plt.plot(hop_vec[a:b], power_law(hop_vec[a:b], *params), linewidth=10, label=f'fit curve: $y={a_fit:.4f}x^{{{k_fit:.4f}}}$',
#         #                  color='red')
#
#
#         ave_dev_dict[ED] = ave_deviation_vec
#         std_deviation_dict[ED] = std_deviation_vec
#         hop_dict[ED] = hop_vec
#
#         colorindex += 1
#         count += 1
#
#     # ave_df = pd.DataFrame(ave_dev_dict, index=kvec)
#     # std_df = pd.DataFrame(std_dev_dict, index=kvec)
#     # ave_df.index.name = None
#     # std_df.index.name = None
#     # ave_df.index = ave_df.index.astype(int)
#     # std_df.index = std_df.index.astype(int)
#     # print(ave_df)
#     # ave_df = ave_df.dropna(how='all')
#     # std_df = std_df.dropna(how = "all")
#     # ave_df.to_csv(filefolder_name + f"RelativeDeviation_Average_beta{beta}.csv",index = True,encoding='utf-8')
#     # std_df.to_csv(filefolder_name + f"RelativeDeviation_Std_beta{beta}.csv",index = True,encoding='utf-8')
#
#     # plt.xscale('log')
#     # plt.yscale('log')
#     # plt.xlabel('hopcount', fontsize=32)
#     # plt.ylabel('Relative deviation', fontsize=32)
#     # plt.xticks(fontsize=26)
#     # plt.yticks(fontsize=26)
#     # # plt.legend(fontsize=26, loc="best",ncol=2)
#     # plt.tick_params(axis='both', which="both", length=6, width=1)
#     #
#     # picname = filefolder_name + "Relative_dev_vs_hop_beta{beta}.pdf".format(beta=beta)
#     # # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
#     # plt.show()
#     # plt.close()
#
#     fig, ax = plt.subplots(figsize=(12, 8))
#     count = 0
#     for key, value in ave_dev_dict.items():
#         ED = key
#         plt.errorbar(hop_dict[ED], value, std_deviation_dict[ED],
#                      label=f'ED: {ED}', linestyle="--", linewidth=3, elinewidth=1,
#                      capsize=5, marker='o', markersize=16,alpha = 0.5)
#         a = 10
#         b = -20
#
#
#         Avec = [0.0035,0.0036,0.0039,0.0042,0.0044,0.0055]
#         tau_vec = [0.5 for i in range(len(Avec))]
#         a_fit = Avec[count]
#         k_fit = tau_vec[count]
#         params = np.array([a_fit, k_fit])
#
#         hop_vec = hop_dict[ED]
#
#         plt.plot(hop_vec[a:b], power_law(hop_vec[a:b], *params), linewidth=10, label=f'fit curve: $y={a_fit:.4f}x^{{{k_fit:.4f}}}$',
#                          color='red')
#         count +=1
#
#
#
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.xlabel('hopcount', fontsize=32)
#     plt.ylabel('Deviation', fontsize=32)
#     plt.xticks(fontsize=26)
#     plt.yticks(fontsize=26)
#     # plt.legend(fontsize=26, loc="best", ncol=2)
#     plt.tick_params(axis='both', which="both", length=6, width=1)
#
#     picname = filefolder_name + "Dev_vs_hop_beta{beta}.pdf".format(beta=beta)
#     # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
#     plt.show()
#     plt.close()
#
#     fig, ax1 = plt.subplots(figsize=(8, 6))
#     avg = kvec
#     avg = [int(i) for i in avg]
#     # 第一个y轴（左侧）
#     color1 = 'tab:blue'
#     ax1.set_xlabel('E[D]', fontsize=20)
#     ax1.set_ylabel(r'$\tau$', color=color1, fontsize=20)
#     ax1.plot(avg, np.abs(tau_vec), 'o-', color=color1, label=r'$\tau$')
#     ax1.tick_params(axis='y', labelcolor=color1)
#
#     # 第二个y轴（右侧）
#     ax2 = ax1.twinx()
#     color2 = 'tab:red'
#     ax2.set_ylabel('A', color=color2, fontsize=20)
#     ax2.plot(avg, Avec, 's--', color=color2, label='A')
#     ax2.tick_params(axis='y', labelcolor=color2)
#
#     # 设置对数坐标轴
#     ax2.set_xscale('log')
#     ax1.set_yscale('log')
#     ax2.set_yscale('log')
#     a =1
#     params, covariance = curve_fit(power_law, avg[a:], Avec[a:])
#     # 获取拟合的参数
#     a_fit, k_fit = params
#     print(f"拟合结果: a = {a_fit}, k = {k_fit}")
#     ax2.plot(avg[a:], power_law(avg[a:], *params), linewidth=5, label=f'fit curve: $y={a_fit:.4f}x^{{{k_fit:.4f}}}$',
#              color='green')
#
#
#
#     # 添加辅助text信息
#     plt.text(0.35, 0.6, r'$f(h) = A(h)\cdot <k>^{-\tau}$',
#              transform=ax1.transAxes,
#              fontsize=20,
#              bbox=dict(facecolor='white', alpha=0.5))
#
#     # 添加legend (需要合并两个轴的legend)
#     lines1, labels1 = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=20)
#
#     # 标题和网格
#     # plt.title('Dual Y-axis Log-Log Plot')
#     # ax1.grid(True, which='both', ls='--', alpha=0.5)
#
#     plt.tight_layout()
#     plt.show()




def power_law(x, a, k):
    return a * x ** k


def plot_concave_deviation_andrelativedev_vsdiffk_diffhop():
    """
    plot the concave
    :return:
    """
    ave_dev_dict = {}
    std_dev_dict = {}

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']

    kvec = [6.0,8,10, 12,16, 20,27,34,44,56,72,92,118]   # for all shortest path
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\" # for all shortest path

    kvec = [11,21,30,39,50,91]  # for ONE SP
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\OneSP\\"  # for ONE SP

    beta = 128
    exemptionlist = []
    N = 10000
    colorindex = 0

    ave_dev_dict ={}
    std_deviation_dict ={}
    hop_dict = {}
    diff_relative_dev_dict = {}

    for ED in kvec:
        print("ED:", ED)
        ave_deviation_vec = []
        std_deviation_vec = []
        ave_relative_deviation_vec = []
        std_relative_deviation_vec = []
        hop_vec =[]

        hop_dev_dict, hop_relativedev_dict = filter_data_from_hop_geo_dev(N, ED, beta)
        for hop, dev_data in hop_relativedev_dict.items():
            # print(f"{hop}DATA NUM:", len(dev_data))
            hop_vec.append(hop)
            ave_relative_deviation_vec.append(np.mean(dev_data))
            std_relative_deviation_vec.append(np.std(dev_data))

        # difference of ave_relative_deviation_vec vec[i+1]-vec[i]
        diff_relative_dev_vec = [ave_relative_deviation_vec[i+1] - ave_relative_deviation_vec[i] for i in range(len(ave_relative_deviation_vec)-1)]


        for hop, dev_data in hop_dev_dict.items():
            # print(f"{hop}DATA NUM:", len(dev_data))
            ave_deviation_vec.append(np.mean(dev_data))
            std_deviation_vec.append(np.std(dev_data))

        # plt.errorbar(hop_vec, ave_relative_deviation_vec, std_relative_deviation_vec,
        #              label=f'ED: {ED}', linestyle="--", linewidth=3, elinewidth=1,
        #              capsize=5, marker='o', markersize=16)

        ave_dev_dict[ED] = ave_deviation_vec
        std_deviation_dict[ED] = std_deviation_vec
        hop_dict[ED] = hop_vec

        diff_relative_dev_dict[ED] = diff_relative_dev_vec



        colorindex += +1

    # ave_df = pd.DataFrame(ave_dev_dict, index=kvec)
    # std_df = pd.DataFrame(std_dev_dict, index=kvec)
    # ave_df.index.name = None
    # std_df.index.name = None
    # ave_df.index = ave_df.index.astype(int)
    # std_df.index = std_df.index.astype(int)
    # print(ave_df)
    # ave_df = ave_df.dropna(how='all')
    # std_df = std_df.dropna(how = "all")
    # ave_df.to_csv(filefolder_name + f"RelativeDeviation_Average_beta{beta}.csv",index = True,encoding='utf-8')
    # std_df.to_csv(filefolder_name + f"RelativeDeviation_Std_beta{beta}.csv",index = True,encoding='utf-8')



    fig, ax = plt.subplots(figsize=(12, 8))
    for key, value in diff_relative_dev_dict.items():
        ED = key
        plt.plot(value,
                     label=f'ED: {ED}', linestyle="--", linewidth=3, marker='o', markersize=16)

    with open("toxinhan.json", "w") as f:
        json.dump(diff_relative_dev_dict, f)

    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('hopcount diff', fontsize=32)
    plt.ylabel('Deviation diff', fontsize=32)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.legend(fontsize=26, loc="best", ncol=2)
    plt.tick_params(axis='both', which="both", length=6, width=1)

    picname = filefolder_name + "Dev_vs_hop_beta{beta}.pdf".format(beta=beta)
    # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()



def plot_deviation_vs_hop_withED():
    """
    plot the deviation of the shortest path vs hopcount with different ED
    :return:
    """
    ave_dev_dict = {}
    std_dev_dict = {}

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494',"#A2C7A4","#9DB0C2","#E3B6A4"]

    # kvec = [6.0,8,10, 12,16, 20,27,34,44,56,72,92,118]   # for all shortest path
    # filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\" # for all shortest path

    kvec = [11, 21, 56,107,518,1389]  # for ONE SP
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\OneSP\\"  # for ONE SP

    beta = 4
    exemptionlist = []
    N = 10000
    # fig, ax = plt.subplots(figsize=(12, 8))
    colorindex = 0

    ave_dev_dict ={}
    std_deviation_dict ={}
    hop_dict = {}
    count = 0
    for ED in kvec:
        print("ED:", ED)
        ave_deviation_vec = []
        std_deviation_vec = []
        ave_relative_deviation_vec = []
        std_relative_deviation_vec = []
        hop_vec =[]

        hop_dev_dict, hop_relativedev_dict = filter_data_from_hop_geo_dev(N, ED, beta)
        # for hop, dev_data in hop_relativedev_dict.items():
        #     # print(f"{hop}DATA NUM:", len(dev_data))
        #     hop_vec.append(hop)
        #     ave_relative_deviation_vec.append(np.mean(dev_data))
        #     std_relative_deviation_vec.append(np.std(dev_data))

        for hop, dev_data in hop_dev_dict.items():
            # print(f"{hop}DATA NUM:", len(dev_data))
            if len(dev_data)>500:
                hop_vec.append(hop)
                ave_deviation_vec.append(np.mean(dev_data))
                std_deviation_vec.append(np.std(dev_data))

        ave_dev_dict[ED] = ave_deviation_vec
        std_deviation_dict[ED] = std_deviation_vec
        hop_dict[ED] = hop_vec

        colorindex += 1
        count += 1

    fig, ax = plt.subplots(figsize=(12, 8))
    count = 0
    for key, value in ave_dev_dict.items():
        ED = key
        plt.errorbar(hop_dict[ED], value, std_deviation_dict[ED],
                     label=f'ED: {ED}', linestyle="--", linewidth=3, elinewidth=1,
                     capsize=5, marker='o', markersize=16, color = colors[count])
        count +=1


    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Hopcount, $h$', fontsize=28)
    plt.ylabel(r'Average deviation, $\langle d \rangle$', fontsize=28)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=30, loc="best", ncol=2)
    plt.tick_params(axis='both', which="both", length=6, width=1)

    picname = filefolder_name + "Dev_vs_hop_diffEDbeta{beta}.pdf".format(beta=beta)
    plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()




if __name__ == '__main__':
    """
    function 1: plot relative deviation vs diff hop with differnet given ED and do curve fit
    """
    plot_deviation_andrelativedev_vsdiffhop_withdiffk()
    """
    function 2: plot the concave
    """
    # plot_concave_deviation_andrelativedev_vsdiffk_diffhop()
    """
    function 3: plot deviation vs diff hop given diff ED
    """
    # plot_deviation_vs_hop_withED()