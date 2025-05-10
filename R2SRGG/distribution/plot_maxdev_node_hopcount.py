# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 2024/9/8
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
import math

def plot_maxdev_node_hocount(N, ED, beta):
    SP_hopcount = []
    max_dev_node_hopcount = []

    # # small network only have 0
    # ExternalSimutime = 0
    # SPhopcount_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\SPhopcount_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
    #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    # SP_hopcount = np.loadtxt(SPhopcount_name, dtype=int)
    # max_dev_node_hopcount_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\max_dev_node_hopcount_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
    #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    # max_dev_node_hopcount = np.loadtxt(max_dev_node_hopcount_name, dtype=int)

    for ExternalSimutime in range(20):
        SPhopcount_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\formaxhop\\distancetosinglenode\\hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
        SP_hopcount_foronesimu = np.loadtxt(SPhopcount_name,dtype=int)
        SP_hopcount.extend(SP_hopcount_foronesimu)

        max_dev_node_hopcount_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\formaxhop\\distancetosinglenode\\max_dev_node_hopcount_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
        max_dev_node_hopcount_forone = np.loadtxt(max_dev_node_hopcount_name,dtype=int)
        max_dev_node_hopcount.extend(max_dev_node_hopcount_forone)

    SP_hopcount = [x for x in SP_hopcount if x!=1]
    counted = Counter(SP_hopcount)
    countdic = dict(counted)
    sorted_by_key_desc = dict(sorted(counted.items(), key=lambda x: x[0], reverse=False))

    print(sorted_by_key_desc)
    print(countdic[2])
    data_dic = {}


    for hp_index in range(len(SP_hopcount)):
        if SP_hopcount[hp_index] in data_dic.keys():
            data_dic[SP_hopcount[hp_index]].append(max_dev_node_hopcount[hp_index])
        else:
            data_dic[SP_hopcount[hp_index]] = [max_dev_node_hopcount[hp_index]]
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    # for key, values in data_dic.items():
    for key in [25]:
        values = data_dic[key]
        print(key)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        bins = np.arange(min(values) - 0.5, max(values) + 1.5, 1)  # 间隔为1的bin，确保每个柱中心对齐刻度线
        plt.hist(values, bins=bins,alpha=0.7, color=colors[3], edgecolor=colors[3],density=True)  # 绘制直方图
        plt.xticks(np.arange(min(values), max(values) + 1, 1))
        # plt.xlim([0, 1])
        plt.xticks([1, 5, 9, 13, 17,21,25])
        # plt.yticks([0, 10, 20, 30, 40, 50])

        plt.xlabel(r'x', fontsize=35)
        plt.ylabel(r'$f_{h_{iq}}(x)$', fontsize=35)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\formaxhop\\distancetosinglenode\\maxdev_nodehop{Nn}ED{EDn}Beta{betan}Hop{key}.png".format(
            Nn=N, EDn=ED, betan=beta,key = key)
        plt.savefig(picname, format='png', bbox_inches='tight', dpi=600)
        plt.show()
        # 清空图像，以免影响下一个图
        plt.close()


def plot_maxdev_node_hocount_for_twohop(N, ED, beta):
    SP_hopcount = []
    max_dev_node_hopcount = []

    # # small network only have 0
    # ExternalSimutime = 0
    # SPhopcount_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\SPhopcount_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
    #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    # SP_hopcount = np.loadtxt(SPhopcount_name, dtype=int)
    # max_dev_node_hopcount_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\max_dev_node_hopcount_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
    #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    # max_dev_node_hopcount = np.loadtxt(max_dev_node_hopcount_name, dtype=int)

    for ExternalSimutime in range(20):
        # small network only have 0
        SPhopcount_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\formaxhop\\sphopcountmax_dev_node_hopcount_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
        SP_hopcount_foronesimu = np.loadtxt(SPhopcount_name, dtype=int)
        SP_hopcount.extend(SP_hopcount_foronesimu)

        max_dev_node_hopcount_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\formaxhop\\max_dev_node_hopcount_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
        max_dev_node_hopcount_forone = np.loadtxt(max_dev_node_hopcount_name, dtype=int)
        max_dev_node_hopcount.extend(max_dev_node_hopcount_forone)

    counted = Counter(SP_hopcount)
    countdic = dict(counted)
    sorted_by_key_desc = dict(sorted(counted.items(), key=lambda x: x[0], reverse=False))
    data_dic = {}

    for hp_index in range(len(SP_hopcount)):
        if SP_hopcount[hp_index] in data_dic.keys():
            data_dic[SP_hopcount[hp_index]].append(max_dev_node_hopcount[hp_index])
        else:
            data_dic[SP_hopcount[hp_index]] = [max_dev_node_hopcount[hp_index]]
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4))
    key = 24
    values = data_dic[key]
    bins = np.arange(min(values) - 0.5, max(values) + 1.5, 1)  # 间隔为1的bin，确保每个柱中心对齐刻度线
    ax1.hist(values, bins=bins, alpha=0.7, color=colors[3], edgecolor=colors[3], density=True)  # 绘制直方图
    ax1.set_xticks(np.arange(min(values), max(values) + 1, 1))
    ax1.set_yticks([0, 0.1, 0.2])
    ax1.set_xticks([3,6,9, 12])
    ax1.tick_params(axis='x', labelsize=20)  # x 轴刻度字体大小
    ax1.tick_params(axis='y', labelsize=20)  # y 轴刻度字体大小
    ax1.set_xlabel(r'x', fontsize=24)
    ax1.set_ylabel(r'$f_{h(q_{max})}(x)$', fontsize=24)
    # plt.xticks(fontsize=28)
    # plt.yticks(fontsize=28)

    ax1.text(0.1, 0.8, r'$h_{ij} = 24$',
             transform=ax1.transAxes,
             fontsize=20,
             bbox=dict(facecolor='white', alpha=0.5))


    key = 25
    values = data_dic[key]
    bins = np.arange(min(values) - 0.5, max(values) + 1.5, 1)  # 间隔为1的bin，确保每个柱中心对齐刻度线
    ax2.hist(values, bins=bins, alpha=0.7, color=colors[3], edgecolor=colors[3], density=True)  # 绘制直方图
    ax2.set_xticks(np.arange(min(values), max(values) + 1, 1))
    ax2.set_xticks([3,6,9, 12])
    # ax2.set_yticks([0,0.1,0.2])
    ax2.tick_params(axis='x', labelsize=20)  # x 轴刻度字体大小
    ax2.tick_params(axis='y', labelleft=False)
    ax2.set_xlabel(r'x', fontsize=24)
    # ax2.text(0.2, max(values) * 0.9, r'$h_{ij} = 25$', fontsize=20)
    ax2.text(0.1, 0.8, r'$h_{ij} = 25$',
             transform=ax2.transAxes,
             fontsize=20,
             bbox=dict(facecolor='white', alpha=0.5))

    # plt.ylabel(r'$f_{h(q_{max})}(x)$', fontsize=35)
    # plt.xticks(fontsize=28)
    # plt.yticks(fontsize=28)

    plt.tight_layout()

    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\formaxhop\\maxdev_nodehop{Nn}ED{EDn}Beta{betan}Hop2425.png".format(
            Nn=N, EDn=ED, betan=beta, key=key)

    plt.savefig(picname,transparent=True, format='png', bbox_inches='tight', dpi=600)

    plt.show()


    # for key, values in data_dic.items():
    #     print(key)
    #     fig, ax = plt.subplots(figsize=(6, 3))
    #     # ax.spines['right'].set_visible(False)
    #     # ax.spines['top'].set_visible(False)
    #     bins = np.arange(min(values) - 0.5, max(values) + 1.5, 1)  # 间隔为1的bin，确保每个柱中心对齐刻度线
    #     plt.hist(values, bins=bins, alpha=0.7, color=colors[3], edgecolor=colors[3], density=True)  # 绘制直方图
    #     plt.xticks(np.arange(min(values), max(values) + 1, 1))
    #     # plt.xlim([0, 1])
    #     # plt.yticks([0, 5, 10, 15, 20, 25])
    #     # plt.yticks([0, 10, 20, 30, 40, 50])
    #
    #     plt.xlabel(r'x', fontsize=35)
    #     plt.ylabel(r'$f_{h(q_{max})}(x)$', fontsize=35)
    #     plt.xticks(fontsize=28)
    #     plt.yticks(fontsize=28)
    #     picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\formaxhop\\maxdev_nodehop{Nn}ED{EDn}Beta{betan}Hop{key}.png".format(
    #         Nn=N, EDn=ED, betan=beta, key=key)
    #     plt.show()
    #     plt.savefig(picname, format='png', bbox_inches='tight', dpi=600)
    #     # 清空图像，以免影响下一个图
    #     plt.close()


if __name__ == '__main__':
    # plot_distribution(50)
    # plot_maxdev_node_hocount(10000, 5, 4)
    # radius = math.sqrt(5/((10000-1)*math.pi))
    # print(0.52/radius)

    plot_maxdev_node_hocount(10000, 5, 4)
    # plot_maxdev_node_hocount_for_twohop(10000, 5, 4)