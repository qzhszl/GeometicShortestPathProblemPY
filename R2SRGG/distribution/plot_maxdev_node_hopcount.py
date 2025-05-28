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

from matplotlib.ticker import FormatStrFormatter


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
    if N == 10000:
        for ExternalSimutime in range(1,20):
            SPhopcount_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\formaxhop\\distancetosinglenode\\hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
            SP_hopcount_foronesimu = np.loadtxt(SPhopcount_name,dtype=int)
            SP_hopcount.extend(SP_hopcount_foronesimu)

            max_dev_node_hopcount_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\formaxhop\\distancetosinglenode\\max_dev_node_hopcount_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
            max_dev_node_hopcount_forone = np.loadtxt(max_dev_node_hopcount_name,dtype=int)
            max_dev_node_hopcount.extend(max_dev_node_hopcount_forone)
    else:
        for ExternalSimutime in range(1):
            SPhopcount_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\formaxhop\\distancetosinglenode\\hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
            SP_hopcount_foronesimu = np.loadtxt(SPhopcount_name, dtype=int)
            SP_hopcount.extend(SP_hopcount_foronesimu)

            max_dev_node_hopcount_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\formaxhop\\distancetosinglenode\\max_dev_node_hopcount_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
            max_dev_node_hopcount_forone = np.loadtxt(max_dev_node_hopcount_name, dtype=int)
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
        print([int(i) for i in values])
        plt.hist(values, bins=bins,alpha=0.7, color=colors[3], edgecolor=colors[3],density=True)  # 绘制直方图
        plt.xticks(np.arange(min(values), max(values) + 1, 1))
        # plt.xlim([0, 1])
        # plt.xticks([1, 5, 9, 13, 17,21,25]) # for N=10000, ED5 BETA4 HOP=25
        # plt.xticks([1, 5, 9, 13]) # for N=1000, ED5 BETA4 HOP=13
        # plt.xticks([1, 5, 9, 13, 17]) # for N=1000, ED5 BETA128 HOP=17

        # plt.xticks([1, 4, 7, 10]) # for N=1000, ED5 BETA4 HOP=12

        # plt.yticks([0, 0.03,0.06, 0.09,0.12,0.15, 0.18])  #h = 12

        # plt.yticks([0, 0.2,0.4,0.6,0.8])  #h = 4
        # plt.ylim([0, 0.8])

        # plt.yticks([0, 0.03, 0.06, 0.09, 0.12, 0.15])  # h = 13
        # plt.ylim([0, 0.18])

        text = rf"$h = {key}$"
        plt.text(
            0.85, 0.85,  # 文本位置（轴坐标，0.5 表示图中央，1.05 表示轴上方）
            text,
            transform=ax.transAxes,  # 使用轴坐标
            fontsize=30,  # 字体大小
            ha='center',  # 水平居中对齐
            va='bottom'  # 垂直对齐方式
        )

        plt.xlabel(r'x', fontsize=35)
        plt.ylabel(r'$f_{h_{iq}}(x)$', fontsize=35)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\formaxhop\\distancetosinglenode\\maxdev_nodehop{Nn}ED{EDn}Beta{betan}Hop{key}.png".format(
            Nn=N, EDn=ED, betan=beta,key = key)
        # plt.savefig(picname, format='png', bbox_inches='tight', dpi=600)
        plt.show()
        # 清空图像，以免影响下一个图
        plt.close()


def plot_mindev_node_hocount(N, ED, beta):
    # The data is collected through deviatition_vs_diffNkbeta_SRGG.py
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
    if N == 100000:
        for ExternalSimutime in range(20):
            SPhopcount_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\formaxhop\\distancetosinglenode\\min_hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
            SP_hopcount_foronesimu = np.loadtxt(SPhopcount_name,dtype=int)
            SP_hopcount.extend(SP_hopcount_foronesimu)

            max_dev_node_hopcount_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\formaxhop\\distancetosinglenode\\min_dev_node_hopcount_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
            max_dev_node_hopcount_forone = np.loadtxt(max_dev_node_hopcount_name,dtype=int)
            max_dev_node_hopcount.extend(max_dev_node_hopcount_forone)
    else:
        for ExternalSimutime in range(1):
            SPhopcount_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\formaxhop\\distancetosinglenode\\min_hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
            SP_hopcount_foronesimu = np.loadtxt(SPhopcount_name, dtype=int)
            SP_hopcount.extend(SP_hopcount_foronesimu)

            max_dev_node_hopcount_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\formaxhop\\distancetosinglenode\\min_dev_node_hopcount_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
            max_dev_node_hopcount_forone = np.loadtxt(max_dev_node_hopcount_name, dtype=int)
            max_dev_node_hopcount.extend(max_dev_node_hopcount_forone)

    SP_hopcount = [x for x in SP_hopcount if x!=1]
    counted = Counter(SP_hopcount)
    countdic = dict(counted)
    sorted_by_key_desc = dict(sorted(counted.items(), key=lambda x: x[0], reverse=False))

    print(sorted_by_key_desc)

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
        print(bins)
        print([int(i) for i in values])
        plt.hist(values, bins=bins,alpha=0.7, color=colors[3], edgecolor=colors[3],density=True)  # 绘制直方图
        plt.xticks(np.arange(min(values), max(values) + 1, 1))
        # plt.xlim([0, 1])
        # plt.xticks([1, 5, 9, 13, 17,21,25]) # for N=10000, ED5 BETA4 HOP=25
        # plt.xticks([1, 5, 9, 13]) # for N=1000, ED5 BETA4 HOP=13
        # plt.xticks([1, 5, 9, 13, 17]) # for N=1000, ED5 BETA128 HOP=17

        # plt.xticks([1, 4, 7, 10]) # for N=1000, ED5 BETA4 HOP=12

        # plt.yticks([0, 0.03,0.06, 0.09,0.12,0.15, 0.18])
        # plt.yticks([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
        plt.ylim([0,0.32])

        text = rf"$h = {key}$"
        plt.text(
            0.5, 0.85,  # 文本位置（轴坐标，0.5 表示图中央，1.05 表示轴上方）
            text,
            transform=ax.transAxes,  # 使用轴坐标
            fontsize=30,  # 字体大小
            ha='center',  # 水平居中对齐
            va='bottom'  # 垂直对齐方式
        )

        plt.xlabel(r'x', fontsize=35)
        plt.ylabel(r'$f_{h_{iq}}(x)$', fontsize=35)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\formaxhop\\distancetosinglenode\\mindev_nodehop{Nn}ED{EDn}Beta{betan}Hop{key}.png".format(
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


def plot_maxmindev_node_hocount_together(N, ED, beta):
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 4))
    key = 25
    values = [13, 19, 4, 14, 11, 6, 7, 8, 18, 10, 10, 5, 14, 7, 3, 11, 17, 7, 10, 17, 13, 9, 11, 12, 12, 3, 18, 12, 14, 10, 14, 15, 13, 14, 12, 20, 18, 17, 9, 10, 7, 15, 21, 14, 7, 15, 12, 17, 18, 15, 8, 19, 10, 14, 10, 8, 19, 21, 9, 11, 8, 7, 10, 16, 13, 15, 7, 18, 4, 18, 13, 15, 8, 11, 9, 7, 16, 11, 15, 9, 10, 15, 8, 21, 12, 17, 6, 19, 13, 16, 17, 11, 13, 17, 7, 18, 10, 16, 21, 6, 7, 9, 20, 7, 11, 5, 13, 10, 11, 19, 15, 5, 10, 18, 9, 16, 14, 10, 12, 17, 4, 15, 18, 17, 14, 12, 17, 21, 5, 23, 10, 21, 10, 7, 19, 12, 4, 12, 9, 11, 12, 16, 13, 17, 16, 15, 16, 13, 18, 5, 16, 5, 20, 5, 15, 11, 12, 17, 12, 10, 6, 22, 16, 8, 7, 12, 14, 10, 17, 9, 10, 14, 3, 8, 16, 9, 10, 12, 9, 10, 11, 18, 14, 10, 5, 9, 12, 5, 14, 20, 10, 12, 13, 11, 14, 11, 9, 13, 9, 17, 13, 14, 9, 16, 15, 15, 14, 11, 6, 14, 10, 19, 6, 13, 13, 17, 18, 10, 8, 18, 11, 14, 20, 19, 9, 7, 9, 11, 3, 7, 20, 10, 15, 10, 6, 17, 16, 16, 15, 10, 2, 16, 24, 17, 13, 10, 18, 10, 16, 15, 12, 7, 11, 13, 13, 17, 10, 12, 12, 18, 13, 10, 18, 14, 13, 12, 13, 7, 10, 8, 6, 16, 12, 10, 9, 13, 12, 9, 6, 13, 13, 9, 10, 3, 12, 11, 15, 11, 21, 9, 14, 16, 13, 13, 15, 9, 14, 16, 10, 8, 5, 10, 13, 7, 16, 8, 15, 24, 18, 12, 14, 7, 15, 19, 8, 12, 11, 16, 11, 12, 11, 11, 19, 11, 10, 19, 14, 6, 9, 13, 5, 11, 13, 12, 8, 10, 24, 12, 7, 8, 14, 7, 6, 12, 9, 4, 15, 15, 14, 14, 18, 11, 12, 10, 16, 8, 6, 21, 8, 17, 17, 11, 4, 11, 15, 3, 12, 13, 22, 14, 11, 2, 13, 18, 18, 10, 11, 14, 19, 16, 9, 18, 17, 20, 18, 13, 7, 12, 19, 16, 6, 3, 11, 10, 6, 12, 20, 14, 7, 11, 19, 9, 17, 11, 7, 6, 13, 20, 9, 10, 11, 5, 17, 17, 16, 17, 13, 17, 16, 17, 15, 12, 14, 9, 12, 15, 10, 11, 10, 9, 10, 12, 8, 16, 6, 7, 4, 15, 15, 14, 13, 20, 21, 15, 16, 10, 10, 13, 11, 12, 17, 16, 10, 12, 12, 12, 15, 9, 12, 9, 9, 20, 9, 19, 19, 11, 12, 15, 12, 19, 7, 14, 13, 5, 12, 15, 14, 15, 13, 4, 6, 8, 12, 19, 12, 14, 10, 7, 7, 12, 17, 13, 11, 19, 20, 11, 9, 14, 7, 18, 13, 15, 8, 6, 17, 14, 8, 6, 3, 13, 10, 16, 20, 8, 7, 11, 21, 10, 3, 8, 12, 11, 7, 11, 5, 6, 13, 8, 12, 19, 7, 7, 4, 10, 13, 9, 13, 12, 16, 18, 13, 15, 12, 12, 16, 14, 16, 9, 22, 15, 23, 16, 15, 8, 12, 15, 16, 20, 21, 19, 9, 12, 17, 9, 17, 8, 20, 18, 19, 11, 9, 15, 18, 15, 9, 18, 13, 14, 9, 14, 23, 17, 13, 15, 7, 11, 8, 9, 22, 16, 9, 19, 14, 10, 9, 18, 12, 12, 15, 19, 22, 10, 9, 16, 12, 13, 11, 5, 11, 11, 8, 7, 12, 17, 17, 8, 17, 14, 16, 16, 9, 10, 8, 10, 10, 7, 20, 10, 8, 4, 13, 17, 10, 13, 14, 5, 11, 15, 14, 8, 12, 13, 13, 18, 7, 15, 9, 19, 17, 14, 17, 7, 10, 12, 15, 15, 8, 11, 13, 6, 6, 6, 18, 19, 11, 12, 11, 22, 8, 3, 12, 11, 12, 6, 15, 12, 18, 17, 7, 20, 12, 11, 21, 13, 4, 13, 12, 6, 13, 8, 7, 14, 10, 8, 13, 3, 22, 11, 10, 13, 7, 13, 12, 16, 7, 13, 14, 4, 19, 18, 11, 10, 12, 4, 12, 20, 12, 17, 11, 14, 7, 17, 18, 12, 10, 7, 18, 13, 20, 8, 16, 17, 18, 16, 13, 16, 14, 12, 9, 17, 11, 14, 11, 18, 18, 12, 17, 12, 9, 8, 11, 12, 12, 14, 19, 15, 17, 6, 17, 16, 12, 9, 5, 11, 6, 14, 21, 19, 8, 10, 13, 18, 14, 14, 12, 16, 5, 12, 7, 12, 17, 15, 18, 15, 17, 16, 15, 10, 11, 15, 13, 18, 12, 16, 11, 11, 9, 13, 18, 5, 11, 6, 14, 8, 21, 16, 16, 7, 13, 6, 16, 14, 15, 11, 10, 10, 12, 18, 13, 6, 10, 4, 6, 9, 11, 15, 16, 10, 8, 17, 16, 19, 5, 16, 13, 6, 16, 8, 20, 16, 10, 20, 10, 8, 16, 12, 15, 11, 5, 15, 16, 16, 11, 15, 18, 11, 16, 6, 8, 17, 16, 19, 13, 16, 10, 18, 13, 13, 10, 14, 15, 12, 11, 16, 12, 19, 16, 17, 12, 13, 14, 12, 7, 13, 16, 13, 13, 13, 13, 10, 7, 4, 11, 18, 13, 14, 21, 16, 13, 8, 23, 4, 12, 13, 12, 13, 16, 8, 13, 13, 10, 11, 15, 13, 13, 17, 12, 18, 20, 13, 13, 12, 11, 14, 11, 10, 9, 9, 4, 19, 11, 12, 10, 14, 15, 12, 20, 12, 14, 7, 19, 10, 12, 17, 7, 14, 14, 19, 15, 19, 21, 20, 16, 16, 17, 20, 6, 5, 20, 13, 21, 14, 10, 20, 6, 16, 18, 7, 16, 3, 7, 13, 12, 23, 12, 6, 5, 12, 19, 18, 16, 16, 10, 9, 6, 5, 3, 13, 3, 13, 11, 16, 9, 8, 7, 12, 13, 15, 14, 17, 13, 14, 15, 16, 14, 9, 12, 21, 18, 8, 11, 12, 18, 6, 17, 16, 6, 13, 12, 17, 23, 10, 13, 11, 14, 12, 11, 12, 13, 19, 12, 12, 15, 14, 10, 11, 10, 15, 12, 15, 15, 19, 17, 5, 6, 17, 16, 6, 16, 11, 17, 5, 6, 18, 20, 12, 6, 14, 11, 10, 15, 4, 7, 16, 19, 5, 17, 8, 11, 20, 16, 12, 13, 15, 15, 8, 11, 21, 10, 4, 13, 15, 10, 6, 11, 8, 17, 15, 7, 17, 9, 3, 14, 20, 20, 11, 13, 15, 10, 15, 9, 12, 12, 11, 16, 3, 13, 19, 2, 8, 11, 10, 8, 12, 11, 11, 10, 11, 12, 13, 8, 16, 12, 3, 13, 6, 20, 11, 13, 20, 19, 15, 16, 13, 17, 9, 8, 13, 21, 8, 19, 7, 7, 5, 21, 13, 13, 15, 10, 22, 12, 15, 13, 16, 23, 7, 13, 12, 11, 13, 12, 15, 16, 17, 21, 7, 10, 7, 16, 13, 22, 20, 17, 13, 19, 12, 16, 7, 14, 14, 15, 15, 5, 10, 17, 7, 15, 14, 13, 11, 10, 14, 9, 17, 12, 5, 17, 4, 18, 19, 11, 12, 12, 17, 11, 12, 2, 14, 13, 13, 12, 15, 14, 11, 12, 19, 19, 7, 18, 21, 6, 10, 9, 22, 6, 17, 17, 7, 14, 18, 13, 17, 13, 9, 10, 14, 11, 19, 8, 15, 14, 8, 13, 17, 9, 16, 7, 15, 9, 19, 16, 13, 16, 10, 13, 17, 12, 13, 8, 15, 15, 18, 17, 10, 9, 17, 12, 11, 11, 13, 20, 16, 15, 13, 12, 15, 15, 16, 12, 14, 16, 9, 17, 15, 19, 9, 8, 9, 10, 11, 18, 19, 18, 13, 8, 20, 7, 10, 17, 9, 11, 9, 15, 11, 15, 13, 7, 9, 19, 10, 8, 18, 7, 16, 6, 4, 11, 14, 16, 9, 6, 7, 7, 16, 23, 9, 8, 16, 11, 20, 9, 15, 20, 20, 17, 12, 12, 10, 6, 18, 14, 17, 9, 6, 13, 10, 9, 12, 11, 10, 11, 3, 4, 5, 10, 17, 15, 16, 9, 8, 12, 15, 9, 15, 10, 6, 4, 14, 21, 11, 9, 11, 20, 8, 14, 16, 11, 15, 7, 19, 12, 11, 13, 15, 14, 13, 20, 7, 16, 16, 17, 14, 2, 16, 15, 13, 16, 7, 17, 14, 7, 6, 10, 7, 10, 8, 16, 3, 17, 10, 12, 4, 14, 21, 12, 9, 13, 8, 9, 6, 21, 14, 8, 11, 14, 20, 8, 12, 11, 7, 6, 3, 13, 2, 6, 17, 7, 14, 18, 20, 8, 13, 16, 13, 17, 8, 16, 4, 14, 15, 3, 4, 14, 9, 17, 21, 13, 17, 5, 10, 13, 13, 16, 11, 7, 10, 15, 7, 19, 20, 10, 13, 5, 19, 19, 17, 12, 8, 15, 14, 15, 17, 16, 4, 10, 12, 15, 7, 20, 18, 14, 13, 13, 10, 4, 13, 14, 14, 14, 15, 12, 9, 7, 10, 6, 13, 12, 3, 5, 8, 8, 18, 8, 6, 11, 15, 18, 11, 17, 17, 8, 17, 10, 6, 13, 16, 9, 5, 18, 15, 19, 15, 15, 14, 13, 13, 6, 12, 15, 13, 8, 10, 2, 11, 18, 15, 18, 11, 9, 10, 6, 20, 11, 9, 8, 7, 15, 13, 17, 7, 15, 5, 2, 8, 16, 12, 4, 16, 14, 15, 11, 14, 11, 9, 19, 5, 11, 16, 14, 15, 4, 12, 18, 8, 9, 12, 5, 18, 10, 9, 9, 12, 17, 13, 14, 14, 7, 9, 6, 16, 9, 6, 17, 10, 12, 6, 22, 18, 7, 20, 11, 11, 9, 15, 8, 15, 19, 14, 14, 7, 10, 16, 18, 18, 8, 11, 18, 11, 12, 7, 15, 15, 9, 13, 11, 1, 12, 22, 7, 11, 13, 9, 16, 11, 13, 14, 15, 8, 14, 5, 8, 14, 6, 4, 9, 13, 10, 12, 13, 9, 8, 11, 16, 17, 10, 8, 7, 6, 9, 9, 12, 15, 11, 7, 6, 13, 7, 10, 13, 12, 18, 6, 11, 22, 6, 18, 18, 7, 8, 10, 8, 12, 24, 17, 14, 17, 12, 5, 17, 10, 10, 10, 13, 19, 13, 5, 4, 15, 19, 10, 11, 5, 10, 14, 7, 8, 14, 11, 16, 4, 9, 12, 13, 14, 8, 15, 16, 17, 12, 15, 22, 16, 17, 20, 8, 10, 12, 16, 12, 11, 10, 7, 18, 20, 12, 15, 11, 17, 20, 10, 10, 5, 10, 10, 21, 9, 14, 16, 14, 11, 14, 10, 14, 12, 7, 10, 14, 14, 13, 11, 12, 12, 12, 17, 17, 4, 6, 12, 12, 21, 11, 6, 5, 18, 12, 22, 10, 4, 20, 9, 6, 8, 15, 13, 13, 3, 15, 13, 11, 7, 18, 10, 16, 17, 19, 17, 8, 16, 7, 11, 5, 5, 7, 15, 9, 10, 6, 11, 14, 14, 19, 14, 7, 13, 6, 19, 7, 11, 13, 13, 15, 17, 20, 5, 14, 9, 14, 13, 10, 17, 12, 9, 18, 15, 14, 20, 19, 9, 15, 8, 9, 18, 7, 20, 14, 18, 10, 7, 14, 14, 7, 15, 12, 12, 20, 18, 9, 15, 14, 13, 16, 6, 9, 5, 13, 18, 12, 15, 16, 14, 14, 19, 11, 15, 16, 5, 18, 10, 7, 13, 6, 13, 14, 11, 18, 15, 12, 10, 9, 5, 12, 13, 11, 17, 7, 21, 13, 12, 6, 11, 4, 12, 16, 7, 12, 8, 8, 21, 6, 12, 13, 14, 16, 8, 15, 5, 9, 14, 7, 10, 16, 5, 6, 14, 12, 5, 14, 10, 12, 7, 12, 13, 13, 11, 19, 18, 19, 10, 9, 17, 2, 9, 17, 13, 14, 15, 10, 15, 7, 22, 14, 15, 4, 14, 4, 6, 12, 13, 21, 14, 14, 9, 19, 8, 10, 14, 10, 15, 9, 7, 9, 11, 17, 10, 6, 9, 9, 16, 13, 17, 7, 17, 8, 9, 7, 17, 3, 17, 18, 9, 11, 16, 5, 12, 15, 8, 10, 10, 6, 6, 8, 8, 14, 16, 5, 15, 10, 16, 20, 16, 11, 15, 13, 13, 17, 16, 16, 10, 23, 17, 15, 15, 16, 10, 12, 11, 12, 11, 8, 12, 9, 7, 20, 24, 5, 9, 10, 13, 14, 22, 13, 6, 4, 12, 7, 12, 14, 9, 14, 19, 20, 7, 10, 7, 21, 8, 14, 12, 23, 5, 15, 12, 16, 12, 6, 16, 13, 13, 11, 10, 14, 15, 6, 16, 10, 12, 21, 10, 20, 13, 12, 13, 11, 19, 18, 9, 8, 14, 10, 24, 9, 9, 9, 14, 15, 19, 15, 9, 8, 10, 14, 18, 13, 12, 17, 18, 16, 14, 6, 14, 21, 14, 15, 9, 5, 12, 9, 15, 10, 11, 16, 12, 19, 9, 12, 9, 15, 17, 11, 21, 17, 11, 5, 19, 8, 18, 11, 12, 15, 12, 16, 11, 7, 12, 13, 15, 20, 19, 11, 14, 9, 10, 12, 7, 14, 5, 10, 13, 6, 12, 13, 19, 6, 12, 12, 6, 12, 24, 15, 5, 18, 14, 12, 9, 17, 13, 18, 14, 12, 7, 16, 9, 12, 6, 17, 13, 15, 12, 18, 12, 7, 6, 12, 10, 10, 5, 11, 5, 7, 21, 12, 14, 15, 15, 18, 6, 16, 16, 13, 8, 16, 11, 13, 10, 7, 7, 13, 14, 21, 12, 20, 18, 7, 11, 12, 8, 13, 12, 11, 10, 7, 17, 22, 15, 20, 13, 15, 17, 15, 10, 19, 8, 10, 15, 16, 18, 14, 5, 16, 17, 16, 15, 10, 12, 9, 11, 9, 12, 13, 19, 7, 23, 10, 20, 13, 16, 14, 7, 14, 16, 13, 13, 14, 9, 11, 19, 10, 6, 20, 16, 14, 9, 12, 16, 12, 14, 18, 7, 14, 22, 11, 17, 12, 14, 13, 14, 12, 19, 18, 19, 13, 14, 10, 17, 14, 5, 16, 16, 14, 11, 11, 10, 21, 8, 14, 12, 6, 12, 17, 19, 14, 9, 15, 8, 13, 13, 5, 10, 10, 19, 6, 9, 16, 8, 11, 12, 9, 13, 20, 10, 16, 14, 10, 12, 8, 10, 12, 13, 14, 10, 14, 9, 13, 11, 9, 12, 13, 9, 19, 15, 16, 14, 11, 13, 5, 8, 10, 19, 13, 15, 11, 11, 15, 9, 9, 7, 14, 3, 7, 15, 8, 6, 14, 13, 7, 13, 9, 14, 21, 11, 12, 7, 6, 13, 3, 20, 15, 15, 15, 4, 17, 14, 4, 15, 17, 9, 15, 16, 6, 11, 20, 17, 12, 5, 14, 11, 9, 15, 11, 12, 13, 8, 10, 17, 16, 9, 12, 9, 13, 13, 9, 13, 14, 12, 13, 18, 14, 14, 17, 14, 17, 18, 15, 11, 18, 13, 13, 14, 17, 5, 16, 14, 17, 9, 9, 20, 12, 3, 9, 18, 13, 9, 14, 6, 7, 18, 18, 13, 16, 14, 11, 9, 17, 18, 21, 6, 22, 6, 13, 17, 13, 14, 15, 21, 10, 15, 18, 12, 4, 10, 14, 16, 12, 5, 17, 12, 15, 5, 6, 7, 5, 16, 11, 14, 11, 6, 11, 13, 16, 14, 19, 15, 17, 6, 8, 15, 14, 19, 8, 7, 15, 16, 16, 17, 11, 12, 13, 22, 7, 19, 11, 13, 13, 13, 11, 17, 7, 11, 11, 11, 18, 8, 17, 15, 17, 12, 17, 11, 7, 18, 10, 6, 11, 14, 12, 16, 12, 9, 12, 6, 11, 8, 18, 12, 15, 9, 8, 14, 13, 15, 10, 5, 18, 14, 10, 9, 13, 19, 9, 17, 10, 7, 6, 13, 14, 8, 9, 10, 19, 16, 8, 15, 13, 11, 10, 12, 16, 15, 8, 15, 16, 18, 22, 20, 16, 15, 10, 15, 15, 10, 7, 16, 12, 14, 23, 16, 16, 20, 19, 9, 18, 10, 7, 14, 8, 9, 17, 11, 14, 21, 16, 15, 6, 21, 15, 17, 15, 12, 17, 10, 8, 13, 10, 13, 15, 11, 13, 13, 18, 8, 18, 12, 16, 11, 15, 14, 17, 15, 17, 9, 13, 16, 16, 16, 10, 17, 21, 10, 7, 8, 8, 10, 15, 9, 19, 13, 14, 19, 12, 11, 12, 12, 11, 15, 17, 8, 18, 13, 5, 13, 14, 16, 9, 9, 10, 9, 15, 12, 17, 12, 12, 18, 14, 14, 11, 11, 14, 20, 16, 12, 14, 4, 8, 8, 10, 17, 11, 17, 11, 15, 9, 16, 18, 13, 17, 11, 19, 10, 14, 20, 19, 18, 18, 18, 15, 14, 13, 17, 14, 10, 10, 11, 12, 6, 16, 8, 20, 19, 5, 9, 11, 12, 13, 5, 5, 9, 18, 17, 10, 23, 13, 8, 4, 8, 18, 9, 14, 12, 13, 15, 17, 9, 14, 13, 11, 8, 16, 9, 18, 14, 18, 15, 14, 13, 14, 15, 13, 6, 10, 15, 21, 14, 9, 11, 10, 6, 7, 7, 7, 14, 9, 17, 20, 16, 7, 17, 11, 5, 14, 18, 13, 14, 13, 6, 9, 13, 8, 11, 13, 9, 8, 7, 21, 7, 7, 9, 16, 11, 11, 11, 16, 14, 8, 11, 12, 11, 12, 16, 10, 12, 18, 10, 10, 19, 11, 5, 7, 10, 17, 9, 8, 12, 13, 20, 9, 6, 18, 8, 12, 10, 14, 13, 12, 16, 9, 10, 10, 21, 11, 10, 8, 17, 14, 14, 17, 12, 7, 13, 11, 15, 6, 15, 18, 11, 14, 7, 19, 7, 7, 16, 22, 21, 13, 10, 12, 12, 11, 15, 12, 18, 14, 14, 15, 17, 15, 17, 12, 11, 11, 11, 9, 12, 11, 17, 3, 11, 11, 17, 18, 12, 11, 11, 6, 17, 11, 12, 13, 9, 3, 12, 10, 9, 6, 15, 13, 12, 9, 13, 16, 10, 10, 8, 15, 17, 15, 10, 15, 17, 5, 17, 12, 18, 16, 12, 12, 13, 8, 12, 16, 11, 8, 3, 5, 19, 16, 16, 9, 14, 10, 18, 12, 17, 12, 13, 15, 13, 10, 10, 9, 13, 22, 12, 10, 14, 11, 14, 15, 15, 11, 10, 12, 12, 10, 18, 11, 6, 2, 6, 14, 15, 19, 8, 11, 11, 18, 14, 17, 15, 8, 7, 4, 19, 6, 18, 13, 15, 6, 22, 17, 16, 19, 13, 14, 19, 7, 16, 13, 11, 16, 7, 9, 10, 13, 9, 15, 18, 16, 18, 10, 17, 14, 13, 12, 11, 5, 11, 18, 13, 17, 7, 14, 9, 14, 14, 13, 7, 8, 8, 16, 15, 10, 16, 8, 11, 17, 9, 10, 4, 8, 16, 12, 10, 8, 7, 7, 9, 3, 12, 17, 13, 7, 13, 14, 19, 22, 14, 16, 19, 16, 4, 9, 9, 11, 5, 16, 9, 16, 10, 13, 11, 11, 15, 16, 14, 13, 13, 6, 13, 16, 17, 14, 14, 3, 17, 17, 10, 9, 9, 6, 9, 10, 18, 5, 12, 10, 4, 9, 9, 23, 10, 16, 12, 16, 8, 11, 16, 9, 10, 12, 5, 6, 14, 13, 13, 11, 18, 9, 11, 16, 22, 19, 15, 20, 4, 5, 15, 20, 10, 12, 14, 19, 10, 11, 16, 15, 9, 11, 9, 4, 11, 8, 12, 17, 16, 19, 19, 5, 11, 6, 21, 16, 6, 16, 23, 18, 17, 15, 20, 6, 18, 9, 13, 9, 18, 12, 19, 9, 15, 12, 4, 9, 9, 20, 10, 7, 6, 15, 16, 15, 11, 21, 21, 10, 12, 6, 20, 12, 5, 15, 9, 10, 10, 6, 10, 15, 17, 17, 15, 19, 16, 14, 16, 14, 14, 16, 12, 2, 6, 15, 10, 14, 13, 12, 20, 15, 20, 18, 13, 13, 23, 14, 21, 6, 8, 8, 19, 20, 15, 15, 3, 7, 13, 5, 8, 14, 7, 14, 10, 5, 19, 15, 9, 11, 7, 18, 6, 10, 11, 11, 3, 19, 18, 13, 11, 15, 17, 16, 5, 11, 19, 21, 11, 7, 9, 18, 14, 9, 15, 12, 16, 10, 18, 5, 9, 11, 14, 13, 19, 6, 5, 11, 16, 16, 18, 4, 9, 17, 16, 5, 8, 20, 10, 11, 10, 17, 8, 15, 12, 13, 10, 9, 7, 13, 7, 9, 18, 11, 17, 9, 6, 13, 6, 11, 9, 3, 10, 9, 13, 18, 15, 16, 8, 8, 9, 15, 10, 22, 17, 6, 14, 20, 9, 21, 6, 12, 14, 10, 12, 6, 14, 15, 19, 12, 12, 15, 18, 10, 16, 14, 5, 10, 5, 14, 17, 15, 11, 11, 15, 17, 14, 12, 11, 10, 15, 4, 13, 10, 20, 18, 18, 6, 13, 13, 14, 13, 18, 10, 14, 11, 9, 17, 4, 15, 15, 18, 4, 7, 11, 12, 7, 5, 16, 20, 15, 9, 12, 18, 14, 8, 5, 8, 15, 11, 7, 7, 17, 8, 17, 19, 2, 12, 18, 9, 10, 17, 9, 9, 4, 18, 13, 9, 18, 2, 10, 8, 9, 13, 19, 13, 19, 4, 14, 6, 7, 12, 9, 18, 11, 19, 14, 15, 17, 15, 10, 11, 17, 14, 11, 15, 10, 11, 16, 8, 6, 16, 9, 10, 19, 6, 8, 8, 4, 3, 17, 19, 10, 9, 14, 18, 15, 14, 15, 7, 15, 22, 3, 6, 16, 16, 14, 8, 16, 8, 13, 18, 13, 12, 6, 13, 11, 13, 10, 7, 10, 9, 7, 12, 13, 13, 18, 20, 19, 7, 9, 12, 7, 14, 15, 15, 13, 6, 15, 12, 17, 17, 12, 18, 8, 18, 6, 18, 19, 12, 18, 8, 7, 21, 8, 12, 16, 14, 12, 17, 20, 12, 12, 9, 11, 14, 10, 11, 9, 12, 15, 8, 7, 15, 13, 11, 15, 9, 13, 13, 16, 6, 12, 11, 12, 15, 20, 17, 11, 19, 13, 9, 4, 13, 13, 11, 18, 14, 6, 17, 9, 7, 6, 23, 16, 11, 11, 12, 19, 14, 7, 8, 16, 9, 15, 9, 2, 19, 6, 12, 11, 19, 15, 10, 17, 5, 5, 21, 15, 19, 18, 10, 12, 16, 11, 14, 9, 21, 12, 15, 14, 14, 13, 11, 4, 15, 9, 10, 11, 12, 10, 14, 4, 14, 18, 5, 9, 13, 13, 15, 13, 14, 16, 17, 19, 11, 8, 9, 16, 13, 9, 15, 10, 9, 8, 9, 13, 9, 12, 18, 12, 22, 8, 12, 13, 4, 10, 8, 20, 18, 4, 14, 13, 15, 12, 9, 16, 12, 16, 14, 15, 17, 9, 4, 14, 16, 12, 16, 13, 13, 21, 6, 22, 18, 7, 13, 11, 13, 8, 12, 13, 17, 13, 5, 6, 11, 13, 14, 5, 7, 13, 14, 12, 10, 12, 7, 11, 18, 9, 14, 19, 10, 15, 8, 5, 21, 7, 14, 13, 9, 15, 15, 19, 12, 15, 14, 18, 12, 10, 2, 9, 20, 21, 14, 20, 18, 19, 8, 10, 17, 16, 7, 14, 7, 9, 20, 13, 12, 15, 18, 12, 12, 7, 9, 13, 7, 11, 10, 12, 20, 17, 5, 11, 12, 13, 23, 16, 14, 21, 14, 17, 13, 20, 3, 16, 13, 13, 9, 15, 15, 9, 17, 14, 12, 15, 10, 17, 20, 7, 9, 8, 21, 17, 15, 8, 10, 14, 8, 10, 13, 11, 9, 4, 16, 8, 13, 21, 2, 18, 13, 11, 14, 24, 7, 16, 4, 11, 2, 5, 16, 12, 15, 20, 14, 9, 19, 14, 4, 14, 17, 11, 13, 14, 8, 16, 14, 14, 14, 14, 4, 12, 16, 17, 10, 12, 7, 10, 9, 14, 23, 18, 8, 17, 10, 2, 5, 13, 14, 14, 6, 4, 10, 16, 11, 12, 16, 16, 22, 14, 11, 13, 15, 15, 11, 13, 15, 19, 19, 15, 13, 5, 6, 19]
    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5,
            19.5, 20.5, 21.5, 22.5, 23.5, 24.5
            ]

    ax1.hist(values, bins=bins, alpha=0.7, color=colors[3], edgecolor=colors[3], density=True)  # 绘制直方图
    ax1.set_xticks(np.arange(min(values), max(values) + 1, 1))
    ax1.set_yticks([0, 0.05, 0.10])
    ax1.set_xticks([1, 5, 9, 13, 17,21,25])
    ax1.tick_params(axis='x', labelsize=20)  # x 轴刻度字体大小
    ax1.tick_params(axis='y', labelsize=20)  # y 轴刻度字体大小
    ax1.tick_params(axis='x', labelbottom = False)
    # ax1.set_xlabel(r'x', fontsize=24)
    ax1.set_ylabel(r'$f_X(\text{x})$', fontsize=24)
    # plt.xticks(fontsize=28)
    # plt.yticks(fontsize=28)

    ax1.text(0.61, 0.75, r'$h = 25$',
             transform=ax1.transAxes,
             fontsize=20,
             bbox=None)


    key = 25
    values = [21, 1, 20, 1, 2, 24, 24, 3, 2, 24, 11, 1, 24, 24, 2, 1, 24, 1, 24, 1, 24, 1, 1, 9, 5, 7, 2, 1, 11, 20, 4,
              3, 1, 3, 24, 24, 5, 24, 1, 23, 2, 3, 24, 2, 24, 15, 5, 21, 1, 23, 24, 5, 21, 24, 20, 22, 1, 24, 12, 6, 2,
              5, 24, 12, 17, 3, 1, 1, 23, 24, 9, 1, 24, 3, 1, 2, 23, 24, 13, 17, 24, 16, 1, 15, 24, 24, 24, 12, 13, 1,
              21, 4, 4, 17, 1, 1, 1, 8, 24, 24, 10, 10, 24, 1, 22, 1, 1, 24, 2, 21, 1, 23, 21, 14, 24, 24, 8, 24, 3, 24,
              1, 1, 23, 1, 16, 21, 24, 3, 6, 1, 1, 1, 1, 24, 24, 1, 9, 23, 24, 1, 24, 22, 16, 1, 1, 1, 2, 4, 17, 20, 24,
              1, 23, 3, 15, 1, 17, 24, 21, 1, 3, 20, 1, 4, 24, 24, 24, 24, 1, 24, 24, 24, 2, 3, 23, 24, 1, 22, 10, 24,
              2, 24, 24, 24, 1, 3, 24, 8, 24, 8, 24, 1, 21, 24, 2, 9, 24, 3, 11, 1, 7, 24, 4, 14, 11, 5, 22, 1, 24, 22,
              24, 24, 24, 24, 11, 24, 24, 24, 6, 20, 24, 1, 19, 2, 3, 24, 24, 20, 24, 1, 4, 6, 24, 1, 18, 1, 1, 4, 23,
              19, 24, 3, 2, 24, 1, 24, 22, 24, 24, 24, 24, 1, 21, 21, 1, 1, 10, 1, 24, 10, 1, 24, 1, 19, 24, 8, 24, 23,
              24, 22, 1, 24, 18, 3, 19, 8, 16, 24, 23, 5, 13, 24, 24, 12, 1, 1, 7, 18, 13, 12, 8, 1, 24, 20, 1, 1, 24,
              1, 24, 24, 24, 1, 7, 22, 1, 1, 5, 24, 22, 24, 1, 20, 1, 23, 12, 6, 1, 24, 16, 1, 17, 5, 22, 10, 22, 1, 4,
              24, 1, 13, 1, 12, 1, 21, 24, 2, 2, 1, 23, 17, 1, 24, 4, 1, 24, 24, 24, 1, 1, 9, 20, 1, 5, 3, 5, 1, 24, 8,
              24, 20, 1, 7, 3, 5, 3, 24, 22, 1, 6, 22, 6, 2, 3, 24, 1, 15, 23, 17, 6, 19, 22, 7, 24, 16, 1, 16, 24, 24,
              1, 20, 24, 24, 22, 24, 18, 24, 14, 1, 10, 5, 24, 2, 24, 24, 24, 1, 23, 9, 24, 20, 24, 5, 24, 1, 19, 1, 3,
              24, 24, 3, 15, 24, 24, 1, 1, 24, 1, 24, 1, 11, 15, 17, 24, 24, 1, 2, 22, 9, 24, 24, 24, 24, 1, 23, 24, 2,
              8, 23, 8, 1, 22, 20, 1, 2, 1, 11, 2, 23, 2, 24, 24, 24, 4, 4, 22, 24, 1, 8, 7, 1, 1, 24, 16, 24, 5, 1, 2,
              23, 1, 20, 23, 24, 4, 1, 6, 6, 11, 1, 4, 24, 23, 3, 23, 24, 1, 1, 24, 2, 23, 4, 16, 1, 24, 2, 1, 2, 7, 1,
              4, 24, 1, 20, 24, 3, 24, 1, 17, 4, 1, 1, 24, 2, 24, 1, 19, 1, 24, 18, 23, 5, 24, 2, 3, 17, 1, 6, 1, 24, 5,
              3, 1, 1, 3, 16, 1, 1, 24, 1, 7, 17, 2, 24, 18, 24, 1, 6, 8, 24, 21, 7, 1, 1, 21, 1, 2, 1, 1, 1, 1, 1, 4,
              13, 24, 4, 7, 23, 24, 1, 5, 24, 1, 1, 3, 5, 24, 3, 24, 11, 1, 3, 24, 24, 24, 24, 23, 1, 3, 24, 23, 24, 15,
              11, 24, 24, 1, 5, 11, 24, 1, 1, 11, 12, 4, 21, 24, 1, 15, 1, 24, 1, 24, 20, 8, 6, 1, 2, 4, 24, 5, 8, 9, 1,
              24, 23, 20, 1, 2, 23, 21, 21, 1, 1, 24, 12, 3, 6, 1, 3, 1, 1, 24, 24, 1, 19, 22, 7, 1, 1, 11, 22, 1, 10,
              1, 7, 14, 7, 23, 13, 8, 22, 1, 8, 24, 21, 10, 1, 1, 3, 1, 24, 24, 24, 1, 1, 3, 22, 2, 24, 1, 5, 2, 1, 24,
              24, 1, 4, 24, 1, 24, 1, 1, 1, 10, 11, 3, 20, 5, 8, 1, 24, 1, 2, 13, 18, 1, 22, 13, 4, 13, 24, 1, 24, 18,
              19, 15, 1, 1, 1, 16, 14, 12, 6, 14, 24, 22, 24, 5, 1, 5, 24, 2, 24, 14, 18, 4, 5, 23, 24, 16, 18, 24, 4,
              2, 24, 1, 13, 22, 1, 24, 24, 2, 8, 2, 4, 24, 2, 3, 16, 1, 23, 24, 6, 8, 23, 24, 1, 24, 24, 23, 23, 1, 22,
              24, 4, 1, 24, 23, 1, 13, 1, 24, 10, 24, 9, 1, 3, 18, 1, 16, 24, 1, 17, 24, 23, 1, 24, 24, 24, 16, 23, 24,
              24, 17, 16, 17, 22, 2, 1, 24, 18, 8, 19, 1, 5, 22, 24, 24, 24, 20, 1, 7, 24, 24, 1, 24, 1, 24, 24, 24, 12,
              22, 24, 1, 1, 8, 1, 24, 2, 5, 7, 24, 1, 21, 23, 1, 24, 9, 1, 24, 1, 8, 24, 1, 1, 1, 1, 1, 21, 20, 23, 22,
              7, 2, 24, 17, 23, 24, 18, 2, 7, 21, 4, 4, 24, 1, 14, 9, 19, 1, 24, 20, 24, 15, 1, 24, 5, 22, 24, 24, 1, 6,
              7, 1, 21, 4, 4, 21, 1, 24, 1, 1, 1, 1, 24, 24, 2, 24, 8, 24, 1, 24, 15, 12, 24, 11, 17, 3, 21, 24, 24, 8,
              24, 24, 1, 1, 1, 14, 12, 24, 23, 4, 2, 24, 24, 24, 14, 24, 3, 24, 23, 24, 9, 1, 24, 24, 24, 19, 1, 3, 1,
              17, 24, 21, 24, 1, 1, 18, 24, 20, 14, 1, 1, 19, 13, 1, 17, 24, 2, 24, 24, 1, 4, 17, 1, 24, 24, 18, 1, 1,
              23, 24, 1, 3, 1, 1, 23, 11, 9, 1, 23, 24, 10, 5, 1, 23, 3, 1, 24, 2, 24, 2, 20, 18, 20, 1, 21, 1, 1, 21,
              2, 2, 15, 15, 24, 21, 1, 22, 24, 24, 24, 19, 24, 8, 24, 24, 21, 24, 1, 23, 8, 19, 16, 20, 20, 1, 23, 5,
              24, 19, 8, 1, 24, 17, 6, 23, 19, 1, 14, 3, 24, 1, 1, 24, 1, 18, 6, 24, 16, 17, 12, 21, 1, 22, 24, 1, 1, 5,
              2, 1, 22, 8, 1, 9, 1, 3, 23, 24, 20, 23, 1, 4, 8, 10, 20, 23, 5, 1, 23, 5, 1, 14, 23, 8, 1, 23, 24, 13, 7,
              22, 1, 1, 19, 7, 24, 20, 19, 1, 5, 5, 1, 1, 2, 1, 24, 19, 24, 7, 1, 1, 1, 1, 24, 1, 24, 5, 6, 1, 1, 22,
              23, 23, 24, 15, 21, 1, 10, 1, 12, 1, 22, 23, 1, 7, 2, 1, 1, 11, 1, 21, 24, 24, 3, 24, 22, 2, 24, 7, 19, 1,
              15, 1, 24, 2, 14, 22, 22, 1, 2, 4, 5, 2, 5, 1, 4, 24, 1, 1, 24, 20, 16, 24, 24, 24, 24, 24, 1, 24, 23, 1,
              6, 16, 14, 1, 4, 23, 2, 20, 22, 23, 23, 19, 1, 24, 24, 20, 22, 24, 21, 23, 7, 1, 1, 24, 24, 18, 9, 1, 1,
              24, 1, 24, 24, 1, 4, 1, 24, 1, 1, 24, 24, 1, 4, 12, 1, 23, 23, 12, 1, 8, 24, 1, 24, 8, 1, 24, 12, 22, 23,
              9, 23, 21, 6, 2, 24, 12, 24, 18, 1, 6, 24, 12, 22, 24, 24, 24, 24, 19, 19, 18, 1, 1, 23, 24, 2, 16, 1, 1,
              3, 9, 1, 23, 24, 1, 12, 1, 24, 24, 1, 2, 1, 24, 13, 3, 2, 24, 24, 16, 2, 2, 24, 1, 1, 9, 24, 23, 24, 1, 3,
              21, 22, 14, 19, 24, 1, 12, 24, 24, 2, 13, 2, 24, 1, 12, 10, 8, 1, 24, 1, 24, 24, 24, 22, 11, 6, 22, 23,
              15, 4, 2, 1, 16, 3, 1, 23, 1, 24, 24, 1, 2, 24, 24, 23, 4, 1, 23, 6, 1, 9, 24, 24, 24, 21, 24, 24, 24, 1,
              24, 14, 23, 13, 6, 11, 2, 1, 24, 18, 16, 2, 21, 1, 3, 24, 8, 1, 1, 24, 22, 24, 1, 1, 10, 23, 1, 13, 8, 24,
              15, 17, 24, 24, 1, 8, 1, 23, 24, 24, 24, 24, 3, 3, 11, 19, 22, 1, 24, 1, 1, 23, 3, 21, 21, 6, 1, 11, 4,
              14, 24, 2, 4, 9, 1, 1, 5, 24, 24, 23, 24, 18, 23, 9, 2, 24, 24, 1, 2, 1, 24, 24, 20, 5, 1, 24, 18, 24, 17,
              1, 21, 5, 1, 1, 14, 16, 24, 1, 24, 1, 1, 19, 2, 8, 19, 10, 2, 23, 24, 15, 24, 20, 23, 15, 22, 1, 1, 1, 20,
              1, 24, 21, 17, 1, 17, 6, 5, 24, 5, 1, 19, 22, 24, 4, 21, 1, 20, 24, 24, 5, 9, 3, 10, 1, 24, 2, 1, 24, 24,
              24, 5, 18, 1, 2, 1, 1, 19, 24, 4, 1, 1, 2, 4, 1, 24, 1, 1, 2, 24, 1, 1, 18, 1, 3, 24, 24, 12, 22, 24, 23,
              10, 2, 5, 3, 11, 24, 21, 1, 2, 24, 1, 1, 3, 21, 1, 24, 24, 17, 1, 19, 14, 1, 2, 22, 24, 1, 1, 3, 1, 1, 21,
              24, 3, 1, 24, 24, 24, 13, 7, 22, 7, 4, 9, 24, 8, 19, 24, 1, 1, 6, 23, 1, 10, 24, 14, 23, 4, 11, 1, 24, 24,
              21, 24, 24, 17, 24, 24, 23, 23, 1, 3, 3, 24, 24, 2, 17, 1, 22, 1, 1, 24, 12, 6, 21, 1, 23, 1, 15, 22, 10,
              9, 1, 23, 24, 1, 2, 20, 7, 24, 20, 1, 24, 1, 24, 24, 18, 10, 3, 1, 1, 11, 1, 24, 24, 1, 5, 1, 22, 24, 23,
              2, 1, 13, 1, 1, 13, 24, 9, 8, 1, 22, 6, 19, 24, 1, 1, 12, 24, 19, 21, 2, 23, 1, 1, 7, 22, 24, 1, 1, 1, 6,
              1, 1, 23, 24, 23, 23, 19, 1, 4, 8, 11, 17, 24, 1, 22, 24, 2, 1, 3, 1, 1, 1, 24, 23, 24, 1, 1, 1, 16, 24,
              3, 23, 24, 23, 3, 21, 14, 15, 1, 24, 9, 22, 24, 20, 22, 2, 22, 24, 24, 23, 6, 24, 24, 11, 24, 20, 1, 1,
              18, 14, 1, 1, 23, 1, 23, 2, 15, 1, 24, 24, 19, 1, 22, 24, 2, 22, 2, 11, 1, 14, 24, 24, 21, 1, 1, 1, 24, 1,
              22, 24, 24, 7, 24, 4, 24, 1, 24, 24, 1, 21, 18, 23, 24, 1, 21, 24, 15, 24, 1, 20, 20, 1, 24, 3, 8, 8, 16,
              1, 2, 1, 2, 4, 22, 8, 22, 1, 24, 12, 24, 1, 1, 24, 1, 17, 20, 23, 24, 1, 21, 23, 1, 6, 5, 1, 2, 24, 24,
              19, 13, 1, 24, 6, 24, 1, 24, 1, 3, 6, 14, 24, 4, 22, 24, 11, 1, 24, 1, 2, 23, 1, 1, 1, 1, 1, 18, 4, 24,
              24, 8, 2, 14, 1, 24, 21, 22, 24, 4, 2, 3, 2, 24, 1, 24, 2, 19, 3, 1, 1, 20, 1, 24, 20, 19, 1, 24, 4, 24,
              2, 13, 1, 9, 2, 21, 22, 24, 21, 20, 15, 23, 23, 19, 6, 12, 14, 2, 1, 1, 5, 1, 1, 24, 12, 24, 24, 1, 17,
              24, 1, 24, 18, 19, 6, 20, 24, 1, 13, 24, 2, 13, 22, 20, 16, 24, 2, 5, 1, 1, 2, 1, 24, 24, 2, 20, 5, 1, 9,
              1, 24, 23, 2, 24, 24, 1, 1, 18, 24, 1, 18, 24, 14, 11, 23, 3, 6, 3, 3, 22, 24, 1, 1, 17, 24, 3, 24, 24,
              20, 2, 1, 1, 13, 1, 1, 15, 15, 1, 24, 23, 24, 24, 24, 24, 23, 16, 1, 24, 20, 22, 2, 11, 24, 17, 6, 1, 1,
              24, 24, 2, 15, 16, 24, 1, 20, 24, 2, 24, 1, 3, 1, 24, 9, 1, 1, 23, 1, 3, 24, 9, 24, 8, 1, 11, 24, 7, 24,
              24, 1, 24, 24, 22, 2, 1, 1, 1, 7, 22, 24, 19, 24, 1, 24, 1, 2, 6, 24, 2, 1, 17, 1, 1, 15, 13, 1, 6, 8, 16,
              23, 23, 2, 18, 1, 1, 16, 24, 13, 3, 1, 24, 1, 12, 1, 11, 24, 24, 4, 24, 3, 12, 4, 1, 9, 23, 24, 19, 24,
              24, 1, 1, 1, 2, 6, 22, 24, 20, 24, 3, 24, 24, 1, 11, 3, 12, 24, 23, 17, 8, 21, 2, 24, 1, 1, 1, 1, 13, 6,
              22, 24, 20, 2, 18, 23, 1, 24, 13, 1, 10, 3, 24, 19, 24, 3, 2, 1, 23, 24, 23, 1, 23, 6, 4, 21, 5, 24, 2,
              23, 24, 24, 24, 24, 22, 1, 24, 22, 1, 21, 1, 24, 1, 1, 1, 18, 20, 24, 24, 24, 1, 1, 4, 1, 24, 1, 13, 24,
              11, 19, 8, 2, 24, 24, 24, 19, 1, 10, 23, 14, 18, 19, 1, 24, 1, 1, 1, 19, 11, 16, 1, 23, 22, 1, 24, 8, 11,
              21, 24, 1, 22, 4, 24, 21, 24, 2, 24, 24, 1, 21, 23, 18, 24, 23, 2, 1, 3, 1, 14, 24, 21, 23, 3, 22, 3, 1,
              7, 24, 17, 21, 21, 1, 24, 19, 23, 1, 24, 15, 2, 20, 24, 20, 1, 7, 1, 24, 2, 24, 1, 1, 1, 5, 1, 1, 21, 6,
              24, 2, 24, 20, 1, 24, 17, 24, 1, 24, 24, 2, 5, 1, 24, 19, 11, 24, 4, 24, 24, 1, 1, 9, 23, 1, 24, 1, 21,
              24, 24, 1, 1, 1, 8, 3, 20, 24, 4, 24, 7, 24, 1, 1, 11, 22, 6, 24, 24, 24, 1, 24, 24, 18, 18, 1, 8, 24, 24,
              24, 24, 24, 24, 24, 3, 1, 1, 4, 24, 18, 3, 22, 24, 24, 21, 13, 10, 1, 2, 14, 15, 23, 24, 3, 1, 17, 11, 2,
              24, 8, 14, 1, 3, 7, 11, 14, 15, 24, 1, 22, 24, 24, 22, 16, 24, 24, 9, 23, 2, 12, 24, 1, 12, 1, 24, 22, 24,
              24, 24, 1, 24, 24, 21, 9, 1, 1, 24, 24, 16, 24, 24, 23, 1, 11, 24, 1, 2, 23, 2, 6, 19, 24, 1, 16, 5, 21,
              11, 24, 7, 14, 5, 24, 23, 1, 1, 2, 3, 24, 1, 1, 3, 23, 1, 24, 15, 24, 19, 22, 1, 24, 15, 1, 19, 1, 11, 21,
              2, 22, 24, 24, 18, 1, 2, 1, 4, 2, 13, 1, 5, 1, 24, 2, 1, 23, 4, 1, 4, 19, 1, 1, 12, 1, 24, 1, 1, 24, 23,
              12, 24, 1, 9, 23, 24, 17, 8, 1, 3, 19, 22, 1, 16, 3, 17, 24, 3, 24, 9, 7, 24, 13, 3, 24, 11, 24, 1, 1, 6,
              24, 21, 1, 10, 14, 21, 1, 19, 24, 24, 17, 14, 24, 9, 24, 2, 2, 1, 1, 14, 1, 1, 1, 14, 6, 2, 23, 24, 24,
              23, 24, 3, 4, 24, 1, 1, 24, 1, 24, 24, 1, 1, 24, 22, 2, 13, 1, 21, 22, 14, 1, 14, 21, 11, 23, 24, 24, 1,
              20, 24, 16, 1, 3, 1, 24, 22, 24, 24, 1, 2, 1, 24, 9, 23, 24, 24, 1, 1, 1, 22, 23, 1, 17, 6, 20, 10, 24, 5,
              1, 24, 24, 1, 22, 24, 9, 23, 1, 3, 24, 24, 19, 2, 16, 24, 1, 1, 1, 14, 24, 21, 3, 24, 24, 10, 23, 1, 13,
              24, 24, 4, 1, 1, 22, 21, 24, 2, 24, 1, 1, 3, 24, 1, 2, 1, 7, 24, 1, 4, 24, 19, 3, 20, 3, 1, 24, 12, 1, 5,
              16, 4, 1, 8, 7, 1, 1, 23, 1, 3, 9, 8, 19, 24, 22, 23, 24, 11, 7, 1, 24, 24, 24, 1, 12, 24, 24, 1, 1, 1,
              24, 7, 24, 1, 3, 24, 4, 21, 24, 1, 24, 22, 1, 24, 19, 2, 23, 24, 24, 24, 24, 1, 4, 1, 11, 5, 24, 21, 2,
              24, 24, 11, 2, 1, 1, 1, 14, 18, 22, 1, 1, 18, 3, 1, 15, 1, 24, 1, 23, 24, 24, 1, 1, 6, 20, 24, 1, 2, 17,
              24, 16, 1, 1, 13, 24, 1, 9, 6, 1, 1, 1, 13, 21, 1, 22, 20, 7, 24, 1, 1, 3, 1, 1, 15, 1, 2, 1, 13, 24, 2,
              1, 23, 19, 24, 6, 24, 8, 13, 24, 2, 1, 2, 1, 24, 24, 1, 5, 1, 24, 23, 24, 15, 23, 2, 1, 7, 1, 1, 5, 12,
              24, 23, 1, 1, 24, 1, 4, 1, 24, 23, 1, 2, 19, 1, 20, 2, 1, 3, 12, 1, 4, 1, 24, 1, 9, 2, 20, 3, 21, 12, 7,
              23, 9, 4, 3, 1, 9, 6, 1, 24, 24, 24, 9, 24, 24, 1, 24, 1, 18, 21, 4, 5, 24, 24, 10, 18, 1, 3, 23, 24, 3,
              24, 1, 1, 24, 24, 8, 24, 1, 22, 22, 1, 21, 1, 5, 1, 1, 2, 11, 1, 4, 24, 23, 14, 24, 24, 21, 13, 20, 2, 24,
              19, 1, 22, 1, 13, 3, 1, 24, 8, 2, 23, 10, 1, 1, 24, 22, 24, 3, 24, 6, 24, 10, 1, 21, 11, 14, 6, 2, 22, 3,
              6, 19, 1, 8, 1, 13, 21, 1, 5, 24, 24, 7, 2, 15, 22, 4, 1, 24, 24, 13, 14, 22, 1, 24, 23, 1, 1, 24, 24, 13,
              21, 1, 7, 22, 10, 23, 2, 24, 24, 23, 24, 1, 1, 23, 1, 4, 22, 1, 23, 11, 24, 1, 24, 3, 1, 24, 21, 15, 19,
              24, 6, 7, 23, 18, 1, 18, 24, 19, 23, 16, 24, 24, 1, 1, 1, 13, 23, 24, 15, 1, 3, 1, 23, 24, 1, 21, 22, 1,
              21, 1, 5, 24, 24, 1, 24, 13, 23, 3, 1, 4, 1, 1, 24, 24, 11, 3, 1, 24, 4, 24, 2, 24, 1, 1, 24, 1, 22, 1, 1,
              1, 1, 15, 23, 1, 2, 1, 1, 2, 24, 24, 2, 24, 23, 24, 1, 1, 1, 1, 23, 1, 1, 23, 1, 5, 21, 1, 21, 14, 17, 24,
              2, 8, 24, 14, 1, 24, 13, 24, 20, 2, 1, 1, 1, 1, 24, 13, 20, 1, 17, 19, 11, 1, 6, 21, 1, 22, 13, 1, 1, 24,
              15, 24, 9, 1, 13, 22, 23, 24, 21, 18, 24, 22, 24, 10, 2, 24, 24, 21, 4, 24, 24, 10, 15, 7, 1, 1, 1, 12, 8,
              23, 1, 1, 11, 9, 1, 1, 11, 4, 21, 7, 22, 24, 22, 7, 1, 24, 4, 20, 24, 1, 24, 1, 1, 24, 24, 24, 6, 23, 23,
              2, 23, 17, 24, 2, 1, 1, 1, 5, 23, 24, 24, 1, 24, 24, 24, 1, 24, 1, 21, 24, 16, 22, 10, 17, 4, 12, 1, 6,
              24, 19, 18, 20, 11, 3, 1, 2, 9, 1, 24, 19, 1, 2, 23, 22, 1, 24, 8, 21, 1, 23, 24, 4, 1, 1, 23, 14, 1, 4,
              19, 14, 21, 1, 1, 17, 24, 24, 24, 24, 2, 1, 2, 1, 22, 22, 24, 1, 24, 1, 24, 24, 24, 24, 20, 2, 24, 12, 24,
              1, 3, 15, 2, 1, 7, 20, 2, 24, 4, 1, 8, 16, 8, 17, 24, 8, 5, 1, 19, 24, 23, 1, 24, 1, 9, 23, 12, 2, 24, 2,
              16, 24, 24, 7, 1, 1, 1, 22, 24, 2, 4, 23, 1, 1, 1, 17, 1, 24, 8, 24, 24, 1, 1, 1, 1, 2, 17, 4, 3, 1, 3, 7,
              1, 1, 20, 5, 1, 24, 24, 23, 23, 24, 20, 22, 24, 5, 24, 1, 23, 4, 24, 3, 24, 1, 1, 15, 21, 1, 24, 1, 22,
              24, 12, 16, 23, 23, 1, 5, 6, 12, 6, 1, 6, 5, 1, 24, 24, 23, 9, 15, 6, 14, 2, 4, 1, 20, 11, 24, 1, 22, 13,
              24, 1, 23, 24, 6, 5, 1, 2, 24, 17, 24, 9, 22, 24, 1, 1, 19, 24, 2, 1, 14, 1, 24, 9, 1, 16, 24, 1, 1, 24,
              24, 9, 24, 1, 1, 24, 2, 1, 19, 2, 1, 24, 15, 22, 1, 4, 3, 18, 12, 1, 22, 18, 24, 1, 24, 24, 20, 21, 21, 1,
              24, 24, 1, 20, 17, 3, 1, 3, 24, 24, 16, 1, 1, 6, 24, 24, 1, 24, 24, 24, 20, 5, 23, 24, 2, 18, 24, 14, 18,
              1, 17, 23, 24, 24, 17, 16, 17, 23, 2, 2, 24, 15, 16, 7, 1, 1, 23, 24, 24, 15, 24, 21, 2, 18, 17, 1, 3, 24,
              24, 13, 23, 24, 2, 1, 5, 24, 1, 2, 18, 1, 23, 24, 13, 18, 6, 17, 2, 1, 24, 24, 11, 11, 24, 1, 1, 24, 24,
              24, 1, 2, 24, 24, 1, 1, 24, 1, 22, 1, 24, 24, 19, 23, 4, 12, 24, 1, 2, 24, 1, 2, 19, 1, 21, 2, 9, 3, 18,
              24, 6, 21, 2, 1, 1, 1, 1, 22, 24, 24, 24, 23, 24, 13, 2, 1, 1, 6, 23, 24, 2, 24, 7, 24, 5, 24, 3, 24, 24,
              24, 1, 22, 24, 23, 2, 24, 4, 2, 22, 1, 23, 24, 24, 7, 16, 4, 1, 24, 22, 1, 3, 1, 8, 1, 22, 24, 14, 1, 5,
              16, 24, 1, 18, 10, 24, 1, 9, 23, 23, 24, 1, 23, 1, 1, 2, 4, 20, 3, 1, 23, 24, 1, 2, 24, 1, 2, 3, 23, 24,
              1, 19, 24, 15, 2, 2, 9, 2, 24, 3, 6, 1, 1, 20, 1, 23, 3, 1, 9, 24, 24, 24, 10, 6, 1, 1, 1, 1, 1, 2, 23,
              22, 1, 1, 23, 1, 2, 10, 24, 24, 6, 1, 24, 10, 1, 22, 24, 17, 24, 24, 17, 5, 22, 2, 24, 24, 4, 1, 24, 1, 6,
              24, 16, 20, 16, 2, 24, 14, 24, 10, 4, 22, 24, 1, 16, 1, 24, 1, 15, 24, 2, 24, 24, 20, 20, 1, 1, 1, 1, 1,
              1, 1, 11, 1, 2, 22, 24, 1, 15, 19, 24, 1, 1, 5, 24, 11, 1, 23, 1, 24, 1, 1, 22, 1, 17, 1, 2, 18, 1, 24,
              23, 16, 1, 1, 23, 24, 5, 24, 21, 1, 1, 24, 23, 1, 3, 1, 11, 1, 24, 17, 24, 1, 4, 20, 24, 1, 24, 1, 17, 24,
              24, 1, 1, 24, 23, 24, 24, 17, 1, 2, 1, 11, 13, 20, 24, 24, 2, 2, 24, 23, 1, 2, 24, 9, 3, 10, 1, 24, 24, 1,
              22, 23, 1, 2, 23, 3, 24, 1, 24, 18, 24, 24, 20, 2, 24, 1, 24, 24, 1, 10, 1, 23, 8, 24, 24, 9, 22, 15, 24,
              24, 1, 7, 2, 18, 5, 24, 1, 24, 10, 14, 24, 24, 1, 22, 21, 5, 1, 23, 20, 1, 10, 20, 16, 22, 3, 8, 14, 1,
              10, 24, 22, 1, 23, 13, 23, 24, 24, 24, 1, 6, 1, 21, 1, 1, 3, 24, 17, 1, 13, 24, 24, 2, 2, 14, 13, 22, 22,
              1, 3, 2, 23, 19, 1, 1, 15, 10, 24, 17, 19, 24, 23, 23, 1, 24, 4, 1, 24, 20, 3, 24, 4, 13, 2, 9, 1, 7, 1,
              4, 23, 1, 22, 5, 24, 24, 2, 11, 6, 24, 24, 2, 3, 23, 2, 1, 1, 1, 1, 1, 9, 20, 1, 11, 1, 1, 24, 14, 1, 11,
              24, 23, 24, 23, 24, 4, 20, 1, 1, 11, 22, 3, 7, 1, 24, 23, 19, 16, 7, 17, 1, 11, 8, 24, 1, 1, 12, 21, 24,
              1, 1, 24, 1, 11, 1, 24, 1, 24, 3, 24, 1, 8, 17, 23, 3, 12, 3, 11, 1, 15, 24, 1, 24, 3, 19, 4, 24, 14, 10,
              5, 15, 1, 3, 24, 4, 15, 24, 1, 17, 24, 1, 8, 24, 24, 1, 20, 23, 1, 16, 11, 1, 18, 23, 1, 1, 1, 1, 6, 1, 7,
              21, 2, 14, 24, 1, 24, 24, 18, 24, 1, 24, 17, 13, 24, 24, 1, 24, 12, 1, 20, 23, 2, 15, 23, 1, 1, 1, 15, 23,
              23, 24, 3, 1, 1, 24, 24, 1, 3, 24, 2, 13, 9, 24, 2, 16, 24, 24, 5, 1, 6, 6, 24, 1, 10, 1, 9, 20, 1, 3, 5,
              1, 15, 7, 1, 11, 24, 24, 1, 2, 21, 8, 24, 2, 2, 1, 14, 10, 24, 23, 4, 3, 1, 1, 24, 4, 2, 24, 8, 17, 24,
              24, 16, 24, 4, 24, 2, 1, 1, 23, 22, 24, 1, 1, 16, 21, 23, 24, 5, 2, 24, 2, 1, 17, 3, 23, 1, 2, 1, 24, 21,
              16, 15, 21, 23, 24, 24, 1, 22, 23, 21, 19, 14, 1, 9, 24, 5, 20, 7, 18, 4, 13, 1, 12, 1, 24, 18, 20, 20,
              24, 5, 17, 24, 1, 21, 12, 1, 24, 24, 24, 23, 1, 3, 1, 23, 2, 24, 1, 1, 1, 17, 24, 24, 22, 23, 13, 11, 14,
              3, 23, 1, 2, 2, 4, 9, 24, 22, 20, 24, 1, 1, 19, 24, 1, 1, 24, 24, 24, 24, 24, 14, 22, 24, 24, 24, 24, 6,
              2, 24, 1, 24, 1, 5, 23, 6, 1, 20, 24, 1, 1, 24, 24, 1, 24, 3, 1, 1, 4, 11, 1, 24, 24, 1, 21, 2, 5, 1, 22,
              1, 5, 1, 10, 1, 12, 24, 23, 8, 2, 1, 24]
    ax2.hist(values, bins=bins, alpha=0.7, color=colors[3], edgecolor=colors[3], density=True)  # 绘制直方图
    ax2.set_xticks(np.arange(min(values), max(values) + 1, 1))
    ax2.set_xticks([1, 5, 9, 13, 17,21,25])
    ax2.set_yticks([0,0.10,0.20])
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.tick_params(axis="both", labelsize=20)  # x 轴刻度字体大小
    ax2.set_ylabel(r'$f_X(\text{x})$', fontsize=24)
    ax2.set_xlabel(r'x', fontsize=24)
    # ax2.text(0.2, max(values) * 0.9, r'$h_{ij} = 25$', fontsize=20)
    ax2.text(0.35, 0.2, r'$N = 10^4$'+"\n" +r"$E[D] = 5$"+ "\n"+ r'$\beta = 4$',
             transform=ax2.transAxes,
             fontsize=20,
             bbox=None)

    # plt.ylabel(r'$f_{h(q_{max})}(x)$', fontsize=35)
    # plt.xticks(fontsize=28)
    # plt.yticks(fontsize=28)

    plt.tight_layout()

    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\formaxhop\\distancetosinglenode\\maxdev_nodehop{Nn}ED{EDn}Beta{betan}Hop2425.svg".format(
            Nn=N, EDn=ED, betan=beta, key=key)

    plt.savefig(
        picname,
        format="svg",
        bbox_inches='tight',  # 紧凑边界
        transparent=True  # 背景透明，适合插图叠加
    )

    plt.show()



if __name__ == '__main__':
    # plot_distribution(50)
    # plot_maxdev_node_hocount(10000, 5, 4)
    # radius = math.sqrt(5/((10000-1)*math.pi))
    # print(0.52/radius)

    # plot_maxdev_node_hocount(10000, 5, 4)
    # plot_maxdev_node_hocount_for_twohop(10000, 5, 4)

    # plot_mindev_node_hocount(10000, 5, 4)

    plot_maxmindev_node_hocount_together(10000, 5, 4)