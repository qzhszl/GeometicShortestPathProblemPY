# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 2024/9/8
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

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
        # small network only have 0
        SPhopcount_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\formaxhop\\sphopcountmax_dev_node_hopcount_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
        SP_hopcount_foronesimu = np.loadtxt(SPhopcount_name,dtype=int)
        SP_hopcount.extend(SP_hopcount_foronesimu)

        max_dev_node_hopcount_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\formaxhop\\max_dev_node_hopcount_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
        max_dev_node_hopcount_forone = np.loadtxt(max_dev_node_hopcount_name,dtype=int)
        max_dev_node_hopcount.extend(max_dev_node_hopcount_forone)

    data_dic = {}
    for hp_index in range(len(SP_hopcount)):
        if SP_hopcount[hp_index] in data_dic.keys():
            data_dic[SP_hopcount[hp_index]].append(max_dev_node_hopcount[hp_index])
        else:
            data_dic[SP_hopcount[hp_index]] = [max_dev_node_hopcount[hp_index]]
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    for key, values in data_dic.items():
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        bins = np.arange(min(values) - 0.5, max(values) + 1.5, 1)  # 间隔为1的bin，确保每个柱中心对齐刻度线
        plt.hist(values, bins=bins,alpha=0.7, color=colors[3], edgecolor=colors[3],density=True)  # 绘制直方图
        plt.xticks(np.arange(min(values), max(values) + 1, 1))
        # plt.xlim([0, 1])
        # plt.yticks([0, 5, 10, 15, 20, 25])
        # plt.yticks([0, 10, 20, 30, 40, 50])

        plt.xlabel(r'x', fontsize=35)
        plt.ylabel(r'$f_{h(q_{max})}(x)$', fontsize=35)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\formaxhop\\maxdev_nodehop{Nn}ED{EDn}Beta{betan}Hop{key}.pdf".format(
            Nn=N, EDn=ED, betan=beta,key = key)
        # plt.show()
        plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
        # 清空图像，以免影响下一个图
        plt.close()


if __name__ == '__main__':
    # plot_distribution(50)
    plot_maxdev_node_hocount(10000, 5, 4)