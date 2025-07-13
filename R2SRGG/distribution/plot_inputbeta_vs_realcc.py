# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 2025/05/09
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
import math

from scipy.optimize import curve_fit

from R2SRGG.R2SRGG import loadSRGGandaddnode


def plot_CC_vs_beta(ED):
    # Figure 1(a)
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494', "#A2C7A4", "#9DB0C2", "#E3B6A4"]

    filefolder = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\inputavgbeta\\"
    x = [2.2, 3.0, 4.2, 5.9, 8.3, 11.7, 16.5, 23.2, 32.7, 46.1, 64.9, 91.5, 128.9, 181.7, 256]
    real_cc_vec = [0.028519126984126984, 0.15029302364302363, 0.2909357692307692, 0.3975891785991786,
                   0.4529371162171162, 0.4905946753246753, 0.5108323737373738, 0.519032683982684, 0.523615772005772,
                   0.5256693722943723, 0.5258169408369409, 0.5274510822510823, 0.5267747330447331, 0.5263217388167388,
                   0.5263740187590188]  # ed=5
    N = 10000
    # real_cc_vec = []
    # for beta in x:
    #     print("ED:", ED)
    #     filename = filefolder + f"network_N10000ED{ED}Beta{beta}.txt"
    #     G = loadSRGGandaddnode(N, filename)
    #     real_cc = nx.average_clustering(G)
    #     print("real cc:", real_cc)
    #     real_cc_vec.append(real_cc)
    # print(real_cc_vec)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    text = r"$N = 10^4$" "\n" r"$\mathbb{E}[D] = 5$"
    plt.text(
        0.6, 0.35,  # 文本位置（轴坐标，0.5 表示图中央，1.05 表示轴上方）
        text,
        transform=ax.transAxes,  # 使用轴坐标
        fontsize=28,  # 字体大小
        ha='left',  # 水平居中对齐
        va='bottom'  # 垂直对齐方式
    )
    plt.xscale('log')
    # plt.yscale('log')
    plt.plot(x,real_cc_vec,linewidth=5,color=colors[0])
    plt.xlabel(r'Temperature, $\beta$', fontsize=26)
    plt.ylabel(r"CC, $c_G$", fontsize=26)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)

    # picname = filefolder+ "ccvsbetaN{Nn}ED{ED}.pdf".format(
    #     Nn=10000, ED=ED)
    picname = filefolder + "ccvsbetaN{Nn}ED{ED}.svg".format(
        Nn=10000, ED=ED)
    # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.savefig(
        picname,
        format="svg",
        bbox_inches='tight',  # 紧凑边界
        transparent=True  # 背景透明，适合插图叠加
    )
    plt.show()
    plt.close()


if __name__ == '__main__':
    plot_CC_vs_beta(5)
