# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 2025/2/13
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def plot_hop_distribution(N, ED, beta):
    SP_hopcount = []
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    for ExternalSimutime in range(20):
        # small network only have 0
        SPhopcount_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\formaxhop\\sphopcountmax_dev_node_hopcount_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
        SP_hopcount_foronesimu = np.loadtxt(SPhopcount_name,dtype=int)
        SP_hopcount.extend(SP_hopcount_foronesimu)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bins = np.arange(min(SP_hopcount) - 0.5, max(SP_hopcount) + 1.5, 1)  # 间隔为1的bin，确保每个柱中心对齐刻度线

    plt.hist(SP_hopcount, bins=bins,alpha=0.7, color=colors[3], edgecolor=colors[3],density=True)  # 绘制直方图
    unique_values, counts = np.unique(SP_hopcount, return_counts=True)

    # 打印结果
    print("值:", unique_values)
    print("频数:", counts)

    # plt.xticks(np.arange(min(SP_hopcount), max(SP_hopcount) + 1, 5))
    # plt.xlim([0, 1])
    # plt.yticks([0, 5, 10, 15, 20, 25])
    # plt.yticks([0, 10, 20, 30, 40, 50])

    plt.xlabel(r'x', fontsize=35)
    plt.ylabel(r'$f_{h}(x)$', fontsize=35)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    text = r"$N = 10^4$" "\n" r"$E[D] = 5$" "\n" r"$\beta = 4$"
    plt.text(
        0.05, 0.65,  # 文本位置（轴坐标，0.5 表示图中央，1.05 表示轴上方）
        text,
        transform=ax.transAxes,  # 使用轴坐标
        fontsize=20,  # 字体大小
        ha='left',  # 水平居中对齐
        va='bottom'  # 垂直对齐方式
    )
    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\formaxhop\\hop_distribution{Nn}ED{EDn}Beta{betan}.pdf".format(
        Nn=N, EDn=ED, betan=beta)
    # plt.show()
    # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    # 清空图像，以免影响下一个图
    # plt.close()

def plot_hop_distribution_for_diff_ED(N,beta):
    ED_vec = [5.0, 10, 44, 193]
    # kvec = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    legend_label = [5,10,50,200]
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    count = 0
    excempt_vec  =[]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for ED in ED_vec:
        SP_hopcount = []
        for ExternalSimutime in range(100):
            try:
                SPhopcount_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                SP_hopcount_foronesimu = np.loadtxt(SPhopcount_name, dtype=int)
                SP_hopcount.extend(SP_hopcount_foronesimu)
            except:
                excempt_vec.append((ED,ExternalSimutime))


        bins = np.arange(min(SP_hopcount) - 0.5, max(SP_hopcount) + 1.5, 1)  # 间隔为1的bin，确保每个柱中心对齐刻度线
        plt.hist(SP_hopcount, bins=bins, alpha=0.7, color=colors[count], edgecolor=colors[count], density=True, label=f"ED:{legend_label[count]}")  # 绘制直方图
        unique_values, counts = np.unique(SP_hopcount, return_counts=True)

        # 打印结果
        print("值:", unique_values)
        print("频数:", counts)
        count = count+1

        # plt.xticks(np.arange(min(SP_hopcount), max(SP_hopcount) + 1, 5))
        # plt.xlim([0, 1])
        # plt.yticks([0, 5, 10, 15, 20, 25])
        # plt.yticks([0, 10, 20, 30, 40, 50])
    print(excempt_vec)
    plt.xlabel(r'x', fontsize=35)
    plt.ylabel(r'$f_{h}(x)$', fontsize=35)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    text = r"$N = 10^4$" "\n" r"$\beta = 4$"
    plt.text(
        0.1, 0.65,  # 文本位置（轴坐标，0.5 表示图中央，1.05 表示轴上方）
        text,
        transform=ax.transAxes,  # 使用轴坐标
        fontsize=20,  # 字体大小
        ha='left',  # 水平居中对齐
        va='bottom'  # 垂直对齐方式
    )
    plt.legend(fontsize=26, loc="best")
    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\hop_distribution_diffED{Nn}Beta{betan}.pdf".format(
        Nn=N, betan=beta)
    plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    # 清空图像，以免影响下一个图
    plt.close()





if __name__ == '__main__':
    # plot_distribution(50)
    # plot_hop_distribution(10000, 5, 4)

    plot_hop_distribution_for_diff_ED(10000, 4)