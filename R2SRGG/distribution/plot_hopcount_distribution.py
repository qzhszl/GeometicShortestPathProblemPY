# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 2025/2/13
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.pyplot import figure


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
    ED_vec = [5.0,10,92,1389]
    # kvec = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    legend_label = [5,10,r"$10^2$",r"$10^3$"]
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
    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\hop_distribution_diffED{Nn}Beta{betan}2.pdf".format(
        Nn=N, betan=beta)
    plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    # 清空图像，以免影响下一个图
    plt.close()



def plot_hop_distribution_for_diff_ED_onehop(N,beta):
    ED_vec = [11,107,1389]
    # kvec = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    legend_label = [5,10,r"$10^2$",r"$10^3$"]
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    count = 0
    excempt_vec  =[]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for ED in ED_vec:
        SP_hopcount = []
        for ExternalSimutime in range(5):
            try:
                SPhopcount_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\OneSP\\hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
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
    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\OneSP\\hop_distribution_diffED{Nn}Beta{betan}2.pdf".format(
        Nn=N, betan=beta)
    plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    # 清空图像，以免影响下一个图
    plt.close()



def how_hop_distribution_change_with_diffED():
    """
    1. dev change with ED
    2. hop change with ED
    :return:
    """
    kvec = [11, 12, 14, 16, 18, 20, 21, 23, 24, 27, 28, 30, 32, 33, 34, 35, 39, 42, 44, 47, 50, 56, 68, 71, 73, 74, 79,
            82, 85, 91, 95]
    # kvec = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    N = 10000
    beta = 128
    ave_deviation_vec = []
    std_deviation_vec = []

    for ED in kvec:
        ave_deviation_for_a_para_comb = []
        for ExternalSimutime in range(5):
            try:
                deviation_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\OneSP\\ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                ave_deviation_for_a_para_comb_10times = np.loadtxt(deviation_vec_name)
                ave_deviation_for_a_para_comb.extend(ave_deviation_for_a_para_comb_10times)
            except FileNotFoundError:
                print(ED)
        ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
        std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))

        SP_hopcount = []
        for ExternalSimutime in range(5):
            try:
                SPhopcount_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\OneSP\\hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                SP_hopcount_foronesimu = np.loadtxt(SPhopcount_name, dtype=int)
                SP_hopcount.extend(SP_hopcount_foronesimu)
            except:
                pass
        # try:
        #     plt.figure()
        #     plt.title(f"{ED}")
        #     bins = np.arange(min(SP_hopcount) - 0.5, max(SP_hopcount) + 1.5, 1)  # 间隔为1的bin，确保每个柱中心对齐刻度线
        #     plt.hist(SP_hopcount, bins=bins, alpha=0.7, density=True)  # 绘制直方图
        #     unique_values, counts = np.unique(SP_hopcount, return_counts=True)
        #     plt.show()
        #     plt.close()
        # except:
        #     pass

    plt.figure()
    x = kvec
    y = ave_deviation_vec
    error = std_deviation_vec
    plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=16,
                 label="128")
    plt.xscale("log")
    plt.yscale("log")
    plt.show()



def plot_hop_distribution_for_diff_beta(N,ED):
    beta_vec = [2.2,3.0,4.2,5.9,8.3,11.7,16.5,23.2,32.7,46.1,64.9,91.5,128.9,181.7,256]

    beta_vec = [ 3.0, 4.2, 16.5, 64.9, 128.9]
    # kvec = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    # legend_label = [5,10,r"$10^2$",r"$10^3$"]
    # colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    count = 0
    excempt_vec  =[]
    maxhopnumvec = []
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for beta in beta_vec:
        SP_hopcount = []
        for ExternalSimutime in range(20):
            try:
                SPhopcount_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                SP_hopcount_foronesimu = np.loadtxt(SPhopcount_name, dtype=int)
                SP_hopcount.extend(SP_hopcount_foronesimu)
            except:
                excempt_vec.append((ED,ExternalSimutime))


        bins = np.arange(min(SP_hopcount) - 0.5, max(SP_hopcount) + 1.5, 1)  # 间隔为1的bin，确保每个柱中心对齐刻度线
        plt.hist(SP_hopcount, bins=bins, alpha=0.7, density=True, label=f"beta:{beta}")  # 绘制直方图
        unique_values, counts = np.unique(SP_hopcount, return_counts=True)

        maxidx = np.argmax(counts)
        maxhopnumvec.append(unique_values[maxidx])

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
    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\hop_distribution_diffED{Nn}Beta{betan}2.pdf".format(
        Nn=N, betan=beta)
    # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    # 清空图像，以免影响下一个图
    plt.close()


    # figure()
    # plt.plot(beta_vec, maxhopnumvec)
    # plt.show()


if __name__ == '__main__':
    # plot_distribution(50)
    # plot_hop_distribution(10000, 5, 4)

    # plot_hop_distribution_for_diff_ED(10000, 4)

    # how_hop_distribution_change_with_diffED()

    # plot_hop_distribution_for_diff_ED_onehop(10000, 4)

    plot_hop_distribution_for_diff_beta(10000, 5)