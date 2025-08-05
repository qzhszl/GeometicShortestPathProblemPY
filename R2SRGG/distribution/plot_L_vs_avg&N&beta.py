# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 05-8-2025
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from fontTools.tfmLib import PASSTHROUGH
from scipy.optimize import curve_fit

from R2SRGG.R2SRGG import loadSRGGandaddnode
from collections import defaultdict
import math
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import linregress


def bin_and_compute_stats(length_geodesic_vec, data_vec, num_bins=100):
    """
    将 length_geodesic_vec 分为 num_bins 个均匀区间，计算每个区间中对应 data_vec 的均值和方差。

    参数：
        length_geodesic_vec: 1D numpy array，表示地理距离（或任意连续变量）
        data_vec: 1D numpy array，与 length_geodesic_vec 一一对应的数据值
        num_bins: 分箱数量，默认100

    返回：
        bin_centers: 每个区间的中点（长度为 num_bins）
        bin_means: 每个区间中对应 data_vec 的均值（长度为 num_bins）
        bin_vars: 每个区间中对应 data_vec 的方差（长度为 num_bins）
    """
    # 将输入转为 numpy 数组
    length_geodesic_vec = np.asarray(length_geodesic_vec)
    data_vec = np.asarray(data_vec)

    # 创建等距的分箱
    bin_edges = np.linspace(length_geodesic_vec.min(), length_geodesic_vec.max(), num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 给每个元素分配所属的 bin
    bin_indices = np.digitize(length_geodesic_vec, bins=bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)  # 防止越界

    # 初始化结果数组
    bin_means = np.zeros(num_bins)
    bin_vars = np.zeros(num_bins)

    # 逐个 bin 计算均值和方差
    for i in range(num_bins):
        bin_data = data_vec[bin_indices == i]
        if len(bin_data) > 0:
            bin_means[i] = np.mean(bin_data)
            bin_vars[i] = np.std(bin_data)
        else:
            bin_means[i] = np.nan  # 没有数据的 bin 用 NaN 标记
            bin_vars[i] = np.nan

    return bin_centers, bin_means, bin_vars


def power_law(x, a, b):
    return a * x**b

def load_large_network_results_dev_vs_avg(N, beta, kvec):
    folder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\"
    exemptionlist = []
    for N in [N]:
        ave_deviation_vec = []
        real_ave_degree_vec = []
        std_deviation_vec = []
        ave_edgelength_vec = []
        std_edgelength_vec = []

        ave_hop_vec = []
        std_hop_vec = []
        ave_L_vec = []
        std_L_vec = []


        for beta in [beta]:
            for ED in kvec:
                for ExternalSimutime in [0]:
                    try:
                        # FileNetworkName = folder_name+"network_N{Nn}ED{EDn}Beta{betan}.txt".format(
                        #     Nn=N, EDn=ED, betan=beta)
                        # G = loadSRGGandaddnode(N, FileNetworkName)
                        # real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
                        # # print("real ED:", real_avg)
                        real_avg_name = folder_name + "real_avg_N{Nn}ED{EDn}beta{betan}Simu{ST}.txt".format(
                            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        real_avg = np.loadtxt(real_avg_name)
                        real_ave_degree_vec.append(real_avg)

                        deviation_vec_name = folder_name + "ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        ave_deviation_for_a_para_comb = np.loadtxt(deviation_vec_name)
                        ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
                        std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))

                        edgelength_vec_name = folder_name + "ave_edgelength_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        ave_edgelength_for_a_para_comb = np.loadtxt(edgelength_vec_name)
                        ave_edgelength_vec.append(np.mean(ave_edgelength_for_a_para_comb))
                        std_edgelength_vec.append(np.std(ave_edgelength_for_a_para_comb))

                        hopcount_Name = folder_name + "hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
                            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        hop_vec = np.loadtxt(hopcount_Name, dtype=int)

                        ave_hop_vec.append(np.mean(hop_vec))
                        std_hop_vec.append(np.std(hop_vec))

                        # L = np.multiply(ave_edgelength_for_a_para_comb, hop_vec)
                        L = [x * y for x, y in zip(ave_edgelength_for_a_para_comb, hop_vec)]

                        ave_L_vec.append(np.mean(L))
                        std_L_vec.append(np.std(L))

                    except FileNotFoundError:
                        exemptionlist.append((N, ED, beta, ExternalSimutime))
    print(exemptionlist)
    return real_ave_degree_vec, ave_deviation_vec, std_deviation_vec, ave_edgelength_vec, std_edgelength_vec, ave_hop_vec, std_hop_vec, ave_L_vec, std_L_vec
    # return kvec, real_ave_degree_vec, ave_deviation_vec, std_deviation_vec


def plot_L_with_avg_for_one_network():
    # the x-axis is the input average degree
    N = 10000
    beta = 4
    kvec = [2.2, 2.8, 3.0, 3.4, 3.8, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105,
            9999, 16479, 27081, 44767, 73534, 121205, 199999]
    # kvec = [2.2, 2.8, 3.0, 3.4, 3.8, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105,
    #         9999,
    #         16479, 27081, 44767, 73534, 121205, 199999]
    # real_ave_degree_dict = {}
    # ave_deviation_dict = {}
    # std_deviation_dict = {}
    # kvec_dict = {}

    real_ave_degree_vec, ave_deviation_vec, std_deviation_vec, ave_edgelength_vec, std_edgelength_vec, ave_hop_vec, std_hop_vec, ave_L_vec, std_L_vec = load_large_network_results_dev_vs_avg(
        N, beta, kvec)

    fig, ax = plt.subplots(figsize=(9, 6))

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']

    plt.errorbar(real_ave_degree_vec, ave_L_vec, yerr=std_L_vec, linestyle="-", linewidth=3, elinewidth=1, capsize=5,
                 marker='o', markersize=16,
                 label=r"$\langle L \rangle$", color=colors[3])

    # text = fr"$N = 10^4$, $\beta = {beta}$"
    # ax.text(
    #     0.3, 0.85,  # 文本位置（轴坐标，0.5 表示图中央，1.05 表示轴上方）
    #     text,
    #     transform=ax.transAxes,  # 使用轴坐标
    #     fontsize=26,  # 字体大小
    #     ha='center',  # 水平居中对齐
    #     va='bottom',  # 垂直对齐方式
    # )

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)

    # plt.yticks([0, 0.1, 0.2, 0.3])
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'$\langle D \rangle$', fontsize=28)
    plt.ylabel(r'$\langle L \rangle$', fontsize=28)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    # plt.legend(fontsize=26, loc=(0.5, 0.1))
    plt.tick_params(axis='both', which="both", length=6, width=1)
    # picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\LocalOptimumdiffNBeta{betan}.png".format(
    #     betan=beta)
    # plt.savefig(picname, format='png', bbox_inches='tight', dpi=600,transparent=True)
    folder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\"
    picname = folder_name + f"L_vs_avg_N{N}beta{beta}.svg"
    plt.savefig(
        picname,
        format="svg",
        bbox_inches='tight',  # 紧凑边界
        transparent=True  # 背景透明，适合插图叠加
    )
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.show()
    plt.close()



def plot_L_with_avg():
    # Figure 3(b)
    # the x-axis is the input average degree
    Nvec = [10, 100, 1000, 10000]
    # Nvec = [1000]
    real_ave_degree_dict = {}
    ave_L = {}
    std_L = {}
    kvec_dict = {10000:[2.2, 2.8, 3.0, 3.4, 3.8, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276,
                        3727, 6105,
                        9999, 16479, 27081, 44767, 73534, 121205, 199999]}

    for N in Nvec:
        if N == 10:
            for beta in [4]:
                degree_vec_resort, ave_deviation_vec, std_deviation_vec, _, _, _ = load_resort_data(N, beta)
                real_ave_degree_dict[N] = degree_vec_resort
                ave_L[N] = ave_deviation_vec
                std_L[N] = std_deviation_vec
                kvec = degree_vec_resort
                real_ave_degree_vec = degree_vec_resort
                kvec_dict[N] = kvec
                real_ave_degree_dict[N] = real_ave_degree_vec
        elif N < 200:
            for beta in [4]:
                kvec, real_ave_degree_vec, ave_deviation_vec, std_deviation_vec = load_small_network_results(N, beta)
                real_ave_degree_dict[N] = [1.3878000000000004, 2.0258000000000003, 2.6606, 3.2968, 3.9018, 5.0796,
                                           6.2132000000000005, 7.311599999999999, 8.342, 9.8808, 12.2334, 14.4614,
                                           17.141800000000003, 19.8532, 23.2854, 26.9778, 31.156, 35.631400000000006,
                                           41.68019999999999, 49.1888, 57.876999999999995, 65.243, 72.5054,
                                           79.60860000000001, 85.1024, 89.85379999999999, 92.81960000000001, 95.0446,
                                           96.645, 97.6358, 98.25179999999999, 98.60859999999998]  # for beta = 4
                # real_ave_degree_dict[N] = [1.3878000000000004, 2.0258000000000003, 2.6606, 3.2968, 3.9018, 5.0796,
                #                            6.2132000000000005, 7.311599999999999, 8.342, 9.8808, 12.2334, 14.4614,
                #                            17.141800000000003, 19.8532, 23.2854, 26.9778, 31.156, 35.631400000000006,
                #                            41.68019999999999, 49.1888, 57.876999999999995, 65.243, 72.5054,
                #                            79.60860000000001, 85.1024, 89.85379999999999, 92.81960000000001, 95.0446,
                #                         97.6358, 98.60859999999998]  # for beta = 4
                ave_L[N] = ave_deviation_vec
                std_L[N] = std_deviation_vec
                kvec_dict[N] = kvec
        else:
            for beta in [4]:
                kvec = kvec_dict[N]
                real_ave_degree_vec, ave_deviation_vec, std_deviation_vec, ave_edgelength_vec, std_edgelength_vec, ave_hop_vec, std_hop_vec, ave_L_vec, std_L_vec = load_large_network_results_dev_vs_avg(
                    N, beta, kvec)
                real_ave_degree_dict[N] = real_ave_degree_vec
                ave_L[N] = ave_L_vec
                std_L[N] = std_L_vec

    # plt.plot(kvec,ave_deviation_vec,"o-")
    # plt.xscale('log')
    # plt.show()
    lengend = [r"$N=10$", r"$N=10^2$", r"$N=10^3$", r"$N=10^4$"]
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    # colorvec2 = ['#9FA9C9', '#D36A6A']
    for N_index in range(len(Nvec)):
        N = Nvec[N_index]
        if N == 10:
            # x = real_ave_degree_dict[N]
            # print(len(x))
            # x = x[1:]
            # y = ave_L[N]
            # y = y[1:]
            # error = std_L[N]
            # error = error[1:]
            x = real_ave_degree_dict[N]
            y = ave_L[N]
            error = std_L[N]

        else:
            x = real_ave_degree_dict[N]
            y = ave_L[N]
            error = std_L[N]
        plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=16,
                     label=lengend[N_index], color=colors[N_index])

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # plt.ylim(0, 0.30)
    # plt.yticks([0, 0.1, 0.2, 0.3])
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'Average degree, $\langle D \rangle$', fontsize=26)
    plt.ylabel(r'Average distance, $\langle L \rangle$', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    # plt.legend(fontsize=26, loc=(0.5, 0.5))
    plt.tick_params(axis='both', which="both", length=6, width=1)
    # picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\LocalOptimumdiffNBeta{betan}.png".format(
    #     betan=beta)
    # plt.savefig(picname, format='png', bbox_inches='tight', dpi=600,transparent=True)
    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\L_vs_realavg.svg"
    plt.savefig(
        picname,
        format="svg",
        bbox_inches='tight',  # 紧凑边界
        transparent=True  # 背景透明，适合插图叠加
    )
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.show()
    plt.close()


def plot_devandL_with_geodistance():
    N = 10000
    beta = 128
    ED = 121
    ExternalSimutime = 0
    folder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\"

    deviation_vec_name = folder_name + "ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    ave_deviation_for_a_para_comb = np.loadtxt(deviation_vec_name)

    edgelength_vec_name = folder_name + "ave_edgelength_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    ave_edgelength_for_a_para_comb = np.loadtxt(edgelength_vec_name)


    hopcount_Name = folder_name + "hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    hop_vec = np.loadtxt(hopcount_Name, dtype=int)

    max_deviation_name = folder_name + "max_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)

    max_dev_vec = np.loadtxt(max_deviation_name)

    length_geodesic_name = folder_name + "length_geodesic_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)

    length_geodesic_vec = np.loadtxt(length_geodesic_name)

    L = [x * y for x, y in zip(ave_edgelength_for_a_para_comb, hop_vec)]
    print(min(length_geodesic_vec))
    print(max(length_geodesic_vec))

    # plt.scatter(length_geodesic_vec,ave_deviation_for_a_para_comb)
    fig, ax = plt.subplots(figsize=(9, 6))

    bin_centers, bin_means, bin_vars = bin_and_compute_stats(length_geodesic_vec, L, num_bins=100)
    bin_centers2, bin_means2, bin_vars2 = bin_and_compute_stats(length_geodesic_vec, max_dev_vec, num_bins=100)

    plt.errorbar(bin_centers,bin_means,bin_vars,label ="L")
    plt.errorbar(bin_centers2, bin_means2, bin_vars2, label =r"$d_{max}$")

    mask = (~np.isnan(bin_means2)) & (bin_centers2 > 0) & (bin_means2 > 0)
    x = bin_centers2[mask][10:-10]
    y = bin_means2[mask][10:-10]

    popt, pcov = curve_fit(power_law, x, y)
    a, b = popt
    x_fit = np.linspace(x.min(), x.max(), 500)
    y_fit = power_law(x_fit, *popt)

    plt.loglog(x_fit, y_fit, 'r-', linewidth = 3, label=f'Fit: y = {a:.3f} * x^{b:.3f}')

    mask = ~np.isnan(bin_means)
    x = bin_centers[mask][10:-5]
    y = bin_means[mask][10:-5]
    slope, intercept = np.polyfit(x, y, deg=1)
    x_fit = np.linspace(x.min(), x.max(), 500)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, 'y-',linewidth = 3, label=f'Fit: y = {slope:.3f} * x + {intercept:.3f}')

    text = fr"$N = 10^4$, $\beta = {beta}$,  $ED = {ED}$"
    ax.text(
        0.3, 1.05,  # 文本位置（轴坐标，0.5 表示图中央，1.05 表示轴上方）
        text,
        transform=ax.transAxes,  # 使用轴坐标
        fontsize=26,  # 字体大小
        ha='center',  # 水平居中对齐
        va='bottom'  # 垂直对齐方式
    )

    plt.legend(fontsize=16)
    # plt.xscale("log")
    # plt.yscale("log")
    plt.ylabel(r'$\langle d \rangle$', fontsize=32)
    plt.xlabel(r'$d_{ij}$', fontsize=32)
    plt.show()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """
    # STEP 1  L versus real average degree
    """
    plot_L_with_avg()

    """
    # STEP 1.5  L versus real average degree for one network
    """
    # plot_L_with_avg_for_one_network()


    """
    # STEP 2 plot L vs N
    """
    # plot_devandL_with_geodistance()

    """
    # STEP 3 plot L vs beta
    """
