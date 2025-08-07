# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 22-4-2025
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


def load_large_network_results(N, beta, kvec):
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
                        if N > 200:
                            real_avg_name = folder_name + "real_avg_N{Nn}ED{EDn}beta{betan}Simu{ST}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                            real_avg = np.loadtxt(real_avg_name)
                            real_ave_degree_vec.append(real_avg)
                        else:
                            real_ave_degree_name = folder_name + "real_ave_degree_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                            real_avg = np.loadtxt(real_ave_degree_name)
                            real_ave_degree_vec.append(np.mean(real_avg))

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

def testanalticL(N,k_vals):
    pi = np.pi
    # k 的取值范围
    logN = np.log(N)
    k_vals = np.array(k_vals)
    # 计算 h(k)
    h_vals = (2 / 3) * np.sqrt(k_vals / (N * pi)) * (logN / np.log(k_vals))
    return h_vals

def plot_devandL_withED():
    # the x-axis is the input average degree
    N = 1000
    beta = 256
    kvec_dict = {
        100: [2, 3, 4, 5, 6, 8, 10, 12, 14, 17, 22, 27, 33, 40, 49, 60, 73, 89, 113, 149, 198, 260, 340, 446, 584,
              762, 993, 1292, 1690, 2276, 3142, 4339],
        1000: [2, 3, 4, 5, 6, 7, 8, 11, 15, 20, 28, 40, 58, 83, 118, 169, 241, 344, 490, 700, 999, 1425, 2033, 2900,
               4139, 5909, 8430, 12039, 17177, 24510, 34968, 49887, 71168],
        10000: [2.2, 3.0, 3.8, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276,
                3727, 6105,
                9999, 16479, 27081, 44767, 73534, 121205, 199999]}
    kvec = kvec_dict[N]

    # real_ave_degree_dict = {}
    # ave_deviation_dict = {}
    # std_deviation_dict = {}
    # kvec_dict = {}

    real_ave_degree_vec, ave_deviation_vec, std_deviation_vec, ave_edgelength_vec, std_edgelength_vec, ave_hop_vec, std_hop_vec, ave_L_vec, std_L_vec = load_large_network_results(
        N, beta, kvec)

    fig, ax = plt.subplots(figsize=(16, 12))

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    x = real_ave_degree_vec
    plt.errorbar(x, ave_deviation_vec, yerr=std_deviation_vec, linestyle="-", linewidth=3, elinewidth=1, capsize=5,
                 marker='o', markersize=16,
                 label=r"$\langle d \rangle$", color=colors[0])
    plt.errorbar(x, ave_L_vec, yerr=std_L_vec, linestyle="-", linewidth=3, elinewidth=1, capsize=5,
                 marker='o', markersize=16,
                 label=r"$\langle L \rangle = \langle d_E \rangle h$", color=colors[1])
    plt.plot(x, ave_edgelength_vec, linestyle="-", linewidth=3,
                 marker='s', markersize=16,
                 label=r"$\langle d_E \rangle $", color=colors[2])
    plt.plot(x, ave_hop_vec, linestyle="-", linewidth=3,
                 marker='^', markersize=16,
                 label=r"$\langle h \rangle $", color=colors[3])

    h_a = testanalticL(N, x)
    plt.plot(x, h_a, label=r'$h(k) = \frac{2}{3} \sqrt{\frac{k}{N \pi}} \cdot \frac{\log N}{\log k}$')

    text = fr"$N = 10^4$, $\beta = {beta}$"
    ax.text(
        0.5, 0.85,  # 文本位置（轴坐标，0.5 表示图中央，1.05 表示轴上方）
        text,
        transform=ax.transAxes,  # 使用轴坐标
        fontsize=26,  # 字体大小
        ha='center',  # 水平居中对齐
        va='bottom',  # 垂直对齐方式
    )

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)

    # plt.yticks([0, 0.1, 0.2, 0.3])
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'Expected degree, $\mathbb{E}[D]$', fontsize=26)
    plt.ylabel(r'Average deviation, $\langle d \rangle$', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=26, loc=(0.6, 0.02))
    plt.tick_params(axis='both', which="both", length=6, width=1)
    # picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\LocalOptimumdiffNBeta{betan}.png".format(
    #     betan=beta)
    # plt.savefig(picname, format='png', bbox_inches='tight', dpi=600,transparent=True)
    folder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\"
    picname = folder_name + f"dev_vs_avg_N{N}beta{beta}.svg"
    # plt.savefig(
    #     picname,
    #     format="svg",
    #     bbox_inches='tight',  # 紧凑边界
    #     transparent=True  # 背景透明，适合插图叠加
    # )
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.show()
    plt.close()

    # fig, ax = plt.subplots(figsize=(9, 6))
    # colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    #
    # print(ave_L_vec)
    # print(ave_deviation_vec)
    # print(kvec)
    # print(np.log(kvec))
    #
    # # fitcurve
    # curvefitpoint = 14
    # logx = np.log10(ave_L_vec[curvefitpoint:])
    # logy = np.log10(ave_deviation_vec[curvefitpoint:])
    #
    # # 线性拟合 log-log 数据
    # slope, intercept, r_value, p_value, std_err = linregress(logx, logy)
    #
    # x_fit = np.linspace(min(ave_L_vec[curvefitpoint:]), max(ave_L_vec[curvefitpoint:]), 200)
    # y_fit = 10 ** intercept * x_fit ** slope
    # plt.plot(x_fit, y_fit, 'r--', linewidth=2,
    #          label=fr'Fit: $\langle d \rangle = {10 ** intercept:.2f} \cdot \langle L \rangle^{{{slope:.2f}}}$')
    #
    # # fitcurve
    # curvefitpoint2 = 6
    # logx = np.log10(ave_L_vec[:curvefitpoint2])
    # logy = np.log10(ave_deviation_vec[:curvefitpoint2])
    #
    # # 线性拟合 log-log 数据
    # slope, intercept, r_value, p_value, std_err = linregress(logx, logy)
    #
    # x_fit = np.linspace(min(ave_L_vec[:curvefitpoint2]), max(ave_L_vec[:curvefitpoint2]), 200)
    # y_fit = 10 ** intercept * x_fit ** slope
    # plt.plot(x_fit, y_fit, 'r--', linewidth=2,
    #          label=fr'Fit: $\langle d \rangle = {10 ** intercept:.2f} \cdot \langle L \rangle^{{{slope:.2f}}}$')
    #
    #
    #
    # scatter = plt.scatter(
    #     ave_L_vec,  # x-axis: L
    #     ave_deviation_vec,  # y-axis: d
    #     c=np.log10(kvec),  # color-coded by E[D]
    #     cmap='viridis',  # or try 'plasma', 'coolwarm', etc.
    #     s=150,
    #     marker='o',
    #     edgecolor='k'  # optional: black edge for better visibility
    # )
    # cbar = plt.colorbar(scatter)
    # cbar.set_label('log(E[D])', fontsize=12)
    #
    # highlight_index = 3
    # plt.scatter(
    #     ave_L_vec[highlight_index],
    #     ave_deviation_vec[highlight_index],
    #     color='none',
    #     edgecolor='red',
    #     s=300,
    #     marker='s',
    #     label='Local peak',
    #     zorder=5
    # )
    #
    # subave_L_vec = ave_L_vec[highlight_index:]
    # subave_ave_deviation_vec = ave_deviation_vec[highlight_index:]
    # min_index_in_sub = np.argmin(subave_L_vec)
    # min_index_in_sub2 = np.argmin(subave_ave_deviation_vec)
    # plt.scatter(
    #     ave_L_vec[min_index_in_sub],
    #     subave_ave_deviation_vec[min_index_in_sub],
    #     color='none',
    #     edgecolor='green',
    #     s=300,
    #     marker='^',
    #     label=r'Local min of $\langle L \rangle$',
    #     zorder=5
    # )
    #
    # plt.scatter(
    #     ave_L_vec[min_index_in_sub2],
    #     subave_ave_deviation_vec[min_index_in_sub2],
    #     color='none',
    #     edgecolor='blue',
    #     s=300,
    #     marker='v',
    #     label=r'Local min of $\langle d \rangle$',
    #     zorder=5
    # )
    #
    # text = fr"$N = 10^4$, $\beta = {beta}$"
    # ax.text(
    #     0.3, 1.05,  # 文本位置（轴坐标，0.5 表示图中央，1.05 表示轴上方）
    #     text,
    #     transform=ax.transAxes,  # 使用轴坐标
    #     fontsize=26,  # 字体大小
    #     ha='center',  # 水平居中对齐
    #     va='bottom'  # 垂直对齐方式
    # )
    #
    # # plt.xscale('log')
    # plt.ylabel(r'$\langle d \rangle$', fontsize=32)
    # plt.xlabel(r'$\langle L \rangle$', fontsize=32)
    # plt.xticks(fontsize=26)
    # plt.yticks(fontsize=26)
    #
    # plt.xscale("log")
    # plt.yscale("log")
    #
    # plt.legend(fontsize=20, loc='upper left')
    # plt.tick_params(axis='both', which="both", length=6, width=1)
    # picname = folder_name + f"scattor_L_vs_DEV_N{N}beta{beta}.svg"
    # # plt.savefig(picname, format='svg', bbox_inches='tight', transparent=True)
    # plt.show()


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


def power_law(x, a, b):
    return a * x**b

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """
    # STEP 1 plot local optimum: deviation versus expected degree
    """
    plot_devandL_withED()

    """
    # STEP 2 test Giles
    """
    # plot_devandL_with_geodistance()
