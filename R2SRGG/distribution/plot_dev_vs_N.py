# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 22-8-2024
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from R2SRGG.R2SRGG import loadSRGGandaddnode
from collections import defaultdict
import math
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.special import comb


def plot_local_optimum_with_N(ED, beta):
    # Figure 3 c!
    Nvec = [22, 46, 100, 215, 464, 1000, 2154, 4642, 10000]
    ave_deviation_dict = {}
    std_deviation_dict = {}
    filefolderlarge = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\"
    filefoldersmall = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\"
    for N in Nvec:
        if N < 400:
            for ED in [ED]:
                filename = filefoldersmall+ "ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                Nn=N, EDn=ED, betan=beta, ST=0)
                data = np.loadtxt(filename)
                ave_deviation_dict[N] = np.mean(data)
                std_deviation_dict[N] = np.std(data)
        elif N < 10000:
            for ED in [ED]:
                filename = filefolderlarge + "ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=0)
                data = np.loadtxt(filename)
                ave_deviation_dict[N] = np.mean(data)
                std_deviation_dict[N] = np.std(data)
        else:
            for ED in [ED]:
                ave_deviation_for_a_para_comb = []
                for ExternalSimutime in range(20):
                    try:
                        deviation_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        ave_deviation_for_a_para_comb_10times = np.loadtxt(deviation_vec_name)
                        ave_deviation_for_a_para_comb.extend(ave_deviation_for_a_para_comb_10times)
                    except:
                        pass
                ave_deviation_dict[N] = np.mean(ave_deviation_for_a_para_comb)
                std_deviation_dict[N] = np.std(ave_deviation_for_a_para_comb)


    fig, ax = plt.subplots(figsize=(6, 4.5))
    # colors = [[0, 0.4470, 0.7410],
    #           [0.8500, 0.3250, 0.0980],
    #           [0.9290, 0.6940, 0.1250],
    #           [0.4940, 0.1840, 0.5560],
    #           [0.4660, 0.6740, 0.1880]]
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    y = []
    error =[]
    for N_index in range(len(Nvec)):
        N = Nvec[N_index]
        y.append(ave_deviation_dict[N])
        error.append(std_deviation_dict[N])
    plt.errorbar(Nvec, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=16, color=colors[4])

    text = r"$\mathbb{E}[D] = 10$, $\beta = 8$"
    plt.text(
        0.5, 0.85,  # 文本位置（轴坐标，0.5 表示图中央，1.05 表示轴上方）
        text,
        transform=ax.transAxes,  # 使用轴坐标
        fontsize=30,  # 字体大小
        ha='center',  # 水平居中对齐
        va='bottom'  # 垂直对齐方式
    )
    plt.xscale('log')
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    plt.xlabel(r'$N$', fontsize=28)
    plt.ylabel(r'$\langle d \rangle$', fontsize=28)
    plt.xscale('log')
    # plt.yscale('log')
    plt.xticks(fontsize=30)
    yticks = np.arange(0,0.21,0.1)
    plt.yticks(yticks, fontsize=30)
    # plt.legend(fontsize=26, loc=(0.6, 0.5))
    plt.tick_params(axis='both', which="both", length=6, width=1)

    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\DeviationVsNlogx.svg".format(
        EDn=ED)
    plt.savefig(
        picname,
        format="svg",
        bbox_inches='tight',  # 紧凑边界
        transparent=True  # 背景透明，适合插图叠加
    )

    # picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\DeviationVsNlogx.pdf".format(
    #     EDn=ED)
    # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.show()


def plot_local_optimum_with_N_loglog(ED, beta):
    Nvec = [22, 46, 100, 215, 464, 1000, 2154, 4642, 10000]
    ave_deviation_dict = {}
    std_deviation_dict = {}
    filefolderlarge = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\"
    filefoldersmall = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\"
    for N in Nvec:
        if N < 400:
            for ED in [ED]:
                filename = filefoldersmall+ "ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                Nn=N, EDn=ED, betan=beta, ST=0)
                data = np.loadtxt(filename)
                ave_deviation_dict[N] = np.mean(data)
                std_deviation_dict[N] = np.std(data)
        elif N < 10000:
            for ED in [ED]:
                filename = filefolderlarge + "ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=0)
                data = np.loadtxt(filename)
                ave_deviation_dict[N] = np.mean(data)
                std_deviation_dict[N] = np.std(data)
        else:
            for ED in [ED]:
                ave_deviation_for_a_para_comb = []
                for ExternalSimutime in range(20):
                    try:
                        deviation_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        ave_deviation_for_a_para_comb_10times = np.loadtxt(deviation_vec_name)
                        ave_deviation_for_a_para_comb.extend(ave_deviation_for_a_para_comb_10times)
                    except:
                        pass
                ave_deviation_dict[N] = np.mean(ave_deviation_for_a_para_comb)
                std_deviation_dict[N] = np.std(ave_deviation_for_a_para_comb)


    fig, ax = plt.subplots(figsize=(6, 4.5))
    # colors = [[0, 0.4470, 0.7410],
    #           [0.8500, 0.3250, 0.0980],
    #           [0.9290, 0.6940, 0.1250],
    #           [0.4940, 0.1840, 0.5560],
    #           [0.4660, 0.6740, 0.1880]]
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    y = []
    error =[]
    for N_index in range(len(Nvec)):
        N = Nvec[N_index]
        y.append(ave_deviation_dict[N])
        error.append(std_deviation_dict[N])
    plt.errorbar(Nvec, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=16, color=colors[4])

    # curve_fit:
    popt, pcov = curve_fit(fit_func, Nvec[:-1], y[:-1])
    a_fit  = popt[0]

    # 拟合曲线点
    a_fit = 1
    N_fit = np.linspace(min(Nvec[:-1]), max(Nvec[:-1]), 200)
    y_fit = fit_func(N_fit, a_fit)

    plt.plot(N_fit, y_fit, 'r-', linewidth=2, label=fr'Fit: $y = \log(N) / N^2$')

    popt2, pcov2 = curve_fit(power_law, Nvec[:-1], y[:-1])
    a2, alpha2 = popt2

    # 拟合曲线
    N_fit2 = np.linspace(min(Nvec[:-1]), max(Nvec[:-1]), 200)
    y_fit2 = power_law(N_fit2, a2, alpha2)
    plt.plot(N_fit2, y_fit2, 'r-', linewidth=2,
             label=fr'Fit: $y = {a2:.2f} \cdot N^{{{alpha2:.2f}}}$',color = "blue")

    text = r"$\mathbb{E}[D] = 10$, $\beta = 8$"
    plt.text(
        0.5, 1.05,  # 文本位置（轴坐标，0.5 表示图中央，1.05 表示轴上方）
        text,
        transform=ax.transAxes,  # 使用轴坐标
        fontsize=30,  # 字体大小
        ha='center',  # 水平居中对齐
        va='bottom'  # 垂直对齐方式
    )
    # plt.xscale('log')
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    plt.xlabel(r'$N$', fontsize=28)
    plt.ylabel(r'$\langle d \rangle$', fontsize=28)
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(fontsize=30)
    # yticks = np.arange(0,0.21,0.1)
    plt.yticks( fontsize=30)
    plt.legend(fontsize=20, loc=(0.05, 0.1))
    plt.tick_params(axis='both', which="both", length=6, width=1)

    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\DeviationVsNlogxlogy.svg".format(
        EDn=ED)
    plt.savefig(
        picname,
        format="svg",
        bbox_inches='tight',  # 紧凑边界
        transparent=True  # 背景透明，适合插图叠加
    )

    # picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\DeviationVsNlogx.pdf".format(
    #     EDn=ED)
    # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.show()

def power_law(N, a, alpha):
    return a * N**alpha

def fit_func(N, a):
    return a * np.log(N) / N**2


def equation(R, N):
    if R <= 0:
        return np.inf  # 防止非正数出现在 log 或指数中
    return (1 - np.pi * R**2)**N - 4 * R**2


# 原始方程：
# sum_{k=0}^{n} k * C(n,k) * p^k * (1-p)^{n-k} = 1
# p = (1 - pi * R^2)^N
# n = 1 / (4 * R^2)

def binomial_expectation(R, N, k_max=200):
    if R <= 0:
        return np.inf
    n = int(1 / (4 * R**2))
    if n == 0 or n > k_max:
        return np.inf  # 限制最大项以避免极慢或溢出
    p = (1 - np.pi * R**2)**N
    expectation = 0.0
    for k in range(1, n + 1):
        term = k * comb(n, k) * (p**k) * ((1 - p)**(n - k))
        expectation += term
    return expectation - 1



def plot_dev_with_N_loglog_realavg(ED, beta):
    """
    simurecord
    :param ED:
    :param beta:
    :return:
    simurecord
    1. average deviation does not follow log(N)/N^2
    2. min average deviation does not follow log(N)/N^2
    3. ave deviation at local minimum does not follow log(N)/N^2
    4. exact sum # sum_{k=0}^{n} k * C(n,k) * p^k * (1-p)^{n-k} = 1
                # p = (1 - pi * R^2)^N
                # n = 1 / (4 * R^2)
                is good fitted by (1 - pi * R^2)^N = 4 * R^2
    """


    # Nvec = [22, 46, 100, 215, 464, 1000, 2154, 4642, 10000]
    Nvec = [22, 46, 100, 215, 464, 1000, 2154, 4642]
    ave_deviation_dict = {}
    std_deviation_dict = {}
    min_ave_dict = {}
    filefolderlarge = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\realavg\\"
    filefoldersmall = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\realavg\\"
    for N in Nvec:
        if N < 110:
            for ED in [ED]:
                filename = filefoldersmall+ "ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                Nn=N, EDn=ED, betan=beta, ST=0)
                data = np.loadtxt(filename)
                ave_deviation_dict[N] = np.mean(data)
                std_deviation_dict[N] = np.std(data)
                min_ave_dict[N] = np.min(data)
        else:
            for ED in [ED]:
                filename = filefolderlarge + "ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=0)
                data = np.loadtxt(filename)
                ave_deviation_dict[N] = np.mean(data)
                std_deviation_dict[N] = np.std(data)
                min_ave_dict[N] = np.min(data)


    fig, ax = plt.subplots(figsize=(6, 4.5))
    # colors = [[0, 0.4470, 0.7410],
    #           [0.8500, 0.3250, 0.0980],
    #           [0.9290, 0.6940, 0.1250],
    #           [0.4940, 0.1840, 0.5560],
    #           [0.4660, 0.6740, 0.1880]]
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    y = []
    error =[]
    y1 = []
    for N_index in range(len(Nvec)):
        N = Nvec[N_index]
        y.append(ave_deviation_dict[N])
        y1.append(min_ave_dict[N])
        error.append(std_deviation_dict[N])
    plt.errorbar(Nvec, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=16, color=colors[4])

    # deviation
    # plt.plot(Nvec, y1, linestyle="--", linewidth=3, marker='s', markersize=16,
    #              color=colors[3])


    # local min ave deviation
    # plt.plot([10,100,1000,10000],[0.17,0.1,0.06,0.04], linestyle="--", linewidth=3, marker='s', markersize=16,
    #              color=colors[3])
    # plt.plot([100, 1000, 10000], [0.06, 0.034, 0.018], linestyle="--", linewidth=3, marker='s', markersize=16,
    #          color=colors[3])

    # # sum_{k=0}^{n} k * C(n,k) * p^k * (1-p)^{n-k} = 1
    # # p = (1 - pi * R^2)^N
    # # n = 1 / (4 * R^2)
    # # 设置 N 从 1 到 1000（）
    # N_values = np.arange(10, 10000,10)
    # R_values = []
    # # 初始猜测
    # R0 = 0.1
    # for N in N_values:
    #     try:
    #         R_solution, = fsolve(binomial_expectation, R0, args=(N,))
    #         R_values.append(R_solution)
    #         R0 = R_solution  # 用上一个作为初始猜测
    #     except Exception as e:
    #         R_values.append(np.nan)
    #         print(f"Failed for N = {N}: {e}")
    # plt.plot(N_values, R_values, linewidth=5,color='purple', label='R vs N (exact sum)')

    # (1 - pi * R^2)^N = 4 * R^2
    N_values = np.arange(10, 10001)
    R_values = []
    # 初始猜测值（重要）
    R0 = 0.1
    # 遍历每个 N，数值解方程
    for N in N_values:
        R_solution, = fsolve(equation, R0, args=(N))
        R_values.append(R_solution)
        R0 = R_solution  # 用上一个解作为下一个初始猜测，提高稳定性
    plt.plot(N_values, R_values, color='green', label='R vs N')

    # curve_fit:
    popt, pcov = curve_fit(fit_func, Nvec[:-1], y[:-1])
    a_fit  = popt[0]

    # 拟合曲线点
    a_fit = 1
    N_fit = np.linspace(min(Nvec[:-1]), max(Nvec[:-1]), 200)
    y_fit = fit_func(N_fit, a_fit)

    # plt.plot(N_fit, y_fit, 'r-', linewidth=2, label=fr'Fit: $y = \log(N) / N^2$')

    # curve_fit2:
    popt2, pcov2 = curve_fit(power_law, Nvec[:-1], y[:-1])
    a2, alpha2 = popt2

    # 拟合曲线
    N_fit2 = np.linspace(min(Nvec[:-1]), max(Nvec[:-1]), 200)
    y_fit2 = power_law(N_fit2, a2, alpha2)
    plt.plot(N_fit2, y_fit2, '-', linewidth=2,
             label=fr'Fit: $y = {a2:.2f} \cdot N^{{{alpha2:.2f}}}$',color = "blue")

    text = r"$\langle D \rangle = 10$, $\beta = 128$"
    plt.text(
        0.5, 1.05,  # 文本位置（轴坐标，0.5 表示图中央，1.05 表示轴上方）
        text,
        transform=ax.transAxes,  # 使用轴坐标
        fontsize=30,  # 字体大小
        ha='center',  # 水平居中对齐
        va='bottom'  # 垂直对齐方式
    )
    # plt.xscale('log')
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    plt.xlabel(r'$N$', fontsize=28)
    plt.ylabel(r'$\langle d \rangle$', fontsize=28)
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(fontsize=30)
    # yticks = np.arange(0,0.21,0.1)
    plt.yticks( fontsize=30)
    plt.legend(fontsize=20, loc=(0.05, 0.1))
    plt.tick_params(axis='both', which="both", length=6, width=1)

    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\DeviationVsNlogxlogy_realavg.svg".format(
        EDn=ED)
    plt.savefig(
        picname,
        format="svg",
        bbox_inches='tight',  # 紧凑边界
        transparent=True  # 背景透明，适合插图叠加
    )

    # picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\DeviationVsNlogx.pdf".format(
    #     EDn=ED)
    # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.show()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """
    Plot deviation versus different N
    """
    # plot_local_optimum_with_N(10, 8)

    # plot_local_optimum_with_N_loglog(10, 8)

    plot_dev_with_N_loglog_realavg(10, 128)
