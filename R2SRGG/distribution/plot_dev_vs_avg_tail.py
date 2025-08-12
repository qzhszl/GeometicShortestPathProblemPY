# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 2024/11/17
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import networkx as nx
from R2SRGG.R2SRGG import loadSRGGandaddnode


def load_10000nodenetwork_results_tail(beta):
    # Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    kvec = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    # kvecsupp = [10, 16, 27, 44, 49, 56, 64, 72, 81, 92, 104, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    # kvec = kvecsupp
    # betavec = [2.1, 4, 8, 16, 32, 64, 128]
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\"

    print(beta)
    exemptionlist = []
    for N in [10000]:
        ave_deviation_vec = []
        std_deviation_vec = []
        real_ave_degree_vec = []
        for beta in [beta]:
            for ED in kvec:
                ave_deviation_for_a_para_comb = np.array([])
                for ExternalSimutime in range(50):
                    try:
                        deviation_vec_name = filefolder_name + "ave_deviation_N{Nn}ED{EDn}beta{betan}Simu{ST}.txt".format(
                            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        ave_deviation_for_a_para_comb_10times = np.loadtxt(deviation_vec_name)
                        ave_deviation_for_a_para_comb = np.hstack(
                            (ave_deviation_for_a_para_comb, ave_deviation_for_a_para_comb_10times))
                    except FileNotFoundError:
                        exemptionlist.append((N, ED, beta, ExternalSimutime))

                ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
                std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))
    print(exemptionlist)
    ave_deviation_Name = filefolder_name + "ave_deviation_N{Nn}_beta{betan}.txt".format(
        Nn=N, betan=beta)
    np.savetxt(ave_deviation_Name, ave_deviation_vec)
    std_deviation_Name = filefolder_name + "std_deviation_N{Nn}_beta{betan}.txt".format(Nn=N,
                                                                                        betan=beta)
    np.savetxt(std_deviation_Name, std_deviation_vec)
    return ave_deviation_vec, std_deviation_vec, exemptionlist


def plot_dev_vs_avg_tail(beta):
    """
    the x-axis is the expected degree, the y-axis is the average deviation,
    when use this function, use load_10000nodenetwork_results_tail(beta) before
    the analytic results are loaded from CommonNeighbour_analytic_check.py
    :return:
    """
    N = 10000
    ave_deviation_dict = {}
    std_deviation_dict = {}

    kvec = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    # kvecsupp_geo01 = [10, 16, 27, 44, 49, 56, 64, 72, 81, 92, 104, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    # kvecsupp_geo05 = [10, 16, 27, 44, 49, 56, 64, 72, 81, 92, 104, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    # kvec = kvecsupp
    betavec = [beta]
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\"

    count = 0
    for beta in betavec:
        ave_deviation_Name = filefolder_name + "ave_deviation_N{Nn}_beta{betan}.txt".format(
            Nn=N, betan=beta)
        ave_deviation_vec = np.loadtxt(ave_deviation_Name)
        std_deviation_Name = filefolder_name + "std_deviation_N{Nn}_beta{betan}.txt".format(Nn=N,
                                                                                            betan=beta)
        std_deviation_vec = np.loadtxt(std_deviation_Name)

        ave_deviation_dict[count] = ave_deviation_vec
        std_deviation_dict[count] = std_deviation_vec
        count = count + 1

    # legend = [r"$\beta=2.2$", r"$\beta=2^2$", r"$\beta=2^3$", r"$\beta=2^4$", r"$\beta=2^5$", r"$\beta=2^7$"]
    legend = [r"average devation of shortest path"]
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = [[0, 0.4470, 0.7410],
              [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250],
              [0.4940, 0.1840, 0.5560],
              [0.4660, 0.6740, 0.1880],
              [0.3010, 0.7450, 0.9330]]
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    for count in range(len(betavec)):
        beta = betavec[count]
        x = kvec
        y = ave_deviation_dict[count]
        print(y)
        error = std_deviation_dict[count]
        print(error)
        plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o',
                     markersize=16, label=legend[count], color=colors[3])
        params, covariance = curve_fit(power_law, x[8:15], y[8:15])
        # 获取拟合的参数
        a_fit, k_fit = params
        # print(f"拟合结果: a = {a_fit}, k = {k_fit}")
        # plt.plot(x[8:15], power_law(x[8:15], *params), linewidth=5, label=f'fit curve: $y={a_fit:.6f}x^{{{k_fit:.4f}}}$',
        #          color=colors[0])

        # curve fit no shelter
        x_curve = x[8:15]
        # a_fit = 0.01
        params = a_fit, k_fit
        y_curve = power_law(x[8:15], *params)
        plt.plot(x_curve, y_curve, linewidth=8,
                 label=f'fit curve: $y={a_fit:.4f}x^{{{k_fit:.2f}}}$',
                 color=colors[0])

        # params, covariance = curve_fit(power_law, x[10:13], y[10:13])
        # a_fit, k_fit = params
        # plt.plot(x[10:13], power_law(x[10:13], *params), linewidth=5,
        #          label=f'fit curve: $y={a_fit:.6f}x^{{{k_fit:.4f}}}$',
        #          color='green')

    # analyticy01 = [0.0408034319117630, 0.0494268009073655,
    #                0.0601867525680230, 0.0710647201399454, 0.0826468337716141, 0.0951878331783714, 0.110052892139220,
    #                0.129841960064955, 0.155820990403079, 0.184934349828600, 0.211071618297930, 0.229651984548116,
    #                0.240562299611654, 0.246042131559725, 0.248443310742824]

    analyticy01 = [0.011210490144748496, 0.014362834005337652, 0.01836412087082358, 0.022948784225781036,
                   0.02873677429889581, 0.03580781814761691, 0.04411338516186235, 0.05362436843859807,
                   0.06415204818089597, 0.07537040591851239, 0.08723985015799253, 0.10043013621960226,
                   0.11689120257351479, 0.13907942793358205, 0.1668336102129492]

    kvecanalyticy01 = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]

    plt.plot(kvecanalyticy01, analyticy01, linestyle="--", marker='v', markersize=16, linewidth=5,
             label=f'analytic results from common neighbour model',
             color=colors[4])
    params, covariance = curve_fit(power_law, kvecanalyticy01[4:11], analyticy01[4:11])
    # 获取拟合的参数
    # a_fit, k_fit = params
    # print(f"拟合结果: a = {a_fit}, k = {k_fit}")
    # plt.plot(kvecanalyticy01[4:11], power_law(kvecanalyticy01[4:11], *params), linewidth=5, label=f'fit curve: $y={a_fit:.6f}x^{{{k_fit:.4f}}}$',
    #          color=colors[1])

    text = r"$N = 10^4,\beta = 4$"
    ax.text(
        0.2, 0.88,  # 文本位置（轴坐标，0.5 表示图中央，1.05 表示轴上方）
        text,
        transform=ax.transAxes,  # 使用轴坐标
        fontsize=26,  # 字体大小
        ha='center',  # 水平居中对齐
        va='bottom'  # 垂直对齐方式
    )

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('E[D]', fontsize=26)
    plt.ylabel('Average deviation', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('1000 simulations, N=10000',fontsize=26)
    plt.legend(fontsize=20, loc="lower right")
    plt.tick_params(axis='both', which="both", length=6, width=1)

    picname = filefolder_name + "tailLocalOptimum_dev_vs_avg_beta{beta}.pdf".format(beta=beta)
    # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()


def plot_dev_vs_avg_tail_vshopcountchange(beta):
    # Figure 4 (c)

    """
    the x-axis is the expected degree, the y-axis is the average deviation,
    when use this function, use load_10000nodenetwork_results_tail(beta) before
    the analytic results analyticy01 are loaded from CommonNeighbour_analytic_check.py
    the hopcount results are loaded from plot_hopcount_vsED.py   plot_hopcount_vs_ED(10000,4)
    :return:
    """
    N = 10000
    ave_deviation_dict = {}
    std_deviation_dict = {}

    kvec = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]

    hop_ave = [11.57, 8.125, 6.1675, 4.9025, 4.01, 3.42, 2.9825, 2.665, 2.4375, 2.1775, 1.9884615384615385, 1.8875,
               1.82, 1.7275, 1.61]

    hop_std = [3.566384724058805, 2.3124391883896105, 1.6399523621129972, 1.2541904759644764, 0.9486305919587454,
               0.7606576102294647, 0.6420231693638477, 0.5410868691809106, 0.5577577879330776, 0.48061809162785374,
               0.3560757515310505, 0.32379584617471546, 0.38418745424597095, 0.44524571867677737, 0.4877499359302879]

    # kvecsupp_geo01 = [10, 16, 27, 44, 49, 56, 64, 72, 81, 92, 104, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    # kvecsupp_geo05 = [10, 16, 27, 44, 49, 56, 64, 72, 81, 92, 104, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    # kvec = kvecsupp
    betavec = [beta]
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\"

    count = 0
    for beta in betavec:
        ave_deviation_Name = filefolder_name + "ave_deviation_N{Nn}_beta{betan}.txt".format(
            Nn=N, betan=beta)
        ave_deviation_vec = np.loadtxt(ave_deviation_Name)
        std_deviation_Name = filefolder_name + "std_deviation_N{Nn}_beta{betan}.txt".format(Nn=N,
                                                                                            betan=beta)
        std_deviation_vec = np.loadtxt(std_deviation_Name)

        ave_deviation_dict[count] = ave_deviation_vec
        std_deviation_dict[count] = std_deviation_vec
        count = count + 1

    legend = [r"distance to geodesic"]
    fig, ax1 = plt.subplots(figsize=(12, 8))

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    for count in range(len(betavec)):
        beta = betavec[count]
        x = kvec
        y = ave_deviation_dict[count]
        print(y)
        error = std_deviation_dict[count]
        print(error)
        ax1.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o',
                     markersize=16, label=legend[count], color=colors[3])
        # params, covariance = curve_fit(power_law, x[8:15], y[8:15])
        # # 获取拟合的参数
        # a_fit, k_fit = params
        # x_curve = x[8:15]
        # # a_fit = 0.01
        # params = a_fit,k_fit
        # y_curve = power_law(x[8:15], *params)
        # ax1.plot(x_curve, y_curve, linewidth=8,
        #          label=f'fit curve: $y={a_fit:.4f}x^{{{k_fit:.2f}}}$',
        #          color=colors[0])

    # analyticy01 = [0.011210490144748496, 0.014362834005337652, 0.01836412087082358, 0.022948784225781036,
    #                0.02873677429889581, 0.03580781814761691, 0.04411338516186235, 0.05362436843859807,
    #                0.06415204818089597, 0.07537040591851239, 0.08723985015799253, 0.10043013621960226,
    #                0.11689120257351479, 0.13907942793358205, 0.1668336102129492]  # analytic results for delta = 0.26
    # analyticy01 = [0.011210490144748496, 0.014362834005337652, 0.01836412087082358, 0.022948784225781036, 0.02873677429889581,
    #  0.03580781814761691, 0.04411338516186235, 0.05362436843859807, 0.06415204818089597, 0.07537040591851239,
    #  0.08723985015799253, 0.10043013621960226, 0.11689120257351479, 0.13907942793358205, 0.1668336102129492,
    #  0.19593857047537333, 0.21951189287819609, 0.23504203529784937, 0.24340647418948685, 0.2473534197139448,
    #  0.24899148388677875, 0.24959105981467355, 0.24983551088466646] # analytic results for delta = 0.26 and ED>N-1

    analyticy01 = [0.01346233272869063, 0.01607585979327422, 0.018851987390967894, 0.02277913016890316,
                   0.028632059275195725,
                   0.035652272244433965, 0.04383613629433062, 0.0531669911534992, 0.06344208584313155,
                   0.0743562636773701,
                   0.08595130698082737, 0.09907028029024749, 0.11583355446534334, 0.1385898397958834,
                   0.16682586907758828,
                   0.196150331337391, 0.21974202990006622, 0.23521497602784194, 0.24350849795791418,
                   0.24740141505893926,
                   0.24901109652447787, 0.24959922300786946,
                   0.24983882982895725]  # analytic results for delta = 0.25 and ED>N-1

    kvecanalyticy01 = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999] # analytic results for delta = 0.26 and ED<=N-1

    # kvecanalyticy01 = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999, 16479, 27081, 44767,
    #                    73534, 121205, 199999, 316226, 499999] # analytic results for ED>N-1

    ax1.plot(kvecanalyticy01, analyticy01[0:len(kvecanalyticy01)], linestyle="--", marker='v', markersize=16, linewidth=5,
             label=f'common neighbor',
             color=colors[4])

    text = r"$N = 10^4,\beta = 4$"
    ax1.text(
        0.82, 0.4,  # 文本位置（轴坐标，0.5 表示图中央，1.05 表示轴上方）
        text,
        transform=ax1.transAxes,  # 使用轴坐标
        fontsize=36,  # 字体大小
        ha='center',  # 水平居中对齐
        va='bottom'  # 垂直对齐方式
    )

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'Expected degree, $\mathbb{E}[D]$', fontsize=36)
    ax1.set_ylabel(r'Average distance, $\langle d \rangle$', fontsize=36, color=colors[3])
    ax1.tick_params(axis='both', which="both", length=6, width=1)
    ax1.tick_params(axis='y', labelcolor=colors[3], labelsize=36)
    ax1.tick_params(axis='x', labelsize=36)
    # plt.title('1000 simulations, N=10000',fontsize=26)

    ax2 = ax1.twinx()
    ax2.set_ylabel(r'Average hopcount, $\langle h \rangle$', color=colors[1], fontsize=36)
    ax2.errorbar(kvec, hop_ave, yerr=hop_std, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='s',
                 markersize=16, label="hopcount", color=colors[1])
    ax2.set_yticks([2,4,6,8,10,12,14])
    ax2.tick_params(axis='y', labelcolor=colors[1], labelsize=34)
    # plt.legend(fontsize=20, loc="lower right")
    plt.xlim([8, 12000])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines2 + lines1, labels2 + labels1,loc='upper left',bbox_to_anchor=(0, 1.03), fontsize=30)

    # picname = filefolder_name + "tailLocalOptimum_dev_vs_ED_beta{beta}_withhop.pdf".format(beta=beta)
    # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)

    picname = filefolder_name + "tailLocalOptimum_dev_vs_ED_beta{beta}_withhop.svg".format(beta=beta)
    # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.savefig(
        picname,
        format="svg",
        bbox_inches='tight',  # 紧凑边界
        transparent=True  # 背景透明，适合插图叠加
    )

    plt.show()
    plt.close()


def load_10000nodenetwork_results_tail_fixnode(beta, Geodistance_index):
    # Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    kvec = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    # kvecsupp = [10, 16, 27, 44, 49, 56, 64, 72, 81, 92, 104, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    # kvec = kvecsupp
    # betavec = [2.1, 4, 8, 16, 32, 64, 128]
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\GivenGeodistance\\1000realization\\tail\\"
    distance_list = [[0.491, 0.5, 0.509, 0.5], [0.25, 0.5, 0.75, 0.5], [0.45, 0.5, 0.55, 0.5]]
    x_A = distance_list[Geodistance_index][0]
    y_A = distance_list[Geodistance_index][1]
    x_B = distance_list[Geodistance_index][2]
    y_B = distance_list[Geodistance_index][3]
    geodesic_distance_AB = x_B - x_A
    geodesic_distance_AB = round(geodesic_distance_AB, 2)
    print(beta, geodesic_distance_AB)
    exemptionlist = []
    for N in [10000]:
        ave_deviation_vec = []
        std_deviation_vec = []
        real_ave_degree_vec = []
        for beta in [beta]:
            for ED in kvec:
                ave_deviation_for_a_para_comb = np.array([])
                for ExternalSimutime in range(50):
                    try:
                        deviation_vec_name = filefolder_name + "Givendistanceave_deviation_N{Nn}ED{EDn}beta{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
                            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime, Geodistance=geodesic_distance_AB)
                        ave_deviation_for_a_para_comb_10times = np.loadtxt(deviation_vec_name)
                        ave_deviation_for_a_para_comb = np.hstack(
                            (ave_deviation_for_a_para_comb, ave_deviation_for_a_para_comb_10times))
                    except FileNotFoundError:
                        exemptionlist.append((N, ED, beta, ExternalSimutime))

                ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
                std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))
    print(exemptionlist)
    ave_deviation_Name = filefolder_name + "ave_deviation_beta{betan}coorxA{xA}yA{yA}xB{xB}yB{yB}.txt".format(
        betan=beta, xA=x_A, yA=y_A, xB=x_B, yB=y_B)
    np.savetxt(ave_deviation_Name, ave_deviation_vec)
    std_deviation_Name = filefolder_name + "std_deviation_beta{betan}coorxA{xA}yA{yA}xB{xB}yB{yB}.txt".format(
        betan=beta, xA=x_A, yA=y_A, xB=x_B, yB=y_B)
    np.savetxt(std_deviation_Name, std_deviation_vec)
    return ave_deviation_vec, std_deviation_vec, exemptionlist


def plot_dev_vs_avg_tail_fixnode(beta, Geodistance_index):
    """
    the x-axis is the expected degree, the y-axis is the average deviation, different line is different c_G
    inset is the min(average deviation) vs c_G
    the x-axis is real (approximate) degree
    when use this function, use load_10000nodenetwork_results_tail(beta) before
    :return:
    """
    N = 10000
    distance_list = [[0.491, 0.5, 0.509, 0.5], [0.25, 0.5, 0.75, 0.5], [0.45, 0.5, 0.55, 0.5]]
    x_A = distance_list[Geodistance_index][0]
    y_A = distance_list[Geodistance_index][1]
    x_B = distance_list[Geodistance_index][2]
    y_B = distance_list[Geodistance_index][3]
    geodesic_distance_AB = x_B - x_A
    geodesic_distance_AB = round(geodesic_distance_AB, 2)

    ave_deviation_dict = {}
    std_deviation_dict = {}

    kvec = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    # kvecsupp_geo01 = [10, 16, 27, 44, 49, 56, 64, 72, 81, 92, 104, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    # kvecsupp_geo05 = [10, 16, 27, 44, 49, 56, 64, 72, 81, 92, 104, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    # kvec = kvecsupp_geo01
    betavec = [beta]
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\GivenGeodistance\\1000realization\\tail\\"

    count = 0
    for beta in betavec:
        ave_deviation_Name = filefolder_name + "ave_deviation_beta{betan}coorxA{xA}yA{yA}xB{xB}yB{yB}.txt".format(
            betan=beta, xA=x_A, yA=y_A, xB=x_B, yB=y_B)
        ave_deviation_vec = np.loadtxt(ave_deviation_Name)
        std_deviation_Name = filefolder_name + "std_deviation_beta{betan}coorxA{xA}yA{yA}xB{xB}yB{yB}.txt".format(
            betan=beta, xA=x_A, yA=y_A, xB=x_B, yB=y_B)
        std_deviation_vec = np.loadtxt(std_deviation_Name)

        ave_deviation_dict[count] = ave_deviation_vec
        std_deviation_dict[count] = std_deviation_vec
        count = count + 1

    # legend = [r"$\beta=2.2$", r"$\beta=2^2$", r"$\beta=2^3$", r"$\beta=2^4$", r"$\beta=2^5$", r"$\beta=2^7$"]
    legend = [r"$\beta=2^2$"]
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = [[0, 0.4470, 0.7410],
              [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250],
              [0.4940, 0.1840, 0.5560],
              [0.4660, 0.6740, 0.1880],
              [0.3010, 0.7450, 0.9330]]

    for count in range(len(betavec)):
        beta = betavec[count]
        x = kvec
        y = ave_deviation_dict[count]
        print(y)
        error = std_deviation_dict[count]
        plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o',
                     markersize=16, label=legend[count], color=colors[count])
        # #  for distance = 0.018
        # params, covariance = curve_fit(power_law, x[1:9], y[1:9])
        # # 获取拟合的参数
        # a_fit, k_fit = params
        # print(f"拟合结果: a = {a_fit}, k = {k_fit}")
        # plt.plot(x[1:9], power_law(x[1:9], *params), linewidth=5,
        #          label=f'fit curve: $y={a_fit:.6f}x^{{{k_fit:.4f}}}$',
        #          color='red')

        #  for distance = 0.5
        params, covariance = curve_fit(power_law, x[8:15], y[8:15])
        # 获取拟合的参数
        a_fit, k_fit = params
        print(f"拟合结果: a = {a_fit}, k = {k_fit}")
        plt.plot(x[8:15], power_law(x[8:15], *params), linewidth=5,
                 label=f'fit curve: $y={a_fit:.6f}x^{{{k_fit:.4f}}}$',
                 color='red')

        # #  for distance = 0.1
        # params, covariance = curve_fit(power_law, x[6:12], y[6:12])
        # # 获取拟合的参数
        # a_fit, k_fit = params
        # print(f"拟合结果: a = {a_fit}, k = {k_fit}")
        # plt.plot(x[6:12], power_law(x[6:12], *params), linewidth=5,
        #          label=f'fit curve: $y={a_fit:.6f}x^{{{k_fit:.4f}}}$',
        #          color='red')

    # analyticy001 = [0.0102, 0.0144, 0.0176, 0.0227, 0.0287, 0.0373, 0.0477, 0.0609, 0.0779, 0.0992, 0.1256, 0.1562, 0.1875, 0.2141, 0.2324]
    analyticy05 = [0.0197, 0.0272, 0.0326, 0.0406, 0.0491, 0.0596, 0.0702, 0.0814, 0.0938, 0.1088, 0.1291, 0.1557,
                   0.1851, 0.2113, 0.2299]
    # analyticy01 = [0.0147, 0.0181, 0.0205, 0.0244, 0.0296, 0.0377, 0.0479, 0.0611, 0.0780, 0.0993, 0.1256, 0.1562, 0.1874, 0.2140, 0.2323]
    plt.plot(x[0:12], analyticy05[0:12], linewidth=5, label=f'analytic results of common neighbour model',
             color='green')
    # 0.01
    # params, covariance = curve_fit(power_law, x[1:9], analyticy001[1:9])
    # # 获取拟合的参数
    # a_fit, k_fit = params
    # print(f"拟合结果: a = {a_fit}, k = {k_fit}")
    # plt.plot(x[1:9], power_law(x[1:9], *params), linewidth=5, label=f'fit curve: $y={a_fit:.6f}x^{{{k_fit:.4f}}}$',
    #          color='purple')
    # 05
    params, covariance = curve_fit(power_law, x[7:14], analyticy05[7:14])
    # 获取拟合的参数
    a_fit, k_fit = params
    print(f"拟合结果: a = {a_fit}, k = {k_fit}")
    plt.plot(x[7:14], power_law(x[7:14], *params), linewidth=5, label=f'fit curve: $y={a_fit:.6f}x^{{{k_fit:.4f}}}$',
             color='purple')

    # # 01
    # params, covariance = curve_fit(power_law, x[4:12], analyticy01[4:12])
    # # 获取拟合的参数
    # a_fit, k_fit = params
    # print(f"拟合结果: a = {a_fit}, k = {k_fit}")
    # plt.plot(x[4:12], power_law(x[4:12], *params), linewidth=5, label=f'fit curve: $y={a_fit:.6f}x^{{{k_fit:.4f}}}$',
    #          color='purple')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('E[D]', fontsize=26)
    plt.ylabel('Average deviation of shortest path', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.title(fr'N = 10000, node $i$:({x_A},{y_A}), node $j$:({x_B},{y_B})', fontsize=26)
    plt.legend(fontsize=20, loc="lower right")
    plt.tick_params(axis='both', which="both", length=6, width=1)

    picname = filefolder_name + "tailLocalOptimum_dev_vs_avg_beta{beta}_distance{dis}.pdf".format(beta=beta,
                                                                                                  dis=geodesic_distance_AB)
    # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()


def plot_avg_linkdistance_vs_ED():
    N = 10000
    kvec = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389]
    y = []
    for ED in kvec:
        print(ED)
        for beta in [4]:
            linkweight_filename = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ave_distance_link_radius\\SPlinkweight_N{Nn}ED{EDn}beta{betan}.txt".format(
                Nn=N, EDn=ED, betan=beta)
            link_weight = np.loadtxt(linkweight_filename)
            y.append(np.mean(link_weight))

    plt.xlabel(r'$E[D]$', fontsize=26)
    plt.ylabel('geodistance of shortest path link', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.xscale('log')
    plt.yscale('log')
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    # plt.legend(fontsize=20, loc="upper right")
    plt.tick_params(axis='both', which="both", length=6, width=1)
    plt.plot(kvec, y)
    plt.show()


def power_law(x, a, k):
    return a * x ** k


def plot_hopcountchange(beta):
    """
    the x-axis is the expected degree, the y-axis is the average deviation,
    when use this function, use load_10000nodenetwork_results_tail(beta) before
    the analytic results analyticy01 are loaded from CommonNeighbour_analytic_check.py
    the hopcount results are loaded from plot_hopcount_vsED.py   plot_hopcount_vs_ED(10000,4)
    :return:
    """
    N = 10000
    ave_deviation_dict = {}
    std_deviation_dict = {}

    kvec = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]

    hop_ave = [11.57, 8.125, 6.1675, 4.9025, 4.01, 3.42, 2.9825, 2.665, 2.4375, 2.1775, 1.9884615384615385, 1.8875,
               1.82, 1.7275, 1.61]

    hop_std = [3.566384724058805, 2.3124391883896105, 1.6399523621129972, 1.2541904759644764, 0.9486305919587454,
               0.7606576102294647, 0.6420231693638477, 0.5410868691809106, 0.5577577879330776, 0.48061809162785374,
               0.3560757515310505, 0.32379584617471546, 0.38418745424597095, 0.44524571867677737, 0.4877499359302879]

    # kvecsupp_geo01 = [10, 16, 27, 44, 49, 56, 64, 72, 81, 92, 104, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    # kvecsupp_geo05 = [10, 16, 27, 44, 49, 56, 64, 72, 81, 92, 104, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    # kvec = kvecsupp
    betavec = [beta]
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\"

    fig, ax1 = plt.subplots(figsize=(12, 8))

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    ax1.errorbar(kvec, hop_ave, yerr=hop_std, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='s',
                 markersize=16, label="hopcount", color=colors[1])

    ax1.set_xscale('log')
    # ax1.set_yscale('log')
    ax1.set_xlabel('Expected degree, $E[D]$', fontsize=28)
    ax1.set_ylabel(r'Average hopcount, $\langle h \rangle$', fontsize=28)
    ax1.tick_params(axis='both', which="both", length=6, width=1)
    ax1.tick_params(axis='y', labelsize=26)
    ax1.tick_params(axis='x', labelsize=26)
    # plt.title('1000 simulations, N=10000',fontsize=26)

    ax1.legend(loc='upper left', fontsize=24)

    picname = filefolder_name + "hop_vs_ED_beta{beta}_withhop.png".format(beta=beta)
    plt.savefig(picname, format='png', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()



if __name__ == '__main__':
    """
    # FUNCTION 1 FOR fixed node pair
    """
    # for beta in [4]:
    #     for Geodistance_index in [2]:
    #         load_10000nodenetwork_results_tail_fixnode(beta,Geodistance_index)
    # plot_dev_vs_avg_tail_fixnode(4,1)

    """
    # FUNCTION 2 FOR all node pair
    """
    # for beta in [4]:
    #     load_10000nodenetwork_results_tail(beta)
    # plot_dev_vs_avg_tail(4)

    """
    # FUNCTION 3 FOR link geometric distance
    """
    # plot_avg_linkdistance_vs_ED()

    """
    # FUNCTION 4 FOR all node pair
    """
    # for beta in [4]:
    #     load_10000nodenetwork_results_tail(beta)
    plot_dev_vs_avg_tail_vshopcountchange(4)

    # plot_hopcountchange(4)
