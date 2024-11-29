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
    exemptionlist =[]
    for N in [10000]:
        ave_deviation_vec = []
        std_deviation_vec = []
        real_ave_degree_vec =[]
        for beta in [beta]:
            for ED in kvec:
                ave_deviation_for_a_para_comb=np.array([])
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
    the x-axis is the expected degree, the y-axis is the average deviation, different line is different c_G
    inset is the min(average deviation) vs c_G
    the x-axis is real (approximate) degree
    when use this function, use before
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
        params, covariance = curve_fit(power_law, x[8:15], y[8:15])
        # 获取拟合的参数
        a_fit, k_fit = params
        print(f"拟合结果: a = {a_fit}, k = {k_fit}")
        plt.plot(x[8:15], power_law(x[8:15], *params), linewidth=5, label=f'fit curve: $y={a_fit:.6f}x^{{{k_fit:.4f}}}$',
                 color='red')
        # params, covariance = curve_fit(power_law, x[10:13], y[10:13])
        # a_fit, k_fit = params
        # plt.plot(x[10:13], power_law(x[10:13], *params), linewidth=5,
        #          label=f'fit curve: $y={a_fit:.6f}x^{{{k_fit:.4f}}}$',
        #          color='green')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('E[D]', fontsize=26)
    plt.ylabel('Average deviation of shortest path', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.title('1000 simulations, N=10000',fontsize=26)
    plt.legend(fontsize=20, loc="lower right")
    plt.tick_params(axis='both', which="both", length=6, width=1)

    picname = filefolder_name+"tailLocalOptimum_dev_vs_avg_beta{beta}.pdf".format(beta=beta)
    plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()

def load_10000nodenetwork_results_tail_fixnode(beta,Geodistance_index):
    # Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    kvec = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    # kvecsupp = [10, 16, 27, 44, 49, 56, 64, 72, 81, 92, 104, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    # kvec = kvecsupp
    # betavec = [2.1, 4, 8, 16, 32, 64, 128]
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\GivenGeodistance\\1000realization\\tail\\"
    distance_list = [[0.491, 0.5, 0.509, 0.5], [0.25, 0.5, 0.75, 0.5],[0.45, 0.5, 0.55, 0.5]]
    x_A = distance_list[Geodistance_index][0]
    y_A = distance_list[Geodistance_index][1]
    x_B = distance_list[Geodistance_index][2]
    y_B = distance_list[Geodistance_index][3]
    geodesic_distance_AB = x_B - x_A
    geodesic_distance_AB = round(geodesic_distance_AB, 2)
    print(beta,geodesic_distance_AB)
    exemptionlist =[]
    for N in [10000]:
        ave_deviation_vec = []
        std_deviation_vec = []
        real_ave_degree_vec =[]
        for beta in [beta]:
            for ED in kvec:
                ave_deviation_for_a_para_comb=np.array([])
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
    return ave_deviation_vec,std_deviation_vec,exemptionlist

def plot_dev_vs_avg_tail_fixnode(beta,Geodistance_index):
    """
    the x-axis is the expected degree, the y-axis is the average deviation, different line is different c_G
    inset is the min(average deviation) vs c_G
    the x-axis is real (approximate) degree
    when use this function, use before
    :return:
    """
    N = 10000
    distance_list = [[0.491, 0.5, 0.509, 0.5], [0.25, 0.5, 0.75, 0.5],[0.45, 0.5, 0.55, 0.5]]
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
        ave_deviation_Name = filefolder_name+"ave_deviation_beta{betan}coorxA{xA}yA{yA}xB{xB}yB{yB}.txt".format(
            betan=beta, xA = x_A, yA = y_A, xB = x_B, yB = y_B)
        ave_deviation_vec = np.loadtxt(ave_deviation_Name)
        std_deviation_Name = filefolder_name+"std_deviation_beta{betan}coorxA{xA}yA{yA}xB{xB}yB{yB}.txt".format(
            betan=beta, xA = x_A, yA = y_A, xB = x_B, yB = y_B)
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


        # #  for distance = 0.5
        # params, covariance = curve_fit(power_law, x[8:15], y[8:15])
        # # 获取拟合的参数
        # a_fit, k_fit = params
        # print(f"拟合结果: a = {a_fit}, k = {k_fit}")
        # plt.plot(x[8:15], power_law(x[8:15], *params), linewidth=5, label=f'fit curve: $y={a_fit:.6f}x^{{{k_fit:.4f}}}$',
        #          color='red')

        #  for distance = 0.5
        params, covariance = curve_fit(power_law, x[6:12], y[6:12])
        # 获取拟合的参数
        a_fit, k_fit = params
        print(f"拟合结果: a = {a_fit}, k = {k_fit}")
        plt.plot(x[6:12], power_law(x[6:12], *params), linewidth=5,
                 label=f'fit curve: $y={a_fit:.6f}x^{{{k_fit:.4f}}}$',
                 color='red')


    # analyticy001 = [0.0102, 0.0144, 0.0176, 0.0227, 0.0287, 0.0373, 0.0477, 0.0609, 0.0779, 0.0992, 0.1256, 0.1562, 0.1875, 0.2141, 0.2324]
    # analyticy05 = [0.0197, 0.0272, 0.0326, 0.0406, 0.0491, 0.0596, 0.0702, 0.0814, 0.0938, 0.1088, 0.1291, 0.1557, 0.1851, 0.2113, 0.2299]
    analyticy01 = [0.0147, 0.0181, 0.0205, 0.0244, 0.0296, 0.0377, 0.0479, 0.0611, 0.0780, 0.0993, 0.1256, 0.1562, 0.1874, 0.2140, 0.2323]
    plt.plot(x[0:12], analyticy01[0:12], linewidth=5, label=f'analytic results of common neighbour model',
             color='green')
    #0.01
    # params, covariance = curve_fit(power_law, x[1:9], analyticy001[1:9])
    # # 获取拟合的参数
    # a_fit, k_fit = params
    # print(f"拟合结果: a = {a_fit}, k = {k_fit}")
    # plt.plot(x[1:9], power_law(x[1:9], *params), linewidth=5, label=f'fit curve: $y={a_fit:.6f}x^{{{k_fit:.4f}}}$',
    #          color='purple')
    # # 05
    # params, covariance = curve_fit(power_law, x[7:14], analyticy05[7:14])
    # # 获取拟合的参数
    # a_fit, k_fit = params
    # print(f"拟合结果: a = {a_fit}, k = {k_fit}")
    # plt.plot(x[7:14], power_law(x[7:14], *params), linewidth=5, label=f'fit curve: $y={a_fit:.6f}x^{{{k_fit:.4f}}}$',
    #          color='purple')

    # # 01
    params, covariance = curve_fit(power_law, x[4:12], analyticy01[4:12])
    # 获取拟合的参数
    a_fit, k_fit = params
    print(f"拟合结果: a = {a_fit}, k = {k_fit}")
    plt.plot(x[4:12], power_law(x[4:12], *params), linewidth=5, label=f'fit curve: $y={a_fit:.6f}x^{{{k_fit:.4f}}}$',
             color='purple')


    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('E[D]', fontsize=26)
    plt.ylabel('Average deviation of shortest path', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.title(fr'N = 10000, node $i$:({x_A},{y_A}), node $j$:({x_B},{y_B})', fontsize=26)
    plt.legend(fontsize=20, loc="lower right")
    plt.tick_params(axis='both', which="both", length=6, width=1)

    picname = filefolder_name+"tailLocalOptimum_dev_vs_avg_beta{beta}_distance{dis}.pdf".format(beta=beta,
        dis=geodesic_distance_AB)
    plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
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

if __name__ == '__main__':
    """
    # FUNCTION 1 FOR fixed node pair
    """
    # for beta in [4]:
    #     for Geodistance_index in [2]:
    #         load_10000nodenetwork_results_tail_fixnode(beta,Geodistance_index)
    plot_dev_vs_avg_tail_fixnode(4,2)

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