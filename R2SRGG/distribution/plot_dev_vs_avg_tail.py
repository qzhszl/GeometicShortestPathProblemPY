# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 2024/11/17
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from R2SRGG.R2SRGG import loadSRGGandaddnode

def load_10000nodenetwork_results_tail_fixnode(beta,Geodistance_index):
    # Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    kvec = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    # betavec = [2.1, 4, 8, 16, 32, 64, 128]
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\GivenGeodistance\\1000realization\\tail\\"
    distance_list = [[0.491, 0.5, 0.509, 0.5], [0.25, 0.5, 0.75, 0.5]]
    x_A = distance_list[Geodistance_index][0]
    y_A = distance_list[Geodistance_index][1]
    x_B = distance_list[Geodistance_index][2]
    y_B = distance_list[Geodistance_index][3]
    geodesic_distance_AB = x_B - x_A
    geodesic_distance_AB = round(geodesic_distance_AB, 2)

    exemptionlist =[]
    for N in [10000]:
        ave_deviation_vec = []
        std_deviation_vec = []
        real_ave_degree_vec =[]
        for beta in [beta]:
            for ED in kvec:
                ave_deviation_for_a_para_comb=[]
                for ExternalSimutime in range(50):
                    try:
                        deviation_vec_name = filefolder_name + "Givendistancedeviation_shortest_path_nodes_N{Nn}ED{EDn}beta{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
                            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime, Geodistance=geodesic_distance_AB)
                        ave_deviation_for_a_para_comb_10times = np.loadtxt(deviation_vec_name)
                        ave_deviation_for_a_para_comb.extend(ave_deviation_for_a_para_comb_10times)
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
    distance_list = [[0.491, 0.5, 0.509, 0.5], [0.25, 0.5, 0.75, 0.5]]
    x_A = distance_list[Geodistance_index][0]
    y_A = distance_list[Geodistance_index][1]
    x_B = distance_list[Geodistance_index][2]
    y_B = distance_list[Geodistance_index][3]
    geodesic_distance_AB = x_B - x_A
    geodesic_distance_AB = round(geodesic_distance_AB, 2)

    ave_deviation_dict = {}
    std_deviation_dict = {}

    kvec = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
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
        error = std_deviation_dict[count]
        plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o',
                     markersize=16, label=legend[count], color=colors[count])

    plt.xscale('log')
    plt.xlabel('Input avg', fontsize=26)
    plt.ylabel('Average deviation of shortest path', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=20, loc="upper left")
    plt.tick_params(axis='both', which="both", length=6, width=1)


    picname = filefolder_name+"LocalOptimum_dev_vs_avg_diffbeta_distance{dis}.pdf".format(
        dis=geodesic_distance_AB)
    # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()


if __name__ == '__main__':
    for beta in [4]:
        for Geodistance_index in range(2):
            load_10000nodenetwork_results_tail_fixnode(beta,Geodistance_index)
    # plot_dev_vs_avg_tail_fixnode(4,1)