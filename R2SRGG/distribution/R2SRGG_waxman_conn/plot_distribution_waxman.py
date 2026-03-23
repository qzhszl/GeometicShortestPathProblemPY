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

def load_small_network_results_maxminave(N, ED, beta):
    exemptionlist =[]
    for N in [N]:
        ave_deviation_vec = []
        max_deviation_vec = []
        min_deviation_vec = []
        ran_deviation_vec = []
        for ExternalSimutime in [0]:
            try:
                # clustering_coefficient_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\clustering_coefficient_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                # clustering_coefficient = np.loadtxt(clustering_coefficient_name)
                # print(np.mean(clustering_coefficient))
                #
                # real_ave_degree_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\real_ave_degree_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                # real_ave_degree = np.loadtxt(real_ave_degree_name)
                # print(np.mean(real_ave_degree))

                deviation_vec_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                ave_deviation_for_a_para_comb_10times = np.loadtxt(deviation_vec_name)
                ave_deviation_vec.extend(ave_deviation_for_a_para_comb_10times)

                max_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\max_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                max_deviation_for_a_para_comb_10times = np.loadtxt(max_deviation_name)
                max_deviation_vec.extend(max_deviation_for_a_para_comb_10times)

                min_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\min_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                min_deviation_for_a_para_comb_10times = np.loadtxt(min_deviation_name)
                min_deviation_vec.extend(min_deviation_for_a_para_comb_10times)

                ave_baseline_deviation_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\ave_baseline_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                ave_baseline_deviation_for_a_para_comb_10times = np.loadtxt(ave_baseline_deviation_name)
                ran_deviation_vec.extend(ave_baseline_deviation_for_a_para_comb_10times)
            except FileNotFoundError:
                exemptionlist.append((N, ED, beta, ExternalSimutime))
                print(exemptionlist)
    return ave_deviation_vec, max_deviation_vec, min_deviation_vec, ran_deviation_vec, exemptionlist



def load_large_network_results_maxminave(N, ED, beta,eta):
    exemptionlist = []
    for N in [N]:
        ave_deviation_vec = []
        max_deviation_vec = []
        min_deviation_vec = []
        ran_deviation_vec = []

        file_folder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\waxman\\"
        for ExternalSimutime in [0]:
            try:
                if N ==1000:
                    deviation_vec_name = file_folder_name+"ave_deviation_N{Nn}ED{EDn}Beta{betan}eta{etan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta,etan=eta, ST=ExternalSimutime)
                    ave_deviation_for_a_para_comb_10times = np.loadtxt(deviation_vec_name)
                    ave_deviation_vec.extend(ave_deviation_for_a_para_comb_10times)

                    max_deviation_name = file_folder_name+"max_deviation_N{Nn}ED{EDn}Beta{betan}eta{etan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta,etan=eta, ST=ExternalSimutime)
                    max_deviation_for_a_para_comb_10times = np.loadtxt(max_deviation_name)
                    max_deviation_vec.extend(max_deviation_for_a_para_comb_10times)

                    min_deviation_name = file_folder_name+"min_deviation_N{Nn}ED{EDn}Beta{betan}eta{etan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta,etan=eta, ST=ExternalSimutime)
                    min_deviation_for_a_para_comb_10times = np.loadtxt(min_deviation_name)
                    min_deviation_vec.extend(min_deviation_for_a_para_comb_10times)

                    ave_baseline_deviation_name = file_folder_name+"ave_baseline_deviation_N{Nn}ED{EDn}Beta{betan}eta{etan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta,etan=eta, ST=ExternalSimutime)
                    ave_baseline_deviation_for_a_para_comb_10times = np.loadtxt(ave_baseline_deviation_name)
                    ran_deviation_vec.extend(ave_baseline_deviation_for_a_para_comb_10times)
                elif N ==10000:
                    deviation_vec_name = file_folder_name+"ave_deviation_N{Nn}ED{EDn}Beta{betan}eta{etan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta,etan=eta, ST=ExternalSimutime)
                    ave_deviation_for_a_para_comb_10times = np.loadtxt(deviation_vec_name)
                    ave_deviation_vec.extend(ave_deviation_for_a_para_comb_10times)

                    max_deviation_name = file_folder_name+"max_deviation_N{Nn}ED{EDn}Beta{betan}eta{etan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta,etan=eta, ST=ExternalSimutime)
                    max_deviation_for_a_para_comb_10times = np.loadtxt(max_deviation_name)
                    max_deviation_vec.extend(max_deviation_for_a_para_comb_10times)

                    min_deviation_name = file_folder_name+"min_deviation_N{Nn}ED{EDn}Beta{betan}eta{etan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta,etan=eta, ST=ExternalSimutime)
                    min_deviation_for_a_para_comb_10times = np.loadtxt(min_deviation_name)
                    min_deviation_vec.extend(min_deviation_for_a_para_comb_10times)

                    ave_baseline_deviation_name = file_folder_name+"ave_baseline_deviation_N{Nn}ED{EDn}Beta{betan}eta{etan}Simu{ST}.txt".format(
        Nn=N, EDn=ED, betan=beta,etan=eta, ST=ExternalSimutime)
                    ave_baseline_deviation_for_a_para_comb_10times = np.loadtxt(ave_baseline_deviation_name)
                    ran_deviation_vec.extend(ave_baseline_deviation_for_a_para_comb_10times)
            except FileNotFoundError:
                exemptionlist.append((N, ED, beta, ExternalSimutime))
                print(exemptionlist)
    return ave_deviation_vec, max_deviation_vec, min_deviation_vec, ran_deviation_vec, exemptionlist


def load_10000nodenetwork_maxminave(ED, beta):
    exemptionlist =[]
    for N in [10000]:
        ave_deviation_vec = []
        max_deviation_vec = []
        min_deviation_vec = []
        ran_deviation_vec = []

        # foldername = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\1000realization\\"
        foldername = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\1000000realization\\"

        # FileNetworkName = foler_name+"network_N{Nn}ED{EDn}Beta{betan}.txt".format(
        #     Nn=N, EDn=ED, betan=beta)
        # G = loadSRGGandaddnode(N, FileNetworkName)
        # real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
        # print("real ED:", real_avg)

        for ExternalSimutime in range(20):
            try:
                deviation_vec_name = foldername+ "ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                ave_deviation_for_a_para_comb_10times = np.loadtxt(deviation_vec_name)
                ave_deviation_vec.extend(ave_deviation_for_a_para_comb_10times)

                max_deviation_name = foldername+"max_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                max_deviation_for_a_para_comb_10times = np.loadtxt(max_deviation_name)
                max_deviation_vec.extend(max_deviation_for_a_para_comb_10times)

                min_deviation_name = foldername+"min_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                min_deviation_for_a_para_comb_10times = np.loadtxt(min_deviation_name)
                min_deviation_vec.extend(min_deviation_for_a_para_comb_10times)

                ave_baseline_deviation_name = foldername+"ave_baseline_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                ave_baseline_deviation_for_a_para_comb_10times = np.loadtxt(ave_baseline_deviation_name)
                ran_deviation_vec.extend(ave_baseline_deviation_for_a_para_comb_10times)
            except FileNotFoundError:
                exemptionlist.append((N, ED, beta, ExternalSimutime))
                # print(exemptionlist)
    return ave_deviation_vec, max_deviation_vec, min_deviation_vec, ran_deviation_vec, exemptionlist



def plot_distribution(N, ED, beta,eta,thesis_flag = False):
    """
    Compared maximum, minimum, average deviation with randomly selected nodes
    :return:
    """
    # Figure 10 Appendxi distance to geodesic distribution
    # Nvec = [20,50,100,1000]
    # # Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    # beta = 8
    if N < 200:
        # try:
        #     ave_deviation_vec, max_deviation_vec, min_deviation_vec, ran_deviation_vec, _ = load_resort_data_smallN_maxminave(
        #     N, ED, beta)
        # except:
            ave_deviation_vec, max_deviation_vec, min_deviation_vec, ran_deviation_vec, _ = load_small_network_results_maxminave(N, ED, beta)
    else:
        ave_deviation_vec, max_deviation_vec, min_deviation_vec, ran_deviation_vec, _ = load_large_network_results_maxminave(
            N, ED, beta, eta)

    # cuttail = [9,19,34,24]
    # peakcut = [9,5,5,5]

    data1 = ave_deviation_vec
    # data1 = [0,0,0]
    data2 = max_deviation_vec
    data3 = min_deviation_vec
    data4 = ran_deviation_vec

    fig, ax = plt.subplots(figsize=(6, 4.5))
    # fig, ax = plt.subplots(figsize=(8, 4.5))

    datasets = [data1,data2,data3,data4]

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    labels = ["Avg","Max","Min","Ran"]

    plot_count = 0
    for data, color, label in zip(datasets, colors, labels):
        hvalue, bin_vec = np.histogram(data, bins=60, density=True)

        if plot_count == 1 and ED==5 and eta==1:
            hvalue = np.append(hvalue, 0.001)
            bin_vec = np.append(bin_vec, 0.099)
        elif plot_count == 0 and ED == 5 and eta == 1:
            hvalue = np.append(hvalue, 0.001)
            bin_vec = np.append(bin_vec, 0.0614)
        elif plot_count == 0 and ED == 10 and eta == 1:
            hvalue = np.append(hvalue, 0.001)
            bin_vec = np.append(bin_vec, 0.164)
        elif plot_count == 1 and ED == 10 and eta == 1:
            hvalue = np.append(hvalue, 0.001)
            bin_vec = np.append(bin_vec, 0.294)
        elif plot_count == 1 and ED == 10 and eta == 10:
            hvalue = np.append(hvalue, 0.001)
            bin_vec = np.append(bin_vec, 0.226)
        elif plot_count == 0 and ED == 50 and eta == 2:
            hvalue = np.append(hvalue, 0.001)
            bin_vec = np.append(bin_vec, 0.146)
        elif plot_count == 0 and ED == 5 and eta == 2:
            hvalue = np.append(hvalue, 0.001)
            bin_vec = np.append(bin_vec, 0.157)
        print(bin_vec[1:len(bin_vec)])
        print(hvalue)
        plt.plot(bin_vec[1:len(bin_vec)], hvalue, color=color, label=label, linewidth=5)
        plot_count = plot_count+1


    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.spines['left'].set_position(('data', 0))
    # ax.spines['bottom'].set_position(('data', 0))
    # ax.spines['bottom'].set_position(('outward', 0))
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlim([0,1.2])
    # plt.ylim([0, 50])
    # plt.yticks([0,5,10,15,20,25])
    # plt.yticks([0, 10, 20, 30, 40, 50])


    if N == 10000:
        plt.yscale('log')
        ymin = 0.001  # 设置最低点
        current_ylim = ax.get_ylim()  # 获取当前的 y 轴范围
        ax.set_ylim(ymin, current_ylim[1])  # 保持最大值不变

    # plt.yscale('log')
    plt.xlabel(r'x',fontsize = 32)
    plt.ylabel(r'$f_{d(q,\gamma(i,j))}(x)$',fontsize = 32)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)

    # ytick_dict = {
    #     (5, 4): [0, 0.1, 0.2],
    #     (5, 8): [0, 0.1, 0.2, 0.3, 0.4],
    #     (5, 128): [0, 0.2, 0.4, 0.6, 0.8, 1.0],
    #     (2, 8): [0, 0.2, 0.4, 0.6, 0.8],
    #     (10, 8): [0, 0.1, 0.2, 0.3, 0.4],
    #     (100, 8): [0, 0.1, 0.2, 0.3, 0.4],
    # }
    # ytick_vec = ytick_dict[(N, ED, beta)]
    # plt.yticks(ytick_vec, fontsize=28)

    # fignum_dict = {
    #     (100,5, 4): "a",
    #     (100,10, 4): "b",
    #     (100,50, 4): "c",
    #     (100,5, 8): "d",
    #     (100,5, 128): "e",
    #     (1000,5, 4): "f",
    # }

    fignum_dict = {
    (10000, 5, 1): "a",
    (10000, 10, 1): "b",
    (10000, 50, 1): "c",
    (10000, 5, 2): "d",
    (10000, 10, 2): "e",
    (10000, 50, 2): "f",
    (10000, 5, 10): "g",
    (10000, 10, 10): "h",
    (10000, 50, 10): "i",
    }


    xtextpos_dict = {
        (10000, 5, 1): -0.34,
        (10000, 10, 1): -0.34,
        (10000, 50, 1): -0.34,
        (10000, 5, 2): -0.34,
        (10000, 10, 2): -0.34,
        (10000, 50, 2): -0.34,
        (10000, 5, 10): -0.34,
        (10000, 10, 10): -0.34,
        (10000, 50, 10): -0.34,
    }

    plt.legend(fontsize=26, handlelength=1, handletextpad=0.5, frameon=False,loc='right',bbox_to_anchor=(1.04, 0.64))
    plt.tick_params(axis='both', which="both", length=6, width=1)

    file_folder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\waxman\\"
    if thesis_flag:
        picname = file_folder_name+"DistributionN{Nn}ED{EDn}Beta{betan}_thesis.svg".format(
            Nn=N, EDn=ED, betan=beta)
    else:
        try:
            Nlabel_dict = {100:"10^2",1000:"10^3",10000:"10^4"}
            Nlabel = Nlabel_dict[N]
            fignum = fignum_dict[(N,ED, eta)]
            ax.text(xtextpos_dict[(N,ED, eta)], 1.15, fr'({fignum}) $N = {Nlabel}$, $\mathbb{{E}}[D] = {ED}$, $\eta = {eta}$', transform=ax.transAxes,
                    fontsize=28, verticalalignment='top', horizontalalignment='left')
            # picname = file_folder_name+"DistributionN{Nn}ED{EDn}Beta{betan}.pdf".format(
            #     Nn=N, EDn=ED, betan=beta)
            # plt.savefig(picname,format='pdf', bbox_inches='tight', dpi=600)

            picname = file_folder_name+"DistributionN{Nn}ED{EDn}eta{etan}.svg".format(
                Nn=N, EDn=ED, etan=eta)
        except:
            pass

    # plt.title('Errorbar Curves with Minimum Points after Peak')


    # plt.savefig(picname, format='svg', bbox_inches='tight', transparent=True)

    plt.show()
    plt.close()



def check_distribution(N, beta,eta):
    """
    Compared maximum, minimum, average deviation with randomly selected nodes
    :return:
    """
    # Figure 10 Appendxi distance to geodesic distribution
    # Nvec = [20,50,100,1000]
    # # Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    # beta = 8
    y_ave = []
    y_max = []
    y_min = []
    y_ran = []
    ED_vec  =[5, 10, 50, 100]

    for ED in [5,10,50,100]:
        ave_deviation_vec, max_deviation_vec, min_deviation_vec, ran_deviation_vec, _ = load_large_network_results_maxminave(
            N, ED, beta, eta)

        data1 = np.mean(ave_deviation_vec)
        data2 = np.mean(max_deviation_vec)
        data3 = np.mean(min_deviation_vec)
        data4 = np.mean(ran_deviation_vec)

        y_ave.append(data1)
        y_max.append(data2)
        y_min.append(data3)
        y_ran.append(data4)


    plt.plot(ED_vec,y_ave,label="avg")
    plt.plot(ED_vec, y_max,label="max")
    plt.plot(ED_vec, y_min,label="min")
    plt.plot(ED_vec, y_ran,label="ran")

    plt.yscale('log')
    plt.xlabel(r'ED',fontsize = 32)
    plt.ylabel(r'$y$',fontsize = 32)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.legend()


    plt.show()
    plt.close()





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Figure 10 Appendix distance to geodesic distribution

    # for (N,ED,beta) in [(100, 5, 4),
    #                   (100, 10, 4),
    #                   (100, 50, 4),
    #                   (100, 5, 8),
    #                   (100, 5, 128),
    #                   (1000, 5, 4)]:
    #     plot_distribution(N,ED,beta,thesis_flag=True)
    # plot_distribution(10000,5,1,eta=2,thesis_flag=False)

    check_distribution(10000, 1, 10)





