# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 2024/11/17
This file is for the peak of the shortest path deviation

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import networkx as nx
from R2SRGG.R2SRGG import loadSRGGandaddnode

def load_10000nodenetwork_results_peak(beta):
    # Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    kvec = np.arange(2, 6.1, 0.2)
    kvec = [round(a, 1) for a in kvec]

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
    print(len(exemptionlist))
    # np.savetxt("notrun.txt", exemptionlist)

    ave_deviation_Name = filefolder_name + "peakave_deviation_N{Nn}_beta{betan}.txt".format(
        Nn=N, betan=beta)
    np.savetxt(ave_deviation_Name, ave_deviation_vec)
    std_deviation_Name = filefolder_name + "peakstd_deviation_N{Nn}_beta{betan}.txt".format(Nn=N,
                                                                                        betan=beta)
    np.savetxt(std_deviation_Name, std_deviation_vec)
    return ave_deviation_vec, std_deviation_vec, exemptionlist

def plot_dev_vs_avg_peak(beta):
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

    kvec = np.arange(2, 6.1, 0.2)
    kvec = [round(a, 1) for a in kvec]

    betavec = [beta]
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\"

    count = 0
    for beta in betavec:
        ave_deviation_Name = filefolder_name + "peakave_deviation_N{Nn}_beta{betan}.txt".format(
            Nn=N, betan=beta)
        ave_deviation_vec = np.loadtxt(ave_deviation_Name)
        std_deviation_Name = filefolder_name + "peakstd_deviation_N{Nn}_beta{betan}.txt".format(Nn=N,
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
        y = list(y)
        max_index = y.index(max(y))
        # 根据索引找到对应的 x
        result_x = x[max_index]
        print(beta,result_x)


    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('E[D]', fontsize=26)
    plt.ylabel('Average deviation of shortest path', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.title('1000 simulations, N=10000',fontsize=26)
    plt.legend(fontsize=20, loc="lower right")
    plt.tick_params(axis='both', which="both", length=6, width=1)

    picname = filefolder_name+"peakLocalOptimum_dev_vs_avg_beta{beta}.pdf".format(beta=beta)
    plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()


def load_LCC_second_LCC_data(beta):
    """
    function for loading data for analysis first peak about Lcc AND second Lcc
    :param beta:
    :return:
    """
    kvec = np.arange(2, 6.1, 0.2)
    input_avg_vec = [round(a, 1) for a in kvec]
    N = 10000
    filefolder_name_lcc = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\"

    LCC_vec = []
    LCC_std_vec = []
    second_LCC_vec = []
    second_LCC_std_vec = []
    for ED in input_avg_vec:
        LCC_oneED = []
        second_LCC_oneED = []
        for simutime in range(1):
            LCC_onesimu = []
            second_LCC_onesimu = []
            LCCname = filefolder_name_lcc + "LCC_2LCC_N{Nn}ED{EDn}beta{betan}.txt".format(
                Nn=N, EDn=ED, betan=beta)
            try:
                with open(LCCname, "r") as file:
                    for line in file:
                        if line.startswith("#"):
                            continue
                        else:
                            data = line.strip().split("\t")
                            LCC_onesimu.append(int(data[0]))
                            second_LCC_onesimu.append(int(data[1]))
                LCC_oneED = LCC_oneED + LCC_onesimu
                second_LCC_oneED = second_LCC_oneED + second_LCC_onesimu
            except:
                print("Not data",ED,simutime)
        LCC_vec.append(np.mean(LCC_oneED))
        second_LCC_vec.append(np.mean(second_LCC_oneED))

    return LCC_vec,second_LCC_vec

def find_giant_component(beta):
    LCC_vec, second_LCC_vec = load_LCC_second_LCC_data(beta)
    kvec = np.arange(2, 6.1, 0.2)
    input_avg_vec = [round(a, 1) for a in kvec]
    # print(len(input_avg_vec))
    # print(second_LCC_vec)
    plt.plot(input_avg_vec, LCC_vec)
    plt.plot(input_avg_vec, second_LCC_vec)
    max_index = second_LCC_vec.index(max(second_LCC_vec))
    # 根据索引找到对应的 x
    result_x = input_avg_vec[max_index]
    print("GLCC",result_x)
    plt.show()


def scattor_peakvs_GLCC():
    peak_avg = [4,3.4,5.0,5.0,5.4]
    SLCC_avg = [3.2,3.2,5.0,4.8,5.4]
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = [[0, 0.4470, 0.7410],
              [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250],
              [0.4940, 0.1840, 0.5560],
              [0.4660, 0.6740, 0.1880]]

    plt.scatter(peak_avg, SLCC_avg, marker='o', s=60, color=colors[0], label=r"$N=10^4$")
    x = np.linspace(3,6,10)
    y = np.linspace(3, 6, 10)
    plt.plot(x,y,"--",color=colors[1],label=r"$y=x$")
    # plt.scatter(ave_deviation_vec, spnodenum_vec, marker='o', c=colors[1],markersize=16, label=r"$N=10^2$")

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # plt.ylim(0, 0.30)
    # plt.yticks([0, 0.1, 0.2, 0.3])

    # plt.xscale('log')
    plt.xlabel(r'$E[D]_{dev_{max}}$', fontsize=26)
    plt.ylabel(r'$E[D]_{SLCC_{max}}$', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=20)
    plt.tick_params(axis='both', which="both", length=6, width=1)
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\"
    picname = filefolder_name + "scattor_slcc_vs_peak.pdf"
    plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()


def check_meandistancebetweentwonodes():
    rg = RandomGenerator(-12)
    rseed = random.randint(0, 10000)
    for i in range(rseed):
        rg.ran1()
    kvec = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    N = 10000
    beta = 4
    ED = 9999
    # load a network
    FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\inputavgbeta\\network_N{Nn}ED{EDn}Beta{betan}.txt".format(
        Nn=N, EDn=ED, betan=beta)
    G = loadSRGGandaddnode(N, FileNetworkName)
    # load coordinates with noise
    Coorx = []
    Coory = []

    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\inputavgbeta\\network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt".format(
        Nn=N, EDn=ED, betan=beta)
    with open(FileNetworkCoorName, "r") as file:
        for line in file:
            if line.startswith("#"):
                continue
            data = line.strip().split("\t")  # 使用制表符分割
            Coorx.append(float(data[0]))
            Coory.append(float(data[1]))

    simu1 = []
    unique_pairs = find_k_connected_node_pairs(G, 1000)
    count = 0
    for node_pair in unique_pairs:
        count = count + 1
        nodei = node_pair[0]
        nodej = node_pair[1]
        xSource = Coorx[nodei]
        ySource = Coory[nodei]
        xEnd = Coorx[nodej]
        yEnd = Coory[nodej]
        simu1.append(distR2(xSource, ySource, xEnd, yEnd))

    print(np.mean(simu1))
    simu2 = []
    for simultime in range(1000):
        node1, node2 = random.sample(range(G.number_of_nodes()), 2)
        nodei = node1
        nodej = node2
        xSource = Coorx[nodei]
        ySource = Coory[nodei]
        xEnd = Coorx[nodej]
        yEnd = Coory[nodej]
        simu2.append(distR2(xSource, ySource, xEnd, yEnd))
    print("1:", np.mean(simu1))
    print("2:", np.mean(simu2))



if __name__ == '__main__':
    # load_10000nodenetwork_results_peak(16)

    # betavec = [32]
    # for beta in betavec:
    #     load_10000nodenetwork_results_peak(beta)
    #     plot_dev_vs_avg_peak(beta)
    # find_giant_component(32)
    # scattor_peakvs_GLCC()

    kvec = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    # betavec = [2.1, 4, 8, 16, 32, 64, 128]
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\"

    beta =4
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
                        deviation_vec_name = filefolder_name + "length_geodesic_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        ave_deviation_for_a_para_comb_10times = np.loadtxt(deviation_vec_name)
                        ave_deviation_for_a_para_comb = np.hstack(
                            (ave_deviation_for_a_para_comb, ave_deviation_for_a_para_comb_10times))
                    except FileNotFoundError:
                        exemptionlist.append((N, ED, beta, ExternalSimutime))

                ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
                std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))
    print(exemptionlist)
    plt.xscale('log')
    plt.plot(kvec,ave_deviation_vec)
    plt.show()


