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
    kvec = np.arange(2.5, 5, 0.1)
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
                for ExternalSimutime in range(20):
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

    # kvec = np.arange(2, 7.1, 0.2)
    kvec = np.arange(2.5, 5, 0.1)
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
        # print(y)
        error = std_deviation_dict[count]
        # print(error)
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
    kvec = np.arange(2.5, 5, 0.1)
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
    kvec = np.arange(2.5, 5, 0.1)
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
    peak_avg = [4, 3.4, 5.0, 5.0, 5.8, 5.4, 6, 6.4, 3.2, 3.1, 2.6, 2.7, 2.9, 2.9]
    SLCC_avg = [3.8, 3.2, 5.0, 4.8, 6, 5.6, 6, 6.0, 3.1, 2.9, 2.6, 2.7, 2.8, 2.7]
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = [[0, 0.4470, 0.7410],
              [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250],
              [0.4940, 0.1840, 0.5560],
              [0.4660, 0.6740, 0.1880]]
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']

    plt.scatter(peak_avg, SLCC_avg, marker='o', s=150, color=colors[3], label=r"$E[D]$")
    x = np.linspace(2.5,6,10)
    y = np.linspace(2.5, 6, 10)
    plt.plot(x,y,"--",color=colors[0],label=r"$y=x$",linewidth=5)
    # plt.scatter(ave_deviation_vec, spnodenum_vec, marker='o', c=colors[1],markersize=16, label=r"$N=10^2$")

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # plt.ylim(0, 0.30)
    # plt.yticks([0, 0.1, 0.2, 0.3])
    text = r"$N = 10^4$"
    ax.text(
        0.5, 0.85,  # 文本位置（轴坐标，0.5 表示图中央，1.05 表示轴上方）
        text,
        transform=ax.transAxes,  # 使用轴坐标
        fontsize=26,  # 字体大小
        ha='center',  # 水平居中对齐
        va='bottom'  # 垂直对齐方式
    )

    # plt.xscale('log')
    plt.xlabel(r'$E[D]_{dev_{max}}$', fontsize=32)
    plt.ylabel(r'$E[D]_{SLCC_{max}}$', fontsize=32)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.legend(fontsize=26)
    plt.tick_params(axis='both', which="both", length=6, width=1)
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\"
    picname = filefolder_name + "scattor_slcc_vs_peak.pdf"
    plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()



if __name__ == '__main__':
    # load_10000nodenetwork_results_peak(64)
    # betavec = [3,3.1,3.2,3.3,3.4,3.5,3.6,3.7]
    # for beta in betavec:
    #     load_10000nodenetwork_results_peak(beta)
    #     plot_dev_vs_avg_peak(beta)
    #     find_giant_component(beta)
    scattor_peakvs_GLCC()

