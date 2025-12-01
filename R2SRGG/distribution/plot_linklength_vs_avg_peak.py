# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 2024/11/17
This file is for the peak of the shortest path deviation

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.optimize import curve_fit
import networkx as nx
from R2SRGG.R2SRGG import loadSRGGandaddnode

def load_10000nodenetwork_results_peak(N,beta,kvec):
    folder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\peak\\"
    realL = True
    exclude_hop1_flag  =True
    exemptionlist =[]
    for N in [N]:
        real_ave_degree_vec = []
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

                        real_ave_degree_name = folder_name + "real_ave_degree_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        real_avg = np.loadtxt(real_ave_degree_name)
                        real_ave_degree_vec.append(np.mean(real_avg))


                        if realL:
                            edgelength_vec_name = folder_name + "ave_edgelength_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        else:

                            # if L = <d_e><h> ave  link length* hopcount
                            edgelength_vec_name = folder_name + "ave_graph_edge_length_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)

                        ave_edgelength_for_a_para_comb = np.loadtxt(edgelength_vec_name)
                        ave_edgelength_vec.append(np.mean(ave_edgelength_for_a_para_comb))
                        std_edgelength_vec.append(np.std(ave_edgelength_for_a_para_comb))

                        hopcount_Name = folder_name + "hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
                            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        hop_vec = np.loadtxt(hopcount_Name, dtype=int)

                        ave_hop_vec.append(np.mean(hop_vec))
                        std_hop_vec.append(np.std(hop_vec))

                        hop_vec_no1 = hop_vec[hop_vec != 1]

                        if realL:
                            if len(ave_edgelength_for_a_para_comb) != len(hop_vec_no1):
                                ave_edgelength_for_a_para_comb_no1 = ave_edgelength_for_a_para_comb[hop_vec != 1]
                            #if L = <d_e>h real stretch
                                L = [x * y for x, y in zip(ave_edgelength_for_a_para_comb_no1, hop_vec_no1)]
                            # if we include 1-hop sp
                            #     L = [x * y for x, y in zip(ave_edgelength_for_a_para_comb, hop_vec)]
                            else:
                                L = [x * y for x, y in zip(ave_edgelength_for_a_para_comb, hop_vec_no1)]

                        else:
                            # if L = <d_e><h> ave  link length* hopcount
                            if exclude_hop1_flag == True:
                                L = [np.mean(hop_vec_no1) * np.mean(ave_edgelength_for_a_para_comb)]
                            else:
                                L = [np.mean(hop_vec)*np.mean(ave_edgelength_for_a_para_comb)]

                        # # L = np.multiply(ave_edgelength_for_a_para_comb, hop_vec)
                        # L = [x * y for x, y in zip(ave_edgelength_for_a_para_comb, hop_vec)]

                        ave_L_vec.append(np.mean(L))
                        std_L_vec.append(np.std(L))

                    except FileNotFoundError:
                        exemptionlist.append((N, ED, beta, ExternalSimutime))

    avg_Name = folder_name + "peak_avg_N{Nn}_beta{betan}.txt".format(
        Nn=N, betan=beta)
    np.savetxt(avg_Name, real_ave_degree_vec)

    ave_deviation_Name = folder_name + "peakave_stretch_N{Nn}_beta{betan}.txt".format(
        Nn=N, betan=beta)
    np.savetxt(ave_deviation_Name, ave_L_vec)

    std_deviation_Name = folder_name + "peakstd_stretch_N{Nn}_beta{betan}.txt".format(Nn=N,                                                                                    betan=beta)
    np.savetxt(std_deviation_Name, std_L_vec)
    return ave_L_vec, std_L_vec, exemptionlist

def plot_dev_vs_avg_peak(N,beta,kvec):
    """
    the x-axis is the expected degree, the y-axis is the average deviation, different line is different c_G
    inset is the min(average deviation) vs c_G
    the x-axis is real (approximate) degree
    when use this function, use before
    :return:
    """
    ave_deviation_dict = {}
    std_deviation_dict = {}
    real_avg = {}


    betavec = [beta]
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\peak\\"

    count = 0
    for beta in betavec:
        ave_deviation_Name = filefolder_name + "peakave_stretch_N{Nn}_beta{betan}.txt".format(
            Nn=N, betan=beta)
        ave_deviation_vec = np.loadtxt(ave_deviation_Name)
        std_deviation_Name = filefolder_name + "peakstd_stretch_N{Nn}_beta{betan}.txt".format(Nn=N,
                                                                                            betan=beta)
        std_deviation_vec = np.loadtxt(std_deviation_Name)

        avg_Name = filefolder_name + "peak_avg_N{Nn}_beta{betan}.txt".format(
            Nn=N, betan=beta)
        real_avg[count] = np.loadtxt(avg_Name)
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
        x = real_avg[count]
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


    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('E[D]', fontsize=26)
    plt.ylabel('Average deviation of shortest path', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.title('1000 simulations, N=10000',fontsize=26)
    plt.legend(fontsize=20, loc="lower right")
    plt.tick_params(axis='both', which="both", length=6, width=1)

    picname = filefolder_name+"peakLocalOptimum_dev_vs_avg_beta{beta}.pdf".format(beta=beta)
    # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    # plt.show()
    plt.close()
    return result_x


def load_LCC_second_LCC_data(beta,kvec):
    """
    function for loading data for analysis first peak about Lcc AND second Lcc
    :param beta:
    :return:
    """
    input_avg_vec = [round(a, 1) for a in kvec]
    N = 10000
    filefolder_name_lcc = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\peak\\"
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

def find_giant_component(N,beta,kvec):
    LCC_vec, second_LCC_vec = load_LCC_second_LCC_data(beta,kvec)
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\peak\\"
    avg_Name = filefolder_name + "peak_avg_N{Nn}_beta{betan}.txt".format(
        Nn=N, betan=beta)
    avg_vec = np.loadtxt(avg_Name)
    # print(len(input_avg_vec))
    # print(second_LCC_vec)
    plt.plot(avg_vec, LCC_vec)
    plt.plot(avg_vec, second_LCC_vec)
    max_index = second_LCC_vec.index(max(second_LCC_vec))
    # 根据索引找到对应的 x
    result_x = avg_vec[max_index]
    print("GLCC",result_x)
    # plt.show()
    return result_x



def scattor_peakvs_GLCC():
    # Figure 4(a)
    thesis_flag = False
    import matplotlib.colors as mcolors
    from matplotlib.colors import LogNorm

    peak_avg = [np.float64(1.2492), np.float64(1.188), np.float64(1.1646), np.float64(1.3974), np.float64(1.4334),
                np.float64(1.3498), np.float64(1.4938), np.float64(1.6878), np.float64(1.6494),
                np.float64(1.9978), np.float64(1.8312), np.float64(2.0116), np.float64(2.1534), np.float64(2.6046),
                np.float64(4.3612), np.float64(4.2564), np.float64(5.0898), np.float64(4.8582),
                np.float64(4.7808), np.float64(2.7862), np.float64(3.2416), np.float64(3.0958), np.float64(3.1082),
                np.float64(3.3996), np.float64(3.7156), np.float64(4.0192), np.float64(3.847), np.float64(4.932),
                np.float64(4.8128), np.float64(4.939)]

    SLCC_avg = [np.float64(1.083), np.float64(1.0786), np.float64(1.1646), np.float64(1.2608), np.float64(1.3094),
                np.float64(1.3498), np.float64(1.4938), np.float64(1.6878), np.float64(1.5236),
                np.float64(1.8442), np.float64(1.7046), np.float64(2.0116), np.float64(2.1534), np.float64(2.507),
                np.float64(4.0514), np.float64(4.0514), np.float64(4.9346), np.float64(4.79),
                np.float64(4.709), np.float64(2.6344), np.float64(2.9306), np.float64(2.8142), np.float64(3.2564),
                np.float64(3.2596), np.float64(3.5502), np.float64(3.7502), np.float64(3.7326), np.float64(4.786),
                np.float64(4.9384), np.float64(4.7844)]
    beta_vec = [2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 3.1, 2.9, 3.4, 3.3, 3.5, 3.8, 4,8, 16, 32, 64, 128,4.2, 4.5, 4.8, 5.1, 5.4, 6, 6.3, 7, 80, 128, 128]


    print(len(SLCC_avg))
    print(len(beta_vec))

    print(np.corrcoef(peak_avg,SLCC_avg))
    # peak_avg = [4, 3.4, 5.0, 5.0, 5.8, 5.4, 6, 6.4, 3.2, 3.1, 2.6, 2.7, 2.9, 2.9]
    # SLCC_avg = [3.8, 3.2, 5.0, 4.8, 6, 5.6, 6, 6.0, 3.1, 2.9, 2.6, 2.7, 2.8, 2.7]
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    # colors = ['#ffb2b7', '#f17886', '#e04750', '#b82d36', '#7a1017']

    color2 = ["#EDD4E2", "#922E42"]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_purpleblue", color2)
    base = plt.colormaps.get_cmap('Purples')
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "deep_purples",
        base(np.linspace(0.15, 1.0, 1000))
    )

    if thesis_flag:
        plt.scatter(peak_avg, SLCC_avg, marker='o', s=150, color=colors[3], label=r"$E[D]$")
    else:
        sc = plt.scatter(peak_avg, SLCC_avg, marker='o', s=200, c=beta_vec,cmap=cmap, norm=LogNorm(),edgecolor='black',label=r"$\langle D\rangle$")
    x = np.linspace(1,5.5,10)
    y = np.linspace(1, 5.5, 10)
    plt.plot(x,y,"--",color=colors[0],label=r"$y=x$",linewidth=5)
    # plt.scatter(ave_deviation_vec, spnodenum_vec, marker='o', c=colors[1],markersize=16, label=r"$N=10^2$")

    cbar = plt.colorbar(sc)

    cbar.set_label(r'$\beta$', fontsize=30)  # LaTeX + 字体变大
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(labelsize=26)  # 刻度字体

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # plt.ylim(0, 0.30)
    # plt.yticks([0, 0.1, 0.2, 0.3])
    text = r"$N = 10^4$"
    ax.text(
        0.5, 0.85,  # 文本位置（轴坐标，0.5 表示图中央，1.05 表示轴上方）
        text,
        transform=ax.transAxes,  # 使用轴坐标
        fontsize=30,  # 字体大小
        ha='center',  # 水平居中对齐
        va='bottom'  # 垂直对齐方式
    )

    # plt.xscale('log')
    if thesis_flag:
        plt.xlabel(r'$E[D]_{\langle d \rangle_{local~max}}$', fontsize=36)
        plt.ylabel(r'$E[D]_{SLCC_{max}}$', fontsize=36)
        plt.xticks([2, 3, 4, 5], fontsize=36)
        plt.yticks([2, 3, 4, 5], fontsize=36)
        plt.legend(fontsize=40)
        plt.tick_params(axis='both', which="both", length=6, width=1)
        filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\"
        picname = filefolder_name + "scattor_slcc_vs_peak_thesis.svg"
        # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
        plt.savefig(
            picname,
            format="svg",
            bbox_inches='tight',  # 紧凑边界
            transparent=True  # 背景透明，适合插图叠加
        )
    else:
        # plt.xlabel(r'$\mathbb{E}[D]_{\langle d \rangle_{local~max}}$', fontsize=36)
        # plt.ylabel(r'$\mathbb{E}[D]_{SLCC_{max}}$', fontsize=36)
        plt.xlabel(r'$\langle D\rangle_{max}$', fontsize=36)
        plt.ylabel(r'$D_c$', fontsize=36)
        plt.xticks([1,2,3,4,5],fontsize=36)
        plt.yticks([1,2,3,4,5],fontsize=36)
        legend1 = ax.legend(loc=(0.02, 0.75),  # (x,y) 以 axes 坐标为基准
                            fontsize=30,  # 根据期刊要求调小
                            markerscale=1,
                            handlelength=1.5,
                            labelspacing=0.2,
                            ncol=1,
                            handletextpad=0.3,
                            borderpad=0.1,
                            borderaxespad=0.1
                            )
        plt.tick_params(axis='both', which="both", length=6, width=1)
        filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\hopandedgelength\\"
        picname = filefolder_name + "scattor_slcc_vs_peak.svg"
        # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
        plt.savefig(
            picname,
            format="svg",
            bbox_inches='tight',  # 紧凑边界
            transparent=True  # 背景透明，适合插图叠加
        )

    plt.show()




if __name__ == '__main__':
    """
    plot LCC and SLCC
    """
    # load_10000nodenetwork_results_peak(64)
    # find_giant_component(3.9)


    """
    first time use. need to load_10000nodenetwork_results_peak(N,beta,kvec)
    prepare data for plot the scattor plot Figure 4 (b)
    """
    # N = 10000
    # kvec_dict = {
    #     2.2: np.arange(1.4, 3, 0.2),
    #     2.3: np.arange(1.4, 3, 0.2),
    #     2.4: np.arange(1.4, 3, 0.2),
    #     2.5: np.arange(1.4, 3, 0.2),
    #     2.6: np.arange(1.4, 3, 0.2),
    #     2.7: np.arange(1.4, 3, 0.2),
    #     2.8: np.arange(1.4, 3, 0.2),
    #     2.9: np.arange(1.4, 3, 0.2),
    #     3.1: np.arange(1.4, 3, 0.2),
    #     3.3: np.arange(1.4, 3, 0.2),
    #     3.4: np.arange(1.4, 3, 0.2),
    #     3.5: np.arange(1.4, 3, 0.2),
    #     3.8: np.arange(1.4, 3, 0.2),
    #     4: np.arange(2, 5, 0.2),
    #     8: np.arange(4.1, 8, 0.1),
    #     16: np.arange(4.1, 8, 0.1),
    #     32: np.arange(4, 8, 0.1),
    #     64: np.arange(4.1, 8, 0.1),
    #     128: np.arange(4.1, 8, 0.1),
    #     1024: np.arange(4.1, 8, 0.1),
    # }

    # for beta in [4.2, 4.5, 4.8, 5.1, 5.4, 5.7, 6, 6.3, 7, 7.3, 7.6]:
    #     for N in Nvec:
    #         input_ED_vec = np.arange(3, 8, 0.2)
    #         x_1dec = [float(f"{v:.1f}") for v in input_ED_vec]
    #         for inputED in x_1dec:
    #             tasks.append((N, inputED, beta, 0))
    # for beta in [50, 80, 128, 200]:
    #     for N in Nvec:
    #         input_ED_vec = np.arange(4, 8, 0.2)
    #         x_1dec = [float(f"{v:.1f}") for v in input_ED_vec]
    #         for inputED in x_1dec:
    #             tasks.append((N, inputED, beta, 0))

    # peak_avg = [np.float64(1.2492), np.float64(1.188), np.float64(1.1646), np.float64(1.3974), np.float64(1.4334),
    #             np.float64(1.3498), np.float64(1.4938), np.float64(1.6878), np.float64(1.6494),
    #             np.float64(1.9978), np.float64(1.8312), np.float64(2.0116), np.float64(2.1534), np.float64(2.6046),
    #             np.float64(4.3612), np.float64(4.2564), np.float64(5.0898), np.float64(4.8582),
    #             np.float64(4.7808), np.float64(2.7862), np.float64(3.2416), np.float64(3.0958), np.float64(3.1082),
    #             np.float64(3.3996), np.float64(3.7156), np.float64(4.0192), np.float64(3.847), np.float64(4.932),
    #             np.float64(4.8128), np.float64(4.939)]
    #
    # SLCC_avg = [np.float64(1.083), np.float64(1.0786), np.float64(1.1646), np.float64(1.2608), np.float64(1.3094),
    #             np.float64(1.3498), np.float64(1.4938), np.float64(1.6878), np.float64(1.5236),
    #             np.float64(1.8442), np.float64(1.7046), np.float64(2.0116), np.float64(2.1534), np.float64(2.507),
    #             np.float64(4.0514), np.float64(4.0514), np.float64(4.9346), np.float64(4.79),
    #             np.float64(4.709), np.float64(2.6344), np.float64(2.9306), np.float64(2.8142), np.float64(3.2564),
    #             np.float64(3.2596), np.float64(3.5502), np.float64(3.7502), np.float64(3.7326), np.float64(4.786),
    #             np.float64(4.9384), np.float64(4.7844)]
    #
    #
    # print(kvec_dict.keys())
    # Local_max_degree_vec = []
    # GCConset_degree_vec = []
    # for beta in [8, 16, 32, 64, 128]:
    #     input_ED_vec = kvec_dict[beta]
    #     # input_ED_vec = np.arange(4, 8, 0.2)
    #     kvec = [float(f"{v:.1f}") for v in input_ED_vec]
    #     load_10000nodenetwork_results_peak(N,beta,kvec)
    #     Local_max_degree=plot_dev_vs_avg_peak(N,beta,kvec)
    #     Local_max_degree_vec.append(Local_max_degree)
    #     GCConset_degree = find_giant_component(N,beta,kvec)
    #     GCConset_degree_vec.append(GCConset_degree)
    #
    # print(Local_max_degree_vec)
    # print(GCConset_degree_vec)


    """
    # plot the scattor plot Figure 4 (b)
    """

    scattor_peakvs_GLCC()






