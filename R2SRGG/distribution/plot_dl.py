# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 2025/8/12
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import curve_fit

from R2SRGG.R2SRGG import loadSRGGandaddnode
from R2SRGG.distribution.stretchL_diffNkbeta_SRGG_ub import generate_ED_log_unifrom


def analticdl(N, k_vals):
    pi = np.pi
    # k 的取值范围
    k_vals = np.array(k_vals)
    h_vals = (2 / 3) * np.sqrt(k_vals / (N * pi)) *(1+4/(3*pi)*np.sqrt(k_vals / (N * pi)))
    return h_vals


def power_law(x, a, b):
    return a * x ** b

def MKdlmodel(avg, N, beta):
    R = 2.0  # manually tuned value
    alpha = (2 * N / avg * R * R) * (np.pi / (np.sin(2 * np.pi / beta) * beta))
    alpha = np.sqrt(alpha)
    return (np.sin(2 * np.pi / beta) / np.sin(3 * np.pi / beta)) * (1 / alpha)


def plot_dl_vs_realED(N, beta_vec):
    # Figure 4(b)
    # load and test data
    # filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\" # for ONE SP, load data for beta = 2.05 and 1024
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\hopandedgelength\\"

    # kvec = [8,10, 13, 17, 22, 28, 36, 46, 58, 74, 94, 120, 155]
    # kvec = [2, 3, 5, 8, 10, 13, 17, 22, 28, 36, 46, 58, 74, 94, 120, 155, 266, 457, 787, 1356, 2337, 4028, 6943, 11972, 20647]# for ONE SP, load data for beta = 2.05 and 1024
    kvec0 = [2.2, 3.0, 3.8, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276,
            3727, 6105,
            9999, 16479, 27081, 44767, 73534, 121205, 199999]

    # kvec = [2.2, 3.8, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276,
    #         3727, 6105,
    #         9999,]

    kvec1 = [2, 6, 16, 46, 132, 375, 1067, 3040, 8657, 24657, 70224, 199999]

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    if len(beta_vec)>5:
        colors = plt.get_cmap('tab10').colors[:len(beta_vec)+2]

    count = 0
    fig, ax = plt.subplots(figsize=(8, 8))
    if len(beta_vec) > 5:
        betalabel = ["$2.1$", "$2.3$","$2.5$","3","$2^2$", "$2^3$", "$2^6$", "$2^{10}$"]
    else:
        betalabel = ["$2.1$", "$2.3$", "$2.5$", "3"]
    data_dict = {}
    count = 0
    for beta in beta_vec:
        real_ave_degree_vec = []
        ave_length_edge_vec = []
        std_length_edge_vec = []
        if beta in [2.3,2.5,3]:
            kvec = kvec1
        else:
            kvec = kvec0
        for ED in kvec:
            print(ED)
            for ExternalSimutime in range(1):
                try:
                    ave_length_edge_Name = filefolder_name + "ave_edge_length_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
                        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                    ave_length_edge_vec.append(np.loadtxt(ave_length_edge_Name))

                    std_length_edge_Name = filefolder_name + "std_edge_length_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
                        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                    std_length_edge_vec.append(np.loadtxt(std_length_edge_Name))

                    real_avg_name = filefolder_name + "real_avg_N{Nn}ED{EDn}beta{betan}Simu{ST}.txt".format(
                        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                    real_avg = np.loadtxt(real_avg_name)
                    real_ave_degree_vec.append(real_avg)
                except:
                    print("datalost:", (ED, beta, ExternalSimutime))

            print(list(real_ave_degree_vec))
            print(ave_length_edge_vec)
            print(std_length_edge_vec)
            data_dict[beta] = (real_ave_degree_vec, ave_length_edge_vec, std_length_edge_vec)
        if beta == 2.02:
            plt.plot(real_ave_degree_vec[6:], ave_length_edge_vec[6:], linestyle="--",
                         linewidth=1,
                        marker='o', markersize=16, color=colors[count],
                         label=rf"$N=10^4$, $\beta$ = {betalabel[count]}")
            data =np.column_stack((real_ave_degree_vec[6:], ave_length_edge_vec[6:]))
            filename = f"linklengthvsdegree_N{N}_beta{beta}.txt"
            np.savetxt(filename, data, header="real average degree\tave_link_length", fmt="%.6f")

            popt2, pcov2 = curve_fit(power_law, real_ave_degree_vec[6:-5], ave_length_edge_vec[6:-5])
            a2, alpha2 = popt2
            y_fit = power_law(real_ave_degree_vec[6:], a2, alpha2)
            plt.plot(real_ave_degree_vec[6:], y_fit, '-', linewidth=2, color=colors[beta_vec.index(beta)],
                     label=fr'$f(k) = {a2:.4f} k^{{{alpha2:.2f}}}$')

        else:
            plt.plot(real_ave_degree_vec, ave_length_edge_vec, linestyle="--", linewidth=1,
                         marker='o', markersize=16, color=colors[count],
                         label=rf"$N=10^4$, $\beta$ = {betalabel[count]}")

            data = np.column_stack((real_ave_degree_vec, ave_length_edge_vec))
            filename = f"linklengthvsdegree_N{N}_beta{beta}.txt"
            np.savetxt(filename, data, header="real average degree\tave_link_length", fmt="%.6f")


            if beta in [2.3,2.5,3]:
                popt2, pcov2 = curve_fit(power_law, real_ave_degree_vec[:-5], ave_length_edge_vec[:-5])
                a2, alpha2 = popt2
                x = np.linspace(1, 15000, 50)
                y_fit = power_law(x, a2, alpha2)
                plt.plot(x, y_fit, '-', linewidth=2, color=colors[beta_vec.index(beta)],
                         label=fr'$f(k) = {a2:.4f} k^{{{alpha2:.2f}}}$')


        # if beta == 2.02:
        #     plt.errorbar(real_ave_degree_vec[6:], ave_length_edge_vec[6:], std_length_edge_vec[6:], linestyle="-",
        #                  linewidth=3,
        #                  elinewidth=1, capsize=5, marker='o', markersize=16, color=colors[count],
        #                  label=rf"$N=10^4$, $\beta$ = {betalabel[count]}")
        # else:
        #     plt.errorbar(real_ave_degree_vec, ave_length_edge_vec, std_length_edge_vec, linestyle="-", linewidth=3,
        #                  elinewidth=1,
        #                  capsize=5, marker='o', markersize=16, color=colors[count],
        #                  label=rf"$N=10^4$, $\beta$ = {betalabel[count]}")


        # if beta == 2.02:
        #     plt.plot(kvec, ave_length_edge_vec, linestyle="-",
        #                  linewidth=3,
        #                 marker='o', markersize=16, color=colors[count],
        #                  label=rf"$N=10^4$, $\beta$ = {betalabel[count]}")
        # else:
        #     plt.plot(kvec, ave_length_edge_vec, linestyle="-", linewidth=3,
        #                 marker='o', markersize=16, color=colors[count],
        #                  label=rf"$N=10^4$, $\beta$ = {betalabel[count]}")

        count = count + 1
    # data_dict = {2.05:([0.3842, 0.5602, 0.9056, 1.4132, 1.7166, 2.1752, 2.767, 3.5134, 4.3432, 5.4588, 6.794, 8.3146, 10.328, 12.7314,
    #  15.7576, 19.724, 31.4082, 49.8322, 78.8138, 123.7288, 192.536, 296.478, 450.5532, 675.6858, 996.0806],
    # [np.float64(1.4681295715778475), np.float64(1.978980557015239), np.float64(7.089406304591558), np.float64(22.6721),
    #  np.float64(16.0256), np.float64(12.0946), np.float64(9.6047), np.float64(7.9761), np.float64(6.9364),
    #  np.float64(6.0909), np.float64(5.451), np.float64(4.9801), np.float64(4.5597), np.float64(4.2282),
    #  np.float64(3.9341), np.float64(3.6837), np.float64(3.2397), np.float64(2.8884), np.float64(2.6729),
    #  np.float64(2.4423), np.float64(2.1933), np.float64(2.0248), np.float64(1.9605), np.float64(1.9333),
    #  np.float64(1.9043)],
    # [np.float64(0.7959392445512715), np.float64(1.306804595024895), np.float64(5.25729519424271),
    #  np.float64(6.641986268429046), np.float64(4.176307536568637), np.float64(2.773671004282952),
    #  np.float64(2.0463718894668195), np.float64(1.5954086592469028), np.float64(1.3506128386773169),
    #  np.float64(1.1381727417224505), np.float64(0.9963929947565869), np.float64(0.8981670167624727),
    #  np.float64(0.8066200530609192), np.float64(0.7474789361580699), np.float64(0.690765654907654),
    #  np.float64(0.6409791806291371), np.float64(0.5948478040641992), np.float64(0.4776457264542414),
    #  np.float64(0.48980158227592524), np.float64(0.5208365482567443), np.float64(0.4408345608048444),
    #  np.float64(0.28563081066299556), np.float64(0.2184942791013074), np.float64(0.2503020375466409),
    #  np.float64(0.29417938404993643)]),
    # 3:([1.5328, 2.2936, 3.786, 6.0364, 7.5058, 9.6968, 12.5228, 16.047, 20.2318, 25.7052, 32.5532, 40.562, 51.059, 63.9842,
    #  80.3998, 101.9986, 168.1194, 274.0374, 441.9626, 702.2758, 1095.6854, 1668.7986, 2467.4028, 3515.5474, 4785.3006],
    # [np.float64(18.0887), np.float64(21.8116), np.float64(11.6371), np.float64(8.1725), np.float64(7.1046),
    #  np.float64(6.1869), np.float64(5.4659), np.float64(4.9263), np.float64(4.4903), np.float64(4.1183),
    #  np.float64(3.8086), np.float64(3.5668), np.float64(3.3326), np.float64(3.1366), np.float64(2.9525),
    #  np.float64(2.7963), np.float64(2.5403), np.float64(2.2717), np.float64(2.0473), np.float64(1.9389),
    #  np.float64(1.8974), np.float64(1.8384), np.float64(1.7551), np.float64(1.6471), np.float64(1.526)],
    # [np.float64(12.115908232980308), np.float64(6.3460621995060835), np.float64(2.980738765809577),
    #  np.float64(1.9640630717978484), np.float64(1.6347045115249423), np.float64(1.3767964228599665),
    #  np.float64(1.2038426765985661), np.float64(1.0621997505177638), np.float64(0.952840967843008),
    #  np.float64(0.853290753494962), np.float64(0.7763800873283652), np.float64(0.7195399641437575),
    #  np.float64(0.6839424244773825), np.float64(0.6463284304438417), np.float64(0.5940065235332017),
    #  np.float64(0.5442483899838382), np.float64(0.5342058685563085), np.float64(0.5026719705732556),
    #  np.float64(0.366691573396499), np.float64(0.2697532020199204), np.float64(0.30507907171748117),
    #  np.float64(0.368083468794783), np.float64(0.4300278944440697), np.float64(0.47787193891250823),
    #  np.float64(0.4993235424051224)]),
    #  4: (
    #  [1.7068, 2.3392, 2.9518, 3.411, 4.6492, 7.7432, 12.2706, 20.4646, 33.0194, 53.2308, 85.65, 137.0372,
    #   218.4154, 345.4992, 540.9926, 836.887, 1274.8252, 1901.6242, 2765.4762, 3887.8192, 5251.9086,
    #   6699.4674, 8027.4616, 8988.935, 9552.1832, 9819.8554],
    #  [7.812002823554964, 25.942825451245337, 46.03553553553554, 30.642342342342342, 18.49974977479732,
    #   11.374486832882747, 8.155633450175262, 6.012829507868097, 4.862207492216531, 4.016507297433317,
    #   3.431629973742678, 2.9883260582681963, 2.6750869298425037, 2.4394363278416744, 2.2113391157182147,
    #   2.04766050054407, 2.0020510483135827, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
    #  [5.383462293482874, 14.681831440628555, 18.29254098106746, 11.250025421789491, 6.163356488499396,
    #   3.5020401904670075, 2.3296685590949666, 1.5450976129541034, 1.2110331346917764, 0.9355309714878918,
    #   0.7633027802509961, 0.6275618490624042, 0.514938547321775, 0.4967358517733515, 0.4082583665837819,
    #   0.21304688974955338, 0.045242032606834195, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #  ),
    #  1024: (
    #      [1.5554, 2.3392, 3.8926, 6.192, 7.743, 10.0388, 13.0968, 16.9344, 21.4386, 27.531, 35.0834, 44.058, 55.9208,
    #       70.7122, 89.8856, 115.2898, 194.227, 325.963, 544.0998, 900.1238, 1467.1444, 2343.9226, 3645.9348, 5444.3032,
    #       7617.0924],
    #      [np.float64(2.281541857700798), np.float64(4.5090832892986645), np.float64(45.1786), np.float64(
    #          59.942), np.float64(47.1371), np.float64(38.6387), np.float64(32.3199), np.float64(27.2802), np.float64(
    #          23.6262), np.float64(20.3878), np.float64(17.7595), np.float64(15.6246), np.float64(13.6995), np.float64(
    #          12.0753), np.float64(10.6417), np.float64(9.342), np.float64(7.1553), np.float64(5.5226), np.float64(
    #          4.3037), np.float64(3.3825), np.float64(2.6889), np.float64(2.156), np.float64(1.7654), np.float64(
    #          1.4631), np.float64(1.2414)],
    #      [np.float64(1.6684760224508528), np.float64(3.77469872533102), np.float64(33.783142275993214), np.float64(
    #          27.54113715880301), np.float64(21.77658613258745), np.float64(17.914160943510584), np.float64(
    #          15.070380353196132), np.float64(12.680500304010092), np.float64(10.980959591948238), np.float64(
    #          9.479483696910924), np.float64(8.22567077811895), np.float64(7.216139885007773), np.float64(
    #          6.304553889848195), np.float64(5.5304457243517), np.float64(4.8535678742549795), np.float64(
    #          4.238211415208071), np.float64(3.2008720546126175), np.float64(2.4204729372583365), np.float64(
    #          1.840452745929653), np.float64(1.4043481583994761), np.float64(1.0865158949596643), np.float64(
    #          0.8372956467102883), np.float64(0.6655545357068796), np.float64(0.5171444575744769), np.float64(
    #          0.42793228436284175)]
    #  )}
    # for beta, val in data_dict.items():
    #     plt.errorbar(data_dict[beta][0], data_dict[beta][1], data_dict[beta][2], linestyle="-", linewidth=3, elinewidth=1,
    #                  capsize=5, marker='o', markersize=6, label=rf"N={N},$\beta$ = {beta}")

    # popt2, pcov2 = curve_fit(power_law, data_dict[128][0][:-8], data_dict[128][1][:-8])
    # a2, alpha2 = popt2
    # # 拟合曲线
    # N_fit2 = np.linspace(1, 15000, 50)
    # a2 = 0.0034
    # alpha2 = 0.5
    # y_fit2 = power_law(N_fit2, a2, alpha2)
    # plt.plot(N_fit2, y_fit2, '-', linewidth=2, color="#E6B565",
    #          label=fr'$f(k) = {a2:.4f} k^{{{alpha2:.1f}}}$')
    #
    # # fit Samu Model
    # N_samu = np.linspace(1, 15000, 50)
    # y_samu = [analticdl(10000,i) for i in N_samu]
    # plt.plot(N_samu, y_samu, '-', linewidth=2, color=colors[8],
    #          label=fr'Samu with high order term')
    #
    #
    # # fit Mk MODEL
    # for beta in [4, 128]:
    #     N_fit2 = np.linspace(1, 15000, 50)
    #     y = [MKdlmodel(x, N, beta) for x in N_fit2]
    #     print(y)
    #     plt.plot(N_fit2,y,'-', linewidth=2,
    #          label=fr'MK model, $\beta$ = {beta}',color = colors[beta_vec.index(beta)])



    # plt.legend(fontsize=26, bbox_to_anchor=(0.42, 0.42), markerscale=1, handlelength=1, labelspacing=0.2,
    #            handletextpad=0.3, borderpad=0.1, borderaxespad=0.1)

    if len(beta_vec) > 5:
        plt.legend(fontsize=16, bbox_to_anchor=(0.7, 0.6), markerscale=1, handlelength=1, labelspacing=0.2,
                   handletextpad=0.3, borderpad=0.1, borderaxespad=0.1)
    else:
        plt.legend(fontsize=26, bbox_to_anchor=(0.7, 0.6), markerscale=1, handlelength=1, labelspacing=0.2,
               handletextpad=0.3, borderpad=0.1, borderaxespad=0.1)

    # ax.legend(
    #     fontsize=14,
    #     labelspacing=0.2,  # 行间距（默认 0.5）
    #     handlelength=1.0,  # 图例标记线长度（默认 2.0）
    #     handletextpad=0.4,  # 标记与文字之间的间距（默认 0.8）
    #     borderaxespad=0.3,  # 图例与轴的间距
    #     borderpad=0.3  # 图例边框与内容的间距
    # )

    # plt.xlabel(r'x', fontsize=35)
    # plt.ylabel(r'$f_{h}(x)$', fontsize=35)
    # plt.xticks(fontsize=28)
    # plt.yticks(fontsize=28)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\langle D \rangle$', fontsize=36)
    # plt.xlabel(r'$\mathbb{E}[D]$', fontsize=36)
    plt.ylabel(r'$\langle d_l \rangle$', fontsize=36)
    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)
    plt.tick_params(axis='both', which="both", length=6, width=1)

    # text = r"$N = 10^4$" "\n" r"$\beta = 4$"
    # plt.text(
    #     0.25, 0.65,  # 文本位置（轴坐标，0.5 表示图中央，1.05 表示轴上方）
    #     text,
    #     transform=ax.transAxes,  # 使用轴坐标
    #     fontsize=20,  # 字体大小
    #     ha='left',  # 水平居中对齐
    #     va='bottom'  # 垂直对齐方式
    # )
    # plt.title("average hopcount vs expected degree")

    picname = filefolder_name + "edgelengthvsEDN{Nn}.svg".format(
        Nn=N)
    # plt.savefig(
    #     picname,
    #     format="svg",
    #     bbox_inches='tight',  # 紧凑边界
    #     transparent=True  # 背景透明，适合插图叠加
    # )
    plt.show()
    # 清空图像，以免影响下一个图
    plt.close()



def plot_dl_vs_realED_thesis(N, beta_vec):
    # Figure 4(b)
    # load and test data
    # filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\" # for ONE SP, load data for beta = 2.05 and 1024
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\hopandedgelength\\"

    # kvec = [8,10, 13, 17, 22, 28, 36, 46, 58, 74, 94, 120, 155]
    # kvec = [2, 3, 5, 8, 10, 13, 17, 22, 28, 36, 46, 58, 74, 94, 120, 155, 266, 457, 787, 1356, 2337, 4028, 6943, 11972, 20647]# for ONE SP, load data for beta = 2.05 and 1024
    kvec0 = [2.2, 3.0, 3.8, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276,
            3727, 6105,
            9999, 16479, 27081, 44767, 73534, 121205, 199999]

    # kvec = [2.2, 3.8, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276,
    #         3727, 6105,
    #         9999,]

    kvec1 = [2, 6, 16, 46, 132, 375, 1067, 3040, 8657, 24657, 70224, 199999]

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    if len(beta_vec)>5:
        colors = plt.get_cmap('tab10').colors[:len(beta_vec)+2]

    count = 0
    fig, ax = plt.subplots(figsize=(8, 8))
    if len(beta_vec) > 5:

        betalabel = ["$2.1$", "$2.3$","$2.5$","3","$2^2$", "$2^3$", "$2^6$", "$2^{10}$"]
    else:

        betalabel = ["$2.1$", "$2^2$", "$2^3$", "$2^6$"]
    data_dict = {}
    count = 0
    for beta in beta_vec:
        real_ave_degree_vec = []
        ave_length_edge_vec = []
        std_length_edge_vec = []
        if beta in [2.3,2.5,3]:
            kvec = kvec1
        else:
            kvec = kvec0
        for ED in kvec:
            print(ED)
            for ExternalSimutime in range(1):
                try:
                    ave_length_edge_Name = filefolder_name + "ave_edge_length_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
                        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                    ave_length_edge_vec.append(np.loadtxt(ave_length_edge_Name))

                    std_length_edge_Name = filefolder_name + "std_edge_length_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
                        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                    std_length_edge_vec.append(np.loadtxt(std_length_edge_Name))

                    real_avg_name = filefolder_name + "real_avg_N{Nn}ED{EDn}beta{betan}Simu{ST}.txt".format(
                        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                    real_avg = np.loadtxt(real_avg_name)
                    real_ave_degree_vec.append(real_avg)
                except:
                    print("datalost:", (ED, beta, ExternalSimutime))

            print(list(real_ave_degree_vec))
            print(ave_length_edge_vec)
            print(std_length_edge_vec)
            data_dict[beta] = (real_ave_degree_vec, ave_length_edge_vec, std_length_edge_vec)
        if beta == 2.02:
            plt.plot(real_ave_degree_vec[6:], ave_length_edge_vec[6:], linestyle="--",
                         linewidth=1,
                        marker='o', markersize=16, color=colors[count],
                         label=rf"$N=10^4$, $\beta$ = {betalabel[count]}")
            data =np.column_stack((real_ave_degree_vec[6:], ave_length_edge_vec[6:]))
            filename = f"linklengthvsdegree_N{N}_beta{beta}.txt"
            np.savetxt(filename, data, header="real average degree\tave_link_length", fmt="%.6f")

            popt2, pcov2 = curve_fit(power_law, real_ave_degree_vec[6:-5], ave_length_edge_vec[6:-5])
            a2, alpha2 = popt2
            y_fit = power_law(real_ave_degree_vec[6:], a2, alpha2)
            # plt.plot(real_ave_degree_vec[6:], y_fit, '-', linewidth=2, color=colors[beta_vec.index(beta)],
            #          label=fr'$f(k) = {a2:.4f} k^{{{alpha2:.2f}}}$')

        else:
            plt.plot(real_ave_degree_vec, ave_length_edge_vec, linestyle="--", linewidth=1,
                         marker='o', markersize=16, color=colors[count],
                         label=rf"$N=10^4$, $\beta$ = {betalabel[count]}")

            data = np.column_stack((real_ave_degree_vec, ave_length_edge_vec))
            filename = f"linklengthvsdegree_N{N}_beta{beta}.txt"
            np.savetxt(filename, data, header="real average degree\tave_link_length", fmt="%.6f")


            if beta in [2.3,2.5,3]:
                popt2, pcov2 = curve_fit(power_law, real_ave_degree_vec[:-5], ave_length_edge_vec[:-5])
                a2, alpha2 = popt2
                x = np.linspace(1, 15000, 50)
                y_fit = power_law(x, a2, alpha2)
                plt.plot(x, y_fit, '-', linewidth=2, color=colors[beta_vec.index(beta)],
                         label=fr'$f(k) = {a2:.4f} k^{{{alpha2:.2f}}}$')


        # if beta == 2.02:
        #     plt.errorbar(real_ave_degree_vec[6:], ave_length_edge_vec[6:], std_length_edge_vec[6:], linestyle="-",
        #                  linewidth=3,
        #                  elinewidth=1, capsize=5, marker='o', markersize=16, color=colors[count],
        #                  label=rf"$N=10^4$, $\beta$ = {betalabel[count]}")
        # else:
        #     plt.errorbar(real_ave_degree_vec, ave_length_edge_vec, std_length_edge_vec, linestyle="-", linewidth=3,
        #                  elinewidth=1,
        #                  capsize=5, marker='o', markersize=16, color=colors[count],
        #                  label=rf"$N=10^4$, $\beta$ = {betalabel[count]}")


        # if beta == 2.02:
        #     plt.plot(kvec, ave_length_edge_vec, linestyle="-",
        #                  linewidth=3,
        #                 marker='o', markersize=16, color=colors[count],
        #                  label=rf"$N=10^4$, $\beta$ = {betalabel[count]}")
        # else:
        #     plt.plot(kvec, ave_length_edge_vec, linestyle="-", linewidth=3,
        #                 marker='o', markersize=16, color=colors[count],
        #                  label=rf"$N=10^4$, $\beta$ = {betalabel[count]}")

        count = count + 1
    # data_dict = {2.05:([0.3842, 0.5602, 0.9056, 1.4132, 1.7166, 2.1752, 2.767, 3.5134, 4.3432, 5.4588, 6.794, 8.3146, 10.328, 12.7314,
    #  15.7576, 19.724, 31.4082, 49.8322, 78.8138, 123.7288, 192.536, 296.478, 450.5532, 675.6858, 996.0806],
    # [np.float64(1.4681295715778475), np.float64(1.978980557015239), np.float64(7.089406304591558), np.float64(22.6721),
    #  np.float64(16.0256), np.float64(12.0946), np.float64(9.6047), np.float64(7.9761), np.float64(6.9364),
    #  np.float64(6.0909), np.float64(5.451), np.float64(4.9801), np.float64(4.5597), np.float64(4.2282),
    #  np.float64(3.9341), np.float64(3.6837), np.float64(3.2397), np.float64(2.8884), np.float64(2.6729),
    #  np.float64(2.4423), np.float64(2.1933), np.float64(2.0248), np.float64(1.9605), np.float64(1.9333),
    #  np.float64(1.9043)],
    # [np.float64(0.7959392445512715), np.float64(1.306804595024895), np.float64(5.25729519424271),
    #  np.float64(6.641986268429046), np.float64(4.176307536568637), np.float64(2.773671004282952),
    #  np.float64(2.0463718894668195), np.float64(1.5954086592469028), np.float64(1.3506128386773169),
    #  np.float64(1.1381727417224505), np.float64(0.9963929947565869), np.float64(0.8981670167624727),
    #  np.float64(0.8066200530609192), np.float64(0.7474789361580699), np.float64(0.690765654907654),
    #  np.float64(0.6409791806291371), np.float64(0.5948478040641992), np.float64(0.4776457264542414),
    #  np.float64(0.48980158227592524), np.float64(0.5208365482567443), np.float64(0.4408345608048444),
    #  np.float64(0.28563081066299556), np.float64(0.2184942791013074), np.float64(0.2503020375466409),
    #  np.float64(0.29417938404993643)]),
    # 3:([1.5328, 2.2936, 3.786, 6.0364, 7.5058, 9.6968, 12.5228, 16.047, 20.2318, 25.7052, 32.5532, 40.562, 51.059, 63.9842,
    #  80.3998, 101.9986, 168.1194, 274.0374, 441.9626, 702.2758, 1095.6854, 1668.7986, 2467.4028, 3515.5474, 4785.3006],
    # [np.float64(18.0887), np.float64(21.8116), np.float64(11.6371), np.float64(8.1725), np.float64(7.1046),
    #  np.float64(6.1869), np.float64(5.4659), np.float64(4.9263), np.float64(4.4903), np.float64(4.1183),
    #  np.float64(3.8086), np.float64(3.5668), np.float64(3.3326), np.float64(3.1366), np.float64(2.9525),
    #  np.float64(2.7963), np.float64(2.5403), np.float64(2.2717), np.float64(2.0473), np.float64(1.9389),
    #  np.float64(1.8974), np.float64(1.8384), np.float64(1.7551), np.float64(1.6471), np.float64(1.526)],
    # [np.float64(12.115908232980308), np.float64(6.3460621995060835), np.float64(2.980738765809577),
    #  np.float64(1.9640630717978484), np.float64(1.6347045115249423), np.float64(1.3767964228599665),
    #  np.float64(1.2038426765985661), np.float64(1.0621997505177638), np.float64(0.952840967843008),
    #  np.float64(0.853290753494962), np.float64(0.7763800873283652), np.float64(0.7195399641437575),
    #  np.float64(0.6839424244773825), np.float64(0.6463284304438417), np.float64(0.5940065235332017),
    #  np.float64(0.5442483899838382), np.float64(0.5342058685563085), np.float64(0.5026719705732556),
    #  np.float64(0.366691573396499), np.float64(0.2697532020199204), np.float64(0.30507907171748117),
    #  np.float64(0.368083468794783), np.float64(0.4300278944440697), np.float64(0.47787193891250823),
    #  np.float64(0.4993235424051224)]),
    #  4: (
    #  [1.7068, 2.3392, 2.9518, 3.411, 4.6492, 7.7432, 12.2706, 20.4646, 33.0194, 53.2308, 85.65, 137.0372,
    #   218.4154, 345.4992, 540.9926, 836.887, 1274.8252, 1901.6242, 2765.4762, 3887.8192, 5251.9086,
    #   6699.4674, 8027.4616, 8988.935, 9552.1832, 9819.8554],
    #  [7.812002823554964, 25.942825451245337, 46.03553553553554, 30.642342342342342, 18.49974977479732,
    #   11.374486832882747, 8.155633450175262, 6.012829507868097, 4.862207492216531, 4.016507297433317,
    #   3.431629973742678, 2.9883260582681963, 2.6750869298425037, 2.4394363278416744, 2.2113391157182147,
    #   2.04766050054407, 2.0020510483135827, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
    #  [5.383462293482874, 14.681831440628555, 18.29254098106746, 11.250025421789491, 6.163356488499396,
    #   3.5020401904670075, 2.3296685590949666, 1.5450976129541034, 1.2110331346917764, 0.9355309714878918,
    #   0.7633027802509961, 0.6275618490624042, 0.514938547321775, 0.4967358517733515, 0.4082583665837819,
    #   0.21304688974955338, 0.045242032606834195, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #  ),
    #  1024: (
    #      [1.5554, 2.3392, 3.8926, 6.192, 7.743, 10.0388, 13.0968, 16.9344, 21.4386, 27.531, 35.0834, 44.058, 55.9208,
    #       70.7122, 89.8856, 115.2898, 194.227, 325.963, 544.0998, 900.1238, 1467.1444, 2343.9226, 3645.9348, 5444.3032,
    #       7617.0924],
    #      [np.float64(2.281541857700798), np.float64(4.5090832892986645), np.float64(45.1786), np.float64(
    #          59.942), np.float64(47.1371), np.float64(38.6387), np.float64(32.3199), np.float64(27.2802), np.float64(
    #          23.6262), np.float64(20.3878), np.float64(17.7595), np.float64(15.6246), np.float64(13.6995), np.float64(
    #          12.0753), np.float64(10.6417), np.float64(9.342), np.float64(7.1553), np.float64(5.5226), np.float64(
    #          4.3037), np.float64(3.3825), np.float64(2.6889), np.float64(2.156), np.float64(1.7654), np.float64(
    #          1.4631), np.float64(1.2414)],
    #      [np.float64(1.6684760224508528), np.float64(3.77469872533102), np.float64(33.783142275993214), np.float64(
    #          27.54113715880301), np.float64(21.77658613258745), np.float64(17.914160943510584), np.float64(
    #          15.070380353196132), np.float64(12.680500304010092), np.float64(10.980959591948238), np.float64(
    #          9.479483696910924), np.float64(8.22567077811895), np.float64(7.216139885007773), np.float64(
    #          6.304553889848195), np.float64(5.5304457243517), np.float64(4.8535678742549795), np.float64(
    #          4.238211415208071), np.float64(3.2008720546126175), np.float64(2.4204729372583365), np.float64(
    #          1.840452745929653), np.float64(1.4043481583994761), np.float64(1.0865158949596643), np.float64(
    #          0.8372956467102883), np.float64(0.6655545357068796), np.float64(0.5171444575744769), np.float64(
    #          0.42793228436284175)]
    #  )}
    # for beta, val in data_dict.items():
    #     plt.errorbar(data_dict[beta][0], data_dict[beta][1], data_dict[beta][2], linestyle="-", linewidth=3, elinewidth=1,
    #                  capsize=5, marker='o', markersize=6, label=rf"N={N},$\beta$ = {beta}")

    # popt2, pcov2 = curve_fit(power_law, data_dict[128][0][:-8], data_dict[128][1][:-8])
    # a2, alpha2 = popt2
    # # 拟合曲线
    N_fit2 = np.linspace(1, 15000, 50)
    a2 = 0.0034
    alpha2 = 0.5
    y_fit2 = power_law(N_fit2, a2, alpha2)
    plt.plot(N_fit2, y_fit2, '-', linewidth=2, color="#E6B565",
             label=fr'$f(\langle D \rangle) = {a2:.3f} \langle D \rangle^{{{alpha2:.1f}}}$')
    #
    # # fit Samu Model
    # N_samu = np.linspace(1, 15000, 50)
    # y_samu = [analticdl(10000,i) for i in N_samu]
    # plt.plot(N_samu, y_samu, '-', linewidth=2, color=colors[8],
    #          label=fr'Samu with high order term')
    #
    #
    # # fit Mk MODEL
    # for beta in [4, 128]:
    #     N_fit2 = np.linspace(1, 15000, 50)
    #     y = [MKdlmodel(x, N, beta) for x in N_fit2]
    #     print(y)
    #     plt.plot(N_fit2,y,'-', linewidth=2,
    #          label=fr'MK model, $\beta$ = {beta}',color = colors[beta_vec.index(beta)])



    # plt.legend(fontsize=26, bbox_to_anchor=(0.42, 0.42), markerscale=1, handlelength=1, labelspacing=0.2,
    #            handletextpad=0.3, borderpad=0.1, borderaxespad=0.1)

    if len(beta_vec) > 5:
        plt.legend(fontsize=16, bbox_to_anchor=(0.7, 0.6), markerscale=1, handlelength=1, labelspacing=0.2,
                   handletextpad=0.3, borderpad=0.1, borderaxespad=0.1)
    else:
        plt.legend(fontsize=24, bbox_to_anchor=(0.4, 0.4), markerscale=1, handlelength=1, labelspacing=0.2,
               handletextpad=0.3, borderpad=0.1, borderaxespad=0.1)

    # ax.legend(
    #     fontsize=14,
    #     labelspacing=0.2,  # 行间距（默认 0.5）
    #     handlelength=1.0,  # 图例标记线长度（默认 2.0）
    #     handletextpad=0.4,  # 标记与文字之间的间距（默认 0.8）
    #     borderaxespad=0.3,  # 图例与轴的间距
    #     borderpad=0.3  # 图例边框与内容的间距
    # )

    # plt.xlabel(r'x', fontsize=35)
    # plt.ylabel(r'$f_{h}(x)$', fontsize=35)
    # plt.xticks(fontsize=28)
    # plt.yticks(fontsize=28)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\langle D \rangle$', fontsize=36)
    # plt.xlabel(r'$\mathbb{E}[D]$', fontsize=36)
    plt.ylabel(r'$\langle d_l \rangle$', fontsize=36)
    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)
    plt.tick_params(axis='both', which="both", length=6, width=1)

    # text = r"$N = 10^4$" "\n" r"$\beta = 4$"
    # plt.text(
    #     0.25, 0.65,  # 文本位置（轴坐标，0.5 表示图中央，1.05 表示轴上方）
    #     text,
    #     transform=ax.transAxes,  # 使用轴坐标
    #     fontsize=20,  # 字体大小
    #     ha='left',  # 水平居中对齐
    #     va='bottom'  # 垂直对齐方式
    # )
    # plt.title("average hopcount vs expected degree")

    picname = filefolder_name + "edgelengthvsEDN{Nn}.svg".format(
        Nn=N)
    plt.savefig(
        picname,
        format="svg",
        bbox_inches='tight',  # 紧凑边界
        transparent=True  # 背景透明，适合插图叠加
    )
    plt.show()
    # 清空图像，以免影响下一个图
    plt.close()




def load_large_network_results_dev_vs_avg(N, beta, kvec, realL):
    if realL:
        # if L = <d_e>h real stretch
        folder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\"
    else:
        # if L = <d_e><h> ave  link length* hopcount
        folder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\hopandedgelength\\"

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
                        if N>200:
                            real_avg_name = folder_name + "real_avg_N{Nn}ED{EDn}beta{betan}Simu{ST}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                            real_avg = np.loadtxt(real_avg_name)
                            real_ave_degree_vec.append(real_avg)
                        else:
                            real_ave_degree_name = folder_name + "real_ave_degree_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                            real_avg = np.loadtxt(real_ave_degree_name)
                            real_ave_degree_vec.append(np.mean(real_avg))


                        if realL:
                            #if L = <d_e>h real stretch
                            deviation_vec_name = folder_name + "ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                            ave_deviation_for_a_para_comb = np.loadtxt(deviation_vec_name)
                            ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
                            std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))


                            edgelength_vec_name = folder_name + "ave_edgelength_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        else:
                            # if L = <d_e><h> ave  link length* hopcount
                            edgelength_vec_name = folder_name + "ave_edge_length_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
                                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)


                        # deviation_vec_name = folder_name + "ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                        #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        # ave_deviation_for_a_para_comb = np.loadtxt(deviation_vec_name)
                        # ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
                        # std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))
                        #
                        # edgelength_vec_name = folder_name + "ave_edgelength_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                        #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)


                        ave_edgelength_for_a_para_comb = np.loadtxt(edgelength_vec_name)
                        ave_edgelength_vec.append(np.mean(ave_edgelength_for_a_para_comb))
                        std_edgelength_vec.append(np.std(ave_edgelength_for_a_para_comb))



                        hopcount_Name = folder_name + "hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
                            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        hop_vec = np.loadtxt(hopcount_Name, dtype=int)

                        ave_hop_vec.append(np.mean(hop_vec))
                        std_hop_vec.append(np.std(hop_vec))


                        if realL:
                            #if L = <d_e>h real stretch
                            L = [x * y for x, y in zip(ave_edgelength_for_a_para_comb, hop_vec)]
                        else:
                            # if L = <d_e><h> ave  link length* hopcount
                            L = [np.mean(hop_vec)*np.mean(ave_edgelength_for_a_para_comb)]

                        # # L = np.multiply(ave_edgelength_for_a_para_comb, hop_vec)
                        # L = [x * y for x, y in zip(ave_edgelength_for_a_para_comb, hop_vec)]

                        ave_L_vec.append(np.mean(L))
                        std_L_vec.append(np.std(L))

                    except FileNotFoundError:
                        exemptionlist.append((N, ED, beta, ExternalSimutime))
    print(exemptionlist)
    return real_ave_degree_vec, ave_deviation_vec, std_deviation_vec, ave_edgelength_vec, std_edgelength_vec, ave_hop_vec, std_hop_vec, ave_L_vec, std_L_vec
    # return kvec, real_ave_degree_vec, ave_deviation_vec, std_deviation_vec


def load_large_network_results_dev_vs_avg_locmin_hunter(N, beta, kvec, realL):
    folder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\localmin_hunter\\"

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

                        real_ave_degree_name = folder_name + "real_ave_degree_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        real_avg = np.loadtxt(real_ave_degree_name)
                        real_ave_degree_vec.append(np.mean(real_avg))


                        if realL:
                            #if L = <d_e>h real stretch
                            # deviation_vec_name = folder_name + "ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                            #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                            # ave_deviation_for_a_para_comb = np.loadtxt(deviation_vec_name)
                            # ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
                            # std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))


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
                            L = [np.mean(hop_vec)*np.mean(ave_edgelength_for_a_para_comb)]

                        # # L = np.multiply(ave_edgelength_for_a_para_comb, hop_vec)
                        # L = [x * y for x, y in zip(ave_edgelength_for_a_para_comb, hop_vec)]

                        ave_L_vec.append(np.mean(L))
                        std_L_vec.append(np.std(L))

                    except FileNotFoundError:
                        exemptionlist.append((N, ED, beta, ExternalSimutime))
    print(exemptionlist)
    return real_ave_degree_vec, ave_deviation_vec, std_deviation_vec, ave_edgelength_vec, std_edgelength_vec, ave_hop_vec, std_hop_vec, ave_L_vec, std_L_vec
    # return kvec, real_ave_degree_vec, ave_deviation_vec, std_deviation_vec


def MKmodel_dl(k, beta):
    return k**(beta/2 - 1)



def plot_dl_vs_k_inlarge_network():
    Nvec = [464, 1000, 2154, 4642, 10000]
    Nvec = [10000, 20000, 40000, 60000, 100000]
    # Nvec = [215]
    # beta = 1024
    beta = 2.5
    realL = False

    real_ave_degree_dict = {}
    ave_L = {}
    std_L = {}

    real_ave_degree_dict_0 = {}
    ave_L_0 = {}
    std_L_0 = {}

    k_star_dict = {}
    localmin_dict = {}

    linklength_dict = {}

    if beta == 1024:
        # kvec_dict = {215: list(range(24, 104 + 1, 2)), 464: list(range(30, 154 + 1, 2)),
        #              1000: list(range(39, 229 + 1, 2)),
        #              2154: list(range(52, 364 + 1, 2)), 4642: list(range(67, 272 + 1, 2)),
        #              10000: list(range(118, 316 + 1, 2)),
        #              681: list(range(40, 164 + 1, 2)), 1468: list(range(50, 240 + 1, 2)),
        #              3156: list(range(72, 384 + 1, 2)),
        #              6803: list(range(87, 295 + 1, 2)), 14683: list(range(140, 340 + 1, 2))}
        # kvec_dict = {215: [2, 3, 5, 9, 14] + list(range(24, 104 + 1, 2)) + [170, 278, 455, 746, 1221, 2000],
        #           464: list(range(30, 154 + 1, 2)), 1000: list(range(39, 229 + 1, 2)), 2154: list(
        #         range(52, 364 + 1, 2)), 4642: list(range(67, 272 + 1, 2)), 10000: list(range(118, 316 + 1, 2))}

        kvec_dict_0 = {
            100: [2, 3, 5, 8, 12, 18, 29, 45, 70, 109, 169, 264, 412, 642, 1000],
            215: [2, 3, 5, 9, 14, 24, 39, 63, 104, 170, 278, 455, 746, 1221, 2000],
            464: [2, 3, 6, 10, 18, 30, 52, 89, 154, 265, 456, 785, 1350, 2324, 4000],
            1000: [2, 4, 7, 12, 21, 39, 70, 126, 229, 414, 748, 1353, 2446, 4424, 8000],
            2154: [2, 4, 7, 14, 27, 52, 99, 190, 364, 697, 1335, 2558, 4902, 9393, 18000],
            4642: [2, 4, 8, 16, 33, 67, 135, 272, 549, 1107, 2234, 4506, 9091, 18340, 37000],
            10000: [2.2, 2.8, 3.0, 3.4, 3.8, 4.4, 6.0, 7.0, 8.0, 9.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848,
                    1389,
                    2276,
                    3727, 6105,
                    9999, 16479, 27081, 44767, 73534, 121205, 199999]}
    elif beta == 2.1:
        # kvec_dict = {
        #     464: generate_ED_log_unifrom(2, 1000000, 12),
        #     681: generate_ED_log_unifrom(2, 1000000, 12),
        #     1000: generate_ED_log_unifrom(2, 1000000, 12),
        #     1468: generate_ED_log_unifrom(2, 1000000, 12),
        #     2154: generate_ED_log_unifrom(2, 1000000, 12),
        #     3156: generate_ED_log_unifrom(2, 1000000, 12),
        #     4642: generate_ED_log_unifrom(2, 1000000, 12),
        #     6803: generate_ED_log_unifrom(2, 1000000, 12),
        #     10000: generate_ED_log_unifrom(2, 1000000, 12)+[3296030],
        #     20000: generate_ED_log_unifrom(2, 1000000, 12)+[3296030],
        #     40000: generate_ED_log_unifrom(2, 1000000, 12)+[3296030],
        #     60000: generate_ED_log_unifrom(2, 1000000, 12)+[3296030],
        #     100000: generate_ED_log_unifrom(2, 1000000, 12),
        # }
        kvec_dict = {
            464: generate_ED_log_unifrom(2, 1000000, 12),
            681: generate_ED_log_unifrom(2, 1000000, 12),
            1000: generate_ED_log_unifrom(2, 1000000, 12),
            1468: generate_ED_log_unifrom(2, 1000000, 12),
            2154: generate_ED_log_unifrom(2, 1000000, 12),
            3156: generate_ED_log_unifrom(2, 1000000, 12),
            4642: generate_ED_log_unifrom(2, 1000000, 12),
            6803: generate_ED_log_unifrom(2, 1000000, 12),
            10000: [2, 7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328, 1000000] + [3296030],
            20000: [2, 7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328, 1000000] + [3296030,
                                                                                          10866500,
                                                                                          35826700],
            40000: [2, 7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328, 1000000] + [3296030,
                                                                                          10866500,
                                                                                          35826700],
            60000: [2, 7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328, 1000000] + [3296030,
                                                                                          10866500,
                                                                                          35826700],
            100000: [2, 7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328, 1000000],
        }

    elif beta==2.5:
        kvec_dict = {
            464: generate_ED_log_unifrom(2, 1000000, 12),
            681: generate_ED_log_unifrom(2, 1000000, 12),
            1000: generate_ED_log_unifrom(2, 1000000, 12),
            1468: generate_ED_log_unifrom(2, 1000000, 12),
            2154: generate_ED_log_unifrom(2, 1000000, 12),
            3156: generate_ED_log_unifrom(2, 1000000, 12),
            4642: generate_ED_log_unifrom(2, 1000000, 12),
            6803: generate_ED_log_unifrom(2, 1000000, 12),
            10000: [2, 7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328, 3296030],
            20000: [2, 7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328, 1000000],
            40000: [2, 7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328, 1000000, 10866500],
            60000: [2, 7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328, 1000000],
            100000: [2, 7, 22, 72, 236, 779, 2568, 8465, 27908, 92008, 303328]
        }
    fig, ax = plt.subplots()

    count = 0
    for N in Nvec:
        # kvec = kvec_dict[N]
        # real_ave_degree_vec, _, _, _, _, _, _, ave_L_vec, std_L_vec = load_large_network_results_dev_vs_avg_locmin_hunter(
        #     N, beta, kvec, realL)
        # real_ave_degree_dict[N] = real_ave_degree_vec
        # ave_L[N] = ave_L_vec
        # std_L[N] = std_L_vec
        if beta == 1024:
            kvec2 = kvec_dict_0[N]
            real_ave_degree_vec_0, _, _, linklength_vec_0, _, _, _, ave_L_vec_0, std_L_vec_0 = load_large_network_results_dev_vs_avg(
                N, beta, kvec2, realL)
            linklength_dict[N] = linklength_vec_0
        else:
            kvec2 = kvec_dict[N]
            real_ave_degree_vec_0, _, _, linklength_vec_0, _,_ , _, ave_L_vec_0, std_L_vec_0 = load_large_network_results_dev_vs_avg_locmin_hunter(
                N, beta, kvec2, realL)
            linklength_dict[N] = linklength_vec_0

        plt.plot(real_ave_degree_vec_0, linklength_vec_0, "--o", markersize=25, markerfacecolor='none', linewidth=5,
                 label=f"{N}")

        data = np.column_stack((real_ave_degree_vec_0, linklength_vec_0))
        np.savetxt(f"graph_link_length_vs_degree_N{N}.txt", data, fmt="%.6f", header="real_avg linklength_graph", comments='')

        # plt.plot(kvec2, hop_vec_0, "-o", markersize=25, markerfacecolor='none', linewidth=5,
        #          label=f"{N}")
        # plt.plot([alpha(x, N, beta) for x in kvec2], hop_vec_0, "-o", markersize=25, markerfacecolor='none', linewidth=5,
        #          label=f"{N}")


        count = count + 1


    # fit Mk model
    y_fit = [0.02*MKmodel_dl(k, beta) for k in real_ave_degree_vec_0]
    plt.plot(real_ave_degree_vec_0, y_fit, '-', label=r"fit: $y = 0.02 k^{\beta/2-1}$")

    # fit power law
    popt, pcov = curve_fit(power_law, real_ave_degree_vec_0[:-3], linklength_vec_0[:-3])
    a_fit, b_fit = popt
    print("拟合参数: a = %.4f, b = %.4f" % (a_fit, b_fit))
    plt.plot(real_ave_degree_vec_0, power_law(real_ave_degree_vec_0, a_fit, b_fit), label=f"fit2: y = {a_fit:.2f} * x^{b_fit:.2f}", color='red')

    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.yscale('log')
    plt.xscale('log')
    # plt.xlabel(r'Network size, $N$', fontsize=26)
    plt.xlabel(r'$k$', fontsize=26)
    # plt.xlabel(r'Expected degree $E[D]$', fontsize=26)
    # plt.xlabel(r'$\alpha$', fontsize=26)
    plt.ylabel(r' $\langle d_l \rangle$', fontsize=26)
    plt.legend(fontsize=12, loc=(0.7, 0.2))
    plt.show()


if __name__ == '__main__':
    # step_1: for differnet beta  Figure4 b new version
    # plot_dl_vs_realED(10000, [2.02,2.3,2.5,3, 4, 8, 128])

    plot_dl_vs_realED_thesis(10000, [2.02, 4, 8, 128])

    # step_2: for differnet N
    # plot_dl_vs_k_inlarge_network()


