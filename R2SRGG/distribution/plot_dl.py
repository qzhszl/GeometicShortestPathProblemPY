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


def power_law(x, a, b):
    return a * x ** b

def MKdlmodel(avg, N, beta):
    R = 2.0  # manually tuned value
    alpha = (2 * N / avg * R * R) * (np.pi / (np.sin(2 * np.pi / beta) * beta))
    alpha = np.sqrt(alpha)
    return (np.sin(2 * np.pi / beta) / np.sin(3 * np.pi / beta)) * (1 / alpha)


def plot_dl_vs_realED(N, beta_vec):
    # Figure 4(d)
    # load and test data
    # filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\" # for ONE SP, load data for beta = 2.05 and 1024
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\hopandedgelength\\"

    # kvec = [8,10, 13, 17, 22, 28, 36, 46, 58, 74, 94, 120, 155]
    # kvec = [2, 3, 5, 8, 10, 13, 17, 22, 28, 36, 46, 58, 74, 94, 120, 155, 266, 457, 787, 1356, 2337, 4028, 6943, 11972, 20647]# for ONE SP, load data for beta = 2.05 and 1024
    # kvec = [2.2, 3.0, 3.8, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276,
    #         3727, 6105,
    #         9999, 16479, 27081, 44767, 73534, 121205, 199999]

    kvec = [2.2, 3.8, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276,
            3727, 6105,
            9999,]
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    count = 0
    fig, ax = plt.subplots(figsize=(8, 8))
    betalabel = ["$2.1$", "$2^2$", "$2^3$", "$2^6$", "$2^{10}$"]
    data_dict = {}
    count = 0
    for beta in beta_vec:
        real_ave_degree_vec = []
        ave_length_edge_vec = []
        std_length_edge_vec = []
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
            plt.plot(real_ave_degree_vec[6:], ave_length_edge_vec[6:], linestyle="-",
                         linewidth=3,
                        marker='o', markersize=16, color=colors[count],
                         label=rf"$N=10^4$, $\beta$ = {betalabel[count]}")
        else:
            plt.plot(real_ave_degree_vec, ave_length_edge_vec, linestyle="-", linewidth=3,
                         marker='o', markersize=16, color=colors[count],
                         label=rf"$N=10^4$, $\beta$ = {betalabel[count]}")


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
    plt.plot(N_fit2, y_fit2, '--', linewidth=5, color="#E6B565",
             label=fr'$f(k) = {a2:.4f} k^{{{alpha2:.1f}}}$')

    # fit Mk MODEL
    for beta in [2.02, 4, 8, 128]:
        N_fit2 = np.linspace(1, 15000, 50)
        y = [MKdlmodel(x, N, beta) for x in N_fit2]
        print(y)
        plt.plot(N_fit2,y,'--', linewidth=5,
             label=fr'MK model, $\beta$ = {beta}')



    # plt.legend(fontsize=26, bbox_to_anchor=(0.42, 0.42), markerscale=1, handlelength=1, labelspacing=0.2,
    #            handletextpad=0.3, borderpad=0.1, borderaxespad=0.1)

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


if __name__ == '__main__':
    plot_dl_vs_realED(10000, [2.02, 4, 8, 128])
