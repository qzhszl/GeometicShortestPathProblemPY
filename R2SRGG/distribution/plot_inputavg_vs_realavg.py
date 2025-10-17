# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 2024/9/8
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
import math

from scipy.optimize import curve_fit
from scipy.integrate import quad
from math import pi, sqrt, sin, atan


from R2SRGG.R2SRGG import loadSRGGandaddnode


def load_10000nodenetwork_results(beta):
    # kvec = [16479, 27081, 44767, 73534, 121205, 199999]

    # for beta = 128
    # kvec = [16479, 21121, 27081, 34822, 44767, 57363]
    kvec = [2.2, 2.8, 3.0, 3.4, 3.8, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999,
         16479, 21121, 27081, 34822, 44767, 57363]
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
                        deviation_vec_name = filefolder_name + "ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        ave_deviation_for_a_para_comb_10times = np.loadtxt(deviation_vec_name)
                        ave_deviation_for_a_para_comb = np.hstack(
                            (ave_deviation_for_a_para_comb, ave_deviation_for_a_para_comb_10times))
                    except FileNotFoundError:
                        exemptionlist.append((N, ED, beta, ExternalSimutime))

                print(len(ave_deviation_for_a_para_comb))
                ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
                std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))
    print(exemptionlist)
    # ave_deviation_Name = filefolder_name + "ave_deviation_N{Nn}_beta{betan}.txt".format(
    #     Nn=N, betan=beta)
    # np.savetxt(ave_deviation_Name, ave_deviation_vec)
    # std_deviation_Name = filefolder_name + "std_deviation_N{Nn}_beta{betan}.txt".format(Nn=N,
    #                                                                                     betan=beta)
    # np.savetxt(std_deviation_Name, std_deviation_vec)
    print(ave_deviation_vec)
    print(std_deviation_vec)
    print(list(map(float, ave_deviation_vec)))
    print(list(map(float,std_deviation_vec)))
    return ave_deviation_vec, std_deviation_vec, exemptionlist


def power_law(x, a, b):
    return a * x**b


def plot_inputavg_vs_realavg(beta,thesis_flag):
    # Figure 1 (b)
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494', "#A2C7A4", "#9DB0C2", "#E3B6A4"]
    if beta == 4:

        filefolder = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\inputavgbeta\\"
        x = [2.2, 2.8, 3, 3.4, 3.8, 4.4, 6, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999,
             16479, 27081, 44767, 73534, 121205, 199999,331131,539052,888611,1465694]
        # x = [2.2, 2.8, 3, 3.4, 3.8, 4.4, 6, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
        real_avg_vec = [1.7024, 2.1224, 2.2988, 2.6058, 2.941, 3.3956, 4.6198, 7.6544, 12.1272, 20.414358564143587,
                        32.9682,
                        53.2058, 85.6794, 137.1644, 218.4686, 345.3296, 541.029, 836.6424, 1278.4108, 1902.8332,
                        2783.4186,
                        3911.416, 5253, 6700, 8029, 8990, 9552, 9820,9931,9973,9989,9996]


        # real_avg_vec =[]
        # N = 10000
        # for ED in x:
        #     print("ED:", ED)
        #     filename = filefolder+f"network_N10000ED{ED}Beta{beta}.txt"
        #     G = loadSRGGandaddnode(N, filename)
        #     real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
        #     print("real ED:", real_avg)
        #     real_avg_vec.append(real_avg)
        # print(real_avg_vec)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        text = r"$N = 10^4$" "\n" r"$\beta = 4$"
        plt.text(
            0.1, 0.65,  # 文本位置（轴坐标，0.5 表示图中央，1.05 表示轴上方）
            text,
            transform=ax.transAxes,  # 使用轴坐标
            fontsize=28,  # 字体大小
            ha='left',  # 水平居中对齐
            va='bottom'  # 垂直对齐方式
        )
        plt.xscale('log')
        plt.yscale('log')
        plt.plot(x,real_avg_vec,linewidth=5,color=colors[0])

        popt, pcov = curve_fit(power_law, x[:15], real_avg_vec[:15])
        a, b = popt
        x_fit = np.linspace(min(x), 9999, 500)
        y_fit = power_law(x_fit, *popt)



        if thesis_flag:
            plt.xlabel(r'Expected degree, $E[D]$', fontsize=26)
        else:
            plt.loglog(x_fit, y_fit, 'r-', linewidth=3, label=f'Fit: y = {a:.3f} * x^{b:.3f}')
            plt.legend()
            plt.xlabel(r'Expected degree, $\mathbb{E}[D]$', fontsize=26)
        plt.ylabel(r"Average degree, $\langle D \rangle$", fontsize=26)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)

        # picname = filefolder+ "avgvsEkN{Nn}Beta{betan}2.pdf".format(
        #     Nn=10000, betan=beta)
        # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
        picname = filefolder + "avgvsEkN{Nn}Beta{betan}2.svg".format(
            Nn=10000, betan=beta)
        plt.savefig(
            picname,
            format="svg",
            bbox_inches='tight',  # 紧凑边界
            transparent=True  # 背景透明，适合插图叠加
        )

        plt.show()
        plt.close()
    elif beta==128:
        filefolder = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\inputavgbeta\\"
        x = [2.2, 2.8, 3, 3.4, 3.8, 4.4, 6, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999,
             16479, 21121, 27081, 34822, 44767, 57363]
        real_avg_vec = [1.686, 2.1618, 2.3386, 2.681, 2.9718, 3.4246, 4.6802, 7.7456, 12.339, 20.6666, 33.5638, 54.4372,
                        88.4134, 142.3818, 229.1506, 366.9368, 583.313, 920.0406, 1433.1184, 2194.9722,
                        3295,4793.8644, 6698, 7708, 8670, 9453, 9863, 9980]
        # if loaded network needed
        # real_avg_vec =[]
        # N = 10000
        # x = [2.2, 2.8, 3, 3.4, 3.8, 4.4, 6, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727]
        # for ED in x:
        #     print("ED:", ED)
        #     filename = filefolder+f"network_N10000ED{ED}Beta{beta}.txt"
        #     G = loadSRGGandaddnode(N, filename)
        #     real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
        #     print("real ED:", real_avg)
        #     real_avg_vec.append(real_avg)
        # print(real_avg_vec)

        plt.plot(x, real_avg_vec)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'Expected degree, $E[k]$', fontsize=35)
        plt.ylabel(r"Average degree, $\langle k \rangle$", fontsize=35)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.show()


def plot_inputavg_vs_realavg_100node(beta):
    # Figure 1 (b)
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494', "#A2C7A4", "#9DB0C2", "#E3B6A4"]
    if beta == 4:
        filefolder = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\inputavgbeta\\"
        x = [2, 3, 5, 8, 13, 21, 33, 52, 82, 131, 208, 331, 526, 835, 1999, 3500]
        # x = [2.2, 2.8, 3, 3.4, 3.8, 4.4, 6, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
        real_avg_vec = [1.24, 1.72, 3.34, 4.84, 7.6, 11.96, 16.88, 25.96, 33.66, 46.5, 56.26, 72.14, 82.7, 91.92, 97.32, 98.48]


        # real_avg_vec =[]
        # N = 10000
        # for ED in x:
        #     print("ED:", ED)
        #     filename = filefolder+f"network_N10000ED{ED}Beta{beta}.txt"
        #     G = loadSRGGandaddnode(N, filename)
        #     real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
        #     print("real ED:", real_avg)
        #     real_avg_vec.append(real_avg)
        # print(real_avg_vec)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        text = r"$N = 10^2$" "\n" r"$\beta = 4$"
        plt.text(
            0.1, 0.65,  # 文本位置（轴坐标，0.5 表示图中央，1.05 表示轴上方）
            text,
            transform=ax.transAxes,  # 使用轴坐标
            fontsize=28,  # 字体大小
            ha='left',  # 水平居中对齐
            va='bottom'  # 垂直对齐方式
        )
        plt.xscale('log')
        plt.yscale('log')
        plt.plot(x,real_avg_vec,linewidth=5,color=colors[0])

        popt, pcov = curve_fit(power_law, x[:8], real_avg_vec[:8])
        a, b = popt
        x_fit = np.linspace(min(x), 99, 500)
        y_fit = power_law(x_fit, *popt)

        plt.loglog(x_fit, y_fit, 'r-', linewidth=3, label=f'Fit: y = {a:.3f} * x^{b:.3f}')

        plt.legend()

        plt.xlabel(r'Expected degree, $\mathbb{E}[D]$', fontsize=26)
        plt.ylabel(r"Average degree, $\langle D \rangle$", fontsize=26)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)

        # picname = filefolder+ "avgvsEkN{Nn}Beta{betan}2.pdf".format(
        #     Nn=10000, betan=beta)
        # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
        picname = filefolder + "avgvsEkN{Nn}Beta{betan}2.svg".format(
            Nn=10000, betan=beta)
        plt.savefig(
            picname,
            format="svg",
            bbox_inches='tight',  # 紧凑边界
            transparent=True  # 背景透明，适合插图叠加
        )

        plt.show()
        plt.close()
    elif beta==128:
        filefolder = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\inputavgbeta\\"
        x = [2.2, 2.8, 3, 3.4, 3.8, 4.4, 6, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999,
             16479, 21121, 27081, 34822, 44767, 57363]
        real_avg_vec = [1.686, 2.1618, 2.3386, 2.681, 2.9718, 3.4246, 4.6802, 7.7456, 12.339, 20.6666, 33.5638, 54.4372,
                        88.4134, 142.3818, 229.1506, 366.9368, 583.313, 920.0406, 1433.1184, 2194.9722,
                        3295,4793.8644, 6698, 7708, 8670, 9453, 9863, 9980]
        # if loaded network needed
        # real_avg_vec =[]
        # N = 10000
        # x = [2.2, 2.8, 3, 3.4, 3.8, 4.4, 6, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727]
        # for ED in x:
        #     print("ED:", ED)
        #     filename = filefolder+f"network_N10000ED{ED}Beta{beta}.txt"
        #     G = loadSRGGandaddnode(N, filename)
        #     real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
        #     print("real ED:", real_avg)
        #     real_avg_vec.append(real_avg)
        # print(real_avg_vec)

        plt.plot(x, real_avg_vec)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'Expected degree, $E[k]$', fontsize=35)
        plt.ylabel(r"Average degree, $\langle k \rangle$", fontsize=35)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.show()



def plot_inputavg_vs_realavg_several_beta():
    x = [2.2, 2.8, 3, 3.4, 3.8, 4.4, 6, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999,
         16479, 27081, 44767, 73534, 121205, 199999]
    # x = [2.2, 2.8, 3, 3.4, 3.8, 4.4, 6, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    real_avg_vec = [1.7024, 2.1224, 2.2988, 2.6058, 2.941, 3.3956, 4.6198, 7.6544, 12.1272, 20.414358564143587,
                    32.9682,
                    53.2058, 85.6794, 137.1644, 218.4686, 345.3296, 541.029, 836.6424, 1278.4108, 1902.8332,
                    2783.4186,
                    3911.416, 5253, 6700, 8029, 8990, 9552, 9820]
    plt.plot(x, real_avg_vec)

    x = [2.2, 2.8, 3, 3.4, 3.8, 4.4, 6, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999,
         16479, 21121, 27081, 34822, 44767, 57363]
    real_avg_vec = [1.686, 2.1618, 2.3386, 2.681, 2.9718, 3.4246, 4.6802, 7.7456, 12.339, 20.6666, 33.5638, 54.4372,
                    88.4134, 142.3818, 229.1506, 366.9368, 583.313, 920.0406, 1433.1184, 2194.9722,
                    3295, 4793.8644, 6698, 7708, 8670, 9453, 9863, 9980]

    plt.plot(x, real_avg_vec)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

def power_law(x, a, k):
    return a * x ** k

def plot_dev_vs_read_vag(beta):
    # if beta = 4
    x1 = np.arange(2, 6.1, 0.2)
    x2 = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    x = np.concatenate((x1, x2))

    x1 = np.arange(2, 6.1, 0.2)
    y1 = [0.00770638, 0.01068419, 0.0144987, 0.02114211, 0.03095507, 0.05568157,
          0.08224888, 0.08943058, 0.08294274, 0.07499516, 0.07045126, 0.06704344,
          0.06514699, 0.0639876, 0.06208567, 0.06061299, 0.05922611, 0.05922872,
          0.05914097, 0.06084116, 0.06136418]
    x2 = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    y2 = [0.05891465, 0.05533494, 0.05265912, 0.05150453, 0.05037399, 0.0513435,
          0.05523626, 0.0590772, 0.06754352, 0.07553414, 0.08651017, 0.10125545,
          0.11933796, 0.14024103, 0.1657074]

    y3 = [0.19563515892291106, 0.2267387399917424, 0.2520529070547254, 0.2703523042691398, 0.27450136030794314,
          0.2910258721133285]

    z1 = [0.0066603, 0.00962583, 0.01291115, 0.01930005, 0.02704265, 0.04229857,
          0.06186865, 0.06237527, 0.05695633, 0.04744045, 0.04617877, 0.04206581,
          0.04079666, 0.04185382, 0.04055155, 0.03993016, 0.03869862, 0.03918321,
          0.03763912, 0.04249137, 0.04233556]
    z2 = [0.04165133, 0.04115107, 0.03823124, 0.03346443, 0.03030473, 0.03030013,
          0.03079327, 0.02692964, 0.03021144, 0.03654027, 0.03683304, 0.02206111,
          0.01436064, 0.01017871, 0.00959986]

    z3 = [0.010782733007184256, 0.01650428248848706, 0.02514991673918939, 0.034831895753117105, 0.03724487672187001,
          0.05506431381447785]

    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))
    z = np.concatenate((z1, z2))
    filter_index = [1, 4, 5, 7, 9, 12, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                    30, 31, 32, 33, 34, 35]
    x = [x[a] for a in filter_index]
    y = [y[a] for a in filter_index]
    print(x)
    print(y)
    print(z)
    z = [z[a] for a in filter_index]

    y = np.concatenate((y, y3))
    z = np.concatenate((z, z3))

    real_avg_vec = [1.7024, 2.1224, 2.2988, 2.6058, 2.941, 3.3956, 4.6198, 7.6544, 12.1272, 20.414358564143587, 32.9682,
                    53.2058, 85.6794, 137.1644, 218.4686, 345.3296, 541.029, 836.6424, 1278.4108, 1902.8332, 2783.4186,
                    3911.416, 5253, 6700, 8029, 8990, 9552, 9820]

    params, covariance = curve_fit(power_law, real_avg_vec[18:26], y[18:26])
    # 获取拟合的参数
    a_fit, k_fit = params
    print(f"拟合结果: a = {a_fit}, k = {k_fit}")
    plt.plot(real_avg_vec[18:26], power_law(real_avg_vec[18:26], *params), linewidth=5,
             label=f'fit curve: $y={a_fit:.6f}x^{{{k_fit:.4f}}}$',
             color='red')

    analyticy = [0.011210490144748496, 0.014362834005337652, 0.01836412087082358, 0.022948784225781036,
                 0.02873677429889581,
                 0.03580781814761691, 0.04411338516186235, 0.05362436843859807, 0.06415204818089597,
                 0.07537040591851239,
                 0.08723985015799253, 0.10043013621960226, 0.11689120257351479, 0.13907942793358205, 0.1668336102129492,
                 0.19593857047537333, 0.21951189287819609, 0.23504203529784937, 0.24340647418948685, 0.2473534197139448,
                 0.24899148388677875]

    plt.plot(real_avg_vec[-len(analyticy):],analyticy,"-s", markersize=20,markerfacecolor='none',linewidth=4, color='green',label="common neighbour analytic")

    plt.errorbar(real_avg_vec, y, yerr=z, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=16, label="shortest path")
    plt.legend(fontsize=20, loc="lower right")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('real average degree', fontsize=26)
    plt.ylabel('Average deviation', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.show()


def plot_dev_vs_read_vag_beta128(beta):
    # if beta = 128
    x = [2.2, 2.8, 3.0, 3.4, 3.8, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999,
         16479, 21121, 27081, 34822, 44767, 57363]
    y = [0.0029345139093950932, 0.004109966121375002, 0.00479625179478224, 0.005790742766701017, 0.007213154897222421,
     0.01017775526587543, 0.07434444521427382, 0.025331327772728215, 0.02207886935025605, 0.021513562427822243,
     0.022412692210352126, 0.02418101045773899, 0.027023115723651202, 0.030336763687625844, 0.034310441787031826,
     0.03987705462214779, 0.04579258119548598, 0.055430165269095666, 0.06507214955432139, 0.07922696780660311,
     0.10072021261849186, 0.12505816332021463, 0.17638307090663247, 0.20516804668251162, 0.2336670582705963,
     0.2587096784280022, 0.25809694997043003, 0.24937579369624774]
    z = [0.0019454771051946808, 0.0030404578675205145, 0.0035442118282165676, 0.004487728884584266, 0.00564010291786058,
     0.007407098760784608, 0.05208998096250223, 0.013244967538735938, 0.01023385836909803, 0.009181115746426268,
     0.008622785518813106, 0.009134783434252676, 0.010425527660028931, 0.01128245879196427, 0.013174506559452787,
     0.014974885465433031, 0.01666198119043067, 0.01940936898505464, 0.02182272152609407, 0.027041996379721156,
     0.02879886825511291, 0.03210882381775897, 0.022465399902082343, 0.016854135615137357, 0.010502776256431107,
     0.006185586465971589, 0.004438102695689572, 0.0075005298077803345]
    real_avg_vec = [1.686, 2.1618, 2.3386, 2.681, 2.9718, 3.4246, 4.6802, 7.7456, 12.339, 20.6666, 33.5638, 54.4372,
                    88.4134, 142.3818, 229.1506, 366.9368, 583.313, 920.0406, 1433.1184, 2194.9722,
                    3295, 4793.8644, 6698, 7708, 8670, 9453, 9863, 9980]

    params, covariance = curve_fit(power_law, real_avg_vec[-12:-5], y[-12:-5])
    # 获取拟合的参数
    a_fit, k_fit = params
    print(f"拟合结果: a = {a_fit}, k = {k_fit}")
    plt.plot(real_avg_vec[-12:-3], power_law(real_avg_vec[-12:-3], *params), linewidth=5,
             label=f'fit curve: $y={a_fit:.6f}x^{{{k_fit:.4f}}}$',
             color='red')

    analyticy = [0.006620548972449431, 0.014017954867989511, 0.009299825466112953, 0.020493224097268156, 0.02222066432278047, 0.012972033515545338, 0.022215797545195398, 0.016455455433967002, 0.008386722785826063, 0.010027614592313383, 0.019132173754216254, 0.013185694891261101, 0.0607932308886549, 0.11234193903658657, 0.16692958578813133, 0.21936546885705224, 0.23015080467671672, 0.24289046415473067, 0.24999588130710304, 0.24999999999956696, 0.24999999999999997]

    # analyticx = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999, 16479, 27081, 44767, 73534,
    #              121205, 199999]

    plt.plot(real_avg_vec[-len(analyticy):],analyticy,"-s", markersize=20,markerfacecolor='none',linewidth=4, color='green',label="common neighbour analytic")

    plt.errorbar(real_avg_vec, y, yerr=z, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=16, label="shortest path")
    plt.legend(fontsize=20, loc="lower right")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('real average degree', fontsize=26)
    plt.ylabel('Average deviation', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.show()



def plot_dev_vs_input_avg(beta):
    if beta == 4:
        x1 = np.arange(2, 6.1, 0.2)
        y1 = [0.00770638, 0.01068419, 0.0144987, 0.02114211, 0.03095507, 0.05568157,
              0.08224888, 0.08943058, 0.08294274, 0.07499516, 0.07045126, 0.06704344,
              0.06514699, 0.0639876, 0.06208567, 0.06061299, 0.05922611, 0.05922872,
              0.05914097, 0.06084116, 0.06136418]
        x2 = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
        x3 = [16479, 27081, 44767, 73534, 121205, 199999]
        y2 = [0.05891465, 0.05533494, 0.05265912, 0.05150453, 0.05037399, 0.0513435,
              0.05523626, 0.0590772, 0.06754352, 0.07553414, 0.08651017, 0.10125545,
              0.11933796, 0.14024103, 0.1657074]

        y3 = [0.19563515892291106, 0.2267387399917424, 0.2520529070547254, 0.2703523042691398, 0.27450136030794314,
              0.2910258721133285]

        z1 = [0.0066603, 0.00962583, 0.01291115, 0.01930005, 0.02704265, 0.04229857,
              0.06186865, 0.06237527, 0.05695633, 0.04744045, 0.04617877, 0.04206581,
              0.04079666, 0.04185382, 0.04055155, 0.03993016, 0.03869862, 0.03918321,
              0.03763912, 0.04249137, 0.04233556]
        z2 = [0.04165133, 0.04115107, 0.03823124, 0.03346443, 0.03030473, 0.03030013,
              0.03079327, 0.02692964, 0.03021144, 0.03654027, 0.03683304, 0.02206111,
              0.01436064, 0.01017871, 0.00959986]

        z3 = [0.010782733007184256, 0.01650428248848706, 0.02514991673918939, 0.034831895753117105, 0.03724487672187001,
              0.05506431381447785]

        x = np.concatenate((x1, x2))
        y = np.concatenate((y1, y2))
        z = np.concatenate((z1, z2))
        filter_index = [1, 4, 5, 7, 9, 12, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                        30, 31, 32, 33, 34, 35]
        x = [x[a] for a in filter_index]
        y = [y[a] for a in filter_index]
        print(x)
        print(y)
        print(z)
        z = [z[a] for a in filter_index]
        x = np.concatenate((x, x3))
        y = np.concatenate((y, y3))
        z = np.concatenate((z, z3))

        real_avg_vec = [1.7024, 2.1224, 2.2988, 2.6058, 2.941, 3.3956, 4.6198, 7.6544, 12.1272, 20.414358564143587,
                        32.9682,
                        53.2058, 85.6794, 137.1644, 218.4686, 345.3296, 541.029, 836.6424, 1278.4108, 1902.8332,
                        2783.4186,
                        3911.416, 5253, 6700, 8029, 8990, 9552, 9820]

        params, covariance = curve_fit(power_law, x[18:26], y[18:26])
        # 获取拟合的参数
        a_fit, k_fit = params
        print(f"拟合结果: a = {a_fit}, k = {k_fit}")
        plt.plot(x[18:26], power_law(x[18:26], *params), linewidth=5,
                 label=f'fit curve: $y={a_fit:.6f}x^{{{k_fit:.4f}}}$',
                 color='red')

        analyticy = [0.011210490144748496, 0.014362834005337652, 0.01836412087082358, 0.022948784225781036,
                     0.02873677429889581, 0.03580781814761691, 0.04411338516186235, 0.05362436843859807,
                     0.06415204818089597, 0.07537040591851239, 0.08723985015799253, 0.10043013621960226,
                     0.11689120257351479, 0.13907942793358205, 0.1668336102129492, 0.19593857047537333,
                     0.21951189287819609, 0.23504203529784937, 0.24340647418948685, 0.2473534197139448,
                     0.24899148388677875, 0.24959105981467355, 0.24983551088466646]

        analyticx = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999, 16479, 27081, 44767,
                     73534, 121205, 199999, 316226, 499999]


        plt.plot(analyticx[-len(analyticy):],analyticy,"-s", markersize=20,markerfacecolor='none',linewidth=4, color='green',label="common neighbour analytic")

        plt.errorbar(x, y, yerr=z, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=16, label="shortest path")
        plt.legend(fontsize=20, loc="lower right")
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Input average degree', fontsize=26)
        plt.ylabel('Average deviation', fontsize=26)
        plt.xticks(fontsize=26)
        plt.yticks(fontsize=26)
        plt.show()
    else:
        x = [2.2, 2.8, 3.0, 3.4, 3.8, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105,
             9999,
             16479, 21121, 27081, 34822, 44767, 57363]
        y = [0.0029345139093950932, 0.004109966121375002, 0.00479625179478224, 0.005790742766701017,
             0.007213154897222421,
             0.01017775526587543, 0.07434444521427382, 0.025331327772728215, 0.02207886935025605, 0.021513562427822243,
             0.022412692210352126, 0.02418101045773899, 0.027023115723651202, 0.030336763687625844,
             0.034310441787031826,
             0.03987705462214779, 0.04579258119548598, 0.055430165269095666, 0.06507214955432139, 0.07922696780660311,
             0.10072021261849186, 0.12505816332021463, 0.17638307090663247, 0.20516804668251162, 0.2336670582705963,
             0.2587096784280022, 0.25809694997043003, 0.24937579369624774]
        z = [0.0019454771051946808, 0.0030404578675205145, 0.0035442118282165676, 0.004487728884584266,
             0.00564010291786058,
             0.007407098760784608, 0.05208998096250223, 0.013244967538735938, 0.01023385836909803, 0.009181115746426268,
             0.008622785518813106, 0.009134783434252676, 0.010425527660028931, 0.01128245879196427,
             0.013174506559452787,
             0.014974885465433031, 0.01666198119043067, 0.01940936898505464, 0.02182272152609407, 0.027041996379721156,
             0.02879886825511291, 0.03210882381775897, 0.022465399902082343, 0.016854135615137357, 0.010502776256431107,
             0.006185586465971589, 0.004438102695689572, 0.0075005298077803345]
        real_avg_vec = [1.686, 2.1618, 2.3386, 2.681, 2.9718, 3.4246, 4.6802, 7.7456, 12.339, 20.6666, 33.5638, 54.4372,
                        88.4134, 142.3818, 229.1506, 366.9368, 583.313, 920.0406, 1433.1184, 2194.9722,
                        3295, 4793.8644, 6698, 7708, 8670, 9453, 9863, 9980]

        params, covariance = curve_fit(power_law, x[-12:-5], y[-12:-5])
        # 获取拟合的参数
        a_fit, k_fit = params
        print(f"拟合结果: a = {a_fit}, k = {k_fit}")
        plt.plot(x[-12:-3], power_law(x[-12:-3], *params), linewidth=5,
                 label=f'fit curve: $y={a_fit:.6f}x^{{{k_fit:.4f}}}$',
                 color='red')

        analyticy = [0.006620548972449431, 0.014017954867989511, 0.009299825466112953, 0.020493224097268156,
                     0.02222066432278047, 0.012972033515545338, 0.022215797545195398, 0.016455455433967002,
                     0.008386722785826063, 0.010027614592313383, 0.019132173754216254, 0.013185694891261101,
                     0.0607932308886549, 0.11234193903658657, 0.16692958578813133, 0.21936546885705224,
                     0.24289046415473067, 0.24999999999956696, 0.24999999999999997, 0.24999999999999997,
                     0.24999999999999997, 0.24999999999999997, 0.24999999999999997]

        analyticx = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999, 16479, 27081, 44767, 73534,
                     121205, 199999, 316226, 499999]

        plt.plot(analyticx[-len(analyticy):], analyticy, "-s", markersize=20, markerfacecolor='none', linewidth=4,
                 color='green', label="common neighbour analytic")

        plt.errorbar(x, y, yerr=z, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o',
                     markersize=16, label="shortest path")
        plt.legend(fontsize=20, loc="lower right")
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Input average degree', fontsize=26)
        plt.ylabel('Average deviation', fontsize=26)
        plt.xticks(fontsize=26)
        plt.yticks(fontsize=26)
        plt.show()


def compute_expected_degree(N, E_D, beta):
    """
    计算 SRGG 中的期望度数 <k>

    参数：
        N (int): 节点数量
        E_D (float): 期望度数（用于计算 alpha）
        beta (float): β 参数

    返回：
        float: 计算得到的 <k>
    """
    R = 2  # 给定常数 R=2

    # 计算 alpha
    alpha = (2 * N / E_D * R * R) * (math.pi / (math.sin(2 * math.pi / beta) * beta))
    alpha = math.sqrt(alpha)

    # alpha = sqrt((2 * N / (E_D * R * R)) * (pi / (sin(2 * pi / beta) * beta)))

    # 几何距离的 PDF 函数
    def f_x(x):
        if 0 <= x < 1:
            return 2 * x * (x ** 2 - 4 * x + pi)
        elif 1 <= x <= sqrt(2):
            return 2 * x * (4 * sqrt(x ** 2 - 1) - (x ** 2 + 2 - pi) - 4 * atan(sqrt(x ** 2 - 1)))
        else:
            return 0

    # 连接概率函数 p(x)
    def p_x(x):
        return 1 / (1 + (alpha * x) ** beta)

    # 被积函数 integrand = p(x) * f(x)
    def integrand(x):
        return p_x(x) * f_x(x)

    # 数值积分计算平均度数 <k>
    integral_result, _ = quad(integrand, 0, sqrt(2))
    k_avg = (N - 1) * integral_result

    return k_avg

def test_analytic_result():
    x = [2.2, 2.8, 3, 3.4, 3.8, 4.4, 6, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999,
         16479, 27081, 44767, 73534, 121205, 199999, 331131, 539052, 888611, 1465694]
    a_realavg_vec = []
    for ED in x:
        a_realavg_vec.append(compute_expected_degree(10000, ED, 4))
    real_avg_vec = [1.7024, 2.1224, 2.2988, 2.6058, 2.941, 3.3956, 4.6198, 7.6544, 12.1272, 20.414358564143587,
                    32.9682,
                    53.2058, 85.6794, 137.1644, 218.4686, 345.3296, 541.029, 836.6424, 1278.4108, 1902.8332,
                    2783.4186,
                    3911.416, 5253, 6700, 8029, 8990, 9552, 9820, 9931, 9973, 9989, 9996]

    popt, pcov = curve_fit(power_law, x[:15], real_avg_vec[:15])
    a, b = popt
    x_fit = np.linspace(min(x), 9999, 500)
    y_fit = power_law(x_fit, *popt)
    plt.loglog(x_fit, y_fit, 'r-', linewidth=3, label=f'Fit: y = {a:.3f} * x^{b:.3f}')

    plt.plot(x,real_avg_vec,'o-',label = "avg,simu")
    plt.plot(x,a_realavg_vec,"s--",label = "avg,analytic")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.show()

    print(a_realavg_vec)


if __name__ == '__main__':
    # load_10000nodenetwork_results(4)

    # plot_inputavg_vs_realavg_several_beta()
    # Figure 1!!!!!!!!!!!!!
    plot_inputavg_vs_realavg(4,thesis_flag=True)

    # test_analytic_result()

    # plot_inputavg_vs_realavg_100node(4)

    # plot_dev_vs_read_vag(4)

    # plot_dev_vs_input_avg(4)


    # load_10000nodenetwork_results(128)
    # plot_inputavg_vs_realavg(128)
    # plot_dev_vs_read_vag_beta128(128)

    # plot_dev_vs_input_avg(128)