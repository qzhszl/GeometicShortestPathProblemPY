# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 2025/2/13
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import curve_fit

from R2SRGG.R2SRGG import loadSRGGandaddnode
def power_law(x, a, b):
    return a * x**b


def plot_hopcount_vs_ED(N, beta):
    # filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\"
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\OneSP\\"  # for ONE SP


    kvec = [2.0,3.0, 4.0,5.0,6.0,8, 10,12, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999] # for all sp
    kvec = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105,
            9999]  # for all sp
    kvec = [11, 16, 27, 44, 73, 107, 120, 193, 316, 518, 848, 1389, 2276, 3727, 6105,
            9999]  # for one sp

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    count = 0
    fig, ax = plt.subplots(figsize=(6, 4.5))
    SP_hopcount_ave = []
    SP_hopcount_std = []
    # for ED in kvec:
    #     hopcount_for_a_para_comb = np.array([])
    #     if ED < 15:
    #
    #         for ExternalSimutime in range(50):
    #             try:
    #                 hopcount_vec_name = filefolder_name + "hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
    #                     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    #                 hopcount_for_a_para_comb_10times = np.loadtxt(hopcount_vec_name)
    #                 hopcount_for_a_para_comb = np.hstack(
    #                     (hopcount_for_a_para_comb, hopcount_for_a_para_comb_10times))
    #             except:
    #                 print("datalost:",(ED,beta,ExternalSimutime))
    #     else:
    #         for ExternalSimutime in range(20):
    #             try:
    #                 hopcount_vec_name = filefolder_name + "hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
    #                     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    #                 hopcount_for_a_para_comb_10times = np.loadtxt(hopcount_vec_name)
    #                 hopcount_for_a_para_comb = np.hstack(
    #                     (hopcount_for_a_para_comb, hopcount_for_a_para_comb_10times))
    #             except:
    #                 print("datalost:",(ED,beta,ExternalSimutime))
    #     SP_hopcount_ave.append(np.mean(hopcount_for_a_para_comb))
    #     SP_hopcount_std.append(np.std(hopcount_for_a_para_comb))

    kvec = [11, 16, 27, 44, 73, 107, 120, 193, 316, 518, 848, 1389]
    kvec = [0.886*i**(0.957) for i in kvec]
    # x = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389]
    # a_realavg_vec = []
    #
    # kvec = [7.6544, 12.1272, 20.414358564143587,
    #                 32.9682,
    #                 53.2058, 85.6794, 137.1644, 218.4686, 345.3296, 541.029]


    SP_hopcount_ave = [np.float64(10.68658), np.float64(8.175736), np.float64(6.041096), np.float64(4.825936), np.float64(3.973012),
     np.float64(3.50587), np.float64(3.381164), np.float64(2.959456), np.float64(2.63692), np.float64(2.38936),
     np.float64(2.140096), np.float64(1.958244)]
    SP_hopcount_std = [np.float64(3.2480486301162426), np.float64(2.3264283479840935), np.float64(1.5923162747343884),
     np.float64(1.2057220757305558), np.float64(0.945885642060392), np.float64(0.8104107249413719),
     np.float64(0.7713455808546515), np.float64(0.657793420508293), np.float64(0.5602186301793256),
     np.float64(0.5534462850177965), np.float64(0.4775406901867107), np.float64(0.3507883071939542)]

    plt.errorbar(kvec,SP_hopcount_ave,SP_hopcount_std,linestyle="-", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=6)

    print(kvec)
    print(SP_hopcount_ave)
    print(SP_hopcount_std)

    k_vals  = np.linspace(min(kvec)-1,max(kvec)+100,50)
    f1 = np.full_like(k_vals, np.log(N) / np.log(np.log(N)))
    # 曲线 2: 1 + 1/p = 1 + n / k
    f2 = 1 + N / k_vals
    # 曲线 3: log(n) / log(k)
    f3 = np.log(N) / np.log(k_vals)

    plt.plot(k_vals, f1, label=r'$f_1(k) = \frac{\log n}{\log \log n}$', linestyle='--', color='orange')
    plt.plot(k_vals, f2, label=r'$f_2(k) = 1 + \frac{n}{k}$', color='green')
    plt.plot(k_vals, f3, label=r'$f_3(k) = \frac{\log n}{\log k}$', color='blue')

    # plt.xticks(np.arange(0, 50, 10))
    plt.legend(fontsize=20)
    # plt.xlabel(r'x', fontsize=35)
    # plt.ylabel(r'$f_{h}(x)$', fontsize=35)
    # plt.xticks(fontsize=28)
    # plt.yticks(fontsize=28)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$E[D]$', fontsize=32)
    plt.ylabel(r'$E[h]$', fontsize=32)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)

    text = r"$N = 10^4$" "\n" r"$\beta = 4$"
    plt.text(
        0.25, 0.65,  # 文本位置（轴坐标，0.5 表示图中央，1.05 表示轴上方）
        text,
        transform=ax.transAxes,  # 使用轴坐标
        fontsize=20,  # 字体大小
        ha='left',  # 水平居中对齐
        va='bottom'  # 垂直对齐方式
    )
    plt.title("average hopcount vs expected degree")

    picname =  "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\"+ "hopvsEDN{Nn}beta{betan}.pdf".format(
        Nn=N, betan=beta)
    # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    # 清空图像，以免影响下一个图
    plt.close()


def plot_hopcount_vs_realED(N, beta_vec):
    # Figure 4(c)
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    # colors = ['#ffb2b7', '#f17886', '#e04750', '#b82d36', '#7a1017']
    count = 0
    fig, ax = plt.subplots(figsize=(8, 6))
    betalabel =["$2.5$","$3$","$2^2$","$2^3$","$2^7$"]
    data_dict = {}
    count = 0
    for beta in beta_vec:
        if beta in [4,8,128]:
            kvec = [2.2, 3.0, 3.8, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276,
                    3727, 6105,
                    9999, 16479, 27081, 44767, 73534, 121205, 199999]
        elif beta ==3:
            kvec = [1.2, 1.5,1.8,2,2.2, 3, 3.8, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276,
                    3727, 6105,
                    9999, 16479, 27081, 44767, 73534, 121205, 199999]
        elif beta ==2.5:
            kvec = [1.2, 1.5,2,3, 3.8, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276,
                    3727, 6105,
                    9999, 16479, 27081, 44767, 73534, 121205, 199999]

        SP_hopcount_ave = []
        SP_hopcount_std = []
        real_ave_degree_vec = []
        for ED in kvec:
            print(ED)
            hopcount_for_a_para_comb = np.array([])

            for ExternalSimutime in range(1):
                try:
                    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\hopandedgelength\\"
                    hopcount_vec_name = filefolder_name + "hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
                        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                    hopcount_for_a_para_comb_10times = np.loadtxt(hopcount_vec_name)
                    hopcount_for_a_para_comb = np.hstack(
                        (hopcount_for_a_para_comb, hopcount_for_a_para_comb_10times))
                    # FileNetworkName = filefolder_name +"network_N{Nn}ED{EDn}Beta{betan}.txt".format(
                    #     Nn=N, EDn=ED, betan=beta)
                    real_avg_name = filefolder_name + "real_avg_N{Nn}ED{EDn}beta{betan}Simu{ST}.txt".format(
                        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                    real_avg = np.loadtxt(real_avg_name)
                    real_ave_degree_vec.append(real_avg)
                except:
                    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\approxLrealdiff\\test\\"
                    hopcount_vec_name = filefolder_name + "hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
                        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                    hopcount_for_a_para_comb_10times = np.loadtxt(hopcount_vec_name)
                    hopcount_for_a_para_comb = np.hstack(
                        (hopcount_for_a_para_comb, hopcount_for_a_para_comb_10times))
                    # FileNetworkName = filefolder_name +"network_N{Nn}ED{EDn}Beta{betan}.txt".format(
                    #     Nn=N, EDn=ED, betan=beta)
                    real_avg_name = filefolder_name + "real_ave_degree_N{Nn}ED{EDn}beta{betan}Simu{ST}.txt".format(
                        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                    real_avg = np.loadtxt(real_avg_name)
                    real_ave_degree_vec.append(real_avg)
                    # print("datalost:",(ED,beta,ExternalSimutime))

            SP_hopcount_ave.append(np.mean(hopcount_for_a_para_comb))
            SP_hopcount_std.append(np.std(hopcount_for_a_para_comb))

            # print(list(real_ave_degree_vec))
            # print(SP_hopcount_ave)
            # print(SP_hopcount_std)
            data_dict[beta] =(real_ave_degree_vec,SP_hopcount_ave,SP_hopcount_std)
        # if beta == 2.02:
        #     plt.errorbar(real_ave_degree_vec[4:], SP_hopcount_ave[4:], SP_hopcount_std[4:], linestyle="-", linewidth=3,
        #                  elinewidth=1, capsize=5, marker='o', markersize=16, color=colors[count],
        #                  label=rf"$N=10^4$, $\beta$ = {betalabel[count]}")
        # else:
        #     plt.errorbar(real_ave_degree_vec, SP_hopcount_ave, SP_hopcount_std, linestyle="-", linewidth=3,
        #                  elinewidth=1,
        #                  capsize=5, marker='o', markersize=16, color=colors[count],
        #                  label=rf"$N=10^4$, $\beta$ = {betalabel[count]}")

        if beta == 2.02:
            plt.errorbar(real_ave_degree_vec[4:], SP_hopcount_ave[4:], linestyle="-", linewidth=3,
                        marker='o', markersize=16, color=colors[count],
                         label=rf"$N=10^4$, $\beta$ = {betalabel[count]}")
        else:
            plt.errorbar(real_ave_degree_vec, SP_hopcount_ave, linestyle="--", linewidth=1,
                         marker='o', markersize=16, color=colors[count],
                         label=rf"$\beta$ = {betalabel[count]}")


        count = count+1

    popt2, pcov2 = curve_fit(power_law, data_dict[128][0][10:], data_dict[128][1][10:])
    a2, alpha2 = popt2
    # 拟合曲线
    N_fit2 = np.linspace(1,10000 , 50)
    y_fit2 = power_law(N_fit2, a2, alpha2)
    # plt.plot(N_fit2, y_fit2, '-', linewidth=5, color="#E6B565",
    #          label=fr'$f(k) = {a2:.1f} k^{{{alpha2:.1f}}}$')



    k_vals  = np.linspace(1.1,10000,40000)
    f1 = np.full_like(k_vals, np.log(N) / np.log(np.log(N)))
    # 曲线 2: 1 + 1/p = 1 + n / k
    f2 = 1 + N / k_vals
    # 曲线 3: log(n) / log(k)
    f3 = np.log(N) / np.log(k_vals)

    # plt.plot(k_vals, f1, label=r'$f_1(k) = \frac{\log n}{\log \log n}$', linestyle='--', color='orange')
    # plt.plot(k_vals, f2, label=r'$f_2(k) = 1 + \frac{n}{k}$', color='green')
    # plt.plot(k_vals, f3,"-",linewidth =5 , label=r'$f(\langle D\rangle) = \frac{\log N}{\log \langle D\rangle}$', color="#5CBF9B")

    # 曲线 4: <h> ~ (<k> - <k>_c)^(-a)
    beta=128
    kc = 4.512
    k_vals2 = np.linspace(4.6, 10000, 20000)
    f4 = 100 * (k_vals2 - kc) ** (-0.5)
    plt.plot(k_vals2, f4, linewidth =2,label=r'$\langle h\rangle = 10^2(\langle D\rangle - 4.5)^{-0.5}$', color=colors[beta_vec.index(beta)])


    # plt.xticks(np.arange(0, 50, 10))
    plt.ylim([0.8,400])
    plt.xlim([0.3, 50000])
    plt.legend(fontsize=26, bbox_to_anchor=(0.3, 0.62),markerscale = 1, handlelength = 1,labelspacing = 0.2, handletextpad = 0.3, borderpad = 0.1, borderaxespad=0.1)


    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\langle D \rangle$', fontsize=36)
    plt.ylabel(r'$\langle h \rangle$', fontsize=36)
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
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\hopandedgelength\\"
    picname =  filefolder_name+ "hopvsEDN{Nn}.svg".format(
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


def smallbeta_fit():
    # 数据
    x = np.array([1.3898, 1.6662, 2.0778, 2.6166, 2.9616, 3.9998, 6.5578, 10.1302, 16.6914,
                  26.197, 41.4038, 65.1388, 101.3464, 156.6434, 240.8744, 365.5722, 548.1062, 811.3048, 1180.0008,
                  1683.7022, 2344.7662, 3191.8418, 4192.7072, 5316.3456, 6447.6906, 7495.6608, 8357.897])

    y = np.array([32.23448137765318, 23.064658192373138,
                  15.864545818327331, 12.220566169850954, 10.937575030012004, 8.52836985890123, 6.277238619309655,
                  5.08852836613089, 4.203284598437813, 3.6616880513231758, 3.244424352019289, 2.908862034239678,
                  2.6813353566009104, 2.4677780036592805, 2.2378561180569787, 2.066625220184437, 2.0064434350903135,
                  2.0003259098316133, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

    # 构造函数 f(x) = log(10000)/log(x)
    f = np.log(10000) / np.log(x)

    # 线性最小二乘求解 y = a*f + b
    A = np.vstack([f, np.ones_like(f)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]

    print("a =", a)
    print("b =", b)

    # 生成拟合曲线
    y_fit = a * f + b

    # 作图
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, label='data')
    plt.plot(x, y_fit, label='fit: y = a*log(10000)/log(x) + b')
    plt.xscale('log')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_hopcount_vs_realED_finalversion(N, beta_vec):
    # Figure 4(c)
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    # colors = ['#ffb2b7', '#f17886', '#e04750', '#b82d36', '#7a1017']
    count = 0
    fig, ax = plt.subplots(figsize=(10, 6))
    betalabel =["$2.5$","$3$","$2^2$","$2^3$","$2^7$"]
    betalabel = ["$2.5$", "$3$", "$4$", "$8$", "$128$"]
    data_dict = {}
    count = 0


    for beta in beta_vec:
        if beta in [128]:
            kvec = [2.2, 3.0, 3.8, 4.4,5.1, 5.5,6.0,7.0,8.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276,
                    3727, 6105,
                    9999, 16479, 27081, 44767, 73534, 121205, 199999]
        if beta in [8]:
            kvec = [2.2, 3.0, 3.8, 4.4,5,8, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276,
                    3727, 6105,
                    9999, 16479, 27081, 44767, 73534, 121205, 199999]
        elif beta ==3:
            kvec = [1.2, 1.5,1.8,2,2.4,2.8, 3.8, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276,
                    3727, 6105,
                    9999, 16479, 27081, 44767, 73534, 121205, 199999]
        elif beta ==2.5:
            kvec = [1.2, 1.5,1.6,2,2.4,3, 3.8, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276,
                    3727, 6105,
                    9999, 16479, 27081, 44767, 73534, 121205, 199999]
        elif beta == 4:
            kvec = [1.6,2.2,2.6,2.8, 3.2, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276,
                    3727, 6105,
                    9999, 16479, 27081, 44767, 73534, 121205, 199999]

        SP_hopcount_ave = []
        SP_hopcount_std = []
        real_ave_degree_vec = []
        for ED in kvec:
            # print(ED)
            hopcount_for_a_para_comb = np.array([])

            for ExternalSimutime in range(1):
                try:
                    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\hopandedgelength\\"
                    # filefolder_name = "E:\\GSPP_data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\hopandedgelength\\"
                    hopcount_vec_name = filefolder_name + "hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
                        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                    hopcount_for_a_para_comb_10times = np.loadtxt(hopcount_vec_name)
                    hopcount_for_a_para_comb = np.hstack(
                        (hopcount_for_a_para_comb, hopcount_for_a_para_comb_10times))
                    # FileNetworkName = filefolder_name +"network_N{Nn}ED{EDn}Beta{betan}.txt".format(
                    #     Nn=N, EDn=ED, betan=beta)
                    real_avg_name = filefolder_name + "real_avg_N{Nn}ED{EDn}beta{betan}Simu{ST}.txt".format(
                        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                    real_avg = np.loadtxt(real_avg_name)
                    real_ave_degree_vec.append(real_avg)
                except:
                    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\approxLrealdiff\\test\\"
                    # filefolder_name = "E:\\GSPP_data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\approxLrealdiff\\test\\"
                    hopcount_vec_name = filefolder_name + "hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
                        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                    hopcount_for_a_para_comb_10times = np.loadtxt(hopcount_vec_name)
                    hopcount_for_a_para_comb = np.hstack(
                        (hopcount_for_a_para_comb, hopcount_for_a_para_comb_10times))
                    # FileNetworkName = filefolder_name +"network_N{Nn}ED{EDn}Beta{betan}.txt".format(
                    #     Nn=N, EDn=ED, betan=beta)
                    real_avg_name = filefolder_name + "real_ave_degree_N{Nn}ED{EDn}beta{betan}Simu{ST}.txt".format(
                        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                    real_avg = np.loadtxt(real_avg_name)
                    real_ave_degree_vec.append(real_avg)
                    # print("datalost:",(ED,beta,ExternalSimutime))

            hopcount_for_a_para_comb = hopcount_for_a_para_comb[hopcount_for_a_para_comb!=1]

            SP_hopcount_ave.append(np.mean(hopcount_for_a_para_comb))
            SP_hopcount_std.append(np.std(hopcount_for_a_para_comb))

            # print(list(real_ave_degree_vec))
            # print(SP_hopcount_ave)
            # print(SP_hopcount_std)
            # data_dict[beta] =(real_ave_degree_vec,SP_hopcount_ave,SP_hopcount_std)

        if beta ==128 or beta==2.5:
            print(real_ave_degree_vec)
            print(SP_hopcount_ave)
            # x_128 = [x-4.512 for x in real_ave_degree_vec]
            # plt.errorbar(x_128, SP_hopcount_ave, linestyle="--", linewidth=1,
            #              marker='o', markersize=16, color=colors[count],
            #              label=rf"$\beta$ = {betalabel[count]}")

        if beta == 2.02:
            plt.errorbar(real_ave_degree_vec[4:], SP_hopcount_ave[4:], linestyle="-", linewidth=3,
                        marker='o', markersize=16, color=colors[count],
                         label=rf"$N=10^4$, $\beta$ = {betalabel[count]}")
        else:
            plt.errorbar(real_ave_degree_vec, SP_hopcount_ave, linestyle="--", linewidth=1,
                         marker='o', markersize=16, color=colors[count],
                         label=rf"$\beta$ = {betalabel[count]}")


        count = count+1

    k_vals = np.linspace(1.1, 15000, 40000)
    # 曲线 3: A+b*log(n) / log(k)
    # # f3 = 1.3 * np.log(N) / np.log(k_vals)
    a = 1.17
    b = 0.4
    f3 = a * np.log(N) / np.log(k_vals) + b

    beta = 2.5
    # plt.plot(k_vals, f1, label=r'$f_1(k) = \frac{\log n}{\log \log n}$', linestyle='--', color='orange')
    # plt.plot(k_vals, f2, label=r'$f_2(k) = 1 + \frac{n}{k}$', color='green')
    plt.plot(k_vals, f3, "-", linewidth=2, label=rf'${a}\ln{{N}}/ \ln{{\langle D\rangle}}$+{b}',
             color=colors[beta_vec.index(beta)],zorder=200)

    # a = 1.1
    # f3 = a * np.log(N) / np.log(k_vals)
    #
    # beta = 2.5
    # # plt.plot(k_vals, f1, label=r'$f_1(k) = \frac{\log n}{\log \log n}$', linestyle='--', color='orange')
    # # plt.plot(k_vals, f2, label=r'$f_2(k) = 1 + \frac{n}{k}$', color='green')
    # plt.plot(k_vals, f3, "-", linewidth=2, label=rf'${a}\ln{{N}}/ \ln{{\langle D\rangle}}$',
    #          color=colors[beta_vec.index(beta)], zorder=200)


    # 曲线 4: <h> ~ (<k> - <k>_c)^(-a)
    # beta = 128
    # kc = 4.512
    # k_vals2 = np.linspace(4.6, 15000, 20000)
    # a = 92
    # b = 0.6
    # f4 = a * (k_vals2 - kc) ** (-0.5)+b
    # plt.plot(k_vals2, f4, linewidth=2, label=rf'${a}(\langle D\rangle - 4.5)^{{-0.5}}+{b}$',
    #          color=colors[beta_vec.index(beta)],zorder=200)

    beta = 128
    kc = 4.512
    k_vals2 = np.linspace(4.6, 15000, 20000)
    a = 92
    b = 0.6
    f4 = a * (k_vals2 - kc) ** (-0.5) + b
    plt.plot(k_vals2, f4, linewidth=2, label=rf'${a}(\langle D\rangle - 4.5)^{{-0.5}}+{b}$',
             color=colors[beta_vec.index(beta)], zorder=200)




    handles, labels = ax.get_legend_handles_labels()
    indices_group1 = [0, 1]
    handles1 = [handles[i] for i in indices_group1]
    labels1 = [labels[i] for i in indices_group1]
    handles2 = [h for i, h in enumerate(handles) if i not in indices_group1]
    labels2 = [l for i, l in enumerate(labels) if i not in indices_group1]

    legend1 = ax.legend(handles2, labels2,
                        loc=(0.71, 0.26),  # (x,y) 以 axes 坐标为基准
                        fontsize=26,  # 根据期刊要求调小
                        markerscale=1,
                        handlelength=1.5,
                        labelspacing=0.2,
                        ncol=1,
                        handletextpad=0.3,
                        borderpad=0.1,
                        borderaxespad=0.1
                        )

    # 必须把第一个 legend 加回 axes，否则下一个 legend 会把它覆盖
    ax.add_artist(legend1)

    # 第二块 legend：放右下角（图内）
    legend2 = ax.legend(handles1, labels1,
                        loc=(0.23, 0.78),  # (x,y) 以 axes 坐标为基准
                        fontsize=26,  # 根据期刊要求调小
                        markerscale=4,
                        handlelength=1,
                        labelspacing=0.2,
                        ncol=1,
                        handletextpad=0.3,
                        borderpad=0.1,
                        borderaxespad=0.1
                        )

    for handle in legend1.legend_handles:
        handle.set_linewidth(2)
    for handle in legend2.legend_handles:
        handle.set_linewidth(4)

    # plt.xticks(np.arange(0, 50, 10))
    plt.ylim([0.8,500])
    # plt.xlim([0.3, 50000])
    # plt.legend(fontsize=26, bbox_to_anchor=(0.3, 0.62),markerscale = 1, handlelength = 1,labelspacing = 0.2, handletextpad = 0.3, borderpad = 0.1, borderaxespad=0.1)


    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\langle D \rangle$', fontsize=36)
    plt.ylabel(r'$\langle h \rangle$', fontsize=36)
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
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\hopandedgelength\\"
    picname =  filefolder_name+ "hopvsEDN{Nn}.svg".format(
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




def plot_hopcount_vs_realED_vsPiet(N, beta_vec):

    x = [
    0.8494, 1.0428, 1.1242, 1.3898, 1.6662, 2.0778, 2.6166,
    2.9616, 3.9998, 6.5578, 10.1302, 16.6914, 26.197, 41.4038,
    65.1388, 101.3464, 156.6434, 240.8744, 365.5722, 548.1062,
    811.3048, 1180.0008, 1683.7022, 2344.7662, 3191.8418,
    4192.7072, 5316.3456, 6447.6906, 7495.6608, 8357.897]
    y = [np.float64(3.7979611650485436), np.float64(8.582271406215067), np.float64(18.543789892225593),
     np.float64(32.23448137765318), np.float64(23.064658192373138), np.float64(15.864545818327331),
     np.float64(12.220566169850954), np.float64(10.937575030012004), np.float64(8.52836985890123),
     np.float64(6.277238619309655), np.float64(5.08852836613089), np.float64(4.203284598437813),
     np.float64(3.6616880513231758), np.float64(3.244424352019289), np.float64(2.908862034239678),
     np.float64(2.6813353566009104), np.float64(2.4677780036592805), np.float64(2.2378561180569787),
     np.float64(2.066625220184437), np.float64(2.0064434350903135), np.float64(2.0003259098316133), np.float64(2.0),
     np.float64(2.0), np.float64(2.0), np.float64(2.0), np.float64(2.0), np.float64(2.0), np.float64(2.0),
     np.float64(2.0), np.float64(2.0)]

    plt.plot(x,y)

    k_vals = np.linspace(1.1, 15000, 40000)
    # 曲线 3: A+b*log(n) / log(k)
    # # f3 = 1.3 * np.log(N) / np.log(k_vals)
    a = 1
    b = 0.5
    f3 = a * np.log(N) / np.log(k_vals) + b
    f4 = np.log(N) / np.log(k_vals) + 0.5

    E_logW = []

    for mu in k_vals:
        val = estimate_E_logW(mu, num_sim=5000, max_gen=25)
        E_logW.append(val)

    plt.figure()
    plt.plot(k_vals, E_logW, marker='o')
    plt.xlabel("Mean degree μ")
    plt.ylabel(r"$E[\log W \mid W>0]$")
    plt.title("Curious mean vs μ (Poisson Galton–Watson)")
    plt.show()

    E_logW = np.array(E_logW)

    f4  = np.log(N) / np.log(k_vals) +0.5 - ((0.5772156649-2*np.log(1-1/k_vals))/np.log(k_vals))- 2*E_logW / np.log(k_vals)

    beta = 2.5
    # plt.plot(k_vals, f1, label=r'$f_1(k) = \frac{\log n}{\log \log n}$', linestyle='--', color='orange')
    # plt.plot(k_vals, f2, label=r'$f_2(k) = 1 + \frac{n}{k}$', color='green')
    # plt.plot(k_vals, f3, "-", linewidth=2, label=rf'${a}\ln{{N}}/ \ln{{\langle D\rangle}}$+{b}',
    #          zorder=200)
    plt.plot(k_vals, f4, "-", linewidth=2, label=rf'${a}\ln{{N}}/ \ln{{\langle D\rangle}}$+{b}',
             zorder=200)

    # a = 1.1
    # f3 = a * np.log(N) / np.log(k_vals)
    #
    # beta = 2.5
    # # plt.plot(k_vals, f1, label=r'$f_1(k) = \frac{\log n}{\log \log n}$', linestyle='--', color='orange')
    # # plt.plot(k_vals, f2, label=r'$f_2(k) = 1 + \frac{n}{k}$', color='green')
    # plt.plot(k_vals, f3, "-", linewidth=2, label=rf'${a}\ln{{N}}/ \ln{{\langle D\rangle}}$',
    #          color=colors[beta_vec.index(beta)], zorder=200)


    # 曲线 4: <h> ~ (<k> - <k>_c)^(-a)
    # beta = 128
    # kc = 4.512
    # k_vals2 = np.linspace(4.6, 15000, 20000)
    # a = 92
    # b = 0.6
    # f4 = a * (k_vals2 - kc) ** (-0.5)+b
    # plt.plot(k_vals2, f4, linewidth=2, label=rf'${a}(\langle D\rangle - 4.5)^{{-0.5}}+{b}$',
    #          color=colors[beta_vec.index(beta)],zorder=200)







    # plt.xticks(np.arange(0, 50, 10))
    plt.ylim([0.8,500])
    # plt.xlim([0.3, 50000])
    # plt.legend(fontsize=26, bbox_to_anchor=(0.3, 0.62),markerscale = 1, handlelength = 1,labelspacing = 0.2, handletextpad = 0.3, borderpad = 0.1, borderaxespad=0.1)


    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\langle D \rangle$', fontsize=36)
    plt.ylabel(r'$\langle h \rangle$', fontsize=36)
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

    # )
    plt.show()
    # 清空图像，以免影响下一个图
    plt.close()

def estimate_E_logW(mu, num_sim=5000, max_gen=25):
    """
    Estimate E[log W | W>0] for Poisson(mu) Galton–Watson process
    """
    logs = []

    for _ in range(num_sim):
        Z = 1  # Z_0 = 1
        for _ in range(max_gen):
            if Z == 0:
                break
            Z = np.random.poisson(mu, Z).sum()

        if Z > 0:
            W_hat = Z / (mu ** max_gen)
            if W_hat > 0:
                logs.append(np.log(W_hat))

    return np.mean(logs)


def plot_hopcount_vs_ED_test(N, beta_vec):
    # Figure 4(c)
    # load and test data
    # filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\" # for ONE SP, load data for beta = 2.05 and 1024
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\hopandedgelength\\"

    # kvec = [8,10, 13, 17, 22, 28, 36, 46, 58, 74, 94, 120, 155]
    # kvec = [2, 3, 5, 8, 10, 13, 17, 22, 28, 36, 46, 58, 74, 94, 120, 155, 266, 457, 787, 1356, 2337, 4028, 6943, 11972, 20647]# for ONE SP, load data for beta = 2.05 and 1024
    # kvec = [2.2, 3.0, 3.8, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276,
    #             3727, 6105,
    #             9999, 16479, 27081, 44767, 73534, 121205, 199999]
    kvec = [2.2, 3.0, 3.8, 4.4, 6.0, 10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276,
            3727, 6105,
            9999]
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    count = 0
    fig, ax = plt.subplots(figsize=(8, 8))
    betalabel =["$2.1$","$2^2$","$2^3$","$2^6$"]
    data_dict = {}
    count = 0
    for beta in beta_vec:
        SP_hopcount_ave = []
        SP_hopcount_std = []
        real_ave_degree_vec = []
        for ED in kvec:
            print(ED)
            hopcount_for_a_para_comb = np.array([])

            for ExternalSimutime in range(1):
                try:
                    hopcount_vec_name = filefolder_name + "hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
                        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                    hopcount_for_a_para_comb_10times = np.loadtxt(hopcount_vec_name)
                    hopcount_for_a_para_comb = np.hstack(
                        (hopcount_for_a_para_comb, hopcount_for_a_para_comb_10times))
                    # FileNetworkName = filefolder_name +"network_N{Nn}ED{EDn}Beta{betan}.txt".format(
                    #     Nn=N, EDn=ED, betan=beta)
                    real_avg_name = filefolder_name + "real_avg_N{Nn}ED{EDn}beta{betan}Simu{ST}.txt".format(
                        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                    real_avg = np.loadtxt(real_avg_name)
                    real_ave_degree_vec.append(real_avg)
                except:
                    print("datalost:",(ED,beta,ExternalSimutime))

            SP_hopcount_ave.append(np.mean(hopcount_for_a_para_comb))
            SP_hopcount_std.append(np.std(hopcount_for_a_para_comb))

            print(list(real_ave_degree_vec))
            print(SP_hopcount_ave)
            print(SP_hopcount_std)
            data_dict[beta] =(real_ave_degree_vec,SP_hopcount_ave,SP_hopcount_std)
        if beta == 2.02:
            plt.errorbar(kvec, SP_hopcount_ave, SP_hopcount_std, linestyle="-", linewidth=3,
                         elinewidth=1, capsize=5, marker='o', markersize=16, color=colors[count],
                         label=rf"$N=10^4$, $\beta$ = {betalabel[count]}")
        else:
            plt.errorbar(kvec, SP_hopcount_ave, SP_hopcount_std, linestyle="-", linewidth=3,
                         elinewidth=1,
                         capsize=5, marker='o', markersize=16, color=colors[count],
                         label=rf"$N=10^4$, $\beta$ = {betalabel[count]}")

        count = count+1
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

    popt2, pcov2 = curve_fit(power_law, data_dict[128][0][10:], data_dict[128][1][10:])
    a2, alpha2 = popt2
    # 拟合曲线
    N_fit2 = np.linspace(1,10000 , 50)
    y_fit2 = power_law(N_fit2, a2, alpha2)
    plt.plot(N_fit2, y_fit2, '-', linewidth=5, color="#E6B565",
             label=fr'$f(k) = {a2:.1f} k^{{{alpha2:.1f}}}$')



    k_vals  = np.linspace(1.1,10000,40000)
    f1 = np.full_like(k_vals, np.log(N) / np.log(np.log(N)))
    # 曲线 2: 1 + 1/p = 1 + n / k
    f2 = 1 + N / k_vals
    # 曲线 3: log(n) / log(k)
    f3 = np.log(N) / np.log(k_vals)

    # plt.plot(k_vals, f1, label=r'$f_1(k) = \frac{\log n}{\log \log n}$', linestyle='--', color='orange')
    # plt.plot(k_vals, f2, label=r'$f_2(k) = 1 + \frac{n}{k}$', color='green')
    plt.plot(k_vals, f3,"-",linewidth =5 , label=r'$f(k) = \frac{\log n}{\log k}$', color="#5CBF9B")

    # kvec = [11, 16, 27, 44, 73, 107, 120, 193, 316, 518, 848, 1389]
    # kvec = [0.886 * i ** (0.957) for i in kvec]
    # x = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389]
    # a_realavg_vec = []
    #
    # kvec = [7.6544, 12.1272, 20.414358564143587,
    #                 32.9682,
    #                 53.2058, 85.6794, 137.1644, 218.4686, 345.3296, 541.029]

    # SP_hopcount_ave = [np.float64(10.68658), np.float64(8.175736), np.float64(6.041096), np.float64(4.825936),
    #                    np.float64(3.973012),
    #                    np.float64(3.50587), np.float64(3.381164), np.float64(2.959456), np.float64(2.63692),
    #                    np.float64(2.38936),
    #                    np.float64(2.140096), np.float64(1.958244)]
    # SP_hopcount_std = [np.float64(3.2480486301162426), np.float64(2.3264283479840935), np.float64(1.5923162747343884),
    #                    np.float64(1.2057220757305558), np.float64(0.945885642060392), np.float64(0.8104107249413719),
    #                    np.float64(0.7713455808546515), np.float64(0.657793420508293), np.float64(0.5602186301793256),
    #                    np.float64(0.5534462850177965), np.float64(0.4775406901867107), np.float64(0.3507883071939542)]
    #
    # plt.errorbar(kvec, SP_hopcount_ave, SP_hopcount_std, linestyle="-", linewidth=3, elinewidth=1, capsize=5,
    #              marker='o', markersize=6, label = rf"N=10000,$\beta$ = 4")


    # plt.xticks(np.arange(0, 50, 10))
    plt.ylim([0.8,400])
    plt.xlim([0.3, 50000])
    plt.legend(fontsize=26, bbox_to_anchor=(0.435, 0.47),markerscale = 1, handlelength = 1,labelspacing = 0.2, handletextpad = 0.3, borderpad = 0.1, borderaxespad=0.1)

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
    plt.ylabel(r'$\langle h \rangle$', fontsize=36)
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

    picname =  filefolder_name+ "hopvsrealEDN{Nn}.svg".format(
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


def plot_hopcount_vs_ED_test2(N, beta_vec):
    # load and test data
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\"
    # filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\OneSP\\"  # for ONE SP

    # kvec = [8,10, 13, 17, 22, 28, 36, 46, 58, 74, 94, 120, 155]
    kvec = [2, 3, 5, 8, 10, 13, 17, 22, 28, 36, 46, 58, 74, 94, 120, 155, 266, 457, 787, 1356, 2337, 4028, 6943, 11972, 20647]

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    count = 0
    fig, ax = plt.subplots(figsize=(12, 8))
    # for beta in beta_vec:
    #     SP_hopcount_ave = []
    #     SP_hopcount_std = []
    #     real_ave_degree_vec = []
    #     for ED in kvec:
    #         print(ED)
    #         hopcount_for_a_para_comb = np.array([])
    #
    #         for ExternalSimutime in range(1):
    #             try:
    #                 hopcount_vec_name = filefolder_name + "hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
    #                     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
    #                 hopcount_for_a_para_comb_10times = np.loadtxt(hopcount_vec_name)
    #                 hopcount_for_a_para_comb = np.hstack(
    #                     (hopcount_for_a_para_comb, hopcount_for_a_para_comb_10times))
    #                 FileNetworkName = filefolder_name +"network_N{Nn}ED{EDn}Beta{betan}.txt".format(
    #                     Nn=N, EDn=ED, betan=beta)
    #                 G = loadSRGGandaddnode(N, FileNetworkName)
    #                 real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
    #                 # print("real ED:", real_avg)
    #                 real_ave_degree_vec.append(real_avg)
    #             except:
    #                 print("datalost:",(ED,beta,ExternalSimutime))
    #
    #         SP_hopcount_ave.append(np.mean(hopcount_for_a_para_comb))
    #         SP_hopcount_std.append(np.std(hopcount_for_a_para_comb))
    #
    #         print(real_ave_degree_vec)
    #         print(SP_hopcount_ave)
    #         print(SP_hopcount_std)
    #
    #     plt.errorbar(real_ave_degree_vec,SP_hopcount_ave,SP_hopcount_std,linestyle="-", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=6, label = rf"N={N},$\beta$ = {beta}")

    data_dict = {2.05:([0.3842, 0.5602, 0.9056, 1.4132, 1.7166, 2.1752, 2.767, 3.5134, 4.3432, 5.4588, 6.794, 8.3146, 10.328, 12.7314,
     15.7576, 19.724, 31.4082, 49.8322, 78.8138, 123.7288, 192.536, 296.478, 450.5532, 675.6858, 996.0806],
    [np.float64(1.4681295715778475), np.float64(1.978980557015239), np.float64(7.089406304591558), np.float64(22.6721),
     np.float64(16.0256), np.float64(12.0946), np.float64(9.6047), np.float64(7.9761), np.float64(6.9364),
     np.float64(6.0909), np.float64(5.451), np.float64(4.9801), np.float64(4.5597), np.float64(4.2282),
     np.float64(3.9341), np.float64(3.6837), np.float64(3.2397), np.float64(2.8884), np.float64(2.6729),
     np.float64(2.4423), np.float64(2.1933), np.float64(2.0248), np.float64(1.9605), np.float64(1.9333),
     np.float64(1.9043)],
    [np.float64(0.7959392445512715), np.float64(1.306804595024895), np.float64(5.25729519424271),
     np.float64(6.641986268429046), np.float64(4.176307536568637), np.float64(2.773671004282952),
     np.float64(2.0463718894668195), np.float64(1.5954086592469028), np.float64(1.3506128386773169),
     np.float64(1.1381727417224505), np.float64(0.9963929947565869), np.float64(0.8981670167624727),
     np.float64(0.8066200530609192), np.float64(0.7474789361580699), np.float64(0.690765654907654),
     np.float64(0.6409791806291371), np.float64(0.5948478040641992), np.float64(0.4776457264542414),
     np.float64(0.48980158227592524), np.float64(0.5208365482567443), np.float64(0.4408345608048444),
     np.float64(0.28563081066299556), np.float64(0.2184942791013074), np.float64(0.2503020375466409),
     np.float64(0.29417938404993643)]),
    3:([1.5328, 2.2936, 3.786, 6.0364, 7.5058, 9.6968, 12.5228, 16.047, 20.2318, 25.7052, 32.5532, 40.562, 51.059, 63.9842,
     80.3998, 101.9986, 168.1194, 274.0374, 441.9626, 702.2758, 1095.6854, 1668.7986, 2467.4028, 3515.5474, 4785.3006],
    [np.float64(18.0887), np.float64(21.8116), np.float64(11.6371), np.float64(8.1725), np.float64(7.1046),
     np.float64(6.1869), np.float64(5.4659), np.float64(4.9263), np.float64(4.4903), np.float64(4.1183),
     np.float64(3.8086), np.float64(3.5668), np.float64(3.3326), np.float64(3.1366), np.float64(2.9525),
     np.float64(2.7963), np.float64(2.5403), np.float64(2.2717), np.float64(2.0473), np.float64(1.9389),
     np.float64(1.8974), np.float64(1.8384), np.float64(1.7551), np.float64(1.6471), np.float64(1.526)],
    [np.float64(12.115908232980308), np.float64(6.3460621995060835), np.float64(2.980738765809577),
     np.float64(1.9640630717978484), np.float64(1.6347045115249423), np.float64(1.3767964228599665),
     np.float64(1.2038426765985661), np.float64(1.0621997505177638), np.float64(0.952840967843008),
     np.float64(0.853290753494962), np.float64(0.7763800873283652), np.float64(0.7195399641437575),
     np.float64(0.6839424244773825), np.float64(0.6463284304438417), np.float64(0.5940065235332017),
     np.float64(0.5442483899838382), np.float64(0.5342058685563085), np.float64(0.5026719705732556),
     np.float64(0.366691573396499), np.float64(0.2697532020199204), np.float64(0.30507907171748117),
     np.float64(0.368083468794783), np.float64(0.4300278944440697), np.float64(0.47787193891250823),
     np.float64(0.4993235424051224)]),
     1024: (
         [1.5554, 2.3392, 3.8926, 6.192, 7.743, 10.0388, 13.0968, 16.9344, 21.4386, 27.531, 35.0834, 44.058, 55.9208,
          70.7122, 89.8856, 115.2898, 194.227, 325.963, 544.0998, 900.1238, 1467.1444, 2343.9226, 3645.9348, 5444.3032,
          7617.0924],
         [np.float64(2.281541857700798), np.float64(4.5090832892986645), np.float64(45.1786), np.float64(
             59.942), np.float64(47.1371), np.float64(38.6387), np.float64(32.3199), np.float64(27.2802), np.float64(
             23.6262), np.float64(20.3878), np.float64(17.7595), np.float64(15.6246), np.float64(13.6995), np.float64(
             12.0753), np.float64(10.6417), np.float64(9.342), np.float64(7.1553), np.float64(5.5226), np.float64(
             4.3037), np.float64(3.3825), np.float64(2.6889), np.float64(2.156), np.float64(1.7654), np.float64(
             1.4631), np.float64(1.2414)],
         [np.float64(1.6684760224508528), np.float64(3.77469872533102), np.float64(33.783142275993214), np.float64(
             27.54113715880301), np.float64(21.77658613258745), np.float64(17.914160943510584), np.float64(
             15.070380353196132), np.float64(12.680500304010092), np.float64(10.980959591948238), np.float64(
             9.479483696910924), np.float64(8.22567077811895), np.float64(7.216139885007773), np.float64(
             6.304553889848195), np.float64(5.5304457243517), np.float64(4.8535678742549795), np.float64(
             4.238211415208071), np.float64(3.2008720546126175), np.float64(2.4204729372583365), np.float64(
             1.840452745929653), np.float64(1.4043481583994761), np.float64(1.0865158949596643), np.float64(
             0.8372956467102883), np.float64(0.6655545357068796), np.float64(0.5171444575744769), np.float64(
             0.42793228436284175)]
     )}
    for beta, val in data_dict.items():
        # plt.errorbar(data_dict[beta][0], data_dict[beta][1], data_dict[beta][2], linestyle="-", linewidth=3, elinewidth=1,
        #              capsize=5, marker='o', markersize=6, label=rf"N={N},$\beta$ = {beta}")
        x_vals = data_dict[beta][0]
        # y_vals = [y * x ** 0.5 for x, y in zip(data_dict[beta][0], data_dict[beta][1])]
        y_vals = data_dict[beta][1]
        # y_vals = [np.log(y) for y in data_dict[beta][1]]
        plt.plot(x_vals, y_vals, linestyle="-", linewidth=3,
                 marker='o', markersize=6, label=rf"N={N}, $\beta$ = {beta}")

        # plt.plot(data_dict[beta][0], data_dict[beta][1], linestyle="-", linewidth=3,
        #             marker='o', markersize=6, label=rf"N={N},$\beta$ = {beta}")

    # 拟合曲线
    popt2, pcov2 = curve_fit(power_law, x_vals[10:20], y_vals[10:20])
    a2, alpha2 = popt2
    N_fit2 = np.linspace(2,10000 , 50)
    y_fit2 = power_law(N_fit2, a2, alpha2)
    plt.plot(N_fit2, y_fit2, '-', linewidth=2,
             label=fr'Fit: $y = {a2:.2f} \cdot k^{{{alpha2:.2f}}}$')



    k_vals  = np.linspace(1.5,10000,10000)
    f1 = np.full_like(k_vals, np.log(N) / np.log(np.log(N)))
    # 曲线 2: 1 + 1/p = 1 + n / k
    f2 = 1 + N / k_vals
    # 曲线 3: log(n) / log(k)
    f3 = np.log(N) / np.log(k_vals)

    # <h> ~ (<k> - <k>_c)^(-a)
    kc = 4.512
    k_vals2 = np.linspace(5,10000,10000)
    f4 = 100*(k_vals - kc)**(-0.5)
    plt.plot(k_vals2, f4, label=r'$\langle h\rangle = (\langle k\rangle - %.3f)^{-1/2}$' % kc)


    # plt.plot(k_vals, f1, label=r'$f_1(k) = \frac{\log n}{\log \log n}$', linestyle='--', color='orange')
    # plt.plot(k_vals, f2, label=r'$f_2(k) = 1 + \frac{n}{k}$', color='green')
    # plt.plot(k_vals, f3, label=r'$f(k) = \frac{\log n}{\log k}$')

    # plt.plot(k_vals, f3, label=r'$f(k) = \frac{\log n}{\log k}k^{1/2}$')

    # plt.plot(k_vals, [100 for i in range(len(k_vals))], label=r'$f(k) = 100$')

    # kvec = [11, 16, 27, 44, 73, 107, 120, 193, 316, 518, 848, 1389]
    # kvec = [0.886 * i ** (0.957) for i in kvec]
    # x = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389]
    # a_realavg_vec = []
    #
    # kvec = [7.6544, 12.1272, 20.414358564143587,
    #                 32.9682,
    #                 53.2058, 85.6794, 137.1644, 218.4686, 345.3296, 541.029]

    # SP_hopcount_ave = [np.float64(10.68658), np.float64(8.175736), np.float64(6.041096), np.float64(4.825936),
    #                    np.float64(3.973012),
    #                    np.float64(3.50587), np.float64(3.381164), np.float64(2.959456), np.float64(2.63692),
    #                    np.float64(2.38936),
    #                    np.float64(2.140096), np.float64(1.958244)]
    # SP_hopcount_std = [np.float64(3.2480486301162426), np.float64(2.3264283479840935), np.float64(1.5923162747343884),
    #                    np.float64(1.2057220757305558), np.float64(0.945885642060392), np.float64(0.8104107249413719),
    #                    np.float64(0.7713455808546515), np.float64(0.657793420508293), np.float64(0.5602186301793256),
    #                    np.float64(0.5534462850177965), np.float64(0.4775406901867107), np.float64(0.3507883071939542)]
    #
    # plt.errorbar(kvec, SP_hopcount_ave, SP_hopcount_std, linestyle="-", linewidth=3, elinewidth=1, capsize=5,
    #              marker='o', markersize=6, label = rf"N=10000,$\beta$ = 4")


    # plt.xticks(np.arange(0, 50, 10))
    plt.legend(fontsize=20)
    # plt.xlabel(r'x', fontsize=35)
    # plt.ylabel(r'$f_{h}(x)$', fontsize=35)
    # plt.xticks(fontsize=28)
    # plt.yticks(fontsize=28)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\langle k \rangle$', fontsize=32)
    # plt.ylabel(r'$\langle h \rangle \langle k \rangle^{1/2}$', fontsize=32)
    plt.ylabel(r'$log(h)$', fontsize=32)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)

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

    picname =  "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\"+ "hopvsEDN{Nn}beta{betan}.pdf".format(
        Nn=N, betan=beta)
    # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    # 清空图像，以免影响下一个图
    plt.close()


def text_hop_for_beta128():
    import numpy as np
    import matplotlib.pyplot as plt

    # 您的 X 坐标数据
    # 注意：我们将 'array(...)' 解释为 np.array(...)
    x = [np.array(4.6558), np.array(5.4164), np.array(6.1978),
         np.array(7.6744), np.array(12.3358), np.array(20.6526), np.array(33.5496), np.array(54.21), np.array(88.3386),
         np.array(142.404),
         np.array(229.1266), np.array(367.1896), np.array(583.4436), np.array(920.6452), np.array(1434.103),
         np.array(2196.6802),
         np.array(3298.0032), np.array(4793.6972), np.array(6698.1078), np.array(8672.7838), np.array(9862.7426)]

    # 您的 Y 坐标数据
    y = [ 99.29961908580593, 72.29231692677071, 59.71461461461462, 47.11138353765324, 33.33345012262876, 23.928671608896014,
         18.19666700130509,
         13.885699924604172, 10.727736010898633, 8.405927051671732, 6.649434406510723, 5.332175625939557,
         4.3110202781611635, 3.545569689960901, 2.9602219626168225, 2.5497091144149966, 2.243979057591623,
         2.0474892954456987, 2.0003012501882815, 2.0, 2.0]


    # 将 X 列表中的单元素数组展平为普通的浮点数列表
    x_flat = [a.item() for a in x]

    # 创建图形和子图
    plt.figure(figsize=(10, 6))

    # 使用 plt.loglog() 进行 log-log 坐标绘图
    # 'o-' 表示用圆圈标记数据点并用线连接

    x2 = [x-4.512 for x in x_flat]

    x3 = [(x - 4.512)**(-0.5) for x in x_flat]

    # plt.plot(x_flat, y, 'o-', label='Data Points')


    # plt.plot(x2, y, 'o-', label='Data Points2')
    N = 10000
    y2 = [j * (1 - k / (N - 1) + k / (N - 1)) for (k, j) in zip(x_flat, y)]
    # plt.plot(x_flat,y2)

    plt.plot(x3, y, 'o-', label='Data Points2')
    plt.plot(x3, y2, 'o-', label='Data Points2')

    #
    #
    # coefficients = np.polyfit(x3[5:], y[5:], 1)
    # a_fit = coefficients[0]
    # B_fit = coefficients[1]
    # Z_fit = np.linspace(min(x3), 2, 100)
    # y_fit = a_fit * Z_fit + B_fit
    # plt.plot(Z_fit, y_fit, 'r-', label=f'Linear Fit: $y={a_fit:.2f}Z + {B_fit:.2f}$')

    # 添加标签和标题
    plt.xlabel(r'$1/\sqrt{\langle D \rangle-\langle D \rangle_c}$')
    plt.ylabel('hopcount')

    # plt.xscale('log')
    # plt.yscale('log')
    # 显示网格线，which="both" 表示主次刻度都显示
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    # 显示图例
    plt.legend()

    # 保存图像（可选，如果需要在环境中保存）
    # plt.savefig('loglog_plot.png')

    # 显示图形
    plt.show()  # 如果在交互式环境（如 Jupyter 或 IDE）中运行，通常需要这一行



def test2():
    import numpy as np
    import matplotlib.pyplot as plt

    # --- 1. 原始数据 ---
    # 您的 X 坐标数据
    # 注意：我们使用 np.array(...) 替换了 'array(...)'
    x = [np.array(1.7172), np.array(2.3352), np.array(2.9842), np.array(3.4444), np.array(3.943), np.array(4.2554),
         np.array(4.6558),
         np.array(7.6744), np.array(12.3358), np.array(20.6526), np.array(33.5496), np.array(54.21), np.array(88.3386),
         np.array(142.404),
         np.array(229.1266), np.array(367.1896), np.array(583.4436), np.array(920.6452), np.array(1434.103),
         np.array(2196.6802),
         np.array(3298.0032), np.array(4793.6972), np.array(6698.1078), np.array(8672.7838), np.array(9862.7426)]

    # 您的 Y 坐标数据
    y = [3.4640264558799805, 5.047299336149668, 7.654561147874187, 13.309341419299926, 19.878204468238444,
         50.98821396192203, 99.29961908580593, 47.11138353765324, 33.33345012262876, 23.928671608896014,
         18.19666700130509,
         13.885699924604172, 10.727736010898633, 8.405927051671732, 6.649434406510723, 5.332175625939557,
         4.3110202781611635, 3.545569689960901, 2.9602219626168225, 2.5497091144149966, 2.243979057591623,
         2.0474892954456987, 2.0003012501882815, 2.0, 2.0]

    # 将 X 列表中的单元素数组展平为普通的 NumPy 数组
    x_flat = np.array([a.item() for a in x])
    y_array = np.array(y)

    # --- 2. 变量转换 ---
    # 设定常数
    C = 4.512

    # 检查分母 x_flat - C 是否为正，以确保开方和幂运算有效
    if np.any(x_flat <= C):
        # 找到所有 x <= C 的索引
        invalid_indices = np.where(x_flat <= C)[0]

        # 找出第一个导致问题的 X 值
        first_invalid_x = x_flat[invalid_indices[0]]

        # 由于您的数据中前几项 x < 4.512，我们将只使用 x > 4.512 的数据点进行绘图
        print(f"警告：常数 C = {C} 大于或等于数据中的前几个 X 值。")
        print(f"第一个无效 X 值是 {first_invalid_x}。")
        print("为了避免数学错误（负数开方），我们将仅使用 x > C 的数据点进行转换和绘图。")

        # 过滤数据：只保留 x > C 的点
        valid_indices = x_flat > C
        x_valid = x_flat[valid_indices]
        y_valid = y_array[valid_indices]
    else:
        x_valid = x_flat
        y_valid = y_array

    # 计算新的横轴变量 Z = (x - C)^(-1/2)
    # Z = 1 / sqrt(x - C)
    # 这里使用 ** (-0.5) 来表示 (-1/2) 幂
    Z = (x_valid - C) ** (-0.5)

    # --- 3. 绘图：y vs Z (线性坐标) ---
    plt.figure(figsize=(10, 6))

    # 使用 plt.plot() 在**线性坐标系**下绘图
    plt.plot(Z, y_valid, 'o', label=r'Data: $y$ vs $(x-4.512)^{-1/2}$')
    print(Z)
    print(y_valid)
    # 添加标签和标题
    plt.xlabel(r'Transformed Variable $Z = (x-4.512)^{-1/2}$')  # 使用 LaTeX 语法显示更美观的公式
    plt.ylabel(r'Original Variable $y$')
    plt.title('Plot of $y$ vs Transformed $Z$ (Linear Coordinates)')

    # 显示网格线
    plt.grid(True, linestyle='--', alpha=0.7)

    # 显示图例
    plt.legend()

    # 显示图形
    plt.show()

    # --- 4. 可选：线性拟合（Liner Regression）---
    print("\n--- 线性拟合结果 ---")
    # 使用 numpy.polyfit 进行一阶多项式拟合 (即线性拟合)
    # coefficients[0] 是斜率 a，coefficients[1] 是截距 B
    coefficients = np.polyfit(Z, y_valid, 1)
    a_fit = coefficients[0]
    B_fit = coefficients[1]

    # 打印拟合结果
    print(f"拟合直线方程: y = {a_fit:.4f} * Z + {B_fit:.4f}")
    print(f"拟合斜率 a ≈ {a_fit:.4f}")
    print(f"拟合截距 B ≈ {B_fit:.4f}")

    # 重新绘图并加上拟合直线
    plt.figure(figsize=(10, 6))
    plt.plot(Z, y_valid, 'o', label='Original Data Points')

    # 绘制拟合直线
    Z_fit = np.linspace(Z.min(), Z.max(), 100)
    y_fit = a_fit * Z_fit + B_fit
    plt.plot(Z_fit, y_fit, 'r-', label=f'Linear Fit: $y={a_fit:.2f}Z + {B_fit:.2f}$')

    plt.xlabel(r'Transformed Variable $Z = (x-4.512)^{-1/2}$')
    plt.ylabel(r'Original Variable $y$')
    plt.title('Plot with Linear Regression')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # plot_hopcount_vs_ED(10000,4)

    # plot_hopcount_vs_ED_test(10000,[2.02,4,8,128])

    """"
    Figure 4 (c)
    """
    # plot_hopcount_vs_realED(10000,[2.5,3,4,8,128])

    # plot_hopcount_vs_realED_finalversion(10000,[2.5,3,4,8,128])

    # text_hop_for_beta128()
    # test2()


    # plot_hopcount_vs_ED_test2(10000, [1024])

    # smallbeta_fit()

    plot_hopcount_vs_realED_vsPiet(10000, [2.5])



