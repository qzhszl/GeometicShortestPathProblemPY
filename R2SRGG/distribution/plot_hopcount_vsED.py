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


def plot_hopcount_vs_ED_test(N, beta_vec):
    # load and test data
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\"
    # filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\OneSP\\"  # for ONE SP

    # kvec = [8,10, 13, 17, 22, 28, 36, 46, 58, 74, 94, 120, 155]
    kvec = [2, 3, 5, 8, 10, 13, 17, 22, 28, 36, 46, 58, 74, 94, 120, 155, 266, 457, 787, 1356, 2337, 4028, 6943, 11972, 20647]

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    count = 0
    fig, ax = plt.subplots(figsize=(6, 4.5))
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
                    FileNetworkName = filefolder_name +"network_N{Nn}ED{EDn}Beta{betan}.txt".format(
                        Nn=N, EDn=ED, betan=beta)
                    G = loadSRGGandaddnode(N, FileNetworkName)
                    real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
                    # print("real ED:", real_avg)
                    real_ave_degree_vec.append(real_avg)
                except:
                    print("datalost:",(ED,beta,ExternalSimutime))

            SP_hopcount_ave.append(np.mean(hopcount_for_a_para_comb))
            SP_hopcount_std.append(np.std(hopcount_for_a_para_comb))

            print(real_ave_degree_vec)
            print(SP_hopcount_ave)
            print(SP_hopcount_std)

        plt.errorbar(real_ave_degree_vec,SP_hopcount_ave,SP_hopcount_std,linestyle="-", linewidth=3, elinewidth=1, capsize=5, marker='o', markersize=6, label = rf"N={N},$\beta$ = {beta}")

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
     [1.5554, 2.3392, 3.8926, 6.192, 7.743, 10.0388, 13.0968, 16.9344, 21.4386, 27.531, 35.0834, 44.058,
      55.9208, 70.7122, 89.8856, 115.2898, 194.227, 325.963, 544.0998, 900.1238, 1467.1444, 2343.9226],
     [np.float64(2.281541857700798), np.float64(4.5090832892986645), np.float64(45.1786), np.float64(
         59.942), np.float64(47.1371), np.float64(38.6387), np.float64(32.3199), np.float64(
         27.2802), np.float64(23.6262), np.float64(20.3878), np.float64(17.7595), np.float64(
         15.6246), np.float64(13.6995), np.float64(12.0753), np.float64(10.6417), np.float64(
         9.342), np.float64(7.1553), np.float64(5.5226), np.float64(4.3037), np.float64(3.3825), np.float64(
         2.6889), np.float64(2.156)],
     [np.float64(1.6684760224508528), np.float64(3.77469872533102), np.float64(
         33.783142275993214), np.float64(27.54113715880301), np.float64(21.77658613258745), np.float64(
         17.914160943510584), np.float64(15.070380353196132), np.float64(12.680500304010092), np.float64(
         10.980959591948238), np.float64(9.479483696910924), np.float64(8.22567077811895), np.float64(
         7.216139885007773), np.float64(6.304553889848195), np.float64(5.5304457243517), np.float64(
         4.8535678742549795), np.float64(4.238211415208071), np.float64(3.2008720546126175), np.float64(
         2.4204729372583365), np.float64(1.840452745929653), np.float64(1.4043481583994761), np.float64(
         1.0865158949596643), np.float64(0.8372956467102883)]
     )}
    for beta, val in data_dict.items():
        plt.errorbar(data_dict[beta][0], data_dict[beta][1], data_dict[beta][2], linestyle="-", linewidth=3, elinewidth=1,
                     capsize=5, marker='o', markersize=6, label=rf"N={N},$\beta$ = {beta}")

    popt2, pcov2 = curve_fit(power_law, data_dict[1024][0][6:], data_dict[1024][1][6:])
    a2, alpha2 = popt2
    # 拟合曲线
    N_fit2 = np.linspace(2,10000 , 50)
    y_fit2 = power_law(N_fit2, a2, alpha2)
    plt.plot(N_fit2, y_fit2, '-', linewidth=2,
             label=fr'Fit: $y = {a2:.2f} \cdot N^{{{alpha2:.2f}}}$')



    k_vals  = np.linspace(1.5,10000,10000)
    f1 = np.full_like(k_vals, np.log(N) / np.log(np.log(N)))
    # 曲线 2: 1 + 1/p = 1 + n / k
    f2 = 1 + N / k_vals
    # 曲线 3: log(n) / log(k)
    f3 = np.log(N) / np.log(k_vals)

    # plt.plot(k_vals, f1, label=r'$f_1(k) = \frac{\log n}{\log \log n}$', linestyle='--', color='orange')
    # plt.plot(k_vals, f2, label=r'$f_2(k) = 1 + \frac{n}{k}$', color='green')
    plt.plot(k_vals, f3, label=r'$f(k) = \frac{\log n}{\log k}$', color='blue')

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
    plt.xlabel(r'$<k>$', fontsize=32)
    plt.ylabel(r'$E[h]$', fontsize=32)
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



if __name__ == '__main__':
    # plot_hopcount_vs_ED(10000,4)

    plot_hopcount_vs_ED_test(10000, [1024])