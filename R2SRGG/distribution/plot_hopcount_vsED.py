# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 2025/2/13
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

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


    # plt.xticks(np.arange(0, 50, 10))
    # plt.legend(fontsize=20)
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


if __name__ == '__main__':
    plot_hopcount_vs_ED(10000,4)