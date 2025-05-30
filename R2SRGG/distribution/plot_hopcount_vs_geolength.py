# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 2025/03/10
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def plot_hopcount_vs_geolength(N, ED, beta):
    filefoldername = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\hopcountvsdistance\\"
    geodesic_distance_AB_list = [0.1,0.3,0.5]
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    count = 0
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for geodesic_distance_AB in geodesic_distance_AB_list:
        SP_hopcount = []
        for ExternalSimutime in range(50):
            hopcount_Name = filefoldername+"Givendistancehopcount_sp_N{Nn}ED{EDn}beta{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime, Geodistance=geodesic_distance_AB)
            SP_hopcount_foronesimu = np.loadtxt(hopcount_Name)
            SP_hopcount.extend(SP_hopcount_foronesimu)

        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        bins = np.arange(min(SP_hopcount) - 0.5, max(SP_hopcount) + 1.5, 1)  # 间隔为1的bin，确保每个柱中心对齐刻度线
        plt.hist(SP_hopcount, bins=bins,alpha=0.7, color=colors[count], edgecolor=colors[count],density=True, label=r"$d_{ij}:$"+f"{geodesic_distance_AB}")  # 绘制直方图

            # plt.xlim([0, 1])
            # plt.yticks([0, 5, 10, 15, 20, 25])
            # plt.yticks([0, 10, 20, 30, 40, 50])
        count = count + 1
    plt.xticks(np.arange(1, max(SP_hopcount) + 1, 2))
    plt.legend(fontsize=20)
    # plt.xlabel(r'x', fontsize=35)
    # plt.ylabel(r'$f_{h}(x)$', fontsize=35)
    # plt.xticks(fontsize=28)
    # plt.yticks(fontsize=28)
    plt.xlabel(r'x', fontsize=32)
    plt.ylabel(r'$f_{h}(x)$', fontsize=32)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)

    text = r"$N = 10^4$" "\n" r"$E[D] = 10$" "\n" r"$\beta = 4$"
    plt.text(
        0.05, 0.65,  # 文本位置（轴坐标，0.5 表示图中央，1.05 表示轴上方）
        text,
        transform=ax.transAxes,  # 使用轴坐标
        fontsize=20,  # 字体大小
        ha='left',  # 水平居中对齐
        va='bottom'  # 垂直对齐方式
    )


    picname = filefoldername+ "hopvsgeolength{Nn}ED{EDn}Beta{betan}.pdf".format(
        Nn=N, EDn=ED, betan=beta)
    # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    # 清空图像，以免影响下一个图
    plt.close()



def plot_hopcount_vs_geolength(N, ED, beta):
    filefoldername = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\hopcountvsdistance\\"
    geodesic_distance_AB_list = [0.1,0.3,0.5]
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    count = 0
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for geodesic_distance_AB in geodesic_distance_AB_list:
        SP_hopcount = []
        for ExternalSimutime in range(50):
            hopcount_Name = filefoldername+"Givendistancehopcount_sp_N{Nn}ED{EDn}beta{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime, Geodistance=geodesic_distance_AB)
            SP_hopcount_foronesimu = np.loadtxt(hopcount_Name)
            SP_hopcount.extend(SP_hopcount_foronesimu)

        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        bins = np.arange(min(SP_hopcount) - 0.5, max(SP_hopcount) + 1.5, 1)  # 间隔为1的bin，确保每个柱中心对齐刻度线
        plt.hist(SP_hopcount, bins=bins,alpha=0.7, color=colors[count], edgecolor=colors[count],density=True, label=r"$d_{ij}:$"+f"{geodesic_distance_AB}")  # 绘制直方图

            # plt.xlim([0, 1])
            # plt.yticks([0, 5, 10, 15, 20, 25])
            # plt.yticks([0, 10, 20, 30, 40, 50])
        count = count + 1
    plt.xticks(np.arange(1, max(SP_hopcount) + 1, 2))
    plt.legend(fontsize=20)
    # plt.xlabel(r'x', fontsize=35)
    # plt.ylabel(r'$f_{h}(x)$', fontsize=35)
    # plt.xticks(fontsize=28)
    # plt.yticks(fontsize=28)
    plt.xlabel(r'x', fontsize=32)
    plt.ylabel(r'$f_{h}(x)$', fontsize=32)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)

    text = r"$N = 10^4$" "\n" r"$E[D] = 10$" "\n" r"$\beta = 4$"
    plt.text(
        0.05, 0.65,  # 文本位置（轴坐标，0.5 表示图中央，1.05 表示轴上方）
        text,
        transform=ax.transAxes,  # 使用轴坐标
        fontsize=20,  # 字体大小
        ha='left',  # 水平居中对齐
        va='bottom'  # 垂直对齐方式
    )


    picname = filefoldername+ "hopvsgeolength{Nn}ED{EDn}Beta{betan}.pdf".format(
        Nn=N, EDn=ED, betan=beta)
    # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    # 清空图像，以免影响下一个图
    plt.close()


def plot_hopcount_vs_geolength_withdifferentED(N, beta):
    filefoldername = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\GivenGeodistance\\OneSP\\"
    geodesic_distance_AB_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.71, 0.85]
    geodesic_distance_AB_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    ED_vec = [5, 10, 15, 20]
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    count = 0
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for ED in ED_vec:
        ave_SP_hopcount_ED = []
        std_SP_hopcount_ED = []
        for geodesic_distance_AB in geodesic_distance_AB_list:
            SP_hopcount = []
            for ExternalSimutime in range(250):
                try:
                    hopcount_Name = filefoldername+"Givendistancehopcount_sp_N{Nn}ED{EDn}beta{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
                        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime, Geodistance=geodesic_distance_AB)
                    SP_hopcount_foronesimu = np.loadtxt(hopcount_Name)
                    SP_hopcount.extend(SP_hopcount_foronesimu)
                except:
                    pass

            ave_SP_hopcount_ED.append(np.mean(SP_hopcount))
            std_SP_hopcount_ED.append(np.std(SP_hopcount))
        plt.errorbar(geodesic_distance_AB_list, ave_SP_hopcount_ED, yerr=std_SP_hopcount_ED, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o',
                     markersize=16, label=f"ED:{ED}", color=colors[count])
        count = count + 1
    plt.legend(fontsize=20)
    ax.set_xticks([0.1,0.2,0.3,0.4,0.5])
    plt.xlabel(r'$d_{ij}$', fontsize=26)
    plt.ylabel(r'$h_{ij}$', fontsize=26)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    text = r"$N = 10^4$" "\n" r"$\beta = 4$"
    plt.text(
        0.55, 0.75,  # 文本位置（轴坐标，0.5 表示图中央，1.05 表示轴上方）
        text,
        transform=ax.transAxes,  # 使用轴坐标
        fontsize=20,  # 字体大小
        ha='left',  # 水平居中对齐
        va='bottom'  # 垂直对齐方式
    )


    picname = filefoldername+ "hopvsgeolengthdiffED{Nn}Beta{betan}2.pdf".format(
        Nn=N, betan=beta)
    plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    # 清空图像，以免影响下一个图
    plt.close()


if __name__ == '__main__':
    # plot_distribution(50)
    plot_hopcount_vs_geolength_withdifferentED(10000, 4)