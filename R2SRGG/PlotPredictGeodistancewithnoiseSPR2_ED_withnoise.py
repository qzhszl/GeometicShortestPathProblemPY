# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 6-6-2024
Generate the graph, remove links, blur node coordinates:  x = x + E(A), y = y + E(A),
where E ~ Unif(-A,A), A is noise amplitude. Or do it in a more “kosher” way, uniformly place it within a 2D circle of radius A.

For the node pair ij:
	a) find shortest path nodes using distance to geodesic (with blurred node coordinates).
	b) find shortest path nodes by reconstructing the graph.

Use the same parameter combinations as before.
Vary noise magnitude A, see what happens to predictions.
It is for Euclidean soft random geometric graph

Differnt from PlotPredictGeodistancewithnoiseSPR2(ED and beta), this .py focus on the relation between ED and noise.
"""


import math

import numpy as np
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from matplotlib import cm, colors


def plot_heatmap_precision(betaindex):
    """
    ED vs noise
    :param betaindex:
    :return:
    """
    N = 10000
    ED_list = [2, 4, 8, 16, 32, 64, 128]  # Expected degrees
    betalist = [2.1, 4, 8, 32, 128]
    beta = betalist[betaindex]
    print("beta:", beta)

    noise_amplitude_list = [0, 0.001, 0.01, 0.1, 1]

    exemptionlist =[]
    RGG_matrix = np.zeros((len(ED_list), len(noise_amplitude_list)))
    SRGG_matrix = np.zeros((len(ED_list), len(noise_amplitude_list)))
    Geo_matrix = np.zeros((len(ED_list), len(noise_amplitude_list)))

    for EDindex in range(len(ED_list)):
        ED = ED_list[EDindex]
        print("ED:", ED)
        for noiseindex in range(len(noise_amplitude_list)):
            noise_amplitude = noise_amplitude_list[noiseindex]
            print(noise_amplitude)
            precision_list = []
            PrecisonRGG_specificnoise = []
            for ExternalSimutime in range(20):
                try:
                    precision_RGG_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\PrecisionRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
                        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
                    Precison_RGG_5_times = np.loadtxt(precision_RGG_Name)
                    PrecisonRGG_specificnoise.extend(Precison_RGG_5_times)
                except FileNotFoundError:
                    exemptionlist.append((ED, beta, noise_amplitude, ExternalSimutime))
            # nonzero_indices_geo = find_nonzero_indices(PrecisonRGG_specificnoise)
            # PrecisonRGG_specificnoise = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PrecisonRGG_specificnoise))
            # PrecisonRGG_specificnoise = [PrecisonRGG_specificnoise[x] for x in nonzero_indices_geo]
            RGG_matrix[EDindex][noiseindex] = np.mean(PrecisonRGG_specificnoise)
            print(np.mean(PrecisonRGG_specificnoise))


            PrecisonSRGG_specificnoise = []
            for ExternalSimutime in range(20):
                try:
                    precision_SRGG_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\PrecisionSRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
                        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
                    Precison_SRGG_5_times = np.loadtxt(precision_SRGG_Name)
                    PrecisonSRGG_specificnoise.extend(Precison_SRGG_5_times)
                except FileNotFoundError:
                    pass
            # nonzero_indices_geo = find_nonzero_indices(PrecisonRGG_specificnoise)
            # PrecisonRGG_specificnoise = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PrecisonRGG_specificnoise))
            # PrecisonRGG_specificnoise = [PrecisonRGG_specificnoise[x] for x in nonzero_indices_geo]
            SRGG_matrix[EDindex][noiseindex] = np.mean(PrecisonSRGG_specificnoise)

            PrecisonGeodis_specificnoise = []
            for ExternalSimutime in range(20):
                try:
                    precision_Geodis_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\PrecisionGeodisED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
                        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
                    Precison_Geodis_5_times = np.loadtxt(precision_Geodis_Name)
                    PrecisonGeodis_specificnoise.extend(Precison_Geodis_5_times)
                except FileNotFoundError:
                    pass

            Geo_matrix[EDindex][noiseindex] = np.mean(PrecisonGeodis_specificnoise)

    print(exemptionlist)
    y_labels = ["2", "4", "8","16", "32","64", "128"]  # 横坐标
    x_labels = ["0", "0.001", "0.01", "0.1", "1"]  # 纵坐标
    plt.figure()
    df = pd.DataFrame(RGG_matrix,
                      index=[ED_list],  # DataFrame的行标签设置为大写字母
                      columns=noise_amplitude_list)  # 设置DataFrame的列标签
    h1 = sns.heatmap(data=df, vmin=0, vmax=0.5, annot=True, fmt=".2f", cbar=True, annot_kws={"size": 20},
                cbar_kws={'label': 'Precision'}, xticklabels=x_labels,  # 指定自定义 x 轴标签
    yticklabels=y_labels)
    plt.xticks(fontsize=20)  # x 轴刻度字体大小
    plt.yticks(fontsize=20)  # y 轴刻度字体大小

    cbar = h1.collections[0].colorbar  # 获取颜色条对象
    cbar.ax.tick_params(labelsize=20)  # 设置颜色条刻度字体大小
    cbar.ax.set_ylabel("Precision", fontsize=20)  # 设置颜色条标签字体大小


    plt.ylabel(r"Expected degree $E[D]$",fontsize = 25)
    plt.xlabel(r"Noise amplitude $\alpha$", fontsize = 25)
    RGG_heatmap_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\HeatmapPrecisionRGGbeta{beta}.pdf".format(beta = beta)
    plt.savefig(RGG_heatmap_name,
        format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()

    plt.figure()
    df = pd.DataFrame(SRGG_matrix,
                      index=[ED_list],  # DataFrame的行标签设置为大写字母
                      columns=noise_amplitude_list)  # 设置DataFrame的列标签
    h2 = sns.heatmap(data=df, vmin=0, vmax=0.5, annot=True, fmt=".2f", cbar=True, annot_kws={"size": 20},
                     cbar_kws={'label': 'Precision'}, xticklabels=x_labels,  # 指定自定义 x 轴标签
                     yticklabels=y_labels)
    # plt.title("50% links are removed when computing Nearly Shortest Path Node")
    plt.xticks(fontsize=20)  # x 轴刻度字体大小
    plt.yticks(fontsize=20)  # y 轴刻度字体大小

    cbar = h2.collections[0].colorbar  # 获取颜色条对象
    cbar.ax.tick_params(labelsize=20)  # 设置颜色条刻度字体大小
    cbar.ax.set_ylabel("Precision", fontsize=20)  # 设置颜色条标签字体大小
    # plt.title("50% links are removed when computing Nearly Shortest Path Node")
    plt.ylabel(r"Expected degree $E[D]$", fontsize=25)
    plt.xlabel(r"Noise amplitude $\alpha$", fontsize=25)
    SRGG_heatmap_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\HeatmapPrecisionSRGGbeta{beta}.pdf".format(
        beta=beta)
    plt.savefig(SRGG_heatmap_name,
                format='pdf', bbox_inches='tight', dpi=600)
    plt.show()

    plt.close()

    plt.figure()
    df = pd.DataFrame(Geo_matrix,
                      index=[ED_list],  # DataFrame的行标签设置为大写字母
                      columns=noise_amplitude_list)  # 设置DataFrame的列标签
    h3 = sns.heatmap(data=df, vmin=0, vmax=0.5, annot=True, fmt=".2f", cbar=True, annot_kws={"size": 20},
                     cbar_kws={'label': 'Precision'}, xticklabels=x_labels,  # 指定自定义 x 轴标签
                     yticklabels=y_labels)
    # plt.title("50% links are removed when computing Nearly Shortest Path Node")
    plt.xticks(fontsize=20)  # x 轴刻度字体大小
    plt.yticks(fontsize=20)  # y 轴刻度字体大小

    cbar = h3.collections[0].colorbar  # 获取颜色条对象
    cbar.ax.tick_params(labelsize=20)  # 设置颜色条刻度字体大小
    cbar.ax.set_ylabel("Precision", fontsize=20)  # 设置颜色条标签字体大小
    # plt.title("50% links are removed when computing Nearly Shortest Path Node")
    plt.ylabel(r"Expected degree $E[D]$", fontsize=25)
    plt.xlabel(r"Noise amplitude $\alpha$", fontsize=25)
    Geo_heatmap_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\HeatmapPrecisionGeobeta{beta}.pdf".format(
        beta=beta)
    plt.savefig(Geo_heatmap_name,
                format='pdf', bbox_inches='tight', dpi=600)
    plt.show()


def plot_heatmap_precision_smooth(betaindex):
    """
    ED vs noise
    :param betaindex:
    :return:
    """
    N = 10000
    ED_list = [2, 4, 8, 16, 32, 64, 128]  # Expected degrees
    betalist = [2.1, 4, 8, 32, 128]
    beta = betalist[betaindex]
    print("beta:", beta)

    noise_amplitude_list = [0, 0.001, 0.01, 0.1, 1]

    exemptionlist =[]
    RGG_matrix = np.zeros((len(ED_list), len(noise_amplitude_list)))
    SRGG_matrix = np.zeros((len(ED_list), len(noise_amplitude_list)))
    Geo_matrix = np.zeros((len(ED_list), len(noise_amplitude_list)))

    for EDindex in range(len(ED_list)):
        ED = ED_list[EDindex]
        print("ED:", ED)
        for noiseindex in range(len(noise_amplitude_list)):
            noise_amplitude = noise_amplitude_list[noiseindex]
            print(noise_amplitude)
            precision_list = []
            PrecisonRGG_specificnoise = []
            for ExternalSimutime in range(20):
                try:
                    precision_RGG_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\PrecisionRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
                        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
                    Precison_RGG_5_times = np.loadtxt(precision_RGG_Name)
                    PrecisonRGG_specificnoise.extend(Precison_RGG_5_times)
                except FileNotFoundError:
                    exemptionlist.append((ED, beta, noise_amplitude, ExternalSimutime))
            # nonzero_indices_geo = find_nonzero_indices(PrecisonRGG_specificnoise)
            # PrecisonRGG_specificnoise = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PrecisonRGG_specificnoise))
            # PrecisonRGG_specificnoise = [PrecisonRGG_specificnoise[x] for x in nonzero_indices_geo]
            RGG_matrix[EDindex][noiseindex] = np.mean(PrecisonRGG_specificnoise)
            print(np.mean(PrecisonRGG_specificnoise))


            PrecisonSRGG_specificnoise = []
            for ExternalSimutime in range(20):
                try:
                    precision_SRGG_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\PrecisionSRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
                        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
                    Precison_SRGG_5_times = np.loadtxt(precision_SRGG_Name)
                    PrecisonSRGG_specificnoise.extend(Precison_SRGG_5_times)
                except FileNotFoundError:
                    pass
            # nonzero_indices_geo = find_nonzero_indices(PrecisonRGG_specificnoise)
            # PrecisonRGG_specificnoise = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PrecisonRGG_specificnoise))
            # PrecisonRGG_specificnoise = [PrecisonRGG_specificnoise[x] for x in nonzero_indices_geo]
            SRGG_matrix[EDindex][noiseindex] = np.mean(PrecisonSRGG_specificnoise)

            PrecisonGeodis_specificnoise = []
            for ExternalSimutime in range(20):
                try:
                    precision_Geodis_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\PrecisionGeodisED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
                        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
                    Precison_Geodis_5_times = np.loadtxt(precision_Geodis_Name)
                    PrecisonGeodis_specificnoise.extend(Precison_Geodis_5_times)
                except FileNotFoundError:
                    pass

            Geo_matrix[EDindex][noiseindex] = np.mean(PrecisonGeodis_specificnoise)

    print(exemptionlist)
    y_labels = [r"$2^1$", r"$2^2$", r"$2^3$",r"$2^{4}$", r"$2^{5}$",r"$2^{6}$", r"$2^{7}$"]  # 横坐标
    x_labels = [r"0", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"]  # 纵坐标

    plt.figure()
    df = pd.DataFrame(RGG_matrix,
                      index=[ED_list],  # DataFrame的行标签设置为大写字母
                      columns=noise_amplitude_list)  # 设置DataFrame的列标签
    im = plt.imshow(df,
               cmap='jet',
               interpolation='bicubic',
               origin='lower',
               extent=[1, 5, 1, 7],
               aspect='auto',
               vmin=0,
               vmax=0.5)
    cbar = plt.colorbar(im, label="Precision")
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.set_ylabel("Precision", fontsize=20)
    plt.ylabel(r"Expected degree $E[D]$",fontsize = 25)
    plt.xlabel(r"Noise amplitude $\alpha$", fontsize = 25)
    plt.xticks(ticks=np.arange(len(x_labels))+ 1,labels=x_labels,fontsize=20)  # x 轴刻度字体大小
    plt.yticks(ticks=np.arange(len(y_labels))+ 1,labels=y_labels,fontsize=20)  # y 轴刻度字体大小
    RGG_heatmap_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\SmoothHeatmapPrecisionRGGbeta{beta}.pdf".format(beta = beta)
    plt.savefig(RGG_heatmap_name,
        format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()

    plt.figure()
    df = pd.DataFrame(SRGG_matrix,
                      index=[ED_list],  # DataFrame的行标签设置为大写字母
                      columns=noise_amplitude_list)  # 设置DataFrame的列标签
    im = plt.imshow(df,
                    cmap='jet',
                    interpolation='bicubic',
                    origin='lower',
                    extent=[1, 5, 1, 7],
                    aspect='auto',
                    vmin=0,
                    vmax=0.5)
    cbar = plt.colorbar(im, label="Precision")
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.set_ylabel("Precision", fontsize=20)
    plt.ylabel(r"Expected degree $E[D]$", fontsize=25)
    plt.xlabel(r"Noise amplitude $\alpha$", fontsize=25)
    plt.xticks(ticks=np.arange(len(x_labels)) + 1, labels=x_labels, fontsize=20)  # x 轴刻度字体大小
    plt.yticks(ticks=np.arange(len(y_labels)) + 1, labels=y_labels, fontsize=20)  # y 轴刻度字体大小
    SRGG_heatmap_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\SmoothHeatmapPrecisionSRGGbeta{beta}.pdf".format(
        beta=beta)
    plt.savefig(SRGG_heatmap_name,
                format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()

    plt.figure()
    df = pd.DataFrame(Geo_matrix,
                      index=[ED_list],  # DataFrame的行标签设置为大写字母
                      columns=noise_amplitude_list)  # 设置DataFrame的列标签
    im = plt.imshow(df,
                    cmap='jet',
                    interpolation='bicubic',
                    origin='lower',
                    extent=[1, 5, 1, 7],
                    aspect='auto',
                    vmin=0,
                    vmax=0.5)
    cbar = plt.colorbar(im, label="Precision")
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.set_ylabel("Precision", fontsize=20)
    plt.ylabel(r"Expected degree $E[D]$", fontsize=25)
    plt.xlabel(r"Noise amplitude $\alpha$", fontsize=25)
    plt.xticks(ticks=np.arange(len(x_labels)) + 1, labels=x_labels, fontsize=20)  # x 轴刻度字体大小
    plt.yticks(ticks=np.arange(len(y_labels)) + 1, labels=y_labels, fontsize=20)  # y 轴刻度字体大小
    Geo_heatmap_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\SmoothHeatmapPrecisionGeobeta{beta}.pdf".format(
        beta=beta)
    plt.savefig(Geo_heatmap_name,
                format='pdf', bbox_inches='tight', dpi=600)
    plt.show()




# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # # # # STEP 1 plot the figure
    """
    Plot the heatmap for the precision and recall
    """
    # plot_heatmap_precision(2)
    plot_heatmap_precision_smooth(2)
