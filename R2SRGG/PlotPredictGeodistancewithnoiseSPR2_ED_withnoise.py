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
from matplotlib.colors import LogNorm

matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from skimage import measure
from matplotlib.patches import Rectangle
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
    df1 = pd.DataFrame(RGG_matrix,
                      index=[ED_list],  # DataFrame的行标签设置为大写字母
                      columns=noise_amplitude_list)  # 设置DataFrame的列标签
    df1.to_csv(f"D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\SmoothHeatmapPrecisionRGGbeta{beta}.csv")

    df1 = df1.mask(df1 < 1e-6, 1e-6)
    im = plt.imshow(df1,
               norm=LogNorm(vmin=1e-6, vmax=0.5),
               cmap='jet',
               interpolation='bicubic',
               origin='lower',
               extent=[1, 5, 1, 7],
               aspect='auto',
               )
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
    df2 = pd.DataFrame(SRGG_matrix,
                      index=[ED_list],  # DataFrame的行标签设置为大写字母
                      columns=noise_amplitude_list)  # 设置DataFrame的列标签

    df2.to_csv(
        f"D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\SmoothHeatmapPrecisionSRGGbeta{beta}.csv")

    df2 = df2.mask(df2 < 1e-6, 1e-6)
    im = plt.imshow(df2,
                    norm=LogNorm(vmin=1e-6, vmax=0.5),
                    cmap='jet',
                    interpolation='bicubic',
                    origin='lower',
                    extent=[1, 5, 1, 7],
                    aspect='auto',
                    )
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

    fig, ax = plt.subplots()
    df3 = pd.DataFrame(Geo_matrix,
                      index=[ED_list],  # DataFrame的行标签设置为大写字母
                      columns=noise_amplitude_list)  # 设置DataFrame的列标签
    print(df3-df1)
    print(df3 - df2)


    df3.to_csv(
        f"D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\SmoothHeatmapPrecisionGeobeta{beta}.csv")

    im = ax.imshow(df3,
                    norm=LogNorm(vmin=1e-6, vmax=0.5),
                    cmap='jet',
                    interpolation='bicubic',
                    origin='lower',
                    extent=[1, 5, 1, 7],
                    aspect='auto',
                    )
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


def process_data(betaindex):
    """
    ED vs noise
    :param betaindex:
    :return:
    """
    N = 10000
    ED_list = [2, 4, 8, 16, 32, 64, 128,256,512]  # Expected degrees
    ED_list = [2]  # Expected degrees
    betalist = [2.1, 4, 8, 32, 128]
    beta = betalist[betaindex]
    print("beta:", beta)

    # noise_amplitude_list = [0, 0.001, 0.01, 0.1, 1]
    # noise_amplitude_list = [0.0005, 0.005, 0.05, 0.5]
    # noise_amplitude_list = [0, 0.0005,0.001, 0.005,0.01,0.05, 0.1, 0.5,1]
    noise_amplitude_list = [0.005]

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
            with open(f"D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\Precisionall_ED{ED}_beta{beta}_noise{noise_amplitude}.txt", "w") as f:
                for a, b, c in zip(PrecisonRGG_specificnoise, PrecisonSRGG_specificnoise, PrecisonGeodis_specificnoise):
                    f.write(f"{a}\t{b}\t{c}\n")
    print(exemptionlist)


def plot_heatmap_precision_smooth_fromprocessed_data(betaindex):
    # Figure 6 Heatmap
    N = 10000
    ED_list = [2, 4, 8, 16, 32, 64, 128,256,512]  # Expected degrees
    betalist = [2.1, 4, 8, 32, 128]
    beta = betalist[betaindex]
    print("beta:", beta)

    # noise_amplitude_list = [0, 0.001, 0.01, 0.1, 1]
    noise_amplitude_list = [0, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

    exemptionlist = []
    RGG_matrix = np.zeros((len(ED_list), len(noise_amplitude_list)))
    SRGG_matrix = np.zeros((len(ED_list), len(noise_amplitude_list)))
    Geo_matrix = np.zeros((len(ED_list), len(noise_amplitude_list)))

    for EDindex in range(len(ED_list)):
        ED = ED_list[EDindex]
        print("ED:", ED)
        for noiseindex in range(len(noise_amplitude_list)):
            noise_amplitude = noise_amplitude_list[noiseindex]
            print(noise_amplitude)

            PrecisonRGG_specificnoise = []
            PrecisonSRGG_specificnoise = []
            PrecisonGeodis_specificnoise = []

            with open(f"D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\Precisionall_ED{ED}_beta{beta}_noise{noise_amplitude}.txt", "r") as f:
                for line in f:
                    parts = line.strip().split("\t")  # 按 Tab 分隔
                    if len(parts) == 3:
                        PrecisonRGG_specificnoise.append(float(parts[0]))
                        PrecisonSRGG_specificnoise.append(float(parts[1]))
                        PrecisonGeodis_specificnoise.append(float(parts[2]))

            RGG_matrix[EDindex][noiseindex] = np.mean(PrecisonRGG_specificnoise)
            SRGG_matrix[EDindex][noiseindex] = np.mean(PrecisonSRGG_specificnoise)
            Geo_matrix[EDindex][noiseindex] = np.mean(PrecisonGeodis_specificnoise)

    y_labels = [r"$2^1$", r"$2^2$", r"$2^3$",r"$2^{4}$", r"$2^{5}$",r"$2^{6}$", r"$2^{7}$", r"$2^{8}$", r"$2^{9}$"]
    x_labels = [r"0", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"]
    x_labels = [""] * 9
    x_labels[0] = r"0"
    x_labels[2] = r"$10^{-3}$"
    x_labels[4] = r"$10^{-2}$"
    x_labels[6] = r"$10^{-1}$"
    x_labels[8] = r"$10^{0}$"
    # x_labels = [r"0",r"$10^{-3}$" ,r"$5 \times 10^{-3}$", r"$10^{-2}$",r"$5 \times 10^{-2}$", r"$10^{-1}$",r"$5 \times 10^{-1}$", r"$10^{0}$"]

    plt.figure()
    df1 = pd.DataFrame(RGG_matrix,
                      index=[ED_list],  # DataFrame的行标签设置为大写字母
                      columns=noise_amplitude_list)  # 设置DataFrame的列标签
    df1.to_csv(f"D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\SmoothHeatmapPrecisionRGGbeta{beta}.csv")

    df1 = df1.mask(df1 < 1e-6, 1e-6)
    im = plt.imshow(df1,
               norm=LogNorm(vmin=1e-6, vmax=0.5),
               cmap='jet',
               interpolation='bicubic',
               origin='lower',
               extent=[1, 9, 1, 9],
               aspect='auto',
               )
    cbar = plt.colorbar(im, label="Precision")
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.set_ylabel("Precision", fontsize=20)
    plt.ylabel(r"Expected degree $\mathbb{E}[D]$",fontsize = 25)
    plt.xlabel(r"Noise amplitude $\alpha$", fontsize = 25)
    plt.xticks(ticks=np.arange(len(x_labels))+ 1,labels=x_labels,fontsize=20)  # x 轴刻度字体大小
    plt.yticks(ticks=np.arange(len(y_labels))+ 1,labels=y_labels,fontsize=20)  # y 轴刻度字体大小
    # RGG_heatmap_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\SmoothHeatmapPrecisionRGGbeta{beta}.pdf".format(beta = beta)
    # plt.savefig(RGG_heatmap_name,
    #             format='pdf', bbox_inches='tight', dpi=600)

    RGG_heatmap_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\SmoothHeatmapPrecisionRGGbeta{beta}.svg".format(
        beta=beta)
    plt.savefig(
        RGG_heatmap_name,
        format="svg",
        bbox_inches='tight',  # 紧凑边界
        transparent=True  # 背景透明，适合插图叠加
    )

    plt.show()
    plt.close()

    plt.figure()
    df2 = pd.DataFrame(SRGG_matrix,
                      index=[ED_list],  # DataFrame的行标签设置为大写字母
                      columns=noise_amplitude_list)  # 设置DataFrame的列标签

    df2.to_csv(
        f"D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\SmoothHeatmapPrecisionSRGGbeta{beta}.csv")

    df2 = df2.mask(df2 < 1e-6, 1e-6)
    im = plt.imshow(df2,
                    norm=LogNorm(vmin=1e-6, vmax=0.5),
                    cmap='jet',
                    interpolation='bicubic',
                    origin='lower',
                    extent=[1, 9, 1, 9],
                    aspect='auto',
                    )
    cbar = plt.colorbar(im, label="Precision")
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.set_ylabel("Precision", fontsize=20)
    plt.ylabel(r"Expected degree $\mathbb{E}[D]$", fontsize=25)
    plt.xlabel(r"Noise amplitude $\alpha$", fontsize=25)
    plt.xticks(ticks=np.arange(len(x_labels)) + 1, labels=x_labels, fontsize=20)  # x 轴刻度字体大小
    plt.yticks(ticks=np.arange(len(y_labels)) + 1, labels=y_labels, fontsize=20)  # y 轴刻度字体大小
    # SRGG_heatmap_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\SmoothHeatmapPrecisionSRGGbeta{beta}.pdf".format(
    #     beta=beta)
    # plt.savefig(SRGG_heatmap_name,
    #             format='pdf', bbox_inches='tight', dpi=600)
    SRGG_heatmap_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\SmoothHeatmapPrecisionSRGGbeta{beta}.svg".format(
        beta=beta)
    plt.savefig(
        SRGG_heatmap_name,
        format="svg",
        bbox_inches='tight',  # 紧凑边界
        transparent=True  # 背景透明，适合插图叠加
    )
    plt.show()
    plt.close()

    fig, ax = plt.subplots()
    df3 = pd.DataFrame(Geo_matrix,
                      index=[ED_list],  # DataFrame的行标签设置为大写字母
                      columns=noise_amplitude_list)  # 设置DataFrame的列标签
    print(df3-df1)
    print(df3 - df2)


    df3.to_csv(
        f"D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\SmoothHeatmapPrecisionGeobeta{beta}.csv")

    im = ax.imshow(df3,
                    norm=LogNorm(vmin=1e-6, vmax=0.5),
                    cmap='jet',
                    interpolation='bicubic',
                    origin='lower',
                    extent=[1, 9, 1, 9],
                    aspect='auto',
                    )
    cbar = plt.colorbar(im, label="Precision")
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.set_ylabel("Precision", fontsize=20)
    plt.ylabel(r"Expected degree $\mathbb{E}[D]$", fontsize=25)
    plt.xlabel(r"Noise amplitude $\alpha$", fontsize=25)
    plt.xticks(ticks=np.arange(len(x_labels)) + 1, labels=x_labels, fontsize=20)  # x 轴刻度字体大小
    plt.yticks(ticks=np.arange(len(y_labels)) + 1, labels=y_labels, fontsize=20)  # y 轴刻度字体大小
    Geo_heatmap_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\SmoothHeatmapPrecisionGeobeta{beta}.svg".format(
        beta=beta)
    plt.savefig(
        Geo_heatmap_name,
        format="svg",
        bbox_inches='tight',  # 紧凑边界
        transparent=True  # 背景透明，适合插图叠加
    )
    plt.show()

    a1 = df3 - df2
    a2 = df3 - df1
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(a1)
        print(a2)


def plot_skeleton(beta):
    # Figure 6
    """
    Plot the skeletion where the geo distance beforms better
    """
    np.random.seed(0)
    df1 = pd.read_csv(
        f"D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\SmoothHeatmapPrecisionRGGbeta{beta}.csv",
        index_col=0)
    df2 = pd.read_csv(
        f"D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\SmoothHeatmapPrecisionSRGGbeta{beta}.csv",
        index_col=0)
    df3 = pd.read_csv(
        f"D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\SmoothHeatmapPrecisionGeobeta{beta}.csv",
        index_col=0)

    y_labels = [r"$2^1$", r"$2^2$", r"$2^3$", r"$2^{4}$", r"$2^{5}$", r"$2^{6}$", r"$2^{7}$", r"$2^{8}$", r"$2^{9}$"]  # 横坐标
    x_labels = [r"0",'', r"$10^{-3}$",'', r"$10^{-2}$",'', r"$10^{-1}$",'', r"$10^{0}$"]  # 纵坐标

    # Step 1: 判断 df3 >= df1 和 df3 >= df2（允许小误差）
    # 容差判断
    tol = 1e-5
    mask = ((df3 > df1) | np.isclose(df3, df1, atol=tol)) & \
           ((df3 > df2) | np.isclose(df3, df2, atol=tol))

    # 转为 float 数组，用于 find_contours
    binary_mask = mask.astype(float)
    fig, ax = plt.subplots()
    im = ax.imshow(binary_mask,
                   norm=LogNorm(vmin=1e-6, vmax=0.5),
                   cmap='jet',
                   interpolation='bilinear',
                   origin='lower',
                   extent=[1, 9, 1, 9],
                   aspect='auto',
                   )
    cbar = plt.colorbar(im, label="Precision")
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.set_ylabel("Precision", fontsize=20)
    plt.ylabel(r"Expected degree $\mathbb{E}[D]$", fontsize=25)
    plt.xlabel(r"Noise amplitude $\alpha$", fontsize=25)
    plt.xticks(ticks=np.arange(len(x_labels)) + 1, labels=x_labels, fontsize=20)  # x 轴刻度字体大小
    plt.yticks(ticks=np.arange(len(y_labels)) + 1, labels=y_labels, fontsize=20)  # y 轴刻度字体大小
    Geo_heatmap_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\SmoothHeatmapPrecisionGeobeta{beta}Skeletion.svg".format(
        beta=beta)
    plt.savefig(
        Geo_heatmap_name,
        format="svg",
        bbox_inches='tight',  # 紧凑边界
        transparent=True  # 背景透明，适合插图叠加
    )
    plt.show()



    # a1 = df3 - df2
    # a2 = df3 - df1
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
    #     print(a1)
    #     print(a2)


    # plt.title("Closed Contour: df3 ≥ df1 and df2")
    # plt.tight_layout()
    # plt.xticks(ticks=np.arange(len(x_labels)) + 1, labels=x_labels, fontsize=20)  # x 轴刻度字体大小
    # plt.yticks(ticks=np.arange(len(y_labels)) + 1, labels=y_labels, fontsize=20)  # y 轴刻度字体大小
    # plt.show()



def check_which_square_is_bigger(beta):
    # Figure 6
    """
    Plot the skeletion where the geo distance beforms better
    """
    np.random.seed(0)
    df1 = pd.read_csv(
        f"D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\SmoothHeatmapPrecisionRGGbeta{beta}.csv",
        index_col=0)
    df2 = pd.read_csv(
        f"D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\SmoothHeatmapPrecisionSRGGbeta{beta}.csv",
        index_col=0)
    df3 = pd.read_csv(
        f"D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\SmoothHeatmapPrecisionGeobeta{beta}.csv",
        index_col=0)

    y_labels = [r"$2^1$", r"$2^2$", r"$2^3$", r"$2^{4}$", r"$2^{5}$", r"$2^{6}$", r"$2^{7}$", r"$2^{8}$", r"$2^{9}$"]  # 横坐标
    x_labels = [r"0",'', r"$10^{-3}$",'', r"$10^{-2}$",'', r"$10^{-1}$",'', r"$10^{0}$"]  # 纵坐标

    # Step 1: 判断 df3 >= df1 和 df3 >= df2（允许小误差）
    # 容差判断
    tol = 1e-5
    mask = ((df3 > df1) | np.isclose(df3, df1, atol=tol)) & \
           ((df3 > df2) | np.isclose(df3, df2, atol=tol))

    # 转为 float 数组，用于 find_contours
    binary_mask = mask.astype(float)

    # 为了让边界被识别，加 padding
    padded_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)

    # 轮廓检测：level=0.5 表示从 0/1 之间的边界找出轮廓线
    contours = measure.find_contours(padded_mask, level=0.5)

    # 显示热图
    fig, ax = plt.subplots()
    im = ax.imshow(df3,
                   norm=LogNorm(vmin=1e-6, vmax=0.5),
                   cmap='jet',
                   interpolation='bicubic',
                   origin='lower',
                   extent=[1, 9, 1, 9],
                   aspect='auto')
    plt.colorbar(im, ax=ax, label='df3 (log scale)')

    # 原图像 shape
    nrows, ncols = df3.shape
    x_extent = np.linspace(1, 9, ncols)
    y_extent = np.linspace(1, 9, nrows)

    # 将 contour 坐标（行列）映射到 extent 坐标系
    for contour in contours:
        # 去掉 padding 偏移
        contour -= 1

        # contour[:, 0] 是行 → y； contour[:, 1] 是列 → x
        x_coords = np.interp(contour[:, 1], np.arange(ncols), np.linspace(1, 9, ncols))
        y_coords = np.interp(contour[:, 0], np.arange(nrows), np.linspace(1, 9, nrows))

        ax.plot(x_coords, y_coords, color='white', linewidth=2.5)

    a1 = df3 - df2
    a2 = df3 - df1
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(a1)
        print(a2)


    plt.title("Closed Contour: df3 ≥ df1 and df2")
    plt.tight_layout()
    plt.xticks(ticks=np.arange(len(x_labels)) + 1, labels=x_labels, fontsize=20)  # x 轴刻度字体大小
    plt.yticks(ticks=np.arange(len(y_labels)) + 1, labels=y_labels, fontsize=20)  # y 轴刻度字体大小
    plt.show()


def check_simple_case():
    """
    this .m focus on data for one square
    :return:
    """
    N = 10000
    ED_list = [32]  # Expected degrees
    beta = 128
    print("beta:", beta)

    # noise_amplitude_list = [0, 0.001, 0.01, 0.1, 1]
    # noise_amplitude_list = [0.0005, 0.005, 0.05, 0.5]
    # noise_amplitude_list = [0, 0.0005,0.001, 0.005,0.01,0.05, 0.1, 0.5,1]
    noise_amplitude_list = [0.1]

    exemptionlist = []
    RGG_matrix = np.zeros((len(ED_list), len(noise_amplitude_list)))
    SRGG_matrix = np.zeros((len(ED_list), len(noise_amplitude_list)))
    Geo_matrix = np.zeros((len(ED_list), len(noise_amplitude_list)))
    inputfolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\test\\"
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
                    precision_RGG_Name = inputfolder_name+"PrecisionRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
                        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
                    Precison_RGG_5_times = np.loadtxt(precision_RGG_Name)
                    PrecisonRGG_specificnoise.extend(Precison_RGG_5_times)
                except FileNotFoundError:
                    exemptionlist.append((ED, beta, noise_amplitude, ExternalSimutime))
            # nonzero_indices_geo = find_nonzero_indices(PrecisonRGG_specificnoise)
            # PrecisonRGG_specificnoise = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PrecisonRGG_specificnoise))
            # PrecisonRGG_specificnoise = [PrecisonRGG_specificnoise[x] for x in nonzero_indices_geo]
            RGG_matrix[EDindex][noiseindex] = np.mean(PrecisonRGG_specificnoise)
            print("RGG",np.mean(PrecisonRGG_specificnoise))

            PrecisonSRGG_specificnoise = []
            for ExternalSimutime in range(20):
                try:
                    precision_SRGG_Name = inputfolder_name+"PrecisionSRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
                        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
                    Precison_SRGG_5_times = np.loadtxt(precision_SRGG_Name)
                    PrecisonSRGG_specificnoise.extend(Precison_SRGG_5_times)
                except FileNotFoundError:
                    pass
            # nonzero_indices_geo = find_nonzero_indices(PrecisonRGG_specificnoise)
            # PrecisonRGG_specificnoise = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PrecisonRGG_specificnoise))
            # PrecisonRGG_specificnoise = [PrecisonRGG_specificnoise[x] for x in nonzero_indices_geo]
            SRGG_matrix[EDindex][noiseindex] = np.mean(PrecisonSRGG_specificnoise)
            print("SRGG", np.mean(PrecisonSRGG_specificnoise))
            PrecisonGeodis_specificnoise = []
            for ExternalSimutime in range(20):
                try:
                    precision_Geodis_Name = inputfolder_name+"PrecisionGeodisED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
                        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
                    Precison_Geodis_5_times = np.loadtxt(precision_Geodis_Name)
                    PrecisonGeodis_specificnoise.extend(Precison_Geodis_5_times)
                except FileNotFoundError:
                    pass

            Geo_matrix[EDindex][noiseindex] = np.mean(PrecisonGeodis_specificnoise)
            print("GEO", np.mean(PrecisonGeodis_specificnoise))
            with open(
                    f"D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\test\\Precisionall_ED{ED}_beta{beta}_noise{noise_amplitude}.txt",
                    "w") as f:
                for a, b, c in zip(PrecisonRGG_specificnoise, PrecisonSRGG_specificnoise, PrecisonGeodis_specificnoise):
                    f.write(f"{a}\t{b}\t{c}\n")
    print(exemptionlist)






# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # # # # STEP 1 plot the figure
    """
    Plot the heatmap for the precision and recall
    """
    # plot_heatmap_precision(2)

    """
    Plot the heatmap for the precision and recall smoothly 
    """
    # for beta in [2]:
    #     # process_data(beta)
    #     plot_heatmap_precision_smooth_fromprocessed_data(beta)
    #     # plot_heatmap_precision_smooth(beta)

    """
    Plot the heatmap for the precision and recall smoothly from the final data
    Figure 6
    """
    # check_simple_case()   # check one square

    # for beta in [2]:
    #     # process_data(beta)
    #     plot_heatmap_precision_smooth_fromprocessed_data(beta)

    """
    Plot the heatmap for the skeleton Figure 6
    """
    # check_which_square_is_bigger(4)
    plot_skeleton(128)





