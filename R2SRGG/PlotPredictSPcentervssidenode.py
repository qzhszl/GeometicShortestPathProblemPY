# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 22*07*2025
SP NODES ARE divided into two parts
"""
import csv
import math

import numpy as np
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

def datatest(Edindex, betaindex,legendpara):
    # Figure 5 test
    # plot PRECISION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ED_list = [2, 5, 10, 100, 1000]  # Expected degrees
    ED = ED_list[Edindex]
    beta_list = [2.1, 4, 8, 16, 32, 64, 128]
    beta = beta_list[betaindex]
    RGG_precision_list_all_ave = []
    SRGG_precision_list_all_ave = []
    Geo_precision_list_all_ave = []
    RGG_precision_list_all_std = []
    SRGG_precision_list_all_std = []
    Geo_precision_list_all_std = []
    exemptionlist = []

    for noise_amplitude in [1]:
        PrecisonRGG_specificnoise = []
        for ExternalSimutime in range(20):
            try:
                precision_RGG_Name = file_folder+"RecallRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
                    EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime,no2=noise_amplitude)
                Precison_RGG_5_times = np.loadtxt(precision_RGG_Name)
                PrecisonRGG_specificnoise.extend(Precison_RGG_5_times)
            except FileNotFoundError:
                exemptionlist.append((ED,beta,noise_amplitude,ExternalSimutime))
        # nonzero_indices_geo = find_nonzero_indices(PrecisonRGG_specificnoise)
        # PrecisonRGG_specificnoise = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PrecisonRGG_specificnoise))
        # PrecisonRGG_specificnoise = [PrecisonRGG_specificnoise[x] for x in nonzero_indices_geo]
        RGG_precision_list_all_ave.append(np.mean(PrecisonRGG_specificnoise))
        RGG_precision_list_all_std.append(np.std(PrecisonRGG_specificnoise))
    # print("lenpre", len(PrecisonRGG_specificnoise))
        PrecisonSRGG_specificnoise = []
        for ExternalSimutime in range(20):
            try:
                precision_SRGG_Name = file_folder+"RecallSRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
                    EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime,no2=noise_amplitude)
                Precison_SRGG_5_times = np.loadtxt(precision_SRGG_Name)
                PrecisonSRGG_specificnoise.extend(Precison_SRGG_5_times)
            except FileNotFoundError:
                pass
        # nonzero_indices_geo = find_nonzero_indices(PrecisonRGG_specificnoise)
        # PrecisonRGG_specificnoise = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PrecisonRGG_specificnoise))
        # PrecisonRGG_specificnoise = [PrecisonRGG_specificnoise[x] for x in nonzero_indices_geo]
        SRGG_precision_list_all_ave.append(np.mean(PrecisonSRGG_specificnoise))
        SRGG_precision_list_all_std.append(np.std(PrecisonSRGG_specificnoise))
        # print("lenpre", len(PrecisonRGG_specificnoise))

        PrecisonGeodis_specificnoise = []
        for ExternalSimutime in range(20):
            try:
                precision_Geodis_Name = file_folder+"RecallGeodisED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
                    EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime,no2=noise_amplitude)
                Precison_Geodis_5_times = np.loadtxt(precision_Geodis_Name)
                PrecisonGeodis_specificnoise.extend(Precison_Geodis_5_times)
            except FileNotFoundError:
                pass
        # nonzero_indices_geo = find_nonzero_indices(PrecisonRGG_specificnoise)
        # PrecisonRGG_specificnoise = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PrecisonRGG_specificnoise))
        # PrecisonRGG_specificnoise = [PrecisonRGG_specificnoise[x] for x in nonzero_indices_geo]
        Geo_precision_list_all_ave.append(np.mean(PrecisonGeodis_specificnoise))
        Geo_precision_list_all_std.append(np.std(PrecisonGeodis_specificnoise))
        # print("lenpre", len(PrecisonRGG_specificnoise))

        rows = zip(PrecisonRGG_specificnoise, PrecisonSRGG_specificnoise, PrecisonGeodis_specificnoise)

        # 写入 CSV 文件
        with open("precision_results.csv", "w", newline='') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow(["PrecisonRGG_specificnoise", "PrecisonSRGG_specificnoise", "PrecisonGeodis_specificnoise"])
            # 写入数据行
            writer.writerows(rows)


def plot_predict_geo_sidevscenter(Edindex, betaindex, legendpara):
    # Figure 7
    # plot PRECISION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ED_list = [4,8]  # Expected degrees
    ED = ED_list[Edindex]
    beta_list = [4]
    beta = beta_list[betaindex]
    RGG_precision_list_all_ave = []
    center_precision_list_all_ave = []
    Geo_precision_list_all_ave = []
    RGG_precision_list_all_std = []
    SRGG_precision_list_all_std = []
    Geo_precision_list_all_std = []
    exemptionlist = []
    file_folder = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\predictcenternodevssidenode\\"
    for noise_amplitude in [0]:
        Precisongeocenter_specificnoise = []  # data for geo center not SRGG
        for ExternalSimutime in range(20):
            try:
                precision_SRGG_Name = file_folder+"PrecisionGeodisED{EDn}Beta{betan}Noise{no}PYSimu{ST}_center.txt".format(
                    EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime,no2=noise_amplitude)
                Precison_SRGG_5_times = np.loadtxt(precision_SRGG_Name)
                Precisongeocenter_specificnoise.extend(Precison_SRGG_5_times)
            except FileNotFoundError:
                pass
        # nonzero_indices_geo = find_nonzero_indices(PrecisonRGG_specificnoise)
        # PrecisonRGG_specificnoise = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PrecisonRGG_specificnoise))
        # PrecisonRGG_specificnoise = [PrecisonRGG_specificnoise[x] for x in nonzero_indices_geo]
        center_precision_list_all_ave.append(np.mean(Precisongeocenter_specificnoise))
        SRGG_precision_list_all_std.append(np.std(Precisongeocenter_specificnoise))
        # print("lenpre", len(PrecisonRGG_specificnoise))

        PrecisonGeodis_specificnoise = []
        for ExternalSimutime in range(20):
            try:
                precision_Geodis_Name = file_folder+"PrecisionGeodisED{EDn}Beta{betan}Noise{no}PYSimu{ST}_side.txt".format(
                    EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime,no2=noise_amplitude)
                Precison_Geodis_5_times = np.loadtxt(precision_Geodis_Name)
                PrecisonGeodis_specificnoise.extend(Precison_Geodis_5_times)
            except FileNotFoundError:
                pass
        # nonzero_indices_geo = find_nonzero_indices(PrecisonRGG_specificnoise)
        # PrecisonRGG_specificnoise = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PrecisonRGG_specificnoise))
        # PrecisonRGG_specificnoise = [PrecisonRGG_specificnoise[x] for x in nonzero_indices_geo]
        Geo_precision_list_all_ave.append(np.mean(PrecisonGeodis_specificnoise))
        Geo_precision_list_all_std.append(np.std(PrecisonGeodis_specificnoise))
        # print("lenpre", len(PrecisonRGG_specificnoise))

    fig, ax = plt.subplots(figsize=(5, 5))
    # Data
    y1 = center_precision_list_all_ave
    y2 = Geo_precision_list_all_ave
    y_error_lower = [p*0 for p in y1]

    # X axis labels
    x_labels = ['Mid-path', "Near-endpoint"]
    # x_labels = ['0', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$']

    # X axis positions for each bar group
    x = [0.23,0.78]

    # Width of each bar
    width = 0.2

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    # Plotting the bars
    bar1 = ax.bar(x[0], y1, width, label='center',yerr=(y_error_lower,SRGG_precision_list_all_std), capsize=5, color=colors[3])
    bar2 = ax.bar(x[1], y2, width, label='side', yerr=(y_error_lower,Geo_precision_list_all_std), capsize=5, color=colors[0])
    # bar3 = ax.bar(x + width, y3, width, label='Geo', yerr=(y_error_lower,Geo_precision_list_all_std), capsize=5, color=colors[2])

    # Adding labels and title
    ax.set_xlim(0,1)
    # ax.set_xlabel(r'Noise amplitude, $\alpha$', fontsize = 25)
    ax.set_ylabel('Precision',fontsize = 30)
    # ax.set_yscale("log")
    # title_name = "beta:{beta_n}, E[D]:{ed_n}".format(ed_n=ED, beta_n = beta)
    # ax.set_title(title_name)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_yticks([0,0.1])
    ax.text(0.1, 0.55, r'$N = 10^4$' + "\n" + r'$\beta = 4$' + "\n" + r"$\mathbb{E}[D] = 5$",transform=ax.transAxes,
             fontsize=34)
    ax.tick_params(axis='x', pad=15)
    plt.tick_params(axis='both', which="both", length=6, width=1)

    # if legendpara ==1:
    #     ax.legend(fontsize=22,loc = (0.5, 0.6))
    ax.tick_params(direction='out')

    plt.xticks(fontsize=26)
    plt.yticks(fontsize=34)

    # Display the plot
    # plt.show()
    # figname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\PrecisionGeoVsRGGSRGGED{EDn}Beta{betan}N.pdf".format(
    #             EDn=ED, betan=beta)
    #
    # fig.savefig(figname, format='pdf', bbox_inches='tight', dpi=600)
    figname = file_folder+"PrecisionGeoCenterVsSideED{EDn}Beta{betan}.svg".format(
                EDn=ED, betan=beta)
    plt.savefig(
        figname,
        format="svg",
        bbox_inches='tight',  # 紧凑边界
        transparent=True  # 背景透明，适合插图叠加
    )
    plt.show()
    plt.close()
    print(exemptionlist)



def plot_predict_geodistance_Vs_reconstructionRGG_SRGGcenterside(Edindex, betaindex, legendpara):
    # Figure unknown
    # plot PRECISION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ED_list = [4,8]  # Expected degrees
    ED = ED_list[Edindex]
    beta_list = [4]
    beta = beta_list[betaindex]
    RGG_precision_list_all_ave = []
    SRGG_precision_list_all_ave = []
    Geo_precision_list_all_ave = []
    RGG_precision_list_all_std = []
    SRGG_precision_list_all_std = []
    Geo_precision_list_all_std = []
    exemptionlist = []
    file_folder = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\predictcenternodevssidenode\\"
    for noise_amplitude in [0]:
        PrecisonRGG_specificnoise = []
        for ExternalSimutime in range(20):
            try:
                precision_RGG_Name = file_folder+"PrecisionRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}_side.txt".format(
                    EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime,no2=noise_amplitude)
                Precison_RGG_5_times = np.loadtxt(precision_RGG_Name)
                PrecisonRGG_specificnoise.extend(Precison_RGG_5_times)
            except FileNotFoundError:
                exemptionlist.append((ED,beta,noise_amplitude,ExternalSimutime))
        # nonzero_indices_geo = find_nonzero_indices(PrecisonRGG_specificnoise)
        # PrecisonRGG_specificnoise = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PrecisonRGG_specificnoise))
        # PrecisonRGG_specificnoise = [PrecisonRGG_specificnoise[x] for x in nonzero_indices_geo]
        RGG_precision_list_all_ave.append(np.mean(PrecisonRGG_specificnoise))
        RGG_precision_list_all_std.append(np.std(PrecisonRGG_specificnoise))
    # print("lenpre", len(PrecisonRGG_specificnoise))
        PrecisonSRGG_specificnoise = []
        for ExternalSimutime in range(20):
            try:
                precision_SRGG_Name = file_folder+"PrecisionSRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}_side.txt".format(
                    EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime,no2=noise_amplitude)
                Precison_SRGG_5_times = np.loadtxt(precision_SRGG_Name)
                PrecisonSRGG_specificnoise.extend(Precison_SRGG_5_times)
            except FileNotFoundError:
                pass
        # nonzero_indices_geo = find_nonzero_indices(PrecisonRGG_specificnoise)
        # PrecisonRGG_specificnoise = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PrecisonRGG_specificnoise))
        # PrecisonRGG_specificnoise = [PrecisonRGG_specificnoise[x] for x in nonzero_indices_geo]
        SRGG_precision_list_all_ave.append(np.mean(PrecisonSRGG_specificnoise))
        SRGG_precision_list_all_std.append(np.std(PrecisonSRGG_specificnoise))
        # print("lenpre", len(PrecisonRGG_specificnoise))

        PrecisonGeodis_specificnoise = []
        for ExternalSimutime in range(20):
            try:
                precision_Geodis_Name = file_folder+"PrecisionGeodisED{EDn}Beta{betan}Noise{no}PYSimu{ST}_side.txt".format(
                    EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime,no2=noise_amplitude)
                Precison_Geodis_5_times = np.loadtxt(precision_Geodis_Name)
                PrecisonGeodis_specificnoise.extend(Precison_Geodis_5_times)
            except FileNotFoundError:
                pass
        # nonzero_indices_geo = find_nonzero_indices(PrecisonRGG_specificnoise)
        # PrecisonRGG_specificnoise = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PrecisonRGG_specificnoise))
        # PrecisonRGG_specificnoise = [PrecisonRGG_specificnoise[x] for x in nonzero_indices_geo]
        Geo_precision_list_all_ave.append(np.mean(PrecisonGeodis_specificnoise))
        Geo_precision_list_all_std.append(np.std(PrecisonGeodis_specificnoise))
        # print("lenpre", len(PrecisonRGG_specificnoise))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    # Data
    y1 = RGG_precision_list_all_ave
    y2 = SRGG_precision_list_all_ave
    y3 = Geo_precision_list_all_ave
    y_error_lower = [p*0 for p in y1]

    # X axis labels
    x_labels = ['0']
    # x_labels = ['0', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$']

    # X axis positions for each bar group
    x = np.arange(len(x_labels))

    # Width of each bar
    width = 0.2

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    # Plotting the bars
    bar1 = ax.bar(x - width, y1, width, label='RGG',yerr=(y_error_lower,RGG_precision_list_all_std), capsize=5, color=colors[3])
    bar2 = ax.bar(x, y2, width, label='SRGG', yerr=(y_error_lower,SRGG_precision_list_all_std), capsize=5, color=colors[0])
    bar3 = ax.bar(x + width, y3, width, label='Geo', yerr=(y_error_lower,Geo_precision_list_all_std), capsize=5, color=colors[2])

    # Adding labels and title
    # ax.set_ylim(0,1)
    ax.set_xlabel(r'Noise amplitude, $\alpha$', fontsize = 25)
    ax.set_ylabel('Precision',fontsize = 25)
    ax.set_yscale("log")
    # title_name = "beta:{beta_n}, E[D]:{ed_n}".format(ed_n=ED, beta_n = beta)
    # ax.set_title(title_name)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)

    if legendpara ==1:
        ax.legend(fontsize=22,loc = (0.57, 0.55))
    ax.tick_params(direction='out')

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    # ytick_dict = {
    #     (5, 4): [0, 0.1, 0.2],
    #     (5, 8): [0, 0.1, 0.2, 0.3, 0.4],
    #     (5, 128): [0, 0.2, 0.4, 0.6, 0.8, 1.0],
    #     (2, 8): [0, 0.2, 0.4, 0.6, 0.8],
    #     (10, 8): [0, 0.1, 0.2, 0.3, 0.4],
    #     (100, 8): [0, 0.1, 0.2, 0.3, 0.4],
    # }
    # ytick_vec = ytick_dict[(ED, beta)]
    # plt.yticks(ytick_vec, fontsize=22)

    fignum_dict = {
        (4, 4): "a",
        (8, 4): "b",
        (5, 128): "c",
        (2, 8): "d",
        (10, 8): "e",
        (100, 8): "f",
    }
    fignum = fignum_dict[(ED, beta)]
    ax.text(-0.23, 1.13, fr'({fignum}) $\mathbb{{E}}[D] = {ED}$, $\beta = {beta}$', transform=ax.transAxes,
            fontsize=25, verticalalignment='top', horizontalalignment='left')


    plt.tick_params(axis='both', which="both", length=6, width=1)
    # Display the plot
    # plt.show()
    # figname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\PrecisionGeoVsRGGSRGGED{EDn}Beta{betan}N.pdf".format(
    #             EDn=ED, betan=beta)
    #
    # fig.savefig(figname, format='pdf', bbox_inches='tight', dpi=600)
    figname = file_folder+"PrecisionGeoVsRGGSRGGED{EDn}Beta{betan}N_side.svg".format(
                EDn=ED, betan=beta)
    plt.savefig(
        figname,
        format="svg",
        bbox_inches='tight',  # 紧凑边界
        transparent=True  # 背景透明，适合插图叠加
    )
    plt.show()
    plt.close()
    print(exemptionlist)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # datatest(2, 2, 0)


    # for Edindex,betaindex in [(0,0)]:
    #     plot_predict_geodistance_Vs_reconstructionRGG_SRGGcenterside(Edindex, betaindex, legendpara=1)
    for Edindex, betaindex in [(0, 0)]:
        plot_predict_geo_sidevscenter(Edindex, betaindex, legendpara=1)



