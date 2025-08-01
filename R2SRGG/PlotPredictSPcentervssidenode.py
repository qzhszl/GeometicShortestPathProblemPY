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
    # Figure unknown
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

    fig, ax = plt.subplots(figsize=(6, 4.5))
    # Data
    y1 = center_precision_list_all_ave
    y2 = Geo_precision_list_all_ave
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
    bar1 = ax.bar(x - width, y1, width, label='center',yerr=(y_error_lower,SRGG_precision_list_all_std), capsize=5, color=colors[3])
    bar2 = ax.bar(x, y2, width, label='side', yerr=(y_error_lower,Geo_precision_list_all_std), capsize=5, color=colors[0])
    # bar3 = ax.bar(x + width, y3, width, label='Geo', yerr=(y_error_lower,Geo_precision_list_all_std), capsize=5, color=colors[2])

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
        ax.legend(fontsize=22,loc = (0.5, 0.6))
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



def plot_predict_geodistance_Vs_reconstructionRGG_SRGG_withnoise_SP_R2_clu2(Edindex, betaindex,legendpara):
    # Figure 5 recall
    # plot recall!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ED_list = [2,5, 10, 100, 1000]  # Expected degrees
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

    for noise_amplitude in [0, 0.001, 0.01, 0.1, 1]:
        PrecisonRGG_specificnoise = []
        for ExternalSimutime in range(20):
            try:
                precision_RGG_Name = file_folder+"RecallRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
                    EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime, no2=noise_amplitude)
                Precison_RGG_5_times = np.loadtxt(precision_RGG_Name)
                PrecisonRGG_specificnoise.extend(Precison_RGG_5_times)
            except:
                exemptionlist.append((ED, beta, noise_amplitude, ExternalSimutime))
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
                    EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime, no2=noise_amplitude)
                Precison_SRGG_5_times = np.loadtxt(precision_SRGG_Name)
                PrecisonSRGG_specificnoise.extend(Precison_SRGG_5_times)
            except:
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
                    EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime, no2=noise_amplitude)
                Precison_Geodis_5_times = np.loadtxt(precision_Geodis_Name)
                PrecisonGeodis_specificnoise.extend(Precison_Geodis_5_times)
            except:
                pass
        # nonzero_indices_geo = find_nonzero_indices(PrecisonRGG_specificnoise)
        # PrecisonRGG_specificnoise = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PrecisonRGG_specificnoise))
        # PrecisonRGG_specificnoise = [PrecisonRGG_specificnoise[x] for x in nonzero_indices_geo]
        Geo_precision_list_all_ave.append(np.mean(PrecisonGeodis_specificnoise))
        Geo_precision_list_all_std.append(np.std(PrecisonGeodis_specificnoise))
        # print("lenpre", len(PrecisonRGG_specificnoise))

    fig, ax = plt.subplots(figsize=(6, 4.5))

    # ax.set_ylim(0,1.1)
    # Data
    y1 = RGG_precision_list_all_ave
    y2 = SRGG_precision_list_all_ave
    y3 = Geo_precision_list_all_ave
    y_error_lower = [p * 0 for p in y1]
    # X axis labels
    x_labels = ['0', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$']

    # X axis positions for each bar group
    x = np.arange(len(x_labels))

    # Width of each bar
    width = 0.2
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # Plotting the bars
    bar1 = ax.bar(x - width, y1, width, label='RGG', yerr=(y_error_lower, RGG_precision_list_all_std), capsize=5,color=colors[3])
    bar2 = ax.bar(x, y2, width, label='SRGG', yerr=(y_error_lower, SRGG_precision_list_all_std), capsize=5,color=colors[0])
    bar3 = ax.bar(x + width, y3, width, label='Geo', yerr=(y_error_lower, Geo_precision_list_all_std), capsize=5,color=colors[2])

    # Adding labels and title
    ax.set_xlabel(r'Noise amplitude, $\alpha$',fontsize = 25)
    ax.set_ylabel('Recall',fontsize = 25)

    # title_name = "beta:{beta_n}, E[D]:{ed_n}".format(ed_n=ED, beta_n = beta)
    # ax.set_title(title_name)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_yscale("log")
    if legendpara == 1:
        ax.legend(fontsize=22,loc = (0.57,0.55))
    ax.tick_params(direction='out')
    plt.xticks(fontsize=22)

    # ytick_dict = {
    #     (5, 4): [0,0.1,0.2],
    #     (5, 8): [0,0.1,0.2,0.3,0.4],
    #     (5, 128): [0,0.2,0.4,0.6,0.8,1.0],
    #     (2, 8): [0,0.2,0.4,0.6,0.8],
    #     (10, 8):[0,0.1,0.2,0.3,0.4,0.5],
    #     (100, 8): [0,0.1,0.2,0.3,0.4,0.5,0.6],
    # }
    # ytick_vec = ytick_dict[(ED, beta)]
    #
    # plt.yticks(ytick_vec,fontsize=22)
    plt.tick_params(axis='both', which="both", length=6, width=1)

    fignum_dict = {
        (5,4): "a",
        (5,8): "b",
        (5,128): "c",
        (2, 8): "d",
        (10, 8): "e",
        (100, 8): "f",
    }
    fignum = fignum_dict[(ED,beta)]
    ax.text(-0.23, 1.13, fr'({fignum}) $\mathbb{{E}}[D] = {ED}$, $\beta = {beta}$', transform=ax.transAxes,
            fontsize=25, verticalalignment='top', horizontalalignment='left')
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # Display the plot

    # figname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\RecallGeoVsRGGSRGGED{EDn}Beta{betan}N.pdf".format(
    #     EDn=ED, betan=beta)
    # plt.savefig(figname, format='pdf', bbox_inches='tight', dpi=600)
    figname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\RecallGeoVsRGGSRGGED{EDn}Beta{betan}N.svg".format(
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


def check_data_wehavenow():
    ED_list = [5, 10, 20, 40]  # Expected degrees

    beta_list = [2.1, 4, 8, 16, 32, 64, 128]

    exemptionlist = []
    for ED_index in range(4):
        for beta_index in range(7):
            ED = ED_list[ED_index]
            beta = beta_list[beta_index]
            for noise_amplitude in [0]:
                for ExternalSimutime in range(20):
                    try:
                        precision_RGG_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise0\\PrecisionSRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
                            EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
                        Precison_RGG_5_times = np.loadtxt(precision_RGG_Name)
                    except FileNotFoundError:
                        exemptionlist.append((ED_index, beta_index, noise_amplitude, ExternalSimutime))
    print(exemptionlist)
    # np.savetxt("notrun.txt",exemptionlist)


def plot_heatmap_precision(noiseindex):

    N = 10000
    ED_list = [2, 3.5, 5, 10, 100, 1000, N - 1]  # Expected degrees
    ED_list = [2, 3.5, 5, 10, 100,1000]  # Expected degrees
    beta_list = [2.1, 4, 8, 32, 128]

    noise_amplitude_list = [0, 0.001, 0.01, 0.1, 1]
    noise_amplitude = noise_amplitude_list[noiseindex]
    print("noise amplitude:", noise_amplitude)

    exemptionlist =[]
    RGG_matrix = np.zeros((len(ED_list), len(beta_list)))
    SRGG_matrix = np.zeros((len(ED_list), len(beta_list)))
    Geo_matrix = np.zeros((len(ED_list), len(beta_list)))
    for EDindex in range(len(ED_list)):
        ED = ED_list[EDindex]
        print("ED:", ED)

        for betaindex in range(len(beta_list)):
            beta = beta_list[betaindex]
            print(beta)
            precision_list = []
            if ED in [0.1]:
                PrecisonRGG_specificnoise = []
                for ExternalSimutime in range(20):
                    try:
                        precision_RGG_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise{no2}\\PrecisionRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
                            EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime, no2=noise_amplitude)
                        Precison_RGG_5_times = np.loadtxt(precision_RGG_Name)
                        PrecisonRGG_specificnoise.extend(Precison_RGG_5_times)
                    except FileNotFoundError:
                        exemptionlist.append((ED, beta, noise_amplitude, ExternalSimutime))
                # nonzero_indices_geo = find_nonzero_indices(PrecisonRGG_specificnoise)
                # PrecisonRGG_specificnoise = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PrecisonRGG_specificnoise))
                # PrecisonRGG_specificnoise = [PrecisonRGG_specificnoise[x] for x in nonzero_indices_geo]
                RGG_matrix[EDindex][betaindex] = np.mean(PrecisonRGG_specificnoise)


                PrecisonSRGG_specificnoise = []
                for ExternalSimutime in range(20):
                    try:
                        precision_SRGG_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise{no2}\\PrecisionSRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
                            EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime, no2=noise_amplitude)
                        Precison_SRGG_5_times = np.loadtxt(precision_SRGG_Name)
                        PrecisonSRGG_specificnoise.extend(Precison_SRGG_5_times)
                    except FileNotFoundError:
                        pass
                # nonzero_indices_geo = find_nonzero_indices(PrecisonRGG_specificnoise)
                # PrecisonRGG_specificnoise = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PrecisonRGG_specificnoise))
                # PrecisonRGG_specificnoise = [PrecisonRGG_specificnoise[x] for x in nonzero_indices_geo]
                SRGG_matrix[EDindex][betaindex] = np.mean(PrecisonSRGG_specificnoise)

                PrecisonGeodis_specificnoise = []
                for ExternalSimutime in range(20):
                    try:
                        precision_Geodis_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise{no2}\\PrecisionGeodisED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
                            EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime, no2=noise_amplitude)
                        Precison_Geodis_5_times = np.loadtxt(precision_Geodis_Name)
                        PrecisonGeodis_specificnoise.extend(Precison_Geodis_5_times)
                    except FileNotFoundError:
                        pass

                Geo_matrix[EDindex][betaindex] = np.mean(PrecisonGeodis_specificnoise)
            else:
                PrecisonRGG_specificnoise = []
                for ExternalSimutime in range(20):
                    try:
                        precision_RGG_Name = file_folder+"PrecisionRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
                            EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime, no2=noise_amplitude)
                        Precison_RGG_5_times = np.loadtxt(precision_RGG_Name)
                        PrecisonRGG_specificnoise.extend(Precison_RGG_5_times)
                    except FileNotFoundError:
                        exemptionlist.append((ED, beta, noise_amplitude, ExternalSimutime))
                # nonzero_indices_geo = find_nonzero_indices(PrecisonRGG_specificnoise)
                # PrecisonRGG_specificnoise = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PrecisonRGG_specificnoise))
                # PrecisonRGG_specificnoise = [PrecisonRGG_specificnoise[x] for x in nonzero_indices_geo]
                RGG_matrix[EDindex][betaindex] = np.mean(PrecisonRGG_specificnoise)

                PrecisonSRGG_specificnoise = []
                for ExternalSimutime in range(20):
                    try:
                        precision_SRGG_Name = file_folder+"PrecisionSRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
                            EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime, no2=noise_amplitude)
                        Precison_SRGG_5_times = np.loadtxt(precision_SRGG_Name)
                        PrecisonSRGG_specificnoise.extend(Precison_SRGG_5_times)
                    except FileNotFoundError:
                        pass
                # nonzero_indices_geo = find_nonzero_indices(PrecisonRGG_specificnoise)
                # PrecisonRGG_specificnoise = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PrecisonRGG_specificnoise))
                # PrecisonRGG_specificnoise = [PrecisonRGG_specificnoise[x] for x in nonzero_indices_geo]
                SRGG_matrix[EDindex][betaindex] = np.mean(PrecisonSRGG_specificnoise)

                PrecisonGeodis_specificnoise = []
                for ExternalSimutime in range(20):
                    try:
                        precision_Geodis_Name = file_folder+"PrecisionGeodisED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
                            EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime, no2=noise_amplitude)
                        Precison_Geodis_5_times = np.loadtxt(precision_Geodis_Name)
                        PrecisonGeodis_specificnoise.extend(Precison_Geodis_5_times)
                    except FileNotFoundError:
                        pass

                Geo_matrix[EDindex][betaindex] = np.mean(PrecisonGeodis_specificnoise)

    print(exemptionlist)
    x_labels = ["2.1", "4", "8", "32", "128"]  # 横坐标
    # y_labels = ["100", "10", "5", "3.5", "2"]  # 纵坐标
    y_labels = ["2", "3.5", "5", "10", "100","1000"]  # 纵坐标
    plt.figure()
    df = pd.DataFrame(RGG_matrix,
                      index=[ED_list],  # DataFrame的行标签设置为大写字母
                      columns=beta_list)  # 设置DataFrame的列标签
    # h1 = sns.heatmap(data=df, vmin=0, vmax=0.9, annot=True, fmt=".2f", cbar=True, annot_kws={"size": 20},
    #                  cbar_kws={'label': 'Precision'})
    # plt.show()

    h1 = sns.heatmap(data=df, vmin=0, vmax=0.9, annot=True, fmt=".2f", cbar=True, annot_kws={"size": 20},
                cbar_kws={'label': 'Precision'}, xticklabels=x_labels,  # 指定自定义 x 轴标签
    yticklabels=y_labels)
    # plt.title("50% links are removed when computing Nearly Shortest Path Node")
    plt.xticks(fontsize=20)  # x 轴刻度字体大小
    plt.yticks(fontsize=20)  # y 轴刻度字体大小

    cbar = h1.collections[0].colorbar  # 获取颜色条对象
    cbar.ax.tick_params(labelsize=20)  # 设置颜色条刻度字体大小
    cbar.ax.set_ylabel("Precision", fontsize=20)  # 设置颜色条标签字体大小


    plt.xlabel(r"Temperature $\beta$",fontsize = 25)
    plt.ylabel(r"Expected degree $E[D]$", fontsize = 25)
    RGG_heatmap_name = file_folder+"HeatmapPrecisionRGGNoise{No}.pdf".format(No = noise_amplitude)
    plt.savefig(RGG_heatmap_name,
        format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()

    plt.figure()
    df = pd.DataFrame(SRGG_matrix,
                      index=[ED_list],  # DataFrame的行标签设置为大写字母
                      columns=beta_list)  # 设置DataFrame的列标签
    h2 = sns.heatmap(data=df, vmin=0, vmax=0.9, annot=True, fmt=".2f", cbar=True, annot_kws={"size": 20},
                     cbar_kws={'label': 'Precision'}, xticklabels=x_labels,  # 指定自定义 x 轴标签
                     yticklabels=y_labels)
    # plt.title("50% links are removed when computing Nearly Shortest Path Node")
    plt.xticks(fontsize=20)  # x 轴刻度字体大小
    plt.yticks(fontsize=20)  # y 轴刻度字体大小

    cbar = h2.collections[0].colorbar  # 获取颜色条对象
    cbar.ax.tick_params(labelsize=20)  # 设置颜色条刻度字体大小
    cbar.ax.set_ylabel("Precision", fontsize=20)  # 设置颜色条标签字体大小
    # plt.title("50% links are removed when computing Nearly Shortest Path Node")
    plt.xlabel(r"Temperature $\beta$", fontsize=25)
    plt.ylabel(r"Expected degree $E[D]$", fontsize=25)
    SRGG_heatmap_name = file_folder+"HeatmapPrecisionSRGGNoise{No}.pdf".format(
        No=noise_amplitude)
    plt.savefig(SRGG_heatmap_name,
                format='pdf', bbox_inches='tight', dpi=600)
    plt.show()

    plt.close()

    plt.figure()
    df = pd.DataFrame(Geo_matrix,
                      index=[ED_list],  # DataFrame的行标签设置为大写字母
                      columns=beta_list)  # 设置DataFrame的列标签
    h3 = sns.heatmap(data=df, vmin=0, vmax=0.9, annot=True, fmt=".2f", cbar=True, annot_kws={"size": 20},
                     cbar_kws={'label': 'Precision'}, xticklabels=x_labels,  # 指定自定义 x 轴标签
                     yticklabels=y_labels)
    # plt.title("50% links are removed when computing Nearly Shortest Path Node")
    plt.xticks(fontsize=20)  # x 轴刻度字体大小
    plt.yticks(fontsize=20)  # y 轴刻度字体大小

    cbar = h3.collections[0].colorbar  # 获取颜色条对象
    cbar.ax.tick_params(labelsize=20)  # 设置颜色条刻度字体大小
    cbar.ax.set_ylabel("Precision", fontsize=20)  # 设置颜色条标签字体大小
    # plt.title("50% links are removed when computing Nearly Shortest Path Node")
    plt.xlabel(r"Temperature $\beta$", fontsize=25)
    plt.ylabel(r"Expected degree $E[D]$", fontsize=25)
    Geo_heatmap_name = file_folder+"HeatmapPrecisionGeoNoise{No}.pdf".format(
        No=noise_amplitude)
    plt.savefig(Geo_heatmap_name,
                format='pdf', bbox_inches='tight', dpi=600)
    plt.show()





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # datatest(2, 2, 0)


    # # # # STEP 1 plot the figure
    # for Edindex in range(5):
    #     for betaindex in range(7):
    #         plot_predict_geodistance_Vs_reconstructionRGG_SRGG_withnoise_SP_R2_clu(Edindex, betaindex,legendpara=0)
    #
    #
    # for Edindex,betaindex in [(1,1),(1,6),(1,2),(2,2),(3,2)]:
    #     plot_predict_geodistance_Vs_reconstructionRGG_SRGG_withnoise_SP_R2_clu(Edindex, betaindex, legendpara=0)
    # for Edindex in [0]:
    #     for betaindex in [2]:
    #         plot_predict_geodistance_Vs_reconstructionRGG_SRGG_withnoise_SP_R2_clu(Edindex, betaindex, legendpara=1)
    # for Edindex,betaindex in [(0,0)]:
    #     plot_predict_geodistance_Vs_reconstructionRGG_SRGGcenterside(Edindex, betaindex, legendpara=1)
    for Edindex, betaindex in [(1, 0)]:
        plot_predict_geo_sidevscenter(Edindex, betaindex, legendpara=1)

    # # STEP 2 plot the recall
    # for Edindex in range(5):
    #     for betaindex in range(7):
    #         plot_predict_geodistance_Vs_reconstructionRGG_SRGG_withnoise_SP_R2_clu2(Edindex, betaindex,legendpara=0)
    # ED_list = [2, 5, 10, 100, 1000]  # Expected degrees
    # beta_list = [2.1, 4, 8, 16, 32, 64, 128]

    # for Edindex,betaindex in [(1,2),(1,6),(1,1),(2,2),(3,2)]:
    #     plot_predict_geodistance_Vs_reconstructionRGG_SRGG_withnoise_SP_R2_clu2(Edindex, betaindex, legendpara=0)

    # for Edindex in [0]:
    #     for betaindex in [2]:
    #         plot_predict_geodistance_Vs_reconstructionRGG_SRGG_withnoise_SP_R2_clu2(Edindex, betaindex,legendpara=1)


    # for Edindex in [2]:
    #     for betaindex in [2]:
    #         plot_predict_geodistance_Vs_reconstructionRGG_SRGG_withnoise_SP_R2_clu2(Edindex, betaindex,legendpara=0)

    """
    Plot figure for netsci
    """
    # for Edindex in [0]:
    #     for betaindex in [2]:
    #         plot_predict_geodistance_Vs_reconstructionSRGG_withnoise_SP_R2_Netsci(Edindex, betaindex, legendpara=1)
    # x = [1, 2, 3, 4]
    # y1 = [1, 4, 9, 16]
    # y2 = [1, 3, 6, 10]
    #
    # colorvec = ["#D08082","#C89FBF","#62ABC7","#7A7DB1",'#6FB494']
    # colorvec2 = ['#9FA9C9','#D36A6A']
    #
    # # 使用Hex颜色代码
    # for i in range(10):
    # plt.plot(x, y1, color="#D08082", label="Line 1")  # 橙色
    # plt.plot(x, y2, color="#C89FBF", label="Line 2")  # 绿色
    #
    # # 添加标题和图例
    # plt.title("Hex Color Example")
    # plt.legend()
    # plt.show()

    """
    Plot the heatmap for the precision and recall
    """
    # plot_heatmap_precision(4)

