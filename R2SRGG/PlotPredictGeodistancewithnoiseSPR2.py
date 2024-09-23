# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 6-6-2024
Generate the graph, remove links, blur node coordinates:  x = x + E(A), y = y + E(A),
where E ~ Unif(0,A), A is noise amplitude. Or do it in a more “kosher” way, uniformly place it within a 2D circle of radius A.

For the node pair ij:
	a) find shortest path nodes using distance to geodesic (with blurred node coordinates).
	b) find shortest path nodes by reconstructing the graph.

Use the same parameter combinations as before.
Vary noise magnitude A, see what happens to predictions.
It is for Euclidean soft random geometric graph
"""


import numpy as np
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt


def plot_predict_geodistance_Vs_reconstructionRGG_SRGG_withnoise_SP_R2_clu(Edindex, betaindex,legendpara):
    # plot PRECISION
    ED_list = [5, 10, 20, 40]  # Expected degrees
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

    for noise_amplitude in [0, 0.001,0.01,0.1, 1]:
        PrecisonRGG_specificnoise = []
        for ExternalSimutime in range(20):
            try:
                precision_RGG_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise{no2}\\PrecisionRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
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
                precision_SRGG_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise{no2}\\PrecisionSRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
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
                precision_Geodis_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise{no2}\\PrecisionGeodisED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
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
    # x_labels = ['0', '0.001', '0.01']
    x_labels = ['0', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$']

    # X axis positions for each bar group
    x = np.arange(len(x_labels))

    # Width of each bar
    width = 0.2

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)

    # Plotting the bars
    bar1 = ax.bar(x - width, y1, width, label='RGG',yerr=(y_error_lower,RGG_precision_list_all_std), capsize=5)
    bar2 = ax.bar(x, y2, width, label='SRGG', yerr=(y_error_lower,SRGG_precision_list_all_std), capsize=5)
    bar3 = ax.bar(x + width, y3, width, label='Deviation', yerr=(y_error_lower,Geo_precision_list_all_std), capsize=5)

    # Adding labels and title
    # ax.set_ylim(0,1.1)
    ax.set_xlabel(r'Noise amplitude, $\alpha$', fontsize = 25)
    ax.set_ylabel('Precision',fontsize = 25)
    # title_name = "beta:{beta_n}, E[D]:{ed_n}".format(ed_n=ED, beta_n = beta)
    # ax.set_title(title_name)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)

    if legendpara ==1:
        ax.legend(fontsize=22)
    ax.tick_params(direction='out')
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.tick_params(axis='both', which="both", length=6, width=1)
    # Display the plot
    # plt.show()
    figname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\PrecisionGeoVsRGGSRGGED{EDn}Beta{betan}N.pdf".format(
                EDn=ED, betan=beta)

    fig.savefig(figname, format='pdf', bbox_inches='tight', dpi=600)
    plt.close()
    print(exemptionlist)


def plot_predict_geodistance_Vs_reconstructionRGG_SRGG_withnoise_SP_R2_clu2(Edindex, betaindex,legendpara):
    # plot recall
    ED_list = [5, 10, 20, 40]  # Expected degrees
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
                precision_RGG_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise{no2}\\RecallRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
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
                precision_SRGG_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise{no2}\\RecallSRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
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
                precision_Geodis_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise{no2}\\RecallGeodisED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
                    EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime, no2=noise_amplitude)
                Precison_Geodis_5_times = np.loadtxt(precision_Geodis_Name)
                PrecisonGeodis_specificnoise.extend(Precison_Geodis_5_times)
            except:
                pass
        # nonzero_indices_geo = find_nonzero_indices(PrecisonRGG_specificnoise)
        # PrecisonRGG_specificnoise = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PrecisonRGG_specificnoise))
        # PrecisonRGG_specificnoise = [PrecisonRGG_specificnoise[x] for x in nonzero_indices_geo]
        Geo_precision_list_all_ave.append(np.mean(PrecisonRGG_specificnoise))
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

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # Plotting the bars
    bar1 = ax.bar(x - width, y1, width, label='RGG', yerr=(y_error_lower, RGG_precision_list_all_std), capsize=5)
    bar2 = ax.bar(x, y2, width, label='SRGG', yerr=(y_error_lower, SRGG_precision_list_all_std), capsize=5)
    bar3 = ax.bar(x + width, y3, width, label='Deviation', yerr=(y_error_lower, Geo_precision_list_all_std), capsize=5)

    # Adding labels and title
    ax.set_xlabel(r'Noise amplitude, $\alpha$',fontsize = 25)
    ax.set_ylabel('Recall',fontsize = 25)

    # title_name = "beta:{beta_n}, E[D]:{ed_n}".format(ed_n=ED, beta_n = beta)
    # ax.set_title(title_name)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    if legendpara == 1:
        ax.legend(fontsize=22)
    ax.tick_params(direction='out')
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.tick_params(axis='both', which="both", length=6, width=1)
    # Display the plot
    # plt.show()
    figname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\RecallGeoVsRGGSRGGED{EDn}Beta{betan}N.pdf".format(
        EDn=ED, betan=beta)
    plt.savefig(figname, format='pdf', bbox_inches='tight', dpi=600)
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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # # # STEP 3 plot the figure
    # for Edindex in range(4):
    #     for betaindex in range(7):
    #         plot_predict_geodistance_Vs_reconstructionRGG_SRGG_withnoise_SP_R2_clu(Edindex, betaindex,legendpara=0)
    #
    #
    # for Edindex in [0]:
    #     for betaindex in [1]:
    #         plot_predict_geodistance_Vs_reconstructionRGG_SRGG_withnoise_SP_R2_clu(Edindex, betaindex, legendpara=1)


    # STEP 3 plot the recall
    for Edindex in range(4):
        for betaindex in range(7):
            plot_predict_geodistance_Vs_reconstructionRGG_SRGG_withnoise_SP_R2_clu2(Edindex, betaindex,legendpara=0)

    for Edindex in [0]:
        for betaindex in [1]:
            plot_predict_geodistance_Vs_reconstructionRGG_SRGG_withnoise_SP_R2_clu2(Edindex, betaindex,legendpara=1)




