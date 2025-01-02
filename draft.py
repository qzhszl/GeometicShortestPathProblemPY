# plot PRECISION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    # Plotting the bars
    bar1 = ax.bar(x - width, y1, width, label='RGG',yerr=(y_error_lower,RGG_precision_list_all_std), capsize=5, color=colors[3])
    bar2 = ax.bar(x, y2, width, label='SRGG', yerr=(y_error_lower,SRGG_precision_list_all_std), capsize=5, color=colors[0])
    bar3 = ax.bar(x + width, y3, width, label='Deviation', yerr=(y_error_lower,Geo_precision_list_all_std), capsize=5, color=colors[2])

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
