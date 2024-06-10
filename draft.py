def plot_GeovsRGG_precsion_withnoise():
    PRAUC_matrix = np.zeros((2, 2))
    PRAUC_std_matrix = np.zeros((2, 2))
    PRAUC_fre_matrix = np.zeros((2, 2))
    PRAUC_fre_std_matrix = np.zeros((2, 2))
    noise_amplitude = 0.001
    for EDindex in [0, 1]:
        ED_list = [5, 20]  # Expected degrees
        ED = ED_list[EDindex]
        print("ED:", ED)

        for betaindex in [0, 1]:
            beta_list = [4, 100]
            beta = beta_list[betaindex]
            print(beta)
            PRAUC_list = []
            PRAUC_fre_list = []
            for ExternalSimutime in range(20):
                if ExternalSimutime not in [4,13]:
                    precision_Geodis_Name = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\RGG\\PrecisionGeodisED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
                        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
                    PRAUC_list_10times = np.loadtxt(precision_Geodis_Name)
                    PRAUC_list.extend(PRAUC_list_10times)

                    precision_fre_Name = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\RGG\\PrecisionRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
                        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
                    PRAUC_fre_list_10times = np.loadtxt(precision_fre_Name)
                    PRAUC_fre_list.extend(PRAUC_fre_list_10times)

            nonzero_indices_geo = find_nonzero_indices(PRAUC_list)
            # PRAUC_list = list(filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PRAUC_list))
            PRAUC_list = [PRAUC_list[x] for x in nonzero_indices_geo]

            print("lenpre", len(PRAUC_list))
            mean_PRAUC = np.mean(PRAUC_list)

            PRAUC_matrix[EDindex][betaindex] = mean_PRAUC
            PRAUC_std_matrix[EDindex][betaindex] = np.std(PRAUC_list)
            print(mean_PRAUC)

            # PRAUC_fre_list = list(
            #     filter(lambda x: not (math.isnan(x) if isinstance(x, float) else False), PRAUC_fre_list))
            PRAUC_fre_list = [PRAUC_fre_list[x] for x in nonzero_indices_geo]
            print(PRAUC_fre_list)
            print("lenPRE", len(PRAUC_fre_list))
            mean_fre_PRAUC = np.mean(PRAUC_fre_list)
            PRAUC_fre_matrix[EDindex][betaindex] = mean_fre_PRAUC
            PRAUC_fre_std_matrix[EDindex][betaindex] = np.std(PRAUC_list)
            print(mean_fre_PRAUC)

    plt.figure()
    df = pd.DataFrame(PRAUC_matrix,
                      index=[5, 20],  # DataFrame的行标签设置为大写字母
                      columns=[4, 100])  # 设置DataFrame的列标签
    sns.heatmap(data=df, vmin=0, vmax=0.8, annot=True, fmt=".2f", cbar=True,
                cbar_kws={'label': 'precision'})
    plt.title("Geo distance")
    plt.xlabel("beta")
    plt.ylabel("average degree")
    precision_Geodis_fig_Name = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\RGG\\PrecisionGeodisED{EDn}Beta{betan}Noise{no}PY.pdf".format(
        EDn=ED, betan=beta, no=noise_amplitude)
    plt.savefig(precision_Geodis_fig_Name,
        format='pdf', bbox_inches='tight', dpi=600)
    plt.close()

    plt.figure()
    df = pd.DataFrame(PRAUC_fre_matrix,
                      index=[5, 20],  # DataFrame的行标签设置为大写字母
                      columns=[4, 100])  # 设置DataFrame的列标签
    sns.heatmap(data=df, vmin=0, vmax=0.8, annot=True, fmt=".2f", cbar=True,
                cbar_kws={'label': 'precision'})
    plt.title("RGG")
    plt.xlabel("beta")
    plt.ylabel("average degree")

    precision_RGG_fig_Name = "D:\\data\\geometric shortest path problem\\SSRGG\\Noise\\RGG\\PrecisionRGGED{EDn}Beta{betan}Noise{no}PY.pdf".format(
        EDn=ED, betan=beta, no=noise_amplitude)
    plt.savefig(
        precision_RGG_fig_Name,
        format='pdf', bbox_inches='tight', dpi=600)
    plt.close()