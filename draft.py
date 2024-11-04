real_ave_degree_dict = {}
    ave_deviation_dict = {}
    std_deviation_dict = {}
    Nvec = [10,100,1000,10000]
    # Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    # beta = 8
    for N in Nvec:
        if N < 200:
            degree_vec_resort, ave_deviation_resort, std_deviation_resort, _, _, _=load_resort_data(N,beta)
            real_ave_degree_dict[N] = degree_vec_resort
            ave_deviation_dict[N] = ave_deviation_resort
            std_deviation_dict[N] = std_deviation_resort
        # elif N < 200:
        #     for beta in [4]:
        #         real_ave_degree_vec, ave_deviation_vec, std_deviation_vec = load_small_network_results(N,beta)
        #         real_ave_degree_dict[N] = real_ave_degree_vec
        #         ave_deviation_dict[N] = ave_deviation_vec
        #         std_deviation_dict[N] = std_deviation_vec
        elif N < 10000:
            for beta in [beta]:
                real_ave_degree_vec, ave_deviation_vec, std_deviation_vec = load_large_network_results(N, beta)
                real_ave_degree_dict[N] = real_ave_degree_vec
                ave_deviation_dict[N] = ave_deviation_vec
                std_deviation_dict[N] = std_deviation_vec
        else:
            for beta in [beta]:
                real_ave_degree_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\real_ave_degree_Beta{betan}.txt".format(
                    betan=beta)
                real_ave_degree_vec = np.loadtxt(real_ave_degree_Name)
                ave_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\ave_deviation_Beta{betan}.txt".format(
                    betan=beta)
                ave_deviation_vec = np.loadtxt(ave_deviation_Name)
                std_deviation_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\10000node\\std_deviation_Beta{betan}.txt".format(
                    betan=beta)
                std_deviation_vec = np.loadtxt(std_deviation_Name)
                # real_ave_degree_vec, ave_deviation_vec, std_deviation_vec = load_10000nodenetwork_results(beta)
                real_ave_degree_dict[N] = real_ave_degree_vec
                ave_deviation_dict[N] = ave_deviation_vec
                std_deviation_dict[N] = std_deviation_vec

    lengend = [r"$N=10$",r"$N=10^2$",r"$N=10^3$",r"$N=10^4$"]
    fig, ax = plt.subplots(figsize=(9, 6))

    # colors = [[0.3059, 0.4745, 0.6549],
    #           [0.9490, 0.5569, 0.1686],
    #           [0.8824, 0.3412, 0.3490],
    #           [0.4627, 0.7176, 0.6980],
    #           [0.9294, 0.7882, 0.2824],
    #           [0.6902, 0.4784, 0.6314],
    #           [1.0000, 0.6157, 0.6549],
    #           [0.6118, 0.4588, 0.3725],
    #           [0.7294, 0.6902, 0.6745]]
    colors = [[0, 0.4470, 0.7410],
              [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250],
              [0.4940, 0.1840, 0.5560],
              [0.4660, 0.6740, 0.1880]]
    cuttail = [5,34,23,23]
    # peakcut = [9,5,5,5,5]
    for N_index in range(len(Nvec)):
        N = Nvec[N_index]
        if N==100:
            x = real_ave_degree_dict[N]
            print(len(x))
            x = x[0:cuttail[N_index]]
            print(x)
            y = ave_deviation_dict[N]
            y = y[0:cuttail[N_index]]
            error = std_deviation_dict[N]
            error = error[0:cuttail[N_index]]

            filter_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16,  18,  20,  22,  24,  26,  28,
             30]
            x = [x[a] for a in filter_index]
            y = [y[a] for a in filter_index]
            print(x)
            print(y)
            error = [error[a] for a in filter_index]
        elif N>100:
            x = real_ave_degree_dict[N]
            print(len(x))
            x = x[0:cuttail[N_index]]
            y = ave_deviation_dict[N]
            y = y[0:cuttail[N_index]]
            print(x)
            print(y)
            error = std_deviation_dict[N]
            error = error[0:cuttail[N_index]]
        else:
            x = real_ave_degree_dict[N]
            print(len(x))
            x = x[1:cuttail[N_index]]
            y = ave_deviation_dict[N]
            y = y[1:cuttail[N_index]]
            error = std_deviation_dict[N]
            error = error[1:cuttail[N_index]]
        plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o',markersize=16, label=lengend[N_index], color=colors[N_index])

        # # 找到峰值后最低点的坐标
        # peak_index = np.argmax(y[0:peakcut[N_index]])
        # post_peak_y = y[peak_index:]
        # post_peak_min_index = peak_index + np.argmin(post_peak_y)
        # post_peak_min_x = x[post_peak_min_index]
        # post_peak_min_y = y[post_peak_min_index]

        # 标出最低点
        # plt.plot(post_peak_min_x, post_peak_min_y, 'o', color=colors[N_index], markersize=16, markerfacecolor="none")

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    plt.ylim(0,0.30)
    plt.yticks([0,0.1,0.2,0.3])

    plt.xscale('log')
    plt.xlabel('Expected degree, E[D]',fontsize = 26)
    plt.ylabel('Average deviation',fontsize = 26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=20,loc=(0.68,0.58))
    plt.tick_params(axis='both', which="both",length=6, width=1)
    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\LocalOptimumBeta{betan}2.pdf".format(
        betan=beta)
    # plt.savefig(picname,format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()