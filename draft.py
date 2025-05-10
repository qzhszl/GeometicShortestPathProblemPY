def plot_distribution_10000node(N, ED, beta):
    """
    Compared maximum, minimum, average deviation with randomly selected nodes
    :return:
    """
    # Nvec = [20,50,100,1000]
    # # Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    # beta = 8
    if N < 200:
        ave_deviation_vec, max_deviation_vec, min_deviation_vec, ran_deviation_vec, _ = load_resort_data_smallN_maxminave(
            N, ED, beta)
    elif N < 10000:
        ave_deviation_vec, max_deviation_vec, min_deviation_vec, ran_deviation_vec, _ = load_large_network_results_maxminave(
            N, ED, beta)
    else:
        ave_deviation_vec, max_deviation_vec, min_deviation_vec, ran_deviation_vec, _ = load_10000nodenetwork_maxminave(
            ED, beta)

    # cuttail = [9,19,34,24]
    # peakcut = [9,5,5,5]

    data1 = ave_deviation_vec
    # data1 = [0,0,0]
    data2 = max_deviation_vec
    data3 = min_deviation_vec
    data4 = ran_deviation_vec

    fig, ax = plt.subplots(figsize=(8, 4.5))

    datasets = [data1,data2,data3,data4]
    # colors = [[0, 0.4470, 0.7410],
    #           [0.8500, 0.3250, 0.0980],
    #           [0.9290, 0.6940, 0.1250],
    #           [0.4940, 0.1840, 0.5560]]
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    # colorvec2 = ['#9FA9C9', '#D36A6A']

    labels = ["Ave","Max","Min","Ran"]
    for data, color, label in zip(datasets, colors, labels):
        hvalue, bin_vec = np.histogram(data, bins=60, density=True)
        print(bin_vec[1:len(bin_vec)])
        plt.plot(bin_vec[1:len(bin_vec)], hvalue, color=color, label=label, linewidth=10)

    text = r"$N = 10^4$, $\beta = {beta}$, $E[D] = {ED}$".format(beta=beta, ED=5)
    ax.text(
        0.5, 0.85,  # 文本位置（轴坐标，0.5 表示图中央，1.05 表示轴上方）
        text,
        transform=ax.transAxes,  # 使用轴坐标
        fontsize=26,  # 字体大小
        ha='center',  # 水平居中对齐
        va='bottom'  # 垂直对齐方式
    )


    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.spines['left'].set_position(('data', 0))
    # ax.spines['bottom'].set_position(('data', 0))
    # plt.xscale('log')
    plt.yscale('log')

    plt.xlim([0,1])

    ymin = 0.001  # 设置最低点
    current_ylim = ax.get_ylim()  # 获取当前的 y 轴范围
    ax.set_ylim(ymin, current_ylim[1])  # 保持最大值不变
    # plt.yticks([0,5,10,15,20,25])
    # plt.yticks([0, 10, 20, 30, 40, 50])

    plt.xlabel(r'x',fontsize = 32)
    plt.ylabel(r'$f_{d(q,\gamma(i,j))}(x)$',fontsize = 32)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)

    plt.legend(fontsize=26, handlelength=1, handletextpad=0.5, frameon=False,loc='right',bbox_to_anchor=(1.04, 0.54))
    plt.tick_params(axis='both', which="both",length=6, width=1)
    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\DistributionN{Nn}ED{EDn}Beta{betan}logy.pdf".format(Nn = N, EDn = ED, betan=beta)
    # plt.savefig(picname,format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    # plt.close()