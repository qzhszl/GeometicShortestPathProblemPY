# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 21-10-2024
"""
import numpy as np
import scipy.integrate as integrate
import math
import matplotlib.pyplot as plt
import networkx as nx

def integrand(x, alpha, beta):
    prob = 1 / (1 + math.exp(beta * math.log(alpha * x)))
    return prob

def average_distance_withEDbeta(avg,beta):
    R = 2.0  # manually tuned value
    alpha = (2 * 10000 / avg * R * R) * (math.pi / (math.sin(2 * math.pi / beta) * beta))
    alpha = math.sqrt(alpha)
    result, error = integrate.quad(integrand, 0, 1, args=(alpha, beta))
    return result

def plot_average_distance():
    avg_vec = list(range(2, 20)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    beta_vec = [2.2,4,8,16,32,64,128]
    datadic = {}
    for beta in beta_vec:
        avg_geodistance_link_vec_for_abeta = []
        for avg in avg_vec:
            avg_dis = average_distance_withEDbeta(avg,beta)
            avg_geodistance_link_vec_for_abeta.append(avg_dis)
        datadic[beta] = avg_geodistance_link_vec_for_abeta

    colors = [[0, 0.4470, 0.7410],
              [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250],
              [0.4940, 0.1840, 0.5560],
              [0.4660, 0.6740, 0.1880],
              [0.3010, 0.7450, 0.9330]]
    legend = [r"$\beta=2.2$",r"$\beta=2^2$",r"$\beta=2^3$",r"$\beta=2^4$",r"$\beta=2^5$",r"$\beta=2^6$",r"$\beta=2^7$"]

    count = 0
    for count in range(len(beta_vec)-1):
        beta = beta_vec[count]
        x = avg_vec
        y = datadic[beta]
        plt.plot(x, y, linewidth=3, marker='o',markersize=10, label=legend[count], color=colors[count])


    plt.xscale('log')
    plt.xlabel('Expected degree, E[D]',fontsize = 26)
    plt.ylabel('Average geodistance',fontsize = 26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=20,loc="upper left")
    plt.tick_params(axis='both', which="both",length=6, width=1)


    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ave_geodistance_link_vs_ED_withdiff_beta.pdf".format()
    plt.savefig(picname,format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()


def plot_average_distance_VS_beta():
    avg_vec = [2.2,2.4,2.6,2.8]+ list(range(3, 20)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    beta_vec = [2,5,10,20,50,100]
    datadic = {}
    for beta in beta_vec:
        avg_geodistance_link_vec_for_abeta = []
        for avg in avg_vec:
            avg_dis = average_distance_withEDbeta(beta,avg)
            avg_geodistance_link_vec_for_abeta.append(avg_dis)
        datadic[beta] = avg_geodistance_link_vec_for_abeta

    colors = [[0, 0.4470, 0.7410],
              [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250],
              [0.4940, 0.1840, 0.5560],
              [0.4660, 0.6740, 0.1880],
              [0.3010, 0.7450, 0.9330]]
    legend = [r"$ED=2$",r"$ED = 5$",r"$ED = 10$",r"$ED = 20$",r"$ED = 50$", r"$ED = 100$"]

    count = 0
    for count in range(len(beta_vec)):
        beta = beta_vec[count]
        x = avg_vec
        y = datadic[beta]
        plt.plot(x, y, linewidth=3, marker='o',markersize=10, label=legend[count], color=colors[count])


    plt.xscale('log')
    plt.xlabel(r'$beta$',fontsize = 26)
    plt.ylabel('Average geodistance of link',fontsize = 26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=20,loc="upper left")
    plt.tick_params(axis='both', which="both",length=6, width=1)


    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ave_geodistance_link_vs_beta_withdiff_ed.pdf".format()
    plt.savefig(picname,format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()


def plot_simu_linkdistance_withavg():
    avg_vec = list(range(2, 20)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    beta_vec = [2.2, 4, 8, 16, 32, 64, 128]
    N = 10000
    ExternalSimutime = 0
    data_dic ={}
    error_dic = {}

    for beta in beta_vec:
        average_weight_vec =[]
        error_vec = []
        for ED in avg_vec:
            ave_geodistance_link_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ave_distance_link_radius\\linkweight_N{Nn}ED{EDn}beta{betan}Simu{ST}.txt".format(
                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
            G  = nx.read_edgelist(ave_geodistance_link_name, nodetype=int, data=(("weight", float),))
            linkweight =[]
            for u, v, data in G.edges(data=True):
                linkweight.append(data['weight'])
            average_weight_vec.append(np.mean(linkweight))
            error_vec.append(np.std(linkweight))
        data_dic[beta] = average_weight_vec
        error_dic[beta] = error_vec

    legend = [r"$\beta=2.2$",r"$\beta=2^2$",r"$\beta=2^3$",r"$\beta=2^4$",r"$\beta=2^5$",r"$\beta=2^6$",r"$\beta=2^7$"]
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = [[0, 0.4470, 0.7410],
              [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250],
              [0.4940, 0.1840, 0.5560],
              [0.4660, 0.6740, 0.1880],
              [0.3010, 0.7450, 0.9330],
              [0.6350, 0.0780, 0.1840]
              ]
    for count in range(len(beta_vec)):
        beta = beta_vec[count]
        x = avg_vec
        # print(len(x))
        # x = x[0:cuttail[N_index]]
        y = data_dic[beta]
        # y = y[0:cuttail[N_index]]
        error = error_dic[beta]
        # error = error[0:cuttail[N_index]]
        plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o',markersize=16, label=legend[count], color=colors[count])

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Expected degree, E[D]',fontsize = 26)
    plt.ylabel('Average geometirc distance of link',fontsize = 26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=20,loc="upper left")
    plt.tick_params(axis='both', which="both",length=6, width=1)

    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ave_distance_link_radius\\linkdistance_withavg_diffbeta.pdf"

    plt.savefig(picname,format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()


def plot_simu_linkdistance_withbeta():
    avg_vec = [2, 5, 10, 20, 50, 100]
    beta_vec = [2.2, 2.4, 2.6, 2.8] + list(range(3, 16))
    N = 10000
    ExternalSimutime = 0
    data_dic ={}
    error_dic = {}

    for ED in avg_vec:
        average_weight_vec =[]
        error_vec = []
        for beta in beta_vec:
            ave_geodistance_link_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ave_distance_link_radius\\linkweight_N{Nn}ED{EDn}beta{betan}Simu{ST}.txt".format(
                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
            G  = nx.read_edgelist(ave_geodistance_link_name, nodetype=int, data=(("weight", float),))
            linkweight =[]
            for u, v, data in G.edges(data=True):
                linkweight.append(data['weight'])
            average_weight_vec.append(np.mean(linkweight))
            error_vec.append(np.std(linkweight))
        data_dic[ED] = average_weight_vec
        error_dic[ED] = error_vec

    legend = [r"$ED=2$",r"$ED = 5$",r"$ED = 10$",r"$ED = 20$",r"$ED = 50$", r"$ED = 100$"]
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = [[0, 0.4470, 0.7410],
              [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250],
              [0.4940, 0.1840, 0.5560],
              [0.4660, 0.6740, 0.1880],
              [0.3010, 0.7450, 0.9330],
              [0.6350, 0.0780, 0.1840]
              ]
    for count in range(len(avg_vec)):
        ED = avg_vec[count]
        x = beta_vec
        # print(len(x))
        # x = x[0:cuttail[N_index]]
        y = data_dic[ED]
        # y = y[0:cuttail[N_index]]
        error = error_dic[ED]
        # error = error[0:cuttail[N_index]]
        plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o',markersize=16, label=legend[count], color=colors[count])

    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel(r'$\beta$',fontsize = 26)
    plt.ylabel('Average geometirc distance of link',fontsize = 26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=20,loc="upper right")
    plt.tick_params(axis='both', which="both",length=6, width=1)

    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ave_distance_link_radius\\linkdistance_withbeta_diffavg.pdf"

    plt.savefig(picname,format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()


def plot_simu_radius_withavg():
    avg_vec = list(range(2, 20)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    beta_vec = [2.2, 4, 8, 16, 32, 64, 128]
    N = 10000
    ExternalSimutime = 0
    data_dic ={}
    error_dic = {}

    for beta in beta_vec:
        average_weight_vec =[]
        error_vec = []
        for ED in avg_vec:
            max_weights = {}
            radius_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ave_distance_link_radius\\radius_N{Nn}ED{EDn}beta{betan}Simu{ST}.txt".format(
                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
            with open(radius_name, "r") as f:
                for line in f:
                    # 移除换行符，并按": "拆分每行
                    key, value = line.strip().split(": ")
                    # 将键转换为整数，值转换为浮点数，并添加到字典中
                    max_weights[int(key)] = float(value)
            linkweight = [v for v in max_weights.values()]
            average_weight_vec.append(np.mean(linkweight))
            error_vec.append(np.std(linkweight))
        data_dic[beta] = average_weight_vec
        error_dic[beta] = error_vec

    legend = [r"$\beta=2.2$",r"$\beta=2^2$",r"$\beta=2^3$",r"$\beta=2^4$",r"$\beta=2^5$",r"$\beta=2^6$",r"$\beta=2^7$"]
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = [[0, 0.4470, 0.7410],
              [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250],
              [0.4940, 0.1840, 0.5560],
              [0.4660, 0.6740, 0.1880],
              [0.3010, 0.7450, 0.9330],
              [0.6350, 0.0780, 0.1840]]
    for count in range(len(beta_vec)):
        beta = beta_vec[count]
        x = avg_vec
        # print(len(x))
        # x = x[0:cuttail[N_index]]
        y = data_dic[beta]
        # y = y[0:cuttail[N_index]]
        error = error_dic[beta]
        # error = error[0:cuttail[N_index]]
        plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o',markersize=16, label=legend[count], color=colors[count])

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Expected degree, E[D]',fontsize = 26)
    plt.ylabel('Average radius',fontsize = 26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=20,loc="upper left")
    plt.tick_params(axis='both', which="both",length=6, width=1)

    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ave_distance_link_radius\\radius_withavg_diffbeta.pdf"

    plt.savefig(picname,format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()

def plot_simu_radius_withbeta():
    avg_vec = [2, 5, 10, 20, 50, 100]
    beta_vec = [2.2, 2.4, 2.6, 2.8] + list(range(3, 16))
    N = 10000
    ExternalSimutime = 0
    data_dic ={}
    error_dic = {}

    for ED in avg_vec:
        average_weight_vec =[]
        error_vec = []
        for beta in beta_vec:
            max_weights = {}
            radius_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ave_distance_link_radius\\radius_N{Nn}ED{EDn}beta{betan}Simu{ST}.txt".format(
                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
            with open(radius_name, "r") as f:
                for line in f:
                    # 移除换行符，并按": "拆分每行
                    key, value = line.strip().split(": ")
                    # 将键转换为整数，值转换为浮点数，并添加到字典中
                    max_weights[int(key)] = float(value)
            linkweight = [v for v in max_weights.values()]

            average_weight_vec.append(np.mean(linkweight))
            error_vec.append(np.std(linkweight))
        data_dic[ED] = average_weight_vec
        error_dic[ED] = error_vec

    legend = [r"$ED=2$",r"$ED = 5$",r"$ED = 10$",r"$ED = 20$",r"$ED = 50$", r"$ED = 100$"]
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = [[0, 0.4470, 0.7410],
              [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250],
              [0.4940, 0.1840, 0.5560],
              [0.4660, 0.6740, 0.1880],
              [0.3010, 0.7450, 0.9330],
              [0.6350, 0.0780, 0.1840]
              ]
    for count in range(len(avg_vec)):
        ED = avg_vec[count]
        x = beta_vec
        # print(len(x))
        # x = x[0:cuttail[N_index]]
        y = data_dic[ED]
        # y = y[0:cuttail[N_index]]
        error = error_dic[ED]
        # error = error[0:cuttail[N_index]]
        plt.errorbar(x, y, yerr=error, linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o',markersize=16, label=legend[count], color=colors[count])

    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel(r'$\beta$',fontsize = 26)
    plt.ylabel('Average radius',fontsize = 26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.title('Errorbar Curves with Minimum Points after Peak')
    plt.legend(fontsize=20,loc="upper right")
    plt.tick_params(axis='both', which="both",length=6, width=1)

    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ave_distance_link_radius\\radius_withbeta_diffavg.pdf"

    plt.savefig(picname,format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """
    step1: plot average distance of a link change with avg(analytic result)
    """
    # plot_average_distance()
    """
    step2: plot average distance of a link change with beta(analytic result)
    """
    # plot_average_distance_VS_beta()

    """
    step3: plot average distance of a link change with beta
    """
    plot_simu_linkdistance_withavg()


    # """
    # step4: plot average distance of a link change with beta
    # """
    # plot_simu_linkdistance_withbeta()

    # """
    # step5: plot average distance of a link change with beta
    # """
    # plot_simu_radius_withavg()

    # """
    # step6: plot average distance of a link change with beta
    # """
    # plot_simu_radius_withbeta()