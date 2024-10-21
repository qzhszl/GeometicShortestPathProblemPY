# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 21-10-2024
"""

import scipy.integrate as integrate
import math
import matplotlib.pyplot as plt

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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # plot_average_distance()

    plot_average_distance_VS_beta()
