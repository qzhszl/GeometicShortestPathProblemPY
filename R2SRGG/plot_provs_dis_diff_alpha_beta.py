# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 16-10-2024
"""
import math
import matplotlib.pyplot as plt

def probability_with_diff_beta(avg, beta):
    prob_vec =[]
    for dist in range(1,1001):
        dist = dist*0.001
        N = 10000
        R = 2.0  # manually tuned value
        alpha = (2 * N / avg * R * R) * (math.pi / (math.sin(2 * math.pi / beta) * beta))
        alpha = math.sqrt(alpha)
        try:
            prob = 1 / (1 + math.exp(beta * math.log(alpha * dist)))
        except:
            prob = 0
        prob_vec.append(prob)
    return [0.01*i for i in range(1,1001)],prob_vec

def plot_probability_with_diff_beta(avg):
    count = 0
    beta_vec = [2.2,4,8,16,32,1024]
    legend = [r"$\beta=2.2$", r"$\beta=2^2$", r"$\beta=2^3$", r"$\beta=2^4$", r"$\beta=2^5$", r"$\beta=2^{10}$"]
    for beta in beta_vec:
        x,y = probability_with_diff_beta(avg, beta)
        plt.plot(x,y,linewidth=3,label=legend[count])
        count = count+1
    plt.xscale('log')
    plt.xlabel('distance x', fontsize=26)
    plt.ylabel('Probability P(x)', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    titlename = r"$avg={beta}$".format(
        beta=avg)
    plt.title(titlename,fontsize=26)
    plt.legend(fontsize=20, loc="upper right")
    plt.tick_params(axis='both', which="both", length=6, width=1)
    # picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\Provsdis_avg{avg}.pdf".format(
    #     avg=avg)
    # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)

    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\Provsdis_avg{avg}.png".format(
        avg=avg)
    plt.savefig(picname, format='png', bbox_inches='tight', dpi=600)
    # plt.show()
    plt.close()


def plot_probability_with_diff_avg(beta):
    count = 0
    avg_vec = [2,4,8,16,32,64,128]
    legend = [r"$avg=2$", r"$avg=2^2$", r"$avg=2^3$", r"$avg=2^4$", r"$avg=2^5$", r"$avg=2^{6}$", r"$avg=2^{7}$"]
    for avg in avg_vec:
        x,y = probability_with_diff_beta(avg, beta)
        plt.plot(x,y,linewidth=3,label=legend[count])
        count = count+1
    plt.xscale('log')
    plt.xlabel('distance x', fontsize=26)
    plt.ylabel('Probability P(x)', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    titlename = r"$\beta={beta}$".format(
        beta=beta)
    plt.title(titlename,fontsize=26)
    plt.legend(fontsize=20, loc="upper right")
    plt.tick_params(axis='both', which="both", length=6, width=1)
    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\Provsdis_beta{avg}.pdf".format(
        avg=beta)
    plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    picname = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\Provsdis_beta{avg}.png".format(
        avg=beta)
    plt.savefig(picname, format='png', bbox_inches='tight', dpi=600)
    # plt.show()
    plt.close()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    for avg in [2,5,10,100]:
        plot_probability_with_diff_beta(avg)
    for beta in [2.2,4,8,128]:
        plot_probability_with_diff_avg(beta)
