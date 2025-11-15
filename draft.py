import math
import random
import numpy as np

from R2SRGG.distribution.deviatiton_vs_aveL_diffNkbeta_SRGG import compute_edge_Euclidean_length
from SphericalSoftRandomGeomtricGraph import RandomGenerator
import networkx as nx
import matplotlib.pyplot as plt
from main import find_k_connected_node_pairs



def R2SRGG_withlinkweight(N, avg, beta, rg, Coorx=None, Coory=None, SaveNetworkPath=None):
    """
    Program generates Soft Random Geometric Graph on a 2d unit square
    Connection probability function is (1 + (a*d)^{b})^{-1}
    :param N: number of nodes
    :param avg: <k>
    :param beta: beta (controlling clustering)
    :param rg:  random seed
    :param Coorx: coordinates of x
    :param Coory: coordinates of y
    :param SaveNetworkPath:
    :return: R2SRGG_withlinkweight: the link weights are the Euclidean distance
    """
    # calculate parameters
    assert beta > 2
    assert avg > 0
    assert N > 1

    R = 2.0  # manually tuned value
    alpha = (2 * N / avg * R * R) * (math.pi / (math.sin(2 * math.pi / beta) * beta))
    alpha = math.sqrt(alpha)
    s = []
    t = []
    linkweight =[]

    # Assign coordinates
    if Coorx is not None and Coory is not None:
        xx = Coorx
        yy = Coory
    else:
        xx = []
        yy = []
        for i in range(N):
            xx.append(rg.ran1())
            yy.append(rg.ran1())

    # make connections
    for i in range(N):
        for j in range(i + 1, N):
            dist = math.sqrt((xx[i] - xx[j]) ** 2 + (yy[i] - yy[j]) ** 2)
            assert dist > 0

            try:
                prob = 1 / (1 + math.exp(beta * math.log(alpha * dist)))
            except:
                prob = 0
            if rg.ran1() < prob:
                s.append(i)
                t.append(j)
                linkweight.append(dist)



    if SaveNetworkPath is not None:
        with open(SaveNetworkPath, "w") as file:
            for nodei, nodej, dist in zip(s, t, linkweight):
                file.write(f"{nodei}\t{nodej}\t{dist}\n")


    # Create graph and remove self-loops
    G = nx.Graph()
    # G.add_edges_from(zip(s, t,linkweight))
    for nodei, nodej, dist in zip(s, t, linkweight):
        G.add_edge(nodei, nodej, weight=dist)

    max_edge_weight_per_node = {}

    for node in G.nodes():
        weights = [d["weight"] for _, _, d in G.edges(node, data=True)]
        if weights:  # 避免孤立节点报错
            max_edge_weight_per_node[node] = max(weights)
        else:
            max_edge_weight_per_node[node] = 0

    # 2. 求所有节点最大权重的平均值
    average_max_weight = sum(max_edge_weight_per_node.values()) / nx.number_of_nodes(G)

    if G.number_of_nodes()<N:
        ExpectedNodeList = [i for i in range(0, N)]
        Nodelist = list(G.nodes)
        difference = [item for item in ExpectedNodeList if item not in Nodelist]
        G.add_nodes_from(difference)

    return G,linkweight,average_max_weight, xx, yy, max_edge_weight_per_node.values()





rg = RandomGenerator(-12)
for i in range(random.randint(1, 1000)):
    rg.ran1()



real_ave_degree = []
# LCC_num = []
# clustering_coefficient = []
count_vec = []
ave_graphlinklength_vec = []
# std_length_edge_vec = []

# For each node pair:
ave_deviation = []
max_deviation = []
min_deviation = []
ave_baseline_deviation = []
length_geodesic = []
SP_hopcount = []
max_dev_node_hopcount = []
SPnodenum_vec = []
ave_edge_length = []
length_edge_vec = []


N = 10000
ED = 6
beta = 2.1
G, linkweight_vec, average_max_weight, xx, yy,max_linkweight = R2SRGG_withlinkweight(N, ED, beta, rg)

pos1 = {i: (xx[i], yy[i]) for i in range(N)}

plt.figure()
plt.hist(linkweight_vec, bins=40,density=True, edgecolor='black')   # 直方图，30 个柱子
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram link weight")
plt.show()


plt.figure()
plt.hist(max_linkweight, bins=40,density=True, edgecolor='black')   # 直方图，30 个柱子
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram max link weight")
plt.show()




print(np.mean(linkweight_vec))
print(average_max_weight)



real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)

print("real ED:", real_avg)
real_ave_degree.append(real_avg)

# compute the edge length of the networks
ave_graphlinklength_vec.append(np.mean(linkweight_vec))

# ave_clu = nx.average_clustering(G)
# print("clu:", ave_clu)
# clustering_coefficient.append(ave_clu)
# components = list(nx.connected_components(G))
# largest_component = max(components, key=len)
# LCC_number = len(largest_component)
# print("LCC", LCC_number)
# LCC_num.append(LCC_number)

# pick up all the node pairs in the LCC and save them in the unique_pairs
nodepair_num = 1
unique_pairs = find_k_connected_node_pairs(G, nodepair_num)
count = 0


for node_pair in unique_pairs:
    nodei = node_pair[0]
    nodej = node_pair[1]
    # Find the shortest path nodes
    SPNodelist = nx.shortest_path(G, nodei, nodej)
    path_edges = list(zip(SPNodelist[:-1], SPNodelist[1:]))
    SPnodenum = len(SPNodelist) - 2
    # SPnodenum_vec.append(SPnodenum)
    G1 = nx.Graph()
    G1.add_nodes_from(SPNodelist)
    pos = {i: (xx[i], yy[i]) for i in SPNodelist}

    # hopcount of the SP
    SP_hopcount.append(SPnodenum + 1)

    print(SPnodenum + 1)
    if SPnodenum + 1 > 0:  # for deviation, we restrict ourself for hopcount>2: SPnodenum+1 > 1, but not for the stretch(hop =1 are also included:SPnodenum+1 > 0)
        # compute the length of the edges on the shortest path
        length_edge_for_anodepair = []
        shortest_path_edges = list(zip(SPNodelist[:-1], SPNodelist[1:]))
        for (nodes, nodet) in shortest_path_edges:
            d_E = compute_edge_Euclidean_length(nodes, nodet, xx, yy)
            print(d_E)
            neighbour_dist_list = [compute_edge_Euclidean_length(nodes, nodex, xx, yy) for nodex in nx.neighbors(G,nodes)]
            print(neighbour_dist_list)
            print(f"max_d{max(neighbour_dist_list)}")

            G1.add_edge(nodes, nodet)


            length_edge_for_anodepair.append(d_E)
        length_edge_vec = length_edge_vec + length_edge_for_anodepair
        ave_edge_length.append(np.mean(length_edge_for_anodepair))

        # print(f"realS:{np.mean(length_edge_for_anodepair)*(SPnodenum + 1)}")
        # print(f"realS2:{np.sum(length_edge_for_anodepair)}")




SP_hopcount = np.array(SP_hopcount)
ave_edge_length = np.array(ave_edge_length)



hop_vec_no1 = SP_hopcount[SP_hopcount != 1]

if len(ave_edge_length) != len(hop_vec_no1):
    ave_edgelength_for_a_para_comb_no1 = ave_edge_length[SP_hopcount != 1]
    # if L = <d_e>h real stretch
    L = [x * y for x, y in zip(ave_edgelength_for_a_para_comb_no1, hop_vec_no1)]
else:
    L = [x * y for x, y in zip(ave_edge_length, hop_vec_no1)]


print(f"hop:{np.mean(SP_hopcount)}")
print(f"approxL:{np.mean(SP_hopcount)*average_max_weight}")

print(f"real:{np.mean(L)}")


plt.figure()
nx.draw(G, pos1,
        with_labels=False,       # 显示节点标签（1,2,3）
        node_size=3,
        node_color='lightblue')
plt.axis('equal')               # 坐标比例一致

nx.draw_networkx_edges(
    G, pos1,
    edgelist=path_edges,
    width=1,              # 粗一点
    edge_color='red',     # 红色高亮
)

# ③ 可选：高亮路径上的“点”
nx.draw_networkx_nodes(
    G, pos1,
    nodelist=SPNodelist,
    node_size=3,
    node_color='red'
)

plt.show()


