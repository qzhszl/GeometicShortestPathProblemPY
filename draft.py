from R2SRGG.R2SRGG import loadSRGGandaddnode
import networkx as nx

ED = 5
beta = 4
noise_amplitude = 0.001
N = 10000

ExternalSimutime = 0

FileNetworkName = "NetworkwithNoiseED{EDn}Beta{betan}Noise{no}PYSimu{ST}networksize{N}.txt".format(
            EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime, N = N)

H = loadSRGGandaddnode(N, FileNetworkName)

real_avg = 2 * nx.number_of_edges(H) / nx.number_of_nodes(H)
print("real ED:", real_avg)