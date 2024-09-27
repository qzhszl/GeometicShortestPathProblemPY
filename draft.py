# for 10000 node
betavec = [2.55, 3.2, 3.99, 5.15, 7.99, 300]
beta = betavec[beta_index]
cc = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
C_G = cc[beta_index]

kvec = list(range(2, 20)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
ED_bound = input_ED * 0.05

min_ED = 1
max_ED = N - 1
count = 0
rg = RandomGenerator(-12)
G, Coorx, Coory = R2SRGG(N, input_ED, beta, rg)
real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
ED = input_ED
while abs(input_ED - real_avg) > ED_bound and count < 20:
    count = count + 1
    if input_ED - real_avg > 0:
        min_ED = ED
        ED = min_ED + 0.5 * (max_ED - min_ED)
    else:
        max_ED = ED
        ED = min_ED + 0.5 * (max_ED - min_ED)
        pass
    G, Coorx, Coory = R2SRGG(N, ED, beta, rg)
    real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)

print("input para:", (N, input_ED, beta))
print("real ED:", real_avg)
print("clu:", nx.average_clustering(G))
components = list(nx.connected_components(G))
largest_component = max(components, key=len)
print("LCC", len(largest_component))
if count < 20:
    FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\cleanwithEDCC\\network_N{Nn}ED{EDn}CC{betan}.txt".format(
        Nn=N, EDn=input_ED, betan=C_G)
    nx.write_edgelist(G, FileNetworkName)

    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\cleanwithEDCC\\network_coordinates_N{Nn}ED{EDn}CC{betan}.txt".format(
        Nn=N, EDn=input_ED, betan=C_G)
    with open(FileNetworkCoorName, "w") as file:
        for data1, data2 in zip(Coorx, Coory):
            file.write(f"{data1}\t{data2}\n")