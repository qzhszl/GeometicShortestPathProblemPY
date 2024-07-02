def predict_geodistance_Vs_reconstructionRGG_SRGG_withnoise_SP_R2_clu(Edindex, betaindex, noiseindex, ExternalSimutime):
    """
    :param Edindex: average degree
    :param betaindex: parameter to control the clustering coefficient
    :return: PRAUC control and test simu for diff ED and beta
    4 combination of ED and beta
    ED = 5 and 20 while beta = 4 and 100
    """
    N = 10000
    ED_list = [5, 20]  # Expected degrees
    ED = ED_list[Edindex]
    print("ED:", ED)

    beta_list = [4, 100]
    beta = beta_list[betaindex]
    print("beta:", beta)

    noise_amplitude_list = [0, 0.001, 0.01, 0.1, 0.5]
    noise_amplitude = noise_amplitude_list[noiseindex]
    print("noise amplitude:", noise_amplitude)

    Precision_Geodis_nodepair = []
    Recall_Geodis_nodepair = []
    Precision_RGG_nodepair = []  # save the precision_RGG for each node pair, we selected 100 node pair in total
    Recall_RGG_nodepair = []  # we selected 100 node pair in total
    Precision_SRGG_nodepair = []
    Recall_SRGG_nodepair = []

    SPnum_nodepair = []  # save the Number of nearly shortest path for each node pair
    geodistance_between_nodepair = []  # save the geodeisc length between each node pair

    random.seed(ExternalSimutime)
    rg = RandomGenerator(-12)
    for i in range(random.randint(0, 100)):
        rg.ran1()

    FileOriNetworkName = "/home/zqiu1/GSPP/SSRGGpy/R2/EuclideanSoftRGGnetwork/NetworkOriginalED{EDn}Beta{betan}Noise{no}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude)
    G = loadSRGGandaddnode(N, FileOriNetworkName)

    real_avg = 2*nx.number_of_edges(G)/nx.number_of_nodes(G)
    print("real ED:", real_avg)
    realradius = degree_vs_radius(N, real_avg)

    # load coordinates with noise
    Coorx = []
    Coory = []
    FileOriNetworkCoorName = "/home/zqiu1/GSPP/SSRGGpy/R2/EuclideanSoftRGGnetwork/CoorED{EDn}Beta{betan}Noise{no}mothernetwork.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude)
    with open(FileOriNetworkCoorName, "r") as file:
        for line in file:
            if line.startswith("#"):
                continue
            data = line.strip().split("\t")  # 使用制表符分割
            Coorx.append(float(data[0]))
            Coory.append(float(data[1]))

    nodepair_num = 5

    # Random select nodepair_num nodes in the largest connected component
    components = list(nx.connected_components(G))
    largest_component = max(components, key=len)
    nodes = list(largest_component)
    unique_pairs = set(tuple(sorted(pair)) for pair in itertools.combinations(nodes, 2))
    possible_num_nodepair = len(unique_pairs)
    if possible_num_nodepair > nodepair_num:
        random_pairs = random.sample(sorted(unique_pairs), nodepair_num)
    else:
        random_pairs = random.sample(sorted(unique_pairs), possible_num_nodepair)
    count = 0
    components = []
    largest_component = []
    nodes = []
    unique_pairs = []
    unique_pairs = []
    filename_selecetednodepair = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise\\SelecetedNodepairED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(filename_selecetednodepair, random_pairs, fmt="%i")

    for nodepair in random_pairs:
        count = count + 1
        print("Simunodepair:", count)
        nodei = nodepair[0]
        nodej = nodepair[1]

        tic = time.time()

        # Find shortest path nodes
        SPNodelist = all_shortest_path_node(G, nodei, nodej)
        SPnodenum = len(SPNodelist)

        SPnum_nodepair.append(SPnodenum)

        Predicted_truecase_num = SPnodenum
        toc = time.time() - tic
        print("SP finding time:", toc)
        print("SP num:", SPnodenum)

        # Create label array
        Label_med = np.zeros(N)
        Label_med[SPNodelist] = 1  # True cases

        thetaSource = Coorx[nodei]
        phiSource = Coory[nodei]
        thetaEnd = Coorx[nodej]
        phiEnd = Coory[nodej]
        geodistance_between_nodepair.append(distR2(thetaSource, phiSource, thetaEnd, phiEnd))

        Geodistance = {}
        for NodeC in range(N):
            if NodeC in [nodei, nodej]:
                Geodistance[NodeC] = 0
            else:
                thetaMed = Coorx[NodeC]
                phiMed = Coory[NodeC]
                dist,_ = dist_to_geodesic_R2(thetaMed, phiMed, thetaSource, phiSource, thetaEnd, phiEnd)
                Geodistance[NodeC] = dist

        # Generate an RGG with the coordinates and predict it
        SPNodeList_RGG = SPnodes_inRGG_with_coordinatesR2(N, real_avg, realradius,rg, Coorx, Coory, nodei, nodej)
        # toc2 = time.time() - toc
        # print("RGG generate time:", toc2)

        PredictNSPNodeList_RGG = np.zeros(N)
        PredictNSPNodeList_RGG[SPNodeList_RGG] = 1  # True cases

        precision_RGG = precision_score(Label_med, PredictNSPNodeList_RGG)
        recall_RGG = recall_score(Label_med, PredictNSPNodeList_RGG)

        # Store precision and recall values for RGG
        Precision_RGG_nodepair.append(precision_RGG)
        Recall_RGG_nodepair.append(recall_RGG)


        # Predict sp nodes use distance, where top Predicted_truecase_num nodes will be regarded as predicted nsp according to distance form the geodesic
        Geodistance = sorted(Geodistance.items(), key=lambda kv: (kv[1], kv[0]))
        Geodistance = Geodistance[:Predicted_truecase_num + 2]
        Top100closednode = [t[0] for t in Geodistance]
        Top100closednode = [n for n in Top100closednode if n not in [nodei, nodej]]
        NSPNodeList_Geo = np.zeros(N)
        NSPNodeList_Geo[Top100closednode] = 1  # True cases
        precision_Geo = precision_score(Label_med, NSPNodeList_Geo)
        recall_Geo = recall_score(Label_med, NSPNodeList_Geo)

        # Store precision and recall values
        Precision_Geodis_nodepair.append(precision_Geo)
        Recall_Geodis_nodepair.append(recall_Geo)


        # Predict sp nodes using reconstruction of SRGG
        node_fre = nodeSPfrequency_loaddata_R2_clu(N, ED, beta, noise_amplitude, nodei, nodej)
        _, SPnode_predictedbySRGG = find_top_n_values(node_fre, Predicted_truecase_num)
        SPNodeList_SRGG = np.zeros(N)
        SPNodeList_SRGG[SPnode_predictedbySRGG] = 1  # True cases
        precision_SRGG = precision_score(Label_med, SPNodeList_SRGG)
        recall_SRGG = recall_score(Label_med, SPNodeList_SRGG)
        # Store precision and recall values
        Precision_SRGG_nodepair.append(precision_SRGG)
        Recall_SRGG_nodepair.append(recall_SRGG)



    # Calculate means and standard deviations of AUC
    # AUCWithoutnorMean = np.mean(PRAUC_nodepair[~np.isnan(PRAUC_nodepair)])
    # AUCWithoutnorStd = np.std(PRAUC_nodepair[~np.isnan(PRAUC_nodepair)])

    precision_RGG_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise\\PrecisionRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(precision_RGG_Name, Precision_RGG_nodepair)

    recall_RGG_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise\\RecallRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(recall_RGG_Name, Recall_RGG_nodepair)

    precision_Geodis_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise\\PrecisionGeodisED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(precision_Geodis_Name, Precision_Geodis_nodepair)

    recall_Geodis_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise\\RecallGeodisED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(recall_Geodis_Name, Recall_Geodis_nodepair)

    precision_SRGG_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise\\PrecisionSRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(precision_SRGG_Name, Precision_SRGG_nodepair)

    recall_SRGG_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise\\RecallSRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(recall_SRGG_Name, Recall_SRGG_nodepair)


    NSPnum_nodepairName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise\\NSPNumED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(NSPnum_nodepairName, SPnum_nodepair, fmt="%i")

    geodistance_between_nodepair_Name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\Noise\\GeodistanceBetweenTwoNodesED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
    np.savetxt(geodistance_between_nodepair_Name, geodistance_between_nodepair)

    print("Mean Pre RGG:", np.mean(Precision_RGG_nodepair))
    print("Mean Recall RGG:", np.mean(Recall_RGG_nodepair))
    print("Mean Pre SRGG:", np.mean(Precision_SRGG_nodepair))
    print("Mean Recall SRGG:", np.mean(Recall_SRGG_nodepair))
    print("Mean Pre Geodistance:", np.mean(Precision_Geodis_nodepair))
    print("Mean Recall Geodistance:", np.mean(Recall_Geodis_nodepair))