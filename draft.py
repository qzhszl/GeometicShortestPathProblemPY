for simu_index in range(simu_times):
    ExpectedDegreeVec = np.zeros(simu_times)
    LCCVec = np.zeros(simu_times)
    DistanceWithoutNorVec = []
    DistanceNorVec = []
    AveNorVec = []
    AveWithoutNorVec = []
    MaxNorVec = []
    MaxWithoutNorVec = []
    MinNorVec = []
    MinWithoutNorVec = []

    N = Nvec[counti]
    k = kvec[countj]
    r = np.sqrt(k / ((N - 1) * np.pi))

    if N > k:
        for Simu_i in range(simu_times):
            G, coor = generate_random_geometric_graph(N, r)

            expected_degree = np.mean([d for n, d in G.degree()])
            ExpectedDegreeVec[Simu_i] = expected_degree

            b = nx.connected_components(G)
            bsize = [len(c) for c in b]
            LCCVec[Simu_i] = max(bsize)

            S = dict(nx.all_pairs_shortest_path_length(G))
            nodei_list, nodej_list = [], []

            for i in range(N):
                for j in range(i + 1, N):
                    if j in S[i]:
                        nodei_list.append(i)
                        nodej_list.append(j)

            NumNodePair = 1000000
            if len(nodej_list) > NumNodePair:
                a = np.random.permutation(len(nodej_list))
                nodepair_index_pre = a[:NumNodePair]
                nodei_list = np.array(nodei_list)[nodepair_index_pre]
                nodej_list = np.array(nodej_list)[nodepair_index_pre]
            else:
                NumNodePair = len(nodej_list)

            distance = [None] * NumNodePair
            DistanceWithoutNor = [None] * NumNodePair
            AveNor = [None] * NumNodePair
            AveWithoutNor = [None] * NumNodePair
            MaxNor = [None] * NumNodePair
            MaxWithoutNor = [None] * NumNodePair
            MinNor = [None] * NumNodePair
            MinWithoutNor = [None] * NumNodePair

            SPNumVec = np.zeros(NumNodePair)

            for nodepairindex in range(NumNodePair):
                nodei = nodei_list[nodepairindex]
                nodej = nodej_list[nodepairindex]

                d, spath = bellman_ford_shortest_path(G, nodei)
                spnodej = spath[nodej]
                SPNum = len(spnodej)
                SPNumVec[nodepairindex] = SPNum

                distanceForanodepair = [None] * SPNum
                DistanceWithoutNorForanodepair = [None] * SPNum
                AveNorForanodepair = np.zeros(SPNum)
                AveWithoutNorForanodepair = np.zeros(SPNum)
                MaxNorForanodepair = np.zeros(SPNum)
                MaxWithoutNorForanodepair = np.zeros(SPNum)
                MinNorForanodepair = np.zeros(SPNum)
                MinWithoutNorForanodepair = np.zeros(SPNum)

                for q in range(SPNum):
                    PNodeList = spnodej[q]
                    PLength = len(PNodeList) - 1

                    if PLength > 1:
                        distance_med = np.zeros(PLength - 1)

                        xSource, ySource = coor[nodei]
                        xEnd, yEnd = coor[nodej]
                        disbetweenendnodes = np.sqrt((xSource - xEnd) ** 2 + (ySource - yEnd) ** 2)

                        for PNodeMed_index in range(1, PLength):
                            PNodeMed = PNodeList[PNodeMed_index]
                            xMed, yMed = coor[PNodeMed]
                            dist = perpendicular_distance(xSource, ySource, xEnd, yEnd, xMed, yMed)
                            distance_med[PNodeMed_index - 1] = dist

                        distanceForanodepair[q] = distance_med / disbetweenendnodes
                        DistanceWithoutNorForanodepair[q] = distance_med
                        AveNorForanodepair[q] = np.mean(distance_med / disbetweenendnodes)
                        AveWithoutNorForanodepair[q] = np.mean(distance_med)
                        MaxNorForanodepair[q] = np.max(distance_med / disbetweenendnodes)
                        MaxWithoutNorForanodepair[q] = np.max(distance_med)
                        MinNorForanodepair[q] = np.min(distance_med / disbetweenendnodes)
                        MinWithoutNorForanodepair[q] = np.min(distance_med)

                distance[nodepairindex] = np.concatenate(distanceForanodepair)
                DistanceWithoutNor[nodepairindex] = np.concatenate(DistanceWithoutNorForanodepair)
                AveNor[nodepairindex] = AveNorForanodepair
                AveWithoutNor[nodepairindex] = AveWithoutNorForanodepair
                MaxNor[nodepairindex] = MaxNorForanodepair
                MaxWithoutNor[nodepairindex] = MaxWithoutNorForanodepair
                MinNor[nodepairindex] = MinNorForanodepair
                MinWithoutNor[nodepairindex] = MinWithoutNorForanodepair

            DistanceNor_vecforonegraph = np.concatenate(distance)
            DistanceNorVec.extend(DistanceNor_vecforonegraph)
            DistanceWithoutNor_vecforonegraph = np.concatenate(DistanceWithoutNor)
            DistanceWithoutNorVec.extend(DistanceWithoutNor_vecforonegraph)
            AveNorwith0 = np.concatenate(AveNor)
            AveNorVec.extend(AveNorwith0[AveNorwith0 > 0])
            AveWithoutNorwith0 = np.concatenate(AveWithoutNor)
            AveWithoutNorVec.extend(AveWithoutNorwith0[AveWithoutNorwith0 > 0])
            MaxNorwith0 = np.concatenate(MaxNor)
            MaxNorVec.extend(MaxNorwith0[MaxNorwith0 > 0])
            MaxWithoutNorwith0 = np.concatenate(MaxWithoutNor)
            MaxWithoutNorVec.extend(MaxWithoutNorwith0[MaxWithoutNorwith0 > 0])
            MinNorwith0 = np.concatenate(MinNor)
            MinNorVec.extend(MinNorwith0[MinNorwith0 > 0])
            MinWithoutNorwith0 = np.concatenate(MinWithoutNor)
            MinWithoutNorVec.extend(MinWithoutNorwith0[MinWithoutNorwith0 > 0])

            NodepairLength = np.array([len(AveWithoutNorwith0[AveWithoutNorwith0 > 0])])
            SPcell = np.array(SPNumVec)

            save_to_file(f"NorDeviation_{N}nodek{k}.txt", DistanceNorVec)
            save_to_file(f"WithoutNorDeviation_{N}nodek{k}.txt", DistanceWithoutNorVec)
            save_to_file(f"NorAveDeviation_{N}nodek{k}.txt", AveNorVec)
            save_to_file(f"WithoutNorAveDeviation_{N}nodek{k}.txt", AveWithoutNorVec)
            save_to_file(f"NorMaxDeviation_{N}nodek{k}.txt", MaxNorVec)
            save_to_file(f"WithoutNorMaxDeviation_{N}nodek{k}.txt", MaxWithoutNorVec)
            save_to_file(f"NorMinDeviation_{N}nodek{k}.txt", MinNorVec)
            save_to_file(f"WithoutNorMinDeviation_{N}nodek{k}.txt", MinWithoutNorVec)
            save_to_file(f"LCC_{N}nodek{k}.txt", LCCVec)
            save_to_file(f"Expecteddegree_{N}nodek{k}.txt", ExpectedDegreeVec)
            save_to_file(f"NodepairLength_{N}nodek{k}.txt", NodepairLength)
            save_to_file(f"SPnum_{N}nodek{k}.txt", SPcell)
