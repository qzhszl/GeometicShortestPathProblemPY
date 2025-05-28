# -*- coding UTF-8 -*-

import numpy as np
import networkx as nx


filefoldername ="D:\\data\\geometric shortest path problem\\EuclideanSRGG\\ShortestPathAsActualCase\\EDbase2\\"

for beta in [4,8,128]:
    for ED in [2,4,8,16,32,64,128,256,512]:
        for noise in [0,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1]:
            filename = filefoldername+ "PrecisionSRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
                            EDn=ED, betan=beta, no=noise, ST=0)
            try:
                Precison_RGG_5_times = np.loadtxt(filename)
            except:
                print("Not found",(ED,beta,noise))


# for beta in [8]:
#     for ED in [512]:
#         for noise in [0,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1]:
#             for st in range(20):
#                 filename = filefoldername+ "PrecisionSRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
#                                 EDn=ED, betan=beta, no=noise, ST=st)
#                 try:
#                     Precison_RGG_5_times = np.loadtxt(filename)
#                 except:
#                     print("Not found",(ED,beta,noise,st))


