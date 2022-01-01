# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 14:42:39 2021

@author: Akshay
"""


# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 16:46:27 2021

@author: Akshay
"""

import numpy as np
import extra


class PaRIS_estimator(object):
    model = []  # Standard
    nPart = []
    data = []
    filtMean = []
    nBackDraws = []
    theta = []
    start=[]

    def setup(self, model, nPart, nBackDraws, data,start):
        self.model = model
        self.nPart = nPart
        self.nBackDraws = nBackDraws
        self.data = data
        self.start = start

    def run(self):
        tt = 0;  # akg
        #arr = np.array([0.3,0.9,0.9])  # akg
        arr= np.array(self.start)
        
        self.theta = np.zeros((len(arr), self.data.T))  # akg
        self.theta[:, tt] = arr  # akg
        self.model.updateParams(self.theta[:, tt])  # akg
        #self.model.updateParamsLog(self.theta[:, tt])
        
        xPart = np.zeros((self.nPart, self.model.Xdim))  # Current
        xPartN = np.zeros((self.nPart, self.model.Xdim))  # New
        weights = np.zeros(self.nPart)  # Current
        weightsN = np.zeros(self.nPart)  # New
        #self.filtMean = np.zeros((self.model.dimension, self.data.T))
        tStat = np.zeros((self.model.nPar, self.nPart))
        # Starting the bootstrap particle filter here:

        xPartN = self.model.propagate(xPart, self.data.yt[tt], tt, self.nPart)
        weightsN = self.model.weightFun(xPartN, self.data.yt[tt], tt)

        xPart = xPartN
        weights = weightsN
        for tt in range(1, self.data.T):
            # print(tt)
            # Resample
            print("Iteration: ",tt,"Parameters: ",self.model.par)
            indX = extra.randoind(np.exp(weights), self.nPart)
            # Propagate
            xPartN = self.model.propagate(xPart[indX], self.data.yt[tt], tt, self.nPart)
            weightsN = self.model.weightFun(xPartN, self.data.yt[tt], tt)
            tStatN = np.zeros(np.shape(tStat))
            # Backward draws
            for j in range(0, self.nBackDraws):
                bInd = extra.backwardDraws(weights, xPart, xPartN, self.data, tt, self.model,
                                           int(np.ceil(np.sqrt(self.nPart))))
                tStatN = tStatN + (tStat[:, bInd] + self.model.h_func(xPartN,xPart[bInd], self.data.yt[tt-1],tt,
                                                                           self.nPart)) / self.nBackDraws
                
         #       tStatN = tStatN + (tStat[:, bInd] + self.model.h_func_log(xPartN,xPart[bInd], self.data.yt[tt-1],tt,
         #                                                                  self.nPart)) / self.nBackDraws

            tAvg = np.reshape(np.sum(tStatN, axis=1)/self.nPart,(self.model.nPar,1))
            #print(tAvg)
            #print(tStatN)
            
            #print("Zeta1")
            zeta1 = np.sum(self.model.zeta1(xPartN,self.data.yt[tt],self.nPart,tt), axis=1)/ self.nPart
            #print(zeta1)
            #zeta1 = np.sum(self.model.zeta1_log(xPartN,self.data.yt[tt],self.nPart,tt), axis=1)/ self.nPart
            
            zeta3 = np.sum(np.exp(weightsN), axis=0) / self.nPart

            zeta2 = np.sum((tStatN - np.repeat(tAvg,self.nPart,axis=1))*np.exp(weightsN), axis=1)/ self.nPart

            if (tt<50):
                gamma=0
            elif (tt<200):
                gamma=200**-(0.6)
            else:
                gamma = (tt) ** (-0.6)
            

            
            self.theta[:, tt] = self.theta[:, tt - 1] + gamma * (zeta1 + zeta2) / zeta3
                            
            tStat = tStatN
            #self.filtMean[:, tt] = np.average(tStat, weights=np.exp(weightsN) / np.sum(np.exp(weightsN)),
             #                                 axis=1)
            # Set
            self.model.updateParams(self.theta[:, tt])  # akg
            #self.model.updateParamsLog(self.theta[:, tt])
            xPart = xPartN
            weights = weightsN
            #print(self.model.par[2])
            #print(self.theta[:, tt])
        return self.theta[2:]

    def __str__(self):
        return ('PaRIS algorithm \n N = {0}, K = {1}'.format(self.nPart, self.nBackDraws))
