# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 16:52:31 2021

@author: Akshay
"""

import models_new
import data
import numpy as np

#import targets
import particle_filters_stochastic_volatility
#import matplotlib.pyplot as plt


testModel = models_new.stochVol()
testModel.par[0] = 0.9
testModel.par[1] = np.sqrt(0.1)
testModel.par[2] = 1
testModel.updateMaxPdf()

numParticles = 1000
backwardDraws = 2
np.random.seed(12345)
data = data.data()
data.generate(testModel, 200000)

#m=[]

#for i in [1000,1200]:
    #print("Number of particles:", i)
    #pf = particle_filters_stochastic_volatility.PaRIS_estimator()
    #pf.setup(testModel, i, backwardDraws, data)
    #m.append(pf.run())
    
#starts=[]    
#for j in [np.array([-0.1,np.log(0.7**2),np.log(0.8**2)]),np.array([0.7,np.log(0.2**2),np.log(1.5**2)]),
#          np.array([1.2,np.log(1.2**2),np.log(0.5**2)]),np.array([0.3,np.log(0.9**2),np.log(0.9**2)])]:
#    print("Starts:", j)
#    pf = particle_filters_stochastic_volatility.PaRIS_estimator()
#    pf.setup(testModel, numParticles, backwardDraws, data,j)
#    starts.append(pf.run())
    
    
pf = particle_filters_stochastic_volatility.PaRIS_estimator()
pf.setup(testModel, numParticles, backwardDraws, data,np.array([0.3,np.log(0.9**2),np.log(0.9**2)]))
long=pf.run()
    
    
    
    
    
    
    








