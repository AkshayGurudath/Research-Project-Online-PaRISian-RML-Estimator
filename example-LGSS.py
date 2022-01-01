# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 16:10:24 2021

@author: Akshay
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 14:04:21 2021

@author: Akshay
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 16:52:31 2021

@author: Akshay
"""

import models_new
import data
import numpy as np

#import targets
import particle_filters_lgss
#import matplotlib.pyplot as plt


testModel = models_new.lgss()
testModel.par[0] = 0.7
testModel.par[1] = 0.2
testModel.par[2] = 1
testModel.par[3] = 1
testModel.updateMaxPdf()
np.random.seed(12345)
numParticles = 1000
backwardDraws = 2
data = data.data()

data.generate(testModel, 150000)

pf = particle_filters_lgss.PaRIS_estimator()
pf.setup(testModel, numParticles, backwardDraws, data,start=[0.7,0.2,4,1])
a=pf.run()

#parts=[]    
#for j in [500,800,1000,1200]:
#    print("Starts:", j)
#    pf = particle_filters_lgss.PaRIS_estimator()
#    pf.setup(testModel, j, backwardDraws, data,start=[0.7,0.2,0.5,1])
#    parts.append(pf.run())

#starts=[]    
#for j in [[0.7,0.2,0,1],[0.7,0.2,0.3,1],[0.7,0.2,0.5,1],[0.7,0.2,1.5,1]]:
#    print("Starts:", j)
#    pf = particle_filters_lgss.PaRIS_estimator()
#    pf.setup(testModel, numParticles, backwardDraws, data,start=j)
#    starts.append(pf.run())
    






