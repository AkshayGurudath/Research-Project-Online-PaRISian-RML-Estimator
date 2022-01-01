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

import models_new_autodiff
import data
import numpy as np

#import targets
import particle_filters_autodiff
#import matplotlib.pyplot as plt


testModel = models_new_autodiff.stochVol()
testModel.par[0] = 0.9
testModel.par[1] = np.sqrt(0.1)
testModel.par[2] = 1
testModel.updateMaxPdf()

numParticles = 1000
backwardDraws = 2
data = data.data()
data.generate(testModel, 100000)

pf = particle_filters_autodiff.PaRIS_estimator()
pf.setup(testModel, numParticles, backwardDraws, data)
pf.run()






