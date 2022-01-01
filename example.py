import models_new
import data
import numpy as np
#import targets
import gradients
import particle_filters_new
import matplotlib.pyplot as plt

testModel = models_new.lgss()
testModel.par[0] = 0.7
testModel.par[1] = 0.2
testModel.par[2] = 1
testModel.par[3] = 1
testModel.updateMaxPdf()

numParticles = 500
backwardDraws = 8
target = gradients.awesomeFun()
data = data.data()
data.generate(testModel, 100000)

pf = particle_filters_new.PaRIS_estimator()
pf.setup(testModel, numParticles, backwardDraws, data, target)
pf.run()






