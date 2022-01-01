import numpy as np
import extra
import gradients


class PaRIS_estimator(object):
    model = []  # Standard
    nPart = []
    target = []
    data = []
    filtMean = []
    nBackDraws = []
    theta = []

    def setup(self, model, nPart, nBackDraws, data, target):
        self.model = model
        self.nPart = nPart
        self.nBackDraws = nBackDraws
        self.data = data
        self.target = target

    def run(self):
        tt = 0;  # akg
        arr = np.array([0.7])  # akg
        self.theta = np.zeros((len(arr), self.data.T))  # akg
        self.theta[0, tt] = arr[0]  # akg
        self.model.updateParams(self.theta[:, tt])  # akg
        xPart = np.zeros((self.nPart, self.model.Xdim))  # Current
        xPartN = np.zeros((self.nPart, self.model.Xdim))  # New
        weights = np.zeros(self.nPart)  # Current
        weightsN = np.zeros(self.nPart)  # New
        self.filtMean = np.zeros((self.target.dimension, self.data.T))
        tStat = np.zeros((self.target.dimension, self.nPart))
        # Starting the bootstrap particle filter here:

        xPartN = self.model.propagate(xPart, self.data.yt[tt], tt, self.nPart)
        weightsN = self.model.weightFun(xPartN, self.data.yt[tt], tt)

        xPart = xPartN
        weights = weightsN
        for tt in range(1, self.data.T):
            # print(tt)
            # Resample
            indX = extra.randoind(np.exp(weights), self.nPart)
            # Propagate
            xPartN = self.model.propagate(xPart[indX], self.data.yt[tt], tt, self.nPart)
            weightsN = self.model.weightFun(xPartN, self.data.yt[tt], tt)

            tStatN = np.zeros(np.shape(tStat))
            # Backward draws
            for j in range(0, self.nBackDraws):
                bInd = extra.backwardDraws(weights, xPart, xPartN, self.data, tt, self.model,
                                           int(np.ceil(np.sqrt(self.nPart))))
                tStatN = tStatN + (tStat[:, bInd] + self.target.gradient_1(xPart[bInd], self.data.yt[tt - 1],
                                                                           self.nPart,
                                                                           self.theta[0, tt - 1])) / self.nBackDraws

            tAvg = np.sum(tStatN, axis=1)[0] / self.nPart
            g=-(self.data.yt[tt]-self.model.par[2]*xPartN[:,0])/(self.model.par[3])**2
            zeta1 = -np.sum(xPartN[:, 0]*np.exp(weightsN)*g, axis=0)/ self.nPart
            zeta3 = np.sum(np.exp(weightsN), axis=0) / self.nPart
            zeta2 = np.sum((tStatN - tAvg)*np.exp(weightsN), axis=1)[0] / self.nPart
            gamma = tt ** (-0.6)
            self.theta[0, tt] = self.theta[0, tt - 1] + gamma * (zeta1 + zeta2) / zeta3
            tStat = tStatN
            self.filtMean[:, tt] = np.average(tStat, weights=np.exp(weightsN) / np.sum(np.exp(weightsN)),
                                              axis=1)
            # Set
            self.model.updateParams(self.theta[:, tt])  # akg
            xPart = xPartN
            weights = weightsN
            #print(self.model.par[2])
            print(self.theta[0, tt])

    def __str__(self):
        return ('PaRIS algorithm \n N = {0}, K = {1}'.format(self.nPart, self.nBackDraws))
