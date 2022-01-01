import models_new
import numpy as np


class data(object):
    """Data for model SMC"""
    xt = []
    yt = []
    T = 0

    def generate(self, model, T):
        self.T = T + 1
        yt = np.zeros(self.T)
        xt = np.zeros(self.T)
        for tt in range(0, self.T):
            if tt == 0:
                xt[tt] = model.q(xt[tt], yt[tt], tt) + model.qv(xt[tt], yt[tt], tt) * np.random.randn(1, 1)
                yt[tt] = model.g(xt[tt], yt[tt], tt) + model.gv(xt[tt], yt[tt], tt) * np.random.randn(1, 1)
            else:
                xt[tt] = model.q(xt[tt - 1], yt[tt - 1], tt - 1) + model.qv(xt[tt - 1], yt[tt - 1],
                                                                            tt) * np.random.randn(1, 1)
                yt[tt] = model.g(xt[tt], yt[tt], tt) + model.gv(xt[tt], yt[tt], tt) * np.random.randn(1, 1)
        # print tt
        self.xt = xt
        self.yt = yt

    def setData(self, Y):
        self.yt = Y
        self.T = len(Y)

