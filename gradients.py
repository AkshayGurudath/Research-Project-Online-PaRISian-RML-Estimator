import numpy as np
import extra
class awesomeFun():
    dimension = 1
    def gradient_1(self, xtprev, ytprev, nPart,par):
        val = np.zeros((self.dimension, nPart))
        val[0, :] = ((ytprev-par*xtprev)*xtprev).squeeze()
        return val
    def gradient_2(self, xt, yt,nPart,par):
        val = np.zeros((self.dimension, nPart))
        val[0, :] = -np.exp(extra.logNormPdf(yt-par*xt, 0, 1)).squeeze()
        return val
    