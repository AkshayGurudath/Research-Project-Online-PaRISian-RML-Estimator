# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 19:13:32 2021

@author: Akshay
"""

import numpy as np
import extra
from scipy.stats import multivariate_normal as mvn
from scipy.stats import binom
import tensorflow as tf
from scipy.stats import norm
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)




class stochVol(object):
    """docstring for stochVol"""

    # X(t+1) = q(x(t),y(t),t) + qv(x(t),y(t),t)*v(t)
    # Y(t) = g(x(t),y(t),t) + gv(x(t),y(t),t)*e(t)
    #

    nPar = 3;
    par = np.zeros(3);
    maxPdf = 0;
    Xdim = 1;

    def updateMaxPdf(self):
        # self.maxPdf = - 0.5 * np.log(2*np.pi*self.par[1]**2)
        self.maxPdf = extra.logNormPdf(0, 0, self.par[1])

    def q(self, xt, yt, tt):
        return self.par[0] * xt;

    def qv(self, xt, yt, tt):
        return self.par[1] * np.ones_like(xt)

    def g(self, xt, yt, tt):
        return np.zeros_like(xt)

    def gv(self, xt, yt, tt):
        return self.par[2] * np.exp(0.5 * xt);

    def weightFun(self, xt, yt, tt):
        weights = extra.logNormPdf(yt, self.g(xt, yt, tt), self.gv(xt, yt, tt));
        return weights  # [:,0]

    def propagate(self, xt, yt, tt, nPart):
        return self.q(xt, yt, tt) + self.qv(xt, yt, tt) * np.random.randn(nPart, self.Xdim);

    def logTransProb(self, xtN, xtO, yt, tt):
        return extra.logNormPdf(xtN, self.q(xtO, yt, tt), self.qv(xtO, yt, tt)) - self.maxPdf

    def __str__(self):
        return "Stochastic volatility model \n X(t+1) = {0} * X(t) + {1} * v(t+1) \n Y(t) = {2} * exp(0.5 * X(t)) * e(t)".format(
            self.par[0], self.par[1], self.par[2])
    
    @tf.function
    def g_func(self,xt,yt,diff):
        a=tf.subtract(yt,xt)
        #print(diff)
        b=tf.exp(-tf.divide(tf.square(a),tf.square(tf.multiply(0.5*xt,diff[2]))))
        c=tf.divide(b,((0.5*xt*diff[2])*np.sqrt(2*np.pi)))
        #print(c)
        return c
    
    @tf.function
    def q_func(self,xt,xtprev,diff):
        a=tf.subtract(xt,xtprev*diff[0])
        #print(diff)
        b=tf.exp(-tf.divide(tf.square(a),tf.square(diff[1])))
        c=tf.divide(b,diff[1]*np.sqrt(2*np.pi))
        return c
    
    @tf.function
    def baby(self,xt,xtprev,yt,ytprev,diff):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(diff)
            a=self.g_func(xt,yt,diff)
            b=tf.add(tf.math.log(self.g_func(xt,yt,diff)),tf.math.log(self.q_func(xt,xtprev,diff)))
            
            #y1=tf.math.log(a)
        c=tape.jacobian(b,diff)
        d=tape.jacobian(a,diff)
        return (c,d)
        
    def test(self,xt,xtprev,yt,ytprev):
        xt = tf.constant(xt,dtype="float64")
        yt =tf.constant(yt,dtype="float64")
        xtprev = tf.constant(xtprev,dtype="float64")
        ytprev = tf.constant(ytprev,dtype="float64")
        diff=tf.Variable(self.par,dtype="float64")
        
       # a=np.zeros((self.nPar, nPart))
        ans=self.baby(xt,xtprev,yt,ytprev,diff)
        return (np.transpose(ans[0][:,0,:].numpy()),np.transpose(ans[1][:,0,:].numpy()))
    #   b=tape.jacobian(y2,diff)
        
    
    def h_func(self,xt,xtprev,ytprev,tt,nPart):
        a=np.zeros((self.nPar, nPart))
        a[0,:]=((xt-self.q(xtprev,ytprev,tt))*xtprev/self.par[1]**2).squeeze()
        a[1,:]=((xt-self.q(xtprev,ytprev,tt))**2/self.par[1]**3).squeeze() - 1/(self.par[1])
        a[2,:]=((ytprev**2)/(np.exp(xtprev)*self.par[2]**3)).squeeze() - 1/(self.par[2])
        return a
    
    def zeta1(self,xt,yt,nPart,tt):
        a=np.zeros((self.nPar, nPart))
        num=(np.exp(self.weightFun(xt, yt, tt))*yt**2).squeeze()
        denom=(np.exp(xt)*self.par[2]**3).squeeze()
        a[2,:]=num/denom - (np.exp(self.weightFun(xt, yt, tt))/self.par[2]).squeeze()
        return a
    
    def updateParams(self,arr):
        self.par[0]=arr[0]
        self.par[1]=arr[1]
        self.par[2]=arr[2]
        
    def h_func_log(self,xt,xtprev,ytprev,tt,nPart):
        a=np.zeros((self.nPar,nPart))
        p=self.transform_sigma(self.par[1])
        q=self.transform_beta(self.par[2])
        a[0,:]=((xt-self.q(xtprev,ytprev,tt))*xtprev/self.par[1]**2).squeeze()
        a[1,:]=(-0.5+0.5*(xt-self.par[0]*xtprev)**2*np.exp(-p)).squeeze()
        a[2,:]=(-0.5+0.5*ytprev**2*np.exp(-xt-q)).squeeze()
        return(a)
    
    def transform_sigma(self,sigma):
        return np.log(sigma**2)
    
    
    def transform_beta(self,beta):
        return np.log(beta**2)
    
    
    def zeta1_log(self,xt,yt,nPart,tt):
        a=np.zeros((self.nPar,nPart))
        q=self.transform_beta(self.par[2])
        term_1=np.exp(extra.logNormPdf(yt, np.zeros_like(xt), np.exp((q+xt)/2))).squeeze() #Tricky
        term_2=(0.5*(yt**2)*(np.exp(-(q+xt)))-0.5).squeeze()
        a[2,:]=term_1*term_2
        return(a)
    
    def updateParamsLog(self,arr):
        self.par[0]=arr[0]
        self.par[1]=np.sqrt(np.exp(arr[1]))
        self.par[2]=np.sqrt(np.exp(arr[2]))
        
        


#m=stochVol()
#m.par[0] = 0.9
#m.par[1] = np.sqrt(0.1)
#m.par[2] = 1   
#a=m.test([0.5,0.5,0.7,0.8],[0.6,0.6,0.6,0.6],[1,1,1,1],[1,1,1,1])
#print(a[0])



    

    

    
    

 