# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 19:08:34 2021

@author: Akshay
"""
import numpy as np
import tensorflow as tf

x=tf.Variable(3.0)
with tf.GradientTape() as tape:
    y=x**2
    
dy_dx=tape.gradient(y,x)
print(dy_dx.numpy())

def g(x):
    return 2*x

x=tf.Variable(3.0)



with tf.GradientTape() as tape:
    y=g(x)
    
dy_dx=tape.gradient(y,x)
print(dy_dx.numpy())

x=tf.Variable(np.zeros(3))

def g(x):
    return 2*x[1]

with tf.GradientTape() as tape:
    y=g(x)

dy_dx=tape.gradient(y,x)
print(dy_dx.numpy())

a=np.array([[1,2,3]])
print(a[0][2])




