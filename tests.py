#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 15:05:42 2017

@author: edouardcuny
"""

#%% TEST DU COST - TEST CONCLUANT
nn=NeuralNetwork([4,2])
x,y=np.array([0.5,0.7,0.3,0.9]),np.array([0.3,-0.2])
y.shape=(2,1)
nn.cost(x,y)

#%% TEST POUR SGD
nn=NeuralNetwork([4,2,4])
x=np.array([[1,1,0,0,1,1,1,1],[1,0,1,0,0,0,1,1]])
nn.SGD(x,batch_size=2,epochs=10)

#%%
nn=NeuralNetwork([4,2,4])
x=np.array([[0,0.1,0.2,0.3],[0.4,0.5,0.6,0.7]])
x.shape=(4,2)
Y=np.array([[0.1,0.2],[0.3,0.4],[0,0],[0.2,0.5]])
nn.backprop(x,Y)


#%% je veux prendre un array 
# unroll puis back comme il faut
import numpy as np

list=[np.array([[0,0.1,0.2,0.3],[0.4,0.5,0.6,0.7]]),np.array([[0.1,0.2],[0.3,0.4],[0,0],[0.2,0.5]])]
unrolled=list[0].ravel()
for i in range(1,len(list)):
    u=list[i].ravel()
    unrolled=np.append(unrolled,u)

back=[]
idx=0
for i in range(len(list)):
    shape=list[i].shape
    size=list[i].size
    candidat=unrolled[idx:idx+size,]
    candidat.shape=shape
    back.append(candidat)
    idx+=size
      
#%% gradient checking - fonctionne
nn=NeuralNetwork([4,2,4])
x=np.array([0.7,0.1,0.2,0.3])
x.shape=(1,4)
y=np.array([0.1,0.2,0.2,0.5])
y.shape=(1,4)
nn.gradient_checking(x=x,y=y,epsilon=0.0001)

#%% est-ce que backprop déconne ?  Nooooooon - oui - problème de signe et divergence !!! aaa
# backprop ne déconne pas pour un truc de taille 1
nn=NeuralNetwork([1,1])
print("poids",nn.weights)
print("biais",nn.biases)
x=np.array([0.5])
x.shape=(1,1)
y=np.array([0.7])
y.shape=(1,1)
#print("output",nn.feedforward(x)) - fonctionne
#print("cost",nn.cost(x,y)) - fonctionne
#nn.backprop(x,y) - fonctionne

# est-ce que gradient checking déconne ? Nooooon
nn.gradient_checking(epsilon=0.001,x=x,y=y)

