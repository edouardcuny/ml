#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import os 
import numpy as np
from sklearn.linear_model import LogisticRegression
from datetime import datetime

os.chdir("/Users/edouardcuny/Desktop/10k/cifar/")

# fonction pour pouvoir utiliser mes 
def unpickle(file):
    import _pickle as cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo,encoding="bytes")
    fo.close()
    return dict

#%%
    
# TRAIN
train=unpickle("./cifar-10-batches-py/data_batch_1")
xtrain=np.asarray(train[b'data']) # training data
ytrain=np.asarray(train[b'labels']) # training labels

for i in range(2,6):
    file="./cifar-10-batches-py/data_batch_"+str(i)
    batch=unpickle(file)
    xtrain=np.concatenate((xtrain,batch[b'data']),axis=0)
    ytrain=np.concatenate((ytrain,batch[b'labels']),axis=0)
    
# TEST
test=unpickle("./cifar-10-batches-py/test_batch")
xtest=np.asarray(test[b'data']) # testing data
ytest=np.asarray(test[b'labels']) # testing labels

#%%
# CLASSIFIER
clf=LogisticRegression(verbose=100,solver='lbfgs',n_jobs=-1)
debut=datetime.now()
clf.fit(xtrain,ytrain)
fin=datetime.now()
delta=fin-debut
print(delta.seconds)


#%%

# SCORE
print(clf.score(xtest,ytest))


# sag solver
# faut-il faire du pre-processing ?