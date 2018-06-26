# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 10:07:34 2017

@author: Arjen
"""
import numpy as np
from ACEV2 import XYACE
from sklearn.svm import LinearSVC
from sklearn.cluster import Birch
from mnist import load_mnist

###data
TrainImg, TrainLabels = load_mnist(dataset="training", path = './')
TestImg, TestLabels = load_mnist(dataset="testing", path = './')

##parameter for image
Imgsize = 28
numTrainingSample = 60000
numTestingSample = 10000
window = 7
stride = 1
quantised_size = (Imgsize - window)//stride + 1

#parameters for birch
threshold=np.sqrt(2)
branching_factor=1000

#parameter for ACE
num_feature = 8

###quantisation
quantised_training = {}
quantised_testing = {}


for row in range(0, Imgsize-window, stride):
    for col in range(0, Imgsize-window, stride): #for every subImg
        row_inx = int(row/stride)  #index
        col_inx = int(col/stride)
        print (row_inx, col_inx)
        
        Xtrain = TrainImg[:, row:row+window, col:col+window].reshape(numTrainingSample, -1)
        Xtest = TestImg[:, row:row+window, col:col+window].reshape(numTestingSample, -1)
        
        ##use Birch to quantise
        brc = Birch(threshold=threshold, branching_factor=branching_factor, n_clusters=None, compute_labels=True)
        brc.fit(Xtrain)
        
        ##number of classes in quantisedX
        u = np.unique(brc.labels_)
        if (len(u)!=1):
            ##if there is only one class, no meaning of doing ACE
            ##so we exclude them in final (svm) training
            quantised_training[(row_inx, col_inx)] = brc.labels_
            quantised_testing[(row_inx, col_inx)] = brc.predict(Xtest)

train = []
test = []

##for every random variable, we do ace
Y = TrainLabels
for key in quantised_training:
    Xtrain = quantised_training[key]
    Xtest = quantised_testing[key]
    model = XYACE(Xtrain,Y,num_feature,OutlierControl=True)
    #append f(xtrain) into training_svm
    train.append(model.GetF(Xtrain))
    #append f(xtest) into training_svm
    test.append(model.GetF(Xtest))

train = np.nan_to_num(np.concatenate(train, axis = 1))
test = np.nan_to_num(np.concatenate(test, axis = 1))
print ('Dimensionality: %d' %(test.shape[1]))

print ('supervised')
print ('training begin')
clf = LinearSVC()
clf.fit(train, TrainLabels)
training_error = clf.score(train, TrainLabels)
testing_error = clf.score(test, TestLabels)
print ('training_error: %f' %(training_error))
print ('testing_error: %f' %(testing_error))
print ('threshold= %f, branching_factor = %d, num_feature= %d' %(threshold, branching_factor, num_feature))
print ('window: %d, stride: %d' %(window, stride))