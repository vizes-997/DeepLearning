# -*- coding: utf-8 -*-
"""
Created on 

@author: fame
"""

 
from load_mnist import * 
import hw1_knn  as mlBasics  
import numpy as np 
from sklearn.model_selection import KFold
   
  
# Load data - two class 
X_train, y_train = load_mnist('training' , [0,1] )
X_test, y_test   = load_mnist('testing'  , [0,1] )
#print X_train.shape
#print y_train.shape
# Load data - ALL CLASSES
#X_train, y_train = load_mnist('training'  )
#X_test, y_test = load_mnist('testing'   )



# Reshape the image data into rows  
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
#Load All classes
#X_train, y_train = load_mnist('training'  )
#X_test, y_test = load_mnist('testing'   )

for k in range(1,16):
    pred =0
    kf = KFold(n_splits=5)
    for newTrain, newTest in kf.split(X_train):
        dists =  mlBasics.compute_euclidean_distances(X_train[newTrain],X_train[newTest]) 
        y_test_pred = mlBasics.predict_labels(dists, y_train[newTrain],k )   
        pred+= np.mean(y_test_pred==y_train[newTest])*100
    print ("k="+ str(k) + "...prediction=" + str(pred/5))

"""k = 1    
pred =0
kf = KFold(n_splits=5)
for newTrain, newTest in kf.split(X_train):
    dists =  mlBasics.compute_euclidean_distances(X_train[newTrain],X_train[newTest]) 
    y_test_pred = mlBasics.predict_labels(dists, y_train[newTrain],k )   
    pred+= np.mean(y_test_pred==y_train[newTest])*100
print ("k="+ str(k) + "...prediction=" + str(pred/5))"""
