# -*- coding: utf-8 -*-
"""
Created on  

@author: fame
"""
 
import numpy as np 
 

"""
    Compute the Euclidean distance between two matricess X and Y  
    Input:
    X: N-by-D numpy array 
    Y: M-by-D numpy array 
    
    Should return dist: M-by-N numpy array   
    """  
def compute_euclidean_distances( X, Y ) :
    dists = np.sqrt(-2 * np.dot(Y, X.T) + np.sum(X**2,axis=1) + np.sum(Y**2,axis=1)[:, np.newaxis])
    return dists
    """distances = np.zeros((Y.shape[0], X.shape[0]))
    print distances
    for i in range(Y.shape[0]):
        for j in range(X.shape[0]):
            distances[i][j] = np.sqrt(np.sum((Y[i,:]-X[j,:])**2))
            distances[i][j] = np.linalg.norm(Y[i,:]-X[j,:])
    return distances"""
    
 
"""
    Given a Euclidean distance matrix and associated training labels predict a label for each test point.
    Input:
    dists: M-by-N numpy array 
    labels: is a N dimensional numpy array
    
    Should return  pred_labels: M dimensional numpy array
    """
def predict_labels( dists, labels, k=1):
    min_dists = np.argsort(dists)[:,0:k]
    return np.mean( labels[min_dists] , axis = 1 )
    """min_dists = np.argmin(dists, axis=1)
    return labels[min_dists]"""
    
     
