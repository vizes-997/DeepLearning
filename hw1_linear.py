# -*- coding: utf-8 -*-
"""
Created on  

@author: fame
"""

import numpy as np 

 

def predict(X,W,b):  
    """
    implement the function h(x, W, b) here  
    X: N-by-D array of training data 
    W: D dimensional array of weights
    b: scalar bias

    Should return a N dimensional array  
    """
    W = np.array(W)
    W = W.reshape(W.size, 1)
    X = np.array(X)
    arr = np.dot(X, W)
    arr = np.add(arr, b)
    arr = arr.flatten()
    return [sigmoid(el) for el in arr]
 
def sigmoid(a): 
    """
    implement the sigmoid here
    a: N dimensional numpy array

    Should return a N dimensional array  
    """
    return 1.0 / ( 1 + np.exp(-a) )

def l2loss(X,y,W,b):  
    """
    implement the L2 loss function
    X: N-by-D array of training data 
    y: N dimensional numpy array of labels
    W: D dimensional array of weights
    b: scalar bias

    Should return three variables: (i) the l2 loss: scalar, (ii) the gradient with respect to W, (iii) the gradient with respect to b
     """
    H = predict(X, W, b)
    loss = 0
    for i in range(len(H)):
        loss += (y[i] - H[i]) ** 2
    
    b_grad = 0
    for i in range(len(H)):
        b_grad += 2 * (H[i] - y[i]) * H[i] * (1 - H[i])

    W_grad = np.array([0] * len(W))
    
    for i in range(len(H)):
        sc = 2 * (H[i] - y[i]) * H[i] * (1 - H[i])
        res = np.multiply(sc, np.array(X[i]))
        W_grad = np.add(W_grad, res)
    W_grad = W_grad.tolist()
    
    return (loss, W_grad, b_grad)
    

def train(X,y,W,b, num_iters=1000, eta=0.001):  
    """
    implement the gradient descent here
    X: N-by-D array of training data 
    y: N dimensional numpy array of labels
    W: D dimensional array of weights
    b: scalar bias
    num_iters: (optional) number of steps to take when optimizing
    eta: (optional)  the stepsize for the gradient descent

    Should return the final values of W and b    
     """
    losses = []
    for i in range(num_iters):
        (loss, W_grad, b_grad) = l2loss(X, y, W, b)
        #print loss
        losses.append(loss)
        b = b - eta * b_grad
        W = np.add(np.array(W), np.multiply(-eta, W_grad)).tolist()
    
    return (W, b, losses)

 