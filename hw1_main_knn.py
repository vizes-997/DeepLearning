# -*- coding: utf-8 -*-

from load_mnist import * 
import hw1_knn  as mlBasics  
import numpy as np 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold


# Load data - ALL CLASSES
X_train, y_train = load_mnist('training'  )
X_test, y_test = load_mnist('testing'   )

#getting 1000 samoples ..100 samples from each labels
itemindex = np.zeros((10,100))
for i in range(10):
    itemindex[i] = np.where(y_train == i)[0][0:100]
itemindex = np.reshape(itemindex,1000)
itemindex = itemindex.astype(int)
X_sample = X_train[itemindex]

 # Reshape the image data into rows  
X_train = np.reshape(X_sample, (X_sample.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

# Test on test data   
#1) Compute distances:
dists =  mlBasics.compute_euclidean_distances(X_train,X_test) 
  
#2) Run the code below and predict labels:
 
y_test_pred = mlBasics.predict_labels(dists, y_train[itemindex],1 )
#y_test_pred = mlBasics.predict_labels(dists, y_train[itemindex],5 ) 


#3) Report results
# you should get following message '99.91 of test examples classified correctly.'
print '{0:0.02f}'.format(  np.mean(y_test_pred==y_test)*100), "of test examples classified correctly."
print confusion_matrix(y_test_pred.astype(int), y_test.astype(int))
