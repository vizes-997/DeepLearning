import keras
#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, SGD
from keras.regularizers import l1,l2
import matplotlib.pyplot as plt
from load_mnist import *

import numpy as np
#from Task1 import loss_callback


class LossHistories(keras.callbacks.Callback):
    def on_train_begin(self, data, logs={}):
        #self.x_test = data[0]
        #self.y_test = data[1]
        self.losses = []
        self.val = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val.append(logs.get('val_acc'))
        #self.aucs.append(roc_auc_score(self.validation_data[1], y_pred))
        return

def MLP1():
    model = Sequential()
    model.add(Dropout(0.2))
    model.add(Dense(800, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.5))
    model.add(Dense(800, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.001, decay=1e-6, momentum=0, nesterov=False),
                  metrics=['accuracy'])
    return model

def MLP2():
    model = Sequential()
    model.add(Dropout(0.2))
    model.add(Dense(800, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.5))
    model.add(Dense(800, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
    return model

def MLP3():
    model = Sequential()
    model.add(Dropout(0.2))
    model.add(Dense(800, activation='relu', input_shape=(784,), kernel_regularizer=l1(0.001),
                    bias_regularizer=l1(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(800, activation='relu', kernel_regularizer=l1(0.001),
                    bias_regularizer=l1(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
    return model

def MLP4():
    model = Sequential()
    model.add(Dropout(0.2))
    model.add(Dense(800, activation='relu', input_shape=(784,), kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(800, activation='relu', kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
    return model

def train_model(model):
    loss_callback = LossHistories()

    model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                callbacks=[loss_callback],
                validation_data=(x_test, y_test))
    
    print loss_callback.losses
    print loss_callback.val
    
    return loss_callback.losses, loss_callback.val, loss_callback.val[0]

batch_size = 256
num_classes = 10
epochs = 100

# load the training and test data
x_train, y_train = load_mnist('training')
x_test, y_test = load_mnist('testing')
"""data = np.load('ORL_faces.npz')
x_train = data['trainX']
y_train = data['trainY']
x_test = data['testX']
y_test = data['testY']
""" 
print x_train.shape, x_test.shape 
 
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')

x_train = (x_train-np.mean(x_train, axis=0))/(np.std(x_train, axis=0)+0.0001)
x_test  = (x_test-np.mean(x_test, axis=0))/(np.std(x_test, axis=0)+0.0001)

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = MLP1()
# #model.summary()
loss1, val_acc1, val_acc_final1 = train_model(model)
# 
model = MLP2()
# #model.summary()
loss2, val_acc2, val_acc_final2 = train_model(model)

model = MLP3()
#model.summary()
loss3, val_acc3, val_acc_final3 = train_model(model)

model = MLP4()
#model.summary()
loss4, val_acc4, val_acc_final4 = train_model(model)


t = np.arange(0,epochs)
#s = loss_callback.losses
plt.plot(t,loss1)
plt.plot(t,loss2)
plt.plot(t,loss3)
plt.plot(t,loss4)
plt.legend(['loss for MLP1', 'loss for MLP2', 'loss for MLP3', 'loss for MLP4'])
plt.savefig("Task 3 Plot Loss.png")
plt.show()


t = np.arange(0,epochs)
#s = loss_callback.losses
plt.plot(t,val_acc1)
plt.plot(t,val_acc2)
plt.plot(t,val_acc3)
plt.plot(t,val_acc4)
plt.legend(['Val Acc for MLP1', 'Val Acc for MLP2', 'Val Acc for MLP3', 'Val Acc for MLP4'])
plt.savefig("Task 3 Plot Val Acc.png")
plt.show()

print "Classification Accuracies for Validation Set"
print val_acc_final1, val_acc_final2, val_acc_final3, val_acc_final4 
    
#score = model.evaluate(x_test, y_test, verbose=0)
#print 'Test loss: %f'%score[0]
#print 'Test accuracy:%f'%score[1]