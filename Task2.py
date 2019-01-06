import keras
#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt

import numpy as np


class LossHistories(keras.callbacks.Callback):
    def on_train_begin(self, data, logs={}):
        #self.x_test = data[0]
        #self.y_test = data[1]
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        #y_pred = self.model.predict(self.validation_data[0])
        #self.aucs.append(roc_auc_score(self.validation_data[1], y_pred))
        return

batch_size = 32
num_classes = 20
epochs = 20

# load the training and test data
data = np.load('ORL_faces.npz')
x_train = data['trainX']
y_train = data['trainY']
x_test = data['testX']
y_test = data['testY']
 
print x_train.shape, x_test.shape 
 
#x_train = x_train.reshape(60000, 784)
#x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = (x_train-np.mean(x_train, axis=0))/(np.std(x_train, axis=0)+0.0001)
x_test = (x_test-np.mean(x_test, axis=0))/(np.std(x_test, axis=0)+0.0001)

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(10304,)))
#model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

loss_callback = LossHistories()

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    callbacks=[loss_callback],
                    validation_data=(x_test, y_test))

print loss_callback.losses
"""t = np.arange(0,epochs)
s = loss_callback.losses
plt.plot(t,s)
plt.savefig("task2_plot.png")
plt.show()"""

    
score = model.evaluate(x_test, y_test, verbose=0)
print 'Test loss: %f'%score[0]
print 'Test accuracy:%f'%score[1]