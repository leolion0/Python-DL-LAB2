from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn import preprocessing
from keras.optimizers import SGD, Nadam, rmsprop
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from keras.datasets import boston_housing
from keras.layers.core import Dropout
from keras import regularizers
from keras.callbacks import TensorBoard
from datetime import datetime
import numpy as np
from keras.layers.core import Dense, Activation
dataset = pd.read_csv('heart.csv').values
y =  dataset[:,13]
x = dataset[:,0:13]
#scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(x)
X= scaler.transform(x)

# split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
np.random.seed(155)
#define the model
model = Sequential()
model.add(Dense(16,input_dim=13, kernel_initializer='normal', activation='relu'))
###############################cahnge the activation fuction here
model.add(Dense(1, activation='sigmoid')) # output layer
#change the optimizer and learning rate

sgd = SGD(lr=0.002, momentum=0.9, nesterov=False)
adam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
Rmsprop = rmsprop(lr=0.002)
######################chose the optimizer and the learning rate

model.compile(loss='mean_squared_error', optimizer=Rmsprop,metrics=['acc'])

tensorboard = TensorBoard(log_dir=f".\logs\Tensors")
#######################################to change the patch size and epotches
Batchsize=8
Epotches=100

history = model.fit(x_train, y_train, epochs=Epotches, verbose=True, validation_data=(x_test, y_test), batch_size=Batchsize, callbacks=[tensorboard])
[test_loss, test_acc] = model.evaluate(x_test, y_test)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))
print(history.history.keys())

plt.figure(1)
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['loss'], label='Loss')
plt.legend(loc='upper left')
plt.show()
plt.figure(2)
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.plot(history.history['acc'], label='Accuracy')
plt.legend(loc='upper left')
plt.show()