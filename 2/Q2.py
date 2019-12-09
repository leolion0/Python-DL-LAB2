import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.utils import to_categorical
from keras import backend as K
import keras.callbacks
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
##
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

#Importing the Dataset
dataset = pd.read_csv('heart.csv').values
y =  dataset[:,13]
x = dataset[:,0:13]
#scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit on training set .
scaler.fit(x)
X= scaler.transform(x)

# split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Fitting the Logistic Regression on Traning set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, solver = "lbfgs")
classifier.fit(x_train,y_train)
#Predicting Test set Result
y_pred = classifier.predict(x_test)
#Making the Confussion Matrix and Print Accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Logistic Regression :")
print("Accuracy = ", accuracy)

# #creating network
# model = Sequential()
# model.add(Dense(13, input_dim=13, activation='relu', kernel_initializer='normal'))
# model.add(Dense(13, kernel_initializer='normal', activation='relu'))
# model.add(Dense(2, activation='softmax'))
#
# #compile model
# epochs = 20
# lrate = 0.001
# adam = Adam (lr= lrate)
# model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
# print(model.summary())
#
# #tensor board
# tb = keras.callbacks.TensorBoard(log_dir=f".\logs\Tensors", histogram_freq=0,write_graph=True, write_images=True)
#
# history=model.fit(x_train, Ytrain, validation_data=(x_test, Ytest),epochs=200, batch_size=10, verbose = 10, callbacks=[tb])
#
# scores = model.evaluate(x_test, Ytest, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))
# print('Test  loss:', scores[0]*100)
# model.save('./model' + '.h5')
#
# # Model accuracy
# # plt.figure(1)
# # plt.plot(history.history['accuracy'])
# # plt.plot(history.history['val_accuracy'])
# # plt.title('Model Accuracy')
# # plt.ylabel('accuracy')
# # plt.xlabel('epoch')
# # plt.legend(['train', 'test'])
# # plt.show(block=False)
# #
# # # Model Losss
# # plt.figure(2)
# # plt.plot(history.history['loss'])
# # plt.plot(history.history['val_loss'])
# # plt.title('Model Loss')
# # plt.ylabel('loss')
# # plt.xlabel('epoch')
# # plt.legend(['train', 'test'])
# # plt.show()
#
# #prediction
# pred = np.argmax(model.predict(x_test), axis=1)
#
# print('Results for Logistic Model')
# print(accuracy_score(y_test, pred))
# print(classification_report(y_test, pred))