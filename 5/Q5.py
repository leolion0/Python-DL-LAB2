from keras.models import Sequential
from keras import layers
import keras
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn import preprocessing
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.layers.embeddings import Embedding
from keras.layers import Flatten
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.layers import Dropout,Dense
from keras.layers import Conv1D,MaxPooling1D,LSTM
import matplotlib.pyplot as plt
import io

df_train = pd.read_csv('train.tsv', encoding='latin-1', sep='\t')
df_test = pd.read_csv('test.tsv', encoding='latin-1', sep='\t')

# print(df_train.head())
# print(df_test.head())

sentences_train = df_train['Phrase']
Y_train = df_train['Sentiment'].values
# test data tsv
sentences_test = df_test['Phrase']

#tokenizing data
tokenizer = Tokenizer(num_words=200)
tokenizer.fit_on_texts(sentences_train)
vocab_size = len(tokenizer.word_index) + 1
#getting the vocabulary of data
sentences = tokenizer.texts_to_matrix(sentences_train)

le = preprocessing.LabelEncoder()
y = le.fit_transform(Y_train)
X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)
##
padded_docs = sequence.pad_sequences(sentences,maxlen=200)
##
model = Sequential()
# Add Embedding
model.add(Embedding(vocab_size, 100, input_length=200))
model.add(LSTM(units=12, dropout=0.2, recurrent_dropout=0.2))
model.add(Dropout(0.3))
model.add(layers.Dense(100,input_dim=200, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
history=model.fit(X_train,y_train, epochs=2, verbose=True, validation_data=(X_test,y_test), batch_size=25)

[test_loss, test_acc] = model.evaluate(X_test, y_test)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
