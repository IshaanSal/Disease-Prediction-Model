# -*- coding: utf-8 -*-
"""diseaseModel.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FNQqSUQ9-fXsat1dWy3FmLD5SgTKWj2m
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout

data = pd.read_csv("Training.csv", sep=",")
data2 = pd.read_csv("Testing.csv", sep=',')

df = data.to_numpy()
df2 = data2.to_numpy()

print(df.shape)
print(df2.shape)

x_train = df[0:4920,0:132]
y_train = df[0:4920,132]

print(x_train)
print(y_train)
 
x_test = df2[0:42,0:132]
y_test = df2[0:42,132]
print(x_test)
print(y_test)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

train_labels = []
for val in y_train:
  count = 0
  for val2 in train_labels:
    if (val == val2):
      count += 1
  if count == 0:
    train_labels.append(val)

print(len(train_labels))

for val1 in range(len(y_train)):
  for val2 in range(len(train_labels)):
    if (y_train[val1] == train_labels[val2]):
      df[val1][133] = val2
#print(df)
train_encode = df[0:4920,133]
print(train_encode)

for val1 in range(len(y_test)):
  for val2 in range(len(train_labels)):
    if (y_test[val1] == train_labels[val2]):
      df2[val1][133] = val2
#print(df)
train_encode2 = df2[0:4920,133]
print(train_encode2)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

model = Sequential()
model.add(Dense(4920, input_dim=132, activation='relu'))
model.add(Dense(42, activation="softmax"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

train_encode = train_encode.astype('float32')
train_encode2 = train_encode2.astype('float32')

model.fit(x_train, train_encode, epochs=5)

test_loss, test_acc = model.evaluate(x_test, train_encode2)

final = int(test_acc*100)
print(str(final) + '%')
