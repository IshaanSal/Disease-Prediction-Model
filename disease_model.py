import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.utils import to_categorical

df_train = pd.read_csv("Training.csv")
df_train2 = pd.read_csv("Training.csv")
df_test = pd.read_csv("Testing.csv")

train_label_copy = df_train["prognosis"]

train_labels = []

for val in train_label_copy:
  count = 0
  for val2 in train_labels:
    if (val == val2):
      count += 1
  if count == 0:
    train_labels.append(val)

def prognosis_encode(arr):
  encoded_column = []

  for val1 in range(len(arr)):
    for val2 in range(len(train_labels)):
      if (arr[val1] == train_labels[val2]):
        encoded_column.append(val2)

  return encoded_column

df_train2 = df_train["prognosis"] 
int_encoded_col = prognosis_encode(df_train2)
df_train2 = np.column_stack((df_train2, int_encoded_col))
df_train2 = np.delete(df_train2, 0, 1)
y_train = to_categorical(df_train2, num_classes=41)

X_train = df_train.drop(["prognosis", "encoder"], axis='columns')
X_test = df_test.drop(["prognosis", "encoder"], axis='columns')

X_train = X_train.astype('float32')
y_train = y_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

model = Sequential()
model.add(Dense(64, input_dim=132, activation='relu'))
model.add(Dense(32, activation="relu"))
model.add(Dense(41, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, batch_size=32)


#For saving the model:
'''
import sklearn
import joblib
filename = "trained_disease_model.pkl"
joblib.dump(model, filename)
'''