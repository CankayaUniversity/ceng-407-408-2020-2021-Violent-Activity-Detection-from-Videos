# -*- coding: utf-8 -*-
"""Untitled4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1izhM10tbmfjA-7a4YnzhVpvc0-1Rb-12
"""

#Mount Drive

from google.colab import drive
drive.mount("/content/drive")

from zipfile import ZipFile
file_name = "/path/.zip"

with ZipFile(file_name,'r') as zip:
   zip.extractall("/Videos")
   print('Done')

import numpy as np
 
#fight1= open(, "r")
# Read the array from disk
text_data = np.loadtxt("/path/.txt")

# Note that this returned a 2D array!
print (text_data.shape)

# However, going back to 3D is easy if we know the 
# original shape of the array
data_3D = text_data.reshape((2332,30,1000))
#print (data_3D.shape)
    
# Just to check that they're the same...
#print(data_3D)

import numpy as np
 
#fight1= open(, "r")
# Read the array from disk
label_fight_data = np.loadtxt("FightLabel.txt" , dtype=np.str)
label_normal_data = np.loadtxt("NormalLabel.txt", dtype=np.str)

dataset_label = np.vstack((label_fight_data, label_normal_data))

print(dataset_label.shape)
np.savetxt("Label.txt",dataset_label, fmt='%s')

# Note that this returned a 2D array!
print (dataset_label)

print(dataset_3D_array)

print(dataset_3D_array.shape)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Softmax, MaxPool2D, Conv2D
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout # one of the best regularizers\n",
from tensorflow.keras.regularizers import l1,l2,l1_l2
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

import numpy as np
 
#fight1= open(, "r")
# Read the array from disk
dataset_label_array = np.loadtxt("Label.txt", dtype='str')

# Note that this returned a 2D array!
print (dataset_label_array.shape)

print(dataset_label_array)

dataset_label = []
with open('Label.txt') as inf:
    for line in inf:
        parts = line.split() # split line into parts
        if len(parts) > 1:   # if at least 2 parts/columns
            dataset_label.append(int(parts[1]))
            print (parts[1])

print(dataset_label)

print(len(dataset_label))

X, y = dataset_3D_array, dataset_label

print(X)

print(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, random_state=42)

X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, train_size=0.80, random_state=42)

print(X_train.shape)

print(X_validation.shape)

print(X_train)

print(X_train.shape)

print(y_train)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
y_train = np.array(y_train)
X_validation = np.array(X_validation)
y_validation = np.array(y_validation)

print(X_test.shape)
print(y_test.shape)
print(y_train.shape)
print(X_train.shape)
print(X_validation.shape)
print(y_validation.shape)

print(y_test)

print(X_test)

print(X_train.shape)

from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# model 1
# lstm(128) -         dense(64)- dense(32)


# model 2
#  lstm(128, return_sequence = true) - lstm(64) -        dense(32) 
  

# model 3
# lstm(128) -         dense(100)   dense(50)


# Adam learning rate = 1e-3, 1e-4

# compile-----------------------
# epoch = 50 - 100 - 150
# batch_size = 10 - 16

model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1:]), activation= "tanh", return_sequences= True, recurrent_activation="sigmoid",recurrent_dropout=0.0,unroll=False,use_bias=True))
model.add(Dropout(0.2))

model.add(LSTM(64, input_shape=(X_train.shape[1:]),  activation= "tanh", recurrent_activation="sigmoid",recurrent_dropout=0.0,unroll=False,use_bias=True))
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

model.summary()

opt = tf.keras.optimizers.Adam(learning_rate=1e-4)

model.compile(
    loss='SparseCategoricalCrossentropy',
    optimizer=opt,
    metrics=['accuracy',precision_m, recall_m],
)

history = model.fit(X_train, y_train, epochs=150, batch_size=16, validation_data = (X_validation, y_validation))

import pandas as pd
df = pd.DataFrame(history.history)
df.head()
loss_plot = df.plot(y = 'loss' , title = 'Loss vs Epochs', legend= False)
loss_plot.set(xlabel='Epochs')
        
acc_plot = df.plot(y = 'accuracy' , title = 'Acc vs Epochs', legend= False)
acc_plot.set(xlabel='Epochs')
        
loss_plot = df.plot(y = 'val_loss' , title = 'Loss vs Epochs', legend= False)
loss_plot.set(xlabel='Epochs')
acc_plot = df.plot(y = 'val_accuracy' , title = 'Acc vs Epochs', legend= False)
acc_plot.set(xlabel='Epochs')

# test the model



# evaluate the model
loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=2)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
labels = [1, 0]
test_pred_raw = model.predict(X_test)

test_pred = np.argmax(test_pred_raw, axis=1)

# Calculate the confusion matrix using sklearn.metrics

cm = confusion_matrix(y_test, test_pred, labels)
con_mat_df = pd.DataFrame(cm)
print(cm)
figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues, fmt = 'd')
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()