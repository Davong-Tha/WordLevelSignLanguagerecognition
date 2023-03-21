import os

import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers import Dense, LSTM
import tensorflow as tf
from sklearn.model_selection import train_test_split

from Preprocessing.FrameEqualizer import Sample_frame

X_train = []
y_train = []

dir = '../dataset/lsa64_raw/extracted'
# for i, gloss in enumerate(os.listdir(dir)):
#     gloss_path = os.path.join(dir, gloss)
num_class = 6
for i, vid in enumerate(os.listdir(dir)):
    print(i)
    if i == 600:
        break
    # CSVData = open(os.path.join(dir, vid))
    # Array2d_result = np.loadtxt(CSVData, delimiter=",").tolist()
    # X_train.append(Array2d_result)
    temp = np.zeros(num_class)
    temp[int(i/100)] = 1
    y_train.append(temp)


# for i in range(len(X_train)):
#     X_train[i] = Sample_frame(X_train[i], 80)
X_train = np.load('../dataVisualization/new_X.npy')
# X_train = np.save('X', X_train)
# X_train = tf.ragged.constant(X_train)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
print(X_train.shape)
print(y_train.shape)
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(None,150)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_class, activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2000, callbacks=[tb_callback])
print()