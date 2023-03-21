import os

import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, LSTM
import tensorflow as tf
from sklearn.model_selection import train_test_split

from Preprocessing.FrameEqualizer import Sample_frame
from Preprocessing.RunPreprocess import preprocess

X_train = []
y_train = []

dir = '../../dataset/lsa64_raw/extracted'
# for i, gloss in enumerate(os.listdir(dir)):
#     gloss_path = os.path.join(dir, gloss)
num_class = 6
print('Loading dataset')
for i, vid in enumerate(os.listdir(dir)):

    if i == 600:
        break
    CSVData = open(os.path.join(dir, vid))
    Array2d_result = np.loadtxt(CSVData, delimiter=",").tolist()
    X_train.append(Array2d_result)
    temp = np.zeros(num_class)
    temp[int(i/100)] = 1
    y_train.append(temp)


for i in range(len(X_train)):
    X_train[i] = Sample_frame(X_train[i], 25)
X_train = np.array(X_train)
X_train = preprocess(X_train)[:, :, :]
# X_train = np.save('X', X_train)
# X_train = tf.ragged.constant(X_train)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
print(X_train.shape)
print(y_train.shape)
checkpoint_name='25frame2'
log_dir = f'./checkpoint/{checkpoint_name}/logs'
checkpoint_dir = f'./checkpoint/{checkpoint_name}/' \
                 'model-epoch{epoch:02d}-val_acc{val_categorical_accuracy:.2f}.h5'
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
                                       save_weights_only=True,
                                       verbose=True,
                                       save_best_only=True)
]
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(None,258)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(num_class, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=10e-5), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2000, callbacks=callbacks, batch_size=128)
