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
num_class = 57
print('Loading dataset')
for i, vid in enumerate(os.listdir(dir)):
    print(i)
    if i == 5700:
        break
    if i % 2 == 1:
        continue
    CSVData = open(os.path.join(dir, vid))
    Array2d_result = np.loadtxt(CSVData, delimiter=",").tolist()
    X_train.append(Array2d_result)
    temp = np.zeros(num_class)
    temp[int(i/100)] = 1
    y_train.append(temp)


for i in range(len(X_train)):
    X_train[i] = Sample_frame(X_train[i], 25)
X_train = np.array(X_train)
# #todo may not need preprocessing but need more data to confirm
X_train = preprocess(X_train)[:, :, :]
# # X_train = np.save('X', X_train)
# # X_train = tf.ragged.constant(X_train)
train_ratio = 0.6
validation_ratio = 0.2
test_ratio = 0.2

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=1 - train_ratio)

# test is now 10% of the initial data set
# validation is now 15% of the initial data set
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
X_val = np.array(X_val)
y_val = np.array(y_val)
print(X_train.shape)
print(y_train.shape)
checkpoint_name='57class120epochSimpler32NetworkNoAugmentationPreprocessing'
log_dir = f'./checkpoint/{checkpoint_name}/logs'
checkpoint_dir = f'./checkpoint/{checkpoint_name}/' \
                 'model-epoch{epoch:02d}-val_acc{val_categorical_accuracy:.2f}.h5'
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    # tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
    #                                    save_weights_only=True,
    #                                    verbose=True,
    #                                    save_best_only=True)
]
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(25,258)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
# model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
# model.add(Dense(16, activation='relu'))
model.add(Dense(num_class, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=10e-5), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, callbacks=callbacks, batch_size=128)
results = model.evaluate(X_test, y_test)
#todo 57 class no preprocess have the best accuracy but val is only around 80%
print("test loss, test acc:", results)
