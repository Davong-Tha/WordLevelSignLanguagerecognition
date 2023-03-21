import os

import matplotlib.pyplot as plt
import numpy as np

dir = '../dataset/lsa64_raw/extracted'
# for i, gloss in enumerate(os.listdir(dir)):
#     gloss_path = os.path.join(dir, gloss)
X_train = np.load('./X.npy').tolist()
# num_class = 6
# for i, vid in enumerate(os.listdir(dir)):
#     print(i)
#     if i == 100:
#         break
#     CSVData = open(os.path.join(dir, vid))
#     vidFrames = np.loadtxt(CSVData, delimiter=",").tolist()
#     X_train.append(vidFrames)

# X_train = [v for i, v in enumerate(X_train) if i % 2 == 0]
new_X_train = []

for i, vid in enumerate(X_train):
    new_X_train.append([])
    for frame in vid:
        poseIdx = 33*4
        lhIdx = 33*4 + 21*3
        pose = frame[:poseIdx]
        lh = frame[poseIdx:lhIdx]
        rh = frame[lhIdx:]

        lh = [v for i, v in enumerate(lh) if i % 3 != 0]
        rh = [v for i, v in enumerate(rh) if i % 3 != 0]
        k = 4
        del pose[k - 1::k]
        k = 3
        del pose[k - 1::k]

        row_XY = pose + lh + rh
        new_X_train[i].append(row_XY)

        # X = [v for i, v in enumerate(row_XY) if i % 2 != 0]
        # Y = [v for i, v in enumerate(row_XY) if i % 2 == 0]
np.save('new_X', np.array(new_X_train))
