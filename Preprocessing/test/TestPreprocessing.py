import numpy as np
import pandas as pd

from Preprocessing.CenterRelocator import ShiftFeatureToOrigin
from Preprocessing.DepthEqualizer import ScaleXYcoordinate
from sklearn.preprocessing import StandardScaler

def preprocess(vids):
    columns = []
    processedVid = []
    for i in range(33*4):
        columns.append('pose' +str(i))
    for i in range(21*3):
        columns.append('lh' +str(i))
    for i in range(21*3):
        columns.append('rh' +str(i))
    XY = []
    Depth = []
    Visibility = []

    for i in range(33 * 4):
        if i != 0 and ((i - 2) / 4 + 1).is_integer():
            Depth.append('pose' + str(i))
            continue
        if ((i - 3) / 4 + 1).is_integer():
            Visibility.append('pose' + str(i))
            continue
        XY.append('pose' + str(i))
    for i in range(21 * 3):
        if i != 0 and ((i + 1) / 3).is_integer():
            Depth.append('lh' + str(i))
            continue
        XY.append('lh' + str(i))
    for i in range(21 * 3):
        if i != 0 and ((i + 1) / 3).is_integer():
            Depth.append('rh' + str(i))
            continue
        XY.append('rh' + str(i))
    num = 0
    for i, vid in enumerate(vids):
        print(i)
        df = pd.DataFrame(vid,  columns=columns)
        dfXY = df[XY]
        dfXY = np.array(dfXY.values.tolist())
        D = df[Depth].values.tolist()
        V = df[Visibility].values.tolist()
        X = dfXY[:, ::2].tolist()
        Y = dfXY[:, 1::2].tolist()
        X, Y = ScaleXYcoordinate(X, Y, 1, 1)
        X, Y = ShiftFeatureToOrigin(X, Y, 0)
        D = np.array(D)
        X = np.array(X)
        Y = np.array(Y)
        V = np.array(V)
        X = np.append(X, Y, axis=1)
        X = np.append(X, D, axis=1)
        X = np.append(X, V, axis=1)
        processedVid.append(X.tolist())

    processedVid = np.array(processedVid)
    for i in range(80):
        print(i)
        scaler = StandardScaler()

        temp1 = processedVid[:, :, i]
        temp2 = scaler.fit_transform(np.reshape(temp1, (-1, 1)))
        temp3 = np.reshape(temp2, (processedVid.shape[0], processedVid.shape[1]))
        processedVid[:, :, i] = temp3

    return processedVid