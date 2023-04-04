import numpy as np


def ScaleXYcoordinate(X: list, Y: list, ConWidth: int, ConHeight: int):
    #todo find out if this is x and y from each or all image
    X = np.array(X)
    Y = np.array(Y)
    width = np.max(X, axis=1) - np.min(X, axis=1)
    height = np.max(Y, axis=1) - np.min(Y, axis=1)
    diVx = width/ConWidth
    diVy = height/ConHeight

    X = (X.T/diVx).T
    Y = (Y.T/diVy).T
    return X.tolist(), Y.tolist()


