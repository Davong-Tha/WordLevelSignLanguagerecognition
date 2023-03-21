"""
shift a landmark to the origin and while every landmark shift by the same amount
"""
import numpy as np


def ShiftFeatureToOrigin(X: list, Y: list, idx):

    X = np.array(X)
    Y = np.array(Y)
    point = (X[:, idx], Y[:, idx])

    X = (X.T - point[0]).T
    Y = (Y.T - point[1]).T
    return X.tolist(), Y.tolist()
