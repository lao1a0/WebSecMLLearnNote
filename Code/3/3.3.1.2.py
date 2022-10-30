from sklearn import preprocessing
import numpy as np


X = [[1., -1., 2.],
     [2., 0., 0.],
     [0., 1., -1.]]

min_max_scaler = preprocessing.MinMaxScaler()
x_minmax = min_max_scaler.fit_transform(X)
print(x_minmax)
