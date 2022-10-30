from sklearn import preprocessing
import numpy as np

x = np.array([[1., -1., 2.],
              [2., 0., 0.],
              [0., 1., -1.]])


x_scaled = preprocessing.scale(x)
print(x_scaled)
# 可以查看标准化后的数据的均值与方差，已经变成0,1了
a = x_scaled.mean(axis=0)
print(a)
# axis=1表示对每一行去做这个操作，axis=0表示对每一列做相同的这个操作
b = x_scaled.mean(axis=1)
print(b)
# 同理，看一下标准差
c = x_scaled.std(axis=0)
print(c)
