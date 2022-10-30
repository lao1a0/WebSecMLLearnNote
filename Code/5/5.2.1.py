import numpy as np
from sklearn.neighbors import NearestNeighbors
print(__doc__)


X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)
# 返回值indices：第0列元素为参考点的索引，后面是(n_neighbors - 1)个与之最近的点的索引
# 返回值distances：第0列元素为与自身的距离(为0)，后面是(n_neighbors - 1)个与之最近的点与参考点的距离
print(distances)
print(indices)
print(nbrs.kneighbors_graph(X).toarray())
