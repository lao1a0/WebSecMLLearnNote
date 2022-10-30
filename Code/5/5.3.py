from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from nltk.probability import FreqDist
import nltk
import numpy as np
import warnings
warnings.filterwarnings("ignore")
# 测试样本数
N = 90


def load_user_cmd_new(filename):
    cmd_list = []
    dist = []
    with open(filename) as f:
        i = 0
        x = []
        for line in f:
            line = line.strip('\n')
            x.append(line)
            dist.append(line)
            i += 1
            if i == 100:
                cmd_list.append(x)
                x = []
                i = 0

    fdist = FreqDist(dist)  # 按照频率排序
    # for i in fdist:
    #     print(i, fdist[i])
    return cmd_list, fdist


def load_user_cmd(filename):
    cmd_list = []
    dist_max = []
    dist_min = []
    dist = []
    with open(filename) as f:
        i = 0
        x = []
        for line in f:
            line = line.strip('\n')
            x.append(line)
            dist.append(line)
            i += 1
            if i == 100:
                cmd_list.append(x)
                x = []
                i = 0

    fdist = FreqDist(dist).keys()
    dist_max = set(fdist[0:50])
    dist_min = set(fdist[-50:])
    return cmd_list, dist_max, dist_min


def get_user_cmd_feature(user_cmd_list, dist_max, dist_min):
    user_cmd_feature = []
    for cmd_block in user_cmd_list:
        f1 = len(set(cmd_block))
        fdist = FreqDist(cmd_block).keys()
        f2 = fdist[0:10]
        f3 = fdist[-10:]
        f2 = len(set(f2) & set(dist_max))
        f3 = len(set(f3) & set(dist_min))
        x = [f1, f2, f3]
        user_cmd_feature.append(x)
    return user_cmd_feature


def get_user_cmd_feature_new(user_cmd_list, dist):
    user_cmd_feature = []

    for cmd_list in user_cmd_list:
        v = [0]*len(dist)
        for i in range(0, len(dist)):
            if dist[i] in cmd_list:
                v[i] += 1
        user_cmd_feature.append(v)

    return user_cmd_feature


def get_label(filename, index=0):
    x = []
    with open(filename) as f:
        for line in f:
            line = line.strip('\n')
            x.append(int(line.split()[index]))
    return x


if __name__ == '__main__':
    user_cmd_list, dist = load_user_cmd_new("./Data/MasqueradeDat/User3")
    print(dist)
    print("Dist:(%s)" % dist)
    user_cmd_feature = get_user_cmd_feature_new(user_cmd_list, dist)
    print(user_cmd_feature)
    labels = get_label("./Data/MasqueradeDat/label.txt", 2)
    y = [0]*50+labels

    x_train = user_cmd_feature[0:N]
    y_train = y[0:N]

    x_test = user_cmd_feature[N:150]
    y_test = y[N:150]

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x_train, y_train)
    y_predict = neigh.predict(x_test)

    score = np.mean(y_test == y_predict)*100

    print(score)
    print('y_test\n', y_test)
    print('y_predict\n', y_predict)
    print('score\n', score)

    print('classification_report(y_test, y_predict)\n',
          classification_report(y_test, y_predict))

    print('metrics.confusion_matrix(y_test, y_predict)\n',
          metrics.confusion_matrix(y_test, y_predict))
    print(cross_val_score(
        neigh, user_cmd_feature, y, n_jobs=-1, cv=10))


# 测试样本数
# N = 90

# """
# 数据收集和数据清洗(清洗换行符\n)
# 从scholaon数据集的user3文件导入信息;一百条命令组成一个列表x[],最终组成二维列表cmd_set[[]];
# 返回二维列表,最频繁50条命令,和最不频繁50条命令
# """


# def load_user_cmd(filename):
#     cmd_set = []  # 每100个一组
#     dist_max = []
#     dist_min = []
#     dist = []
#     with open(filename) as f:
#         i = 0
#         x = []
#         for line in f:
#             line = line.strip('\n')
#             x.append(line)
#             dist.append(line)
#             i += 1
#             if i == 100:
#                 cmd_set.append(x)
#                 x = []
#                 i = 0
#     # print(len(dist))
#     fdist = list(FreqDist(dist).keys())
#     # print(len(fdist))
#     dist_max = set(fdist[0:50])
#     dist_min = set(fdist[-50:])
#     return cmd_set, dist_max, dist_min


# """
# 特征化
# 将load_user_cmd函数的输出作为输入;
# 以100个命令为统计单元,作为一个操作序列,去重后的操作命令个数作为特征;(函数FreqDist会统计每个单词的频度,重新整合成一个+1维度的新的列表)
# KNN只能以标量作为输入参数,所以需要将f2和f3表量化,最简单的方式就是和统计的最频繁使用的前50个命令以及最不频繁使用的前50个命令计算重合程度。
# 返回一个150x3的列表;3里的0:不重复单词的个数,1:最频繁单词重合程度<=min{10,50},2最不频繁单词重合程度<=min{10,50}
# """


# def get_user_cmd_feature(user_cmd_set, dist_max, dist_min):
#     user_cmd_feature = []
#     # print(len(user_cmd_set)) # 一共150组
#     for cmd_block in user_cmd_set:
#         f1 = len(set(cmd_block))  # 去重
#         # a = len(cmd_block)
#         fdist = list(FreqDist(cmd_block).keys())

#         f2 = fdist[0:10]
#         f3 = fdist[-10:]
#         f2 = len(set(f2) & set(dist_max))  # 计算重合度
#         f3 = len(set(f3) & set(dist_min))
#         x = [f1, f2, f3]
#         # print(x)
#         # print(f1, a)
#         # print(f2)
#         # print(f3)
#         user_cmd_feature.append(x)
#     return user_cmd_feature


# """
# 训练模型
# 导入标识文件,100x50,正常命令为0,异常命令为1;
# 从标识文件中加载针对操作序列正确/异常的标识
# 返回一个容量为100的list 0/1数值,(只要这一行有1)
# """


# def get_label(filename, index=0):
#     # print(index)
#     x = []
#     with open(filename) as f:
#         for line in f:
#             line = line.strip('\n')  # 清空每行的\n
#             x.append(int(line.split()[index]))  # 每行第一个0/1,这行数据是正/异常数据标识位
#     # print(x)
#     # print(len(x))
#     return x


# if __name__ == '__main__':

#     # user_cmd_set：一共150组，每组100个
#     # user_cmd_dist_max：最常用的前50个
#     # user_cmd_dist_min：最常用的后50个
#     user_cmd_set, user_cmd_dist_max, user_cmd_dist_min = load_user_cmd(
#         "./Data/MasqueradeDat/User3")

#     # user_cmd_feature：特征化之后的值，list,150组
#     user_cmd_feature = get_user_cmd_feature(
#         user_cmd_set, user_cmd_dist_max, user_cmd_dist_min)

#     labels = get_label("./Data/MasqueradeDat/label.txt", 2)

#     y = [0]*50+labels  # y长度150,labels长度100

#     x_train = user_cmd_feature[0:N]
#     y_train = y[0:N]

#     x_test = user_cmd_feature[N:150]
#     y_test = y[N:150]

#     neigh = KNeighborsClassifier(n_neighbors=3)
#     neigh.fit(x_train, y_train)
#     y_predict = neigh.predict(x_test)

#     score = np.mean(y_test == y_predict)*100

#     print('y_test\n', y_test)
#     print('y_predict\n', y_predict)
#     print('score\n', score)

#     print('classification_report(y_test, y_predict)\n',
#           classification_report(y_test, y_predict))

#     print('metrics.confusion_matrix(y_test, y_predict)\n',
#           metrics.confusion_matrix(y_test, y_predict))
