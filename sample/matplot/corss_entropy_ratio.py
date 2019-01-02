import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random


# 值占比交叉熵趋势
def ratio_entropy():
    num = 20  # 样本分类
    total = 5000  # 样本个数
    range = 10  # 样本取值范围
    features = np.matrix(np.random.randint(range, size=(total, num)))
    labels = np.random.randint(1, size=(total, num))

    for label in labels:
        label[random.randint(0, num - 1)] = 1
    labels = np.matrix(labels)

    features_exp = np.exp(features)
    row_sums = features_exp * np.matrix(np.random.randint(low=1, high=2, size=(num, 1)))
    classify = features_exp / row_sums

    softmax_top = (classify * labels.T).diagonal()
    loss = np.log(softmax_top)

    plt.figure()
    plt.scatter(loss.tolist()[0], softmax_top.tolist()[0])
    plt.xlabel("corss entropy")
    plt.ylabel("softmax highest")
    plt.show()


ratio_entropy()
