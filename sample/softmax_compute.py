import numpy as np
import random

'''
根据文章中的步骤执行softmax相关的运算
'''
# 运算值的取值范围(0~Range)
Range = 5
# 500个样本
O = 500
# 20个特征
N = 20
# 5个分类
M = 5
'''指定有500个样本(O)，每个样本有20个特征(N)，对应的分类有5个(M)'''


def build_features():
    """
    随机生成样本特征，特征的形状为(O,N)——O行样本，每个样本N个特征。
    特征的取值限定在0~5。
    真实的权重模型为WX+B，所以增加一个项表示B
    :return: np.matrix shape=(O,N)
    """
    _features = np.random.rand(O, N + 1) * Range
    for f in _features:
        f[0] = 1;
    features = np.matrix(_features)
    print("Matrix Features. type={} shape={}".format(type(features), features.shape))
    return features


def build_labels():
    """
    随机生成标签，标签的形状为(O,M)——O行样本，每个样本对应5个分类之一
    标签是一个一行数据只有一个元素为1，其余元素为0的矩阵。
    :return: np.matrix shape=(O,M)
    """
    _labels = np.zeros((O, M))

    for label in _labels:
        index = random.randint(0, M - 1)
        label[index] = 1

    labels = np.matrix(_labels)

    print("Matrix Labels. type={} shape={}".format(type(labels), labels.shape))

    return labels


def build_weights():
    """
    随机生成权重参数，参数的形状为(M,N)——对应M个分类，N个特征
    真实的权重模型为WX+B，所以增加一个项表示B
    :return:
    """

    weights = np.matrix(np.random.rand(M, N + 1))

    print("Matrix Weight. type={} shape={}".format(type(weights), weights.shape))

    return weights


def build_e():
    """
    获取标准量E
    :return:
    """
    E1 = np.matrix(np.ones((M, 1)))
    E2 = np.matrix(np.ones((O, 1)))

    print("Matrix E1. type={} shape={}".format(type(E1), E1.shape))
    print("Matrix E2. type={} shape={}".format(type(E2), E2.shape))

    return E1, E2


def liner(w, x):
    """
    计算线性结构
    :param w:
    :param x:
    :return:
    """
    return x * w.T


def exp(l):
    """
    指数矩阵运算
    :param l:
    :return:
    """
    return np.exp(l)


if __name__ == '__main__':
    X = build_features()
    P = build_labels()
    W = build_weights()
    E1, E2 = build_e()
    step = .1

    # 1.softmax计算
    Liner = liner(W, X)
    G = exp(Liner)
    S = G * E1
    Q = G / S
    print("Softmax Result. Matrix type={} shape={}".format(type(Q), Q.shape))

    # 验证softmax计算的数据
    print("classify number:{}".format(Liner[0]))
    print("Softmax result:{}".format([str(round(i * 100, 4)) + '%' for i in np.array(Q)[0]]))

    # 2.交叉熵（极大似然估计）计算
    H = (P * np.log(Q.T)).diagonal()
    LOSS = H * E2 / O
    print("LOSS. type={} shape={} data={}".format(type(LOSS), LOSS.shape, LOSS))

    # 3.偏导&更新权重参数计算
    D = ((P - Q).T * X) / O
    print("Weight[0].value={}".format(W[0]))
    W = W + step * D
    print("Weight[0].afterValue={}".format(W[0]))
