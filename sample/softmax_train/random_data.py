import numpy as np
import tempfile
import os
import random


class RandomData:
    """
    用于随机生成 特征和标签
    """

    def __init__(self, total, classify, feature, feature_range=5):
        """
        构造函数
        :param total: 样本总数
        :param classify: softmax分类数
        :param feature: 特征总数
        :param feature_range:  特征取值范围，默然为0~5
        """
        self.__featuresShape = (total, feature)
        self.__labelShape = (total, classify)
        self.__feature_range = feature_range
        _hash = hash((total, classify, feature))

    def __features(self):
        _features = np.random.rand(self.__featuresShape[0], self.__featuresShape[1] + 1) * self.__feature_range
        for f in _features:
            f[0] = 1
        return np.matrix(_features)

    def __label(self):
        _labels = np.zeros(self.__labelShape)
        m = self.__labelShape[1]
        for label in _labels:
            index = random.randint(0, m - 1)
            label[index] = 1
        return np.matrix(_labels)

    def establish_data(self):
        features = self.__features()
        labels = self.__label()
        return features, labels


if __name__ == '__main__':
    data = RandomData(500, 20, 50)
    features, labels = data.establish_data()
    print("features type:{}".format(type(features)))
