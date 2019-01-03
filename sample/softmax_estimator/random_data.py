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
        self.__featuresFileName = '/ml-math-softmax-features' + str(_hash)
        self.__labelFileName = '/ml-math-softmax-label' + str(_hash)

    @staticmethod
    def __exist_read(path):
        if os.path.exists(path):
            with open(path, 'rb') as file:
                np_data = np.load(file)
                file.close()
                return True, np_data
        else:
            return False, False

    @staticmethod
    def __save_write(path, np_data):
        with open(path, 'wb+') as file:
            np.save(file, np_data)
        return np_data

    def __features(self):
        path = tempfile.gettempdir() + self.__featuresFileName
        print("Features Temp File Path:{}".format(path))
        has_data, _features = RandomData.__exist_read(path)
        if has_data:
            return np.matrix(_features)
        else:
            print('Features file not exists! Building...')
            _features = np.random.rand(self.__featuresShape[0], self.__featuresShape[1] + 1) * self.__feature_range
            for f in _features:
                f[0] = 1
            return RandomData.__save_write(path, np.matrix(_features))

    def __label(self):
        path = tempfile.gettempdir() + self.__labelFileName
        print("Labels Temp File Path:{}".format(path))
        has_data, _labels = RandomData.__exist_read(path)
        if has_data:
            return np.matrix(_labels)
        else:
            print('Labels file not exists! Building...')
            _labels = np.zeros(self.__labelShape)
            m = self.__labelShape[1]
            for label in _labels:
                index = random.randint(0, m - 1)
                label[index] = 1

            labels = np.matrix(_labels)
            return RandomData.__save_write(path, labels)

    def establish_data(self):
        features = self.__features()
        labels = self.__label()
        return features, labels


if __name__ == '__main__':
    data = RandomData(500, 20, 50)
    features, labels = data.establish_data()
    print("features type:{}".format(type(features)))
