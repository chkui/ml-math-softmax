import os
import numpy as np
import tempfile


class Softmax:
    def __init__(self, features, labels):
        self.__features = features
        self.__labels = labels
        self.__path = tempfile.gettempdir() + '/ml-math-softmax-weight' + str(
            Softmax.__get_hash(features) + Softmax.__get_hash(labels))
        self.__weight = None
        # 用于 softmax 归一化计算分布的标量矩阵
        self.__e_softmax = np.ones((labels.shape[1], 1))
        # 用于 损失函数计算的标量矩阵
        self.__e_loss = np.ones((features.shape[0], 1))
        # flag用于标记运算符号
        # flag如果是-1,那么损失函数就是求最小值，那么优化器求差值。
        # flag如果是+1损失函数就是求最大值，那么优化器求和
        self.__flag = 1
        self.__init_data()

    @staticmethod
    def __get_hash(matrix):
        _hash = 0
        row_num = 0
        col_num = 0
        row_total = matrix.shape[0]
        col_total = matrix.shape[1]
        for _list in matrix[row_num:row_num + 1, :].tolist():
            for _el in _list:
                _hash = _hash + hash((_el * row_total + row_num) * col_total + col_num)
                col_num = col_num + 1
            row_num = row_num + 1
        return _hash

    def __init_data(self):
        is_exists = self.__read_weight()
        if not is_exists:
            print('Weight file not exists! Building...')
            self.__build_weight()

    def __read_weight(self):
        print("Weight Temp File Path:{}".format(self.__path))
        if os.path.exists(self.__path):
            with open(self.__path, 'rb') as file:
                np_data = np.load(file)
                file.close()
                self.__weight = np_data
            return True
        else:
            return False

    def __build_weight(self):
        classify = self.__labels.shape[1]
        features_num = self.__features.shape[1]
        self.__weight = np.zeros((classify, features_num))
        self.__save_weight()

    def __save_weight(self):
        with open(self.__path, 'wb+') as file:
            np.save(file, self.__weight)

    def __softmax(self):
        liner = self.__features * self.__weight.T
        exp = np.exp(liner)
        den = exp * self.__e_softmax
        q = exp / den
        return q

    def __loss(self, q):
        h = self.__labels * np.log(q.T)
        h = h.diagonal()
        loss = self.__flag * h * self.__e_loss / self.__e_loss.shape[0]
        return loss

    def __optimizer(self, q, step):
        d = ((self.__flag * self.__labels - self.__flag * q).getT() * self.__features) / self.__features.shape[0]
        self.__weight = self.__weight + (self.__flag * step) * d

    def train(self, handle, repeat=2000, save_point=200, step=0.1):
        """
        训练
        :param handle: 单轮训练的回调，用于输出各项数据 (count, loss, )
        :param repeat: 重复的轮次,每轮会执行一次存储 2000
        :param save_point: 多少次进行物理存储，设定为0时不存储，默认200
        :param step: 优化器步近量
        :return:
        """
        print("Weight shape={}".format(self.__weight.shape))
        count = 0
        while count < repeat:
            q = self.__softmax()
            loss = self.__loss(q)
            self.__optimizer(q, step)
            if 0 == count % save_point:
                self.__save_weight()
            count = count + 1
            handle(count, loss)
