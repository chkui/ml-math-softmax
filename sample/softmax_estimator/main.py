from sample.softmax_estimator.random_data import RandomData
from sample.softmax_estimator.softmax_modual import Softmax


def handle(count, loss):
    print("Train Count:{}. Loss Value:{}".format(count, loss))


if __name__ == '__main__':
    data = RandomData(500, 20, 50)
    features, labels = data.establish_data()
    softmax = Softmax(features, labels)
    softmax.train(handle, repeat=5000)
