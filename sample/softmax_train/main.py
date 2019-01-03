from sample.softmax_train.random_data import RandomData
from sample.softmax_train.softmax_modual import Softmax
import matplotlib.pyplot as plt

counts = []
losses = []


def handle(count, loss):
    print("Train Count:{}. Loss Value:{}".format(count, loss))
    counts.append(count)
    losses.append(loss)

if __name__ == '__main__':
    data = RandomData(500, 20, 50)
    features, labels = data.establish_data()
    softmax = Softmax(features, labels)
    softmax.train(handle, repeat=5000)

    plt.figure()
    plt.scatter(counts, losses)
    plt.xlabel("Count")
    plt.ylabel("Loss")
    plt.show()
