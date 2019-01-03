import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# 交叉熵三维显示趋势图
def feature_entropy():
    # 标签是2个分类，这里统计属于第一个分类时与q1、q2的数值关系
    label = np.matrix([1, 0]).T
    origin = np.linspace(0, 1, 100, dtype=float)
    q1 = []
    q2 = []
    h = []
    for _q1 in origin:
        _q2 = 1 - _q1
        q1.append(_q1)
        q2.append(_q2)
        h.append(np.log(np.matrix([_q1, _q2])) * label)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('q1')
    ax.set_ylabel('q2')
    ax.set_zlabel('Corss entropy')
    ax.scatter(q1, q2, h)
    plt.show()


feature_entropy()
