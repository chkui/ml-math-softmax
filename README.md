# ml-math-softmax
## 来源
代码来源于https://www.chkui.com/article/tensorflow/math_in_machine_learning_softmax
## Install
### virtualenv
It is recommended to install project inside a virtualenv

```
$ pip install virtualenv
$ virtualenv venv
$ . ./venv/bin/activate
```
建议安装 python虚拟环境
### install with requirements.txt
```
$ pip3 install -r requirements.txt 
```
安装引入的包。

## 库结构说明
`sample/matplot`是softmax交叉熵算法演示。

`sample/softmax_compute`展现了softmax的一个基本的运算过程。

`sample/softmax_train`展现了一个基础训练模型。

`sample/softmax_estimator`和`sample/softmax_train`类似，但是提供了数据磁盘功能，可以对相同的数据进行反复的训练。