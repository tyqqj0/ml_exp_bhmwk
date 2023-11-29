# -*- CODING: UTF-8 -*-
# @time 2023/11/22 15:58
# @Author tyqqj
# @File main.py
# @
# @Aim 


import numpy as np
import pandas as pd
import sklearn
import sympy as sp
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
# from mlflow import sklearn
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# 读取鸢尾花数据集
def get_data(err_rate=0, train_rate=0.8):
    # 加载鸢尾花数据集
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # 如果train_rate是0，则记成留一个的比例
    if train_rate < 2 * (1 / len(y)):
        train_rate1 = 2 * (1 / len(y)) + 0.001
    else:
        train_rate1 = train_rate
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_rate1, random_state=42)

    # 根据指定的比例打乱训练集中的标签
    if err_rate > 0 and err_rate <= 1:
        n = int(len(y_train) * err_rate)  # 需要打乱的标签数量
        shuffle_index = np.random.permutation(len(y_train))[:n]  # 随机选择n个标签进行打乱
        for i in shuffle_index:
            y_train[i] = np.random.choice(np.delete(np.unique(y), y_train[i]))  # 随机选择一个不同于原标签的新标签

    return [X_train, y_train], [X_test, y_test]


def cal_acc(y_pred, y_true):
    return np.sum(y_pred == y_true) / len(y_pred)


class model:
    def __init__(self, model):
        self.model = model
        self.trained = False
        # 输出模型参数量

    def clear(self):
        # self.model = None
        self.trained = False

    def train(self):
        self.trained = True
        pass

    def acc(self, test):
        if not self.trained:
            raise Exception("Model not trained yet!")
        y_pred = self.model.predict(test[0])
        return cal_acc(y_pred, test[1])


# 方法1: svm
class svm(model):
    def __init__(self, C=1, kernel='linear', max_iter=1000):
        super().__init__(sklearn.svm.SVC(C=C, kernel=kernel, max_iter=max_iter))
        # self.model = sklearn.svm.SVC(C=C, kernel=kernel, max_iter=max_iter)

    def train(self, train_data):
        self.model.fit(train_data[0], train_data[1])
        super().train()


# 方法2: 逻辑回归
class logistic_regression(model):
    def __init__(self, solver='liblinear', multi_class='auto'):
        super().__init__(LogisticRegression(solver=solver, multi_class=multi_class))
        # self.model = LogisticRegression(solver=solver, multi_class=multi_class)

    def train(self, train_data):
        self.model.fit(train_data[0], train_data[1])
        super().train()  # 调用父类的训练方法


# 方法3: 决策树
class decision_tree(model):
    def __init__(self, criterion='gini', max_depth=None):
        super().__init__(DecisionTreeClassifier(criterion=criterion, max_depth=max_depth))
        # self.model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)

    def train(self, train_data):
        self.model.fit(train_data[0], train_data[1])
        super().train()  # 调用父类的训练方法


# class painter:
#     def __init__(self, name="", x_name="data_correct_rate", y_name="test_acc"):
#         self.data = []
#         self.name = name
#         self.x_name = x_name
#         self.y_name = y_name
#
#     def add(self, data):
#         print(self.name, "add data: ", data)
#         self.data.append(data)
#
#     def paint(self):
#         plt.figure()
#         plt.title(self.name)
#         # 如果是二维数组
#         if len(self.data[0]) == 2:
#             print("paint: ", self.data)
#             for i in self.data:
#                 plt.plot(i[0], i[1])
#         # 如果是一维数组
#         elif len(self.data[0]) == 1:
#             for i in self.data:
#                 plt.plot(i[0])
#         else:
#             print("data type: ", type(self.data[0]))
#             print("data shape: ", self.data[0].shape)
#             raise Exception("data type error!")
#
#         # plt.legend()
#
#         plt.xlabel(self.x_name)
#         plt.ylabel(self.y_name)
#         plt.show()


class plots_painters:
    def __init__(self, namett="", plots=[], x_name="data_correct_rate", y_name="test_acc"):
        if not plots:
            raise Exception("plots is empty!")
        self.plots = plots
        self.name = namett
        self.x_name = x_name
        self.y_name = y_name
        self.data = {name: [] for name in plots}
        self.num = None

    def add(self, plot, data):
        if plot not in self.plots:
            raise Exception("plot not in plots!")
        self.data[plot].append(data)

    def paint(self, interpol=False):
        plt.figure()
        plt.title(self.name)
        for plot in self.plots:
            x_data = [i[0] for i in self.data[plot]]
            y_data = [i[1] for i in self.data[plot]]
            num = len(x_data)

            if interpol:
                # 创建插值函数
                # f = interp1d(x_data, y_data, kind='cubic')

                # 创建新的x值
                # xnew = np.linspace(min(x_data), max(x_data), num=num, endpoint=True)

                # 使用插值函数计算新的y值
                # ynew = f(xnew)

                # plt.plot(xnew, ynew, label=plot)
                # x_data = xnew

                # 创建Pandas Series
                s = pd.Series(y_data)

                # 计算滑动平均
                ynew = s.rolling(5).mean()  # 这里的5是滑动窗口的大小，你可以根据你的需要调整
                y_data = ynew
            plt.plot(x_data, y_data, label=plot)

        plt.legend()
        plt.xlabel(self.x_name)
        plt.ylabel(self.y_name)
        plt.show()


# 通过调整数据随机错误比例，测量模型抗噪声能力
def run_err(range=[0, 1], step=0.001):
    painter = plots_painters("noise_test", ["svm", "logistic_regression", "decision_tree"])
    model1 = svm()
    model2 = logistic_regression()
    model3 = decision_tree()
    for rate in np.arange(range[0], range[1], step):
        print("rate: ", rate)
        train, test = get_data(1 - rate)
        model1.train(train)
        model2.train(train)
        model3.train(train)
        painter.add("svm", [rate, model1.acc(test)])
        painter.add("logistic_regression", [rate, model2.acc(test)])
        painter.add("decision_tree", [rate, model3.acc(test)])
        model1.clear()
        model2.clear()
        model3.clear()
    painter.paint(interpol=True)


# 通过控制训练集大小，测量模型的泛化能力
def run_num(range=[0, 1], step=0.001):
    painter = plots_painters("num_test", ["svm", "logistic_regression", "decision_tree"], x_name="train_num")
    model1 = svm()
    model2 = logistic_regression()
    model3 = decision_tree()
    for rate in np.arange(range[0], range[1], step):
        print("rate: ", rate)
        train, test = get_data(0, rate)
        model1.train(train)
        model2.train(train)
        model3.train(train)
        painter.add("svm", [rate, model1.acc(test)])
        painter.add("logistic_regression", [rate, model2.acc(test)])
        painter.add("decision_tree", [rate, model3.acc(test)])
        model1.clear()
        model2.clear()
        model3.clear()
    painter.paint(interpol=True)


if __name__ == "__main__":
    train, test = get_data()
    print(train[0].shape, train[1].shape, test[0].shape, test[1].shape)
    # run_err()
    run_num()
