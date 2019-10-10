import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Parameters
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02

# Load data
iris = load_iris()

# 列表的元素类型可以是数字，字符串，列表
# enumerate()函数，组成索引序列，且不需初始化索引，常用于for循环里
for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3], [2, 3]]):
    # We only take the two corresponding features
    X = iris.data[:, pair]
    y = iris.target

    # Train
    clf = DecisionTreeClassifier().fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].

    # 设置子图，pairidx+1 因为pairidx索引是从0开始的
    plt.subplot(2, 3, pairidx + 1)
    # 设置坐标轴显示的起点和终点
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # meshgrid()函数，生成坐标点网格矩阵
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    # tight_layout()紧密布局函数可以避免子图堆叠
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    # ravel()降维，np.c_()按列堆叠即按行拼接，这里把两个坐标轴数组变成了一个网格点矩阵
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])  # np.c_之后的维度是网格点数*2（1列横坐标，1列纵坐标）
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)  # 这里xx.shape和yy.shape一致，均为网格矩阵的排布
    # contourf()生成等高线图
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])

    # Plot the training points

    # zip()合并后返回一个tuple列表
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)  # 返回满足条件的索引
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

plt.suptitle("Decision surface of a decision tree using paired features")
plt.legend(loc='lower right', borderpad=0, handletextpad=0)
plt.axis("tight")

plt.figure()
clf = DecisionTreeClassifier().fit(iris.data, iris.target)
plot_tree(clf, filled=True)
plt.show()