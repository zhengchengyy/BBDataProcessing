from sklearn.tree import _tree
import numpy as np

# 导入全局变量
import GlobalVariable as gv
action_names = gv.action_names
feature_names = gv.feature_names

# 导入数据
feature_matrix = np.load('feature_matrixs/feature_matrix2.npy')
label_matrix = np.load('feature_matrixs/label_matrix2.npy')

# 定义训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    feature_matrix, label_matrix, test_size=0.25, random_state=0)

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.manifold import TSNE

# 如果超过三维则降维到三维数据
x_reduced = PCA(n_components=3).fit_transform(X_train)
# x_reduced = TSNE(n_components=3).fit_transform(X_train)

# import math
# x_rotate = x_reduced.dot(np.asarray([[math.sqrt(2)/2, -math.sqrt(2)/2],[math.sqrt(2)/2,math.sqrt(2)/2]]))
# plt.scatter(x_reduced[:,0],x_reduced[:,1], c=y_train, label=action_names)
# plt.show()

# 标准化数据
# print(len(X_train))
# print(X_train)
# X_train = preprocessing.scale(X_train)   #标准化，均值为0，方差为1
# min_max_scaler = preprocessing.MinMaxScaler()
# X_train = min_max_scaler.fit_transform(X_train)

# 增加噪音
# X_train = np.random.normal(X_train, scale=0.001)
X_train = np.random.normal(X_train, scale=0.05)
# print(X_train)

fig = plt.figure()
ax = Axes3D(fig)
ax.set_title('Feature Scatter Image', size=14)

# ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train, label=action_names,marker='*')
# ax.legend(labels = action_names, loc='upper right')

n_classes = 8
plot_colors = ['r', 'm', 'c', 'b', 'g', 'lime', 'orange','navy','peru','y']
plot_markers = ['*', 'o', ',', 'v', 'D', 'h','d', 'p','H','s']
for i in range(n_classes):
    idx = np.where(y_train == i)  # 返回满足条件的索引
    ax.scatter(X_train[idx, 0], X_train[idx, 1], X_train[idx, 2], s = 30,
               c=plot_colors[i], label=action_names[i], marker = plot_markers[i])

ax.legend(loc='upper right')
ax.set_xlabel(feature_names[0])
ax.set_ylabel(feature_names[1])
ax.set_zlabel(feature_names[2])
ax.w_xaxis.set_ticklabels(())
ax.w_yaxis.set_ticklabels(())
ax.w_zaxis.set_ticklabels(())
plt.show()
