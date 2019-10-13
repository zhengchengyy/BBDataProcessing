from sklearn.tree import _tree
import numpy as np

feature_names = ["StandardDeviationModule", "VarianceModule", "AverageModule"]
action_names = ["turn_over", "legs_stretch", "hands_stretch",
                "legs_twitch", "hands_twitch", "head_move", "grasp", "kick"]

# 导入数据
feature_matrix = np.load('feature_matrixs/feature_matrix2.npy')
label_matrix = np.load('feature_matrixs/label_matrix2.npy')

# 定义训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    feature_matrix, label_matrix, test_size=0.25, random_state=0)

# 画二维散点图
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.manifold import TSNE

# 如果超过三维则降维到三维数据
# x_reduced = PCA(n_components=2).fit_transform(X_train)
# x_reduced = TSNE(n_components=2).fit_transform(X_train)

# 数据旋转
# import math
# x_rotate = X_train.dot(np.asarray([[math.sqrt(2)/2, -math.sqrt(2)/2],[math.sqrt(2)/2,math.sqrt(2)/2]]))
# plt.scatter(x_rotate[:,0],x_rotate[:,1], c=y_train, label=action_names)
# plt.show()

# 归一化数据
# print(len(X_train))
# print(X_train)
# X_train = preprocessing.scale(X_train)
# min_max_scaler = preprocessing.MinMaxScaler()
# X_train = min_max_scaler.fit_transform(X_train)

# 增加噪音
X_train = np.random.normal(X_train, scale=0.002)
X_test = np.random.normal(X_test, scale=0.002)
# X_train = np.random.normal(X_train, scale=0.05)
# print(X_train)

# ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train, label=action_names,marker='*')
# ax.legend(labels = action_names, loc='upper right')

# 画训练数据的二维散点图
n_classes = 8
plot_colors = ['r', 'm', 'c', 'b', 'g', 'lime', 'y', 'peru', 'navy', 'orange']
plot_markers = ['*', 'o', ',', 'v', 'D', 'h', 'd', 'p', 'H', 's']

for i in range(n_classes):
    idx = np.where(y_train == i)  # 返回满足条件的索引
    plt.scatter(X_train[idx, 0], X_train[idx, 1], s=30,
                c=plot_colors[i], label=action_names[i], marker=plot_markers[i])

plt.legend(loc='upper right')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.show()

# 画测试数据的二维散点图
for i in range(n_classes):
    idx = np.where(y_test == i)  # 返回满足条件的索引
    plt.scatter(X_test[idx, 0], X_test[idx, 1], s=30,
                c=plot_colors[i], label=action_names[i], marker=plot_markers[i])

plt.legend(loc='upper right')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.show()

# 使用seaborn库画散点图
import seaborn as sns
import pandas as pd
x = X_test[:, 0]
y = X_test[:, 1]
df = pd.DataFrame({feature_names[0]: x, feature_names[1]: y})
sns.set(style="ticks", palette="muted")
sns.relplot(feature_names[0], feature_names[1], style=y_test, hue=y_test, sizes=1000,
            palette="Set2", data=df)
plt.show()



