from sklearn.tree import _tree
import numpy as np

# 导入全局变量
import GlobalVariable as gv
action_names = gv.action_names
feature_names = gv.feature_names
# 删除名字后缀
feature_names = [feature[:-6] for feature in feature_names]

# 导入数据
device_no = 1
feature_matrix = np.load('feature_matrixs/feature_matrix'+str(device_no)+'.npy')
label_matrix = np.load('feature_matrixs/label_matrix'+str(device_no)+'.npy')

# 定义训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    feature_matrix, label_matrix, test_size=0.2, random_state=0)

# 画二维散点图
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.manifold import TSNE

# 如果超过二维则降维到二维数据
if(X_train.shape[1] > 2):
    # x_reduced = PCA(n_components=3).fit_transform(X_train)
    X_train = PCA(n_components=3).fit_transform(X_train)
    # x_reduced = TSNE(n_components=3).fit_transform(X_train)

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
# X_train = np.random.normal(X_train, scale=0.002)
# X_test = np.random.normal(X_test, scale=0.002)
# X_train = np.random.normal(X_train, scale=0.05)
# print(X_train)

# ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train, label=action_names,marker='*')
# ax.legend(labels = action_names, loc='upper right')

# 画训练数据的二维散点图
n_classes = len(action_names)
plot_colors = ['r', 'm', 'c', 'b', 'g', 'lime', 'y', 'peru', 'navy', 'orange']
plot_markers = ['*', 'o', ',', 'v', 'D', 'h', 'd', 'p', 'H', 's']

# 定义图片名和标题
fig = plt.figure('device'+str(device_no)+'_train_data_scatter_image')
plt.suptitle('device'+str(device_no)+'_train_data_scatter_image')

for i in range(n_classes):
    idx = np.where(y_train == i)  # 返回满足条件的索引
    plt.scatter(X_train[idx, 0], X_train[idx, 1], s=30,
                c=plot_colors[i], label=action_names[i], marker=plot_markers[i])
    # plt.scatter(x_reduced[idx, 0], x_reduced[idx, 1], s=30,
    #             c=plot_colors[i], label=action_names[i], marker=plot_markers[i])
plt.legend(loc='upper right')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.show()

# 画测试数据的二维散点图
# 定义图片名和标题
fig = plt.figure('device'+str(device_no)+'_test_data_scatter_image')
plt.suptitle('device'+str(device_no)+'_test_data_scatter_image')
for i in range(n_classes):
    idx = np.where(y_test == i)  # 返回满足条件的索引
    plt.scatter(X_test[idx, 0], X_test[idx, 1], s=30,
                c=plot_colors[i], label=action_names[i], marker=plot_markers[i])
plt.legend(loc='upper right')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
# plt.show()

# 使用seaborn库画散点图
# import seaborn as sns
# import pandas as pd
# x = X_test[:, 0]
# y = X_test[:, 1]
# df = pd.DataFrame({feature_names[0]: x, feature_names[1]: y})
# sns.set(style="ticks", palette="muted")
# sns.relplot(feature_names[0], feature_names[1], style=y_test, hue=y_test, sizes=1000,
#             palette="Set2", data=df)
# plt.show()



