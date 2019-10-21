from sklearn.tree import DecisionTreeClassifier
import numpy as np

# ————导入数据————
feature_matrix = np.load('feature_matrixs/feature_matrix2.npy')
label_matrix = np.load('feature_matrixs/label_matrix2.npy')

# 划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    feature_matrix, label_matrix, test_size=0.25, random_state=0)

# 训练和预测
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print(score)

# ————保存模型————
import pickle
feature_num = feature_matrix.shape[1]
with open('models_discard/' + str(round(score,3)) + 'Acc_' + str(feature_num) + 'Fea.pickle', 'wb') as f:
    pickle.dump(clf, f)

# ————决策树可视化————
# 导入全局变量
import GlobalVariable as gv
feature_names = gv.feature_names
# 删除名字后缀
feature_names = [feature[:-6] for feature in feature_names]
action_names = gv.action_names

from IPython.display import Image
from sklearn import tree
import pydotplus
import os

os.environ["PATH"] += os.pathsep + 'D:/Anaconda3/Library/bin/graphviz'

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=feature_names,
                                class_names=action_names,
                                filled=True, rounded=True,
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
# Image(graph.create_png())
graph.write_pdf("result.pdf")

# 直接画出决策树，但是很小看不清
# from sklearn.tree import plot_tree
# plot_tree(clf, filled=True)


# ————特征重要性————
import matplotlib.pyplot as plt  # 导入图形展示库

feature_importance = clf.feature_importances_  # 获得指标重要性
color_list = ['r', 'm', 'c', 'b', 'g', 'lime', 'y', 'peru', 'navy', 'orange', 'deepskyblue', 'pink']


# 按特征输入顺序画出条形图
def plot_feature_importances(feature_importance, feature_names):
    plt.bar(np.arange(feature_importance.shape[0]), feature_importance,
            tick_label=feature_names, color=color_list)
    plt.title('Feature importance')  # 子网格标题
    plt.xlabel('Features')  # x轴标题
    plt.ylabel('Importance')  # y轴标题
    plt.suptitle('Classification result')  # 图形总标题
    # 最大化显示图像窗口
    # plt.get_current_fig_manager().window.showMaximized()
    plt.show()  # 展示图形


# plot_feature_importances(feature_importance, feature_names)


# 按特征重要性排序
def plot_feature_importances_sorted(feature_importances, title, feature_names):
    feature_importances = 100 * (feature_importances / max(feature_importances))
    # 按特征重要性进行排序
    index_sorted = np.flipud(np.argsort(feature_importances))
    pos = np.arange(index_sorted.shape[0]) + 0.8
    plt.figure()
    sorted_color_list = np.asarray(color_list)[index_sorted]
    plt.bar(pos, feature_importances[index_sorted], align='center', color=sorted_color_list)
    plt.xticks(pos, np.array(feature_names)[index_sorted])
    plt.ylabel('Relative Importance')
    plt.title(title)
    plt.show()


plot_feature_importances_sorted(feature_importance, 'Feature importances', feature_names)
