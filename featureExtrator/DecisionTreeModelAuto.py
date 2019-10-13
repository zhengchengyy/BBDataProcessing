from sklearn.tree import DecisionTreeClassifier
import numpy as np

# 导入数据
feature_matrix = np.load('feature_matrixs/feature_matrix2.npy')
label_matrix = np.load('feature_matrixs/label_matrix2.npy')

# 定义训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    feature_matrix, label_matrix, test_size=0.25, random_state=0)
# print(len(X_train),len(X_test))
# print(y_test)

# 训练和预测
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print(score)

# 保存模型
import pickle
feature_num = feature_matrix.shape[1]
with open('models/' + str(round(score,3)) + 'Acc_' + str(feature_num) + 'Fea.pickle', 'wb') as f:
    pickle.dump(clf, f)

# 决策树可视化
feature_names = ["StandardDeviationModule", "AverageModule"]
action_names = ["turn_over", "legs_stretch", "hands_stretch",
                "legs_twitch", "hands_twitch", "head_move", "grasp", "kick"]

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
graph.write_pdf('trees/' + str(round(score,3)) + 'Acc_' + str(feature_num) + 'Fea.pdf')