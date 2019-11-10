from sklearn.tree import DecisionTreeClassifier
import numpy as np

# 导入数据
feature_matrix = np.load('feature_matrixs/feature_random_matrix2.npy')
label_matrix = np.load('feature_matrixs/label_random_matrix2.npy')

# 定义训练集和测试集
train_size = feature_matrix.shape[0] // 4 * 3
test_size = feature_matrix.shape[0] - train_size

trainfea_matrix = feature_matrix[0:train_size]
trainlab_matrix = label_matrix[0:train_size]
test_fea_matrix = feature_matrix[train_size:]
test_lab_matrix = label_matrix[train_size:]

# 训练和预测
# clf = DecisionTreeClassifier(max_depth=9, min_samples_split=10, max_leaf_nodes=15)
clf = DecisionTreeClassifier()
clf.fit(trainfea_matrix, trainlab_matrix)
score = clf.score(test_fea_matrix, test_lab_matrix)
print(score)

# 保存模型
import pickle
feature_num = feature_matrix.shape[1]
with open('models_discard/' + str(round(score,3)) + 'Acc_' + str(feature_num) + 'Fea.pickle', 'wb') as f:
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
graph.write_pdf('trees_discard/' + str(round(score,3)) + 'Acc_' + str(feature_num) + 'Fea.pdf')