from sklearn.tree import DecisionTreeClassifier
import numpy as np

from sklearn.tree import _tree

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)

def shuffle_data(data, labels):
    #Shuffle data and labels.
    idx = np.arange(len(labels))  #和range差不多，但是支持步长为小数
    np.random.shuffle(idx)  #原地洗牌，直接改变值，而无返回值
    return data[idx, ...], labels[idx], idx

feature_matrix = np.load('feature_matrixs/feature_matrix3.npy')
label_matrix = np.load('feature_matrixs/label_matrix3.npy')

feature_matrix, label_matrix, _ = shuffle_data(feature_matrix, label_matrix)

train_size = feature_matrix.shape[0] // 5 * 4
test_size = feature_matrix.shape[0] - train_size

trainfea_matrix = feature_matrix[0:train_size]
trainlab_matrix = label_matrix[0:train_size]
test_fea_matrix = feature_matrix[train_size:]
test_lab_matrix = label_matrix[train_size:]

clf=DecisionTreeClassifier()
clf.fit(trainfea_matrix,trainlab_matrix)
result = clf.predict(test_fea_matrix)
score = clf.score(test_fea_matrix,test_lab_matrix)
print(score)

# correct_count = 0
# for i in range(test_size):
#     if(result[i]==test_lab_matrix[i]):
#         correct_count+=1
# correct = correct_count/test_size
# print(correct)

# 决策树可视化
feature_names = ["Range", "StandardDeviation"]
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
graph.write_pdf("result.pdf")

tree_to_code(clf,feature_names)