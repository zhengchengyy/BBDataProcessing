import sys
import imp
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import json
import copy
import xlwt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from IPython.display import display, Image
import pydotplus
from sklearn import tree
from sklearn.tree import _tree
from sklearn import tree
import collections
import os
from sklearn.tree._tree import TREE_LEAF
from pandas import DataFrame

from sklearn.model_selection import train_test_split
import csv


def rules(clf, features, labels, node_index=0):
    """Structure of rules in a fit decision tree classifier

    Parameters
    ----------
    clf : DecisionTreeClassifier
        A tree that has already been fit.

    features, labels : lists of str
        The names of the features and labels, respectively.

    """
    node = {}
    if clf.tree_.children_left[node_index] == -1:  # 叶子节点
        count_labels = list(zip(clf.tree_.value[node_index, 0], labels))
        node['name'] = ', '.join(('{} of {}'.format(int(count), label)
                                  # 所谓的class_name其实就是在这种地方用到了，这个class_names其实可以理解为类别的取值
                                  for count, label in count_labels))

        node['value'] = [count for count, label in count_labels]  # add by appleyuchi
    else:

        count_labels = list(zip(clf.tree_.value[node_index, 0], labels))  # add by appleyuchi
        node['value'] = [count for count, label in count_labels]  # add by appleyuchi

        feature = features[clf.tree_.feature[node_index]]
        threshold = clf.tree_.threshold[node_index]
        node['name'] = '{} <= {}'.format(feature, threshold)
        left_index = clf.tree_.children_right[node_index]
        right_index = clf.tree_.children_left[node_index]
        node['children'] = [rules(clf, features, labels, right_index),
                            rules(clf, features, labels, left_index)]
    return node

def draw_file(model, dot_file, png_file, feature_names, class_names):
    dot_data = tree.export_graphviz(model, out_file=dot_file,
                                    feature_names=feature_names,
                                    class_names=class_names,
                                    filled=True,
                                    rounded=True,
                                    special_characters=True)

    graph = pydotplus.graph_from_dot_file(dot_file)

    os.environ["PATH"] += os.pathsep + 'D:/Anaconda3/Library/bin/graphviz'

    thisIsTheImage = Image(graph.create_png())
    display(thisIsTheImage)
    # print(dt.tree_.feature)

    from subprocess import check_call
    check_call(['dot', '-Tsvg', dot_file, '-o', png_file])

def model_json():
    # 导入数据
    device_no = 2
    feature_matrix = np.load('feature_matrixs/feature_matrix' + str(device_no) + '.npy')
    label_matrix = np.load('feature_matrixs/label_matrix' + str(device_no) + '.npy')
    # print(feature_matrix)

    # 定义训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix, label_matrix, test_size=0.25, random_state=0)

    print("训练集长度:", len(X_train), len(y_train))
    print("测试集长度：", len(X_test), len(y_test))
    print("分割后的情况：", type(X_train))

    feature_list = ["MeanModule", "SDModule"]
    class_names = ["turn_over", "legs_stretch", "hands_stretch",
          "legs_twitch", "hands_twitch", "head_move", "grasp", "kick"]
    # 删除名字后缀
    feature_list = [feature[:-6] for feature in feature_list]

    clf = DecisionTreeClassifier(random_state=0,
                                 max_depth=11,
                                 max_leaf_nodes=24,
                                 min_impurity_decrease=0.0003,
                                 min_samples_leaf=3,
                                 min_samples_split=7,
                                 splitter='best',
                                 criterion='entropy')

    print("now training,wait please..........")
    clf.fit(X_train, y_train)
    print("train finished")
    result = rules(clf, feature_list, class_names)

    # 保存为json文件
    # with open('structure.json', 'w') as f:
    #     f.write(json.dumps(result))
    print("The json-style model has been stored in structure.json")

    print("now I'm drawing the CART tree,wait please............")

    dot_file = "visualization/T0.dot"
    png_file = "visualization/T0.svg"
    draw_file(clf, dot_file, png_file, feature_list, class_names)
    print("CART tree has been drawn in " + png_file)
    return clf, result, X_train, y_train, X_test, y_test, feature_list, class_names


def sklearn2json_model():
    model, models_json, X_train, y_train, X_test, y_test, feature_names, class_names = \
        model_json()
    print("X_train=", X_train)
    print("y_train=", y_train)
    print("models_json=", models_json)

if __name__ == '__main__':
    sklearn2json_model()
