# 训练完把测试标签和预测分数二值化后画出ROC，后剪枝部分采用的是该方法
# 引入必要的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import pickle
from IPython.display import Image
from sklearn import tree
import pydotplus
import os
# 导入全局变量
import GlobalVariable as gv
action_names = gv.action_names
feature_names = gv.feature_names

ndevices = 5
start = 1
end = ndevices

def drawTreeImage(device_no):
    with open('models/' + 'device_' + str(device_no) + '_post_prune.pickle', 'rb') as f:
        clf = pickle.load(f)
    # 决策树可视化
    os.environ["PATH"] += os.pathsep + 'D:/Anaconda3/Library/bin/graphviz'
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=feature_names,
                                    class_names=action_names,
                                    filled=True, rounded=True,
                                    proportion=False,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('tree_images/' + 'device_' + str(device_no) + '_post_prune.png')

for i in range(start, end + 1):
    drawTreeImage(i)
    print("---------device_" + str(i) + " write tree image success!---------")
