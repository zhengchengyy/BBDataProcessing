from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt

train_times = 100
importance_threshold = 0.1
extract_feature_names = []

# 导入全局变量
import GlobalVariable as gv
feature_names = gv.feature_names
# 删除名字后缀
feature_names = [feature[:-6] for feature in feature_names]
action_names = gv.action_names
color_list = ['r', 'm', 'c', 'b', 'g', 'lime', 'y', 'peru', 'navy', 'orange', 'deepskyblue', 'pink']

# ————导入数据————
device_no = 2
feature_matrix = np.load('feature_matrixs/feature_matrix'+str(device_no)+'.npy')
label_matrix = np.load('feature_matrixs/label_matrix'+str(device_no)+'.npy')

# 划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    feature_matrix, label_matrix, test_size=0.2, random_state=0)

def feature_extractor(feature_names, train_times, importance_thread, X_train, y_train):
    extract_feature_names = []
    feature_importances = []
    for i in range(len(feature_names)):
        feature_importances.append([])
    for i in range(train_times):
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        temp = model.feature_importances_
        for j in range(len(temp)):
            feature_importances[j].append(temp[j])
    for i in range(len(feature_importances)):
        importances = feature_importances[i]
        if np.median(importances) > importance_threshold :
            extract_feature_names.append(feature_names[i])
    return extract_feature_names

extract_feature_names = feature_extractor(feature_names, train_times, importance_threshold, X_train, y_train)
print("原先的特征为: ", feature_names)
print("被提取的特征为: ", extract_feature_names)
