import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures,StandardScaler

device_no = 2
# 导入数据
feature_matrix = np.load('feature_matrixs/feature_matrix' + str(device_no) + '.npy')
label_matrix = np.load('feature_matrixs/label_matrix' + str(device_no) + '.npy')
# print(feature_matrix)

# 定义训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    feature_matrix, label_matrix, test_size=0.25, random_state=0)

CLF=PolynomialFeatures(degree=100)
X_train = CLF.fit_transform(X_train)
print(X_train.shape)
X_test = CLF.transform(X_test)
CLF=StandardScaler()
X_train = CLF.fit_transform(X_train)
X_test = CLF.transform(X_test)

# 训练和预测
clf = LogisticRegression(C=0.01)
clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print('device_' + str(device_no) + '\'s train score:', train_score)
print('device_' + str(device_no) +'\'s test score:', test_score)
