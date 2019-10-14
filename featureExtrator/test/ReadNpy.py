import numpy as np

# 导入数据
feature_matrix = np.load('../feature_matrixs/feature_matrix2.npy')
label_matrix = np.load('../feature_matrixs/label_matrix2.npy')

# 定义训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    feature_matrix, label_matrix, test_size=0.25, random_state=0)

# print(trainfea_matrix)

print(feature_matrix)  # np.save保存时自动为8位小数

np.savetxt('../feature_matrixs/feature_matrix.txt',feature_matrix)
# np.savetxt('feature_matrixs/label_matrix.txt',label_matrix)
