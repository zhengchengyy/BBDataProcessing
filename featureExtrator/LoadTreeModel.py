from sklearn.tree import DecisionTreeClassifier
import numpy as np

# 导入数据
feature_matrix = np.load('feature_matrixs/feature_random_matrix2.npy')
label_matrix = np.load('feature_matrixs/label_random_matrix2.npy')

# 定义训练集和测试集
train_size = feature_matrix.shape[0] // 5 * 4
test_size = feature_matrix.shape[0] - train_size

trainfea_matrix = feature_matrix[0:train_size]
trainlab_matrix = label_matrix[0:train_size]
test_fea_matrix = feature_matrix[train_size:]
test_lab_matrix = label_matrix[train_size:]

# 读取模型
import pickle
with open('models/0.966model.pickle', 'rb') as f:
    model = pickle.load(f)
score = model.score(test_fea_matrix, test_lab_matrix)
print(score)