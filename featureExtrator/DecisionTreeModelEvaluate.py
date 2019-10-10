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

# print(test_lab_matrix)

# 读取模型
import pickle

with open('models/0.978models.pickle', 'rb') as f:
    model = pickle.load(f)
train_score = model.score(trainfea_matrix, trainlab_matrix)
test_score = model.score(test_fea_matrix, test_lab_matrix)
print("train score:", train_score)
print("test score:", test_score)

# 模型评估
from sklearn import metrics

# 将数据分为训练集和测试集
X_train = trainfea_matrix
X_test = test_fea_matrix
y_train = trainlab_matrix
y_test = test_lab_matrix

y_pred = model.predict(X_test)


print(metrics.classification_report(y_test,y_pred))
#法二：通过混淆矩阵验证（横轴：实际值，纵轴：预测值）（理想情况下是个对角阵）
print(metrics.confusion_matrix(y_test, y_pred))

print("精确率:", metrics.accuracy_score(y_test, y_pred))
print("查准率:", metrics.precision_score(y_test, y_pred, average='macro'))
print("召回率:", metrics.recall_score(y_test, y_pred, average='macro'))
print("F1_score:", metrics.f1_score(y_test, y_pred, average='macro'))

print("kappa:", metrics.cohen_kappa_score(y_test, y_pred))
print("ham_distance:", metrics.hamming_loss(y_test, y_pred))
# print("jaccrd_score:", metrics.jaccard_similarity_score(y_test, y_pred))
# print("hinger:", metrics.hinge_loss(y_test, y_pred))