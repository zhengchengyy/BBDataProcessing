import numpy as np

device_no = 1

# 导入数据
feature_matrix = np.load('../feature_matrixs/feature_matrix'+str(device_no)+'.npy')
label_matrix = np.load('../feature_matrixs/label_matrix'+str(device_no)+'.npy')

# 定义训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    feature_matrix, label_matrix, test_size=0.25, random_state=0)

# print(trainfea_matrix)

print(feature_matrix[15:75])  # np.save保存时自动为8位小数

# np.set_printoptions(suppress=True)
# np.savetxt('../feature_matrixs/feature_matrix'+str(device_no)+'.txt',feature_matrix,fmt="%.18f,%.18f")
# np.savetxt('feature_matrixs/label_matrix.txt',label_matrix)


# 读取模型
start = 1
end = start
device_no = 1
import pickle
with open('../models/' + 'device_' + str(device_no) + '_model.pickle', 'rb') as f:
    model = pickle.load(f)

result = model.predict(feature_matrix)
for i in result:
    print(i, end=", ")