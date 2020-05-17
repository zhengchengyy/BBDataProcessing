import numpy as np
from sklearn.model_selection import train_test_split
# 导入全局变量
import GlobalVariable as gv

action_names = gv.action_names
feature_names = gv.feature_names

ndevices = 5
start = 1
end = ndevices


def get_count(label_matrix):
    action_sample_dict = {}
    for i in range(len(action_names)):
        action_sample_dict[i] = 0
    for i in label_matrix:
        action_sample_dict[i] += 1
    for i in range(len(action_names)):
        print(action_names[i], action_sample_dict[i])


for device_no in range(start, end + 1):
    # 导入数据
    feature_matrix = np.load('feature_matrixs/feature_matrix' + str(device_no) + '.npy')
    label_matrix = np.load('feature_matrixs/label_matrix' + str(device_no) + '.npy')

    # 随机化和划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix, label_matrix, test_size=0.2, random_state=0)

    print("--------device_" + str(device_no) + "--------")
    print("总数")
    get_count(label_matrix)
    print("测试数")
    get_count(y_test)