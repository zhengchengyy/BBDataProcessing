from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
from sklearn import metrics

ndevices = 5
start = 1
end = 5

def compute_accuracy(predict_result, true_result):
    if(len(predict_result) != len(true_result)):
        print("Input data is incorrect!")
    correct_count = 0
    for i in range(len(predict_result)):
        if(predict_result[i] == true_result[i]):
            correct_count += 1
    return correct_count / len(predict_result)


def save_model(device_no):
    # 导入数据
    feature_matrix = np.load('feature_matrixs/feature_matrix' + str(device_no) + '.npy')
    label_matrix = np.load('feature_matrixs/label_matrix' + str(device_no) + '.npy')
    # print(feature_matrix)

    # 定义训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix, label_matrix, test_size=0.2, random_state=0)

    y_pred_train = KMeans(n_clusters=9, random_state=0).fit_predict(X_train)
    y_pred_test = KMeans(n_clusters=9, random_state=0).fit_predict(X_test)

    # y_pred_train = MiniBatchKMeans(n_clusters=9, batch_size=1000, random_state=0).fit_predict(X_train)
    # y_pred_test = MiniBatchKMeans(n_clusters=9, batch_size=1000, random_state=0).fit_predict(X_test)
    print("train data accuracy:", compute_accuracy(y_pred_train, y_train))
    print("test data accuracy:", compute_accuracy(y_pred_train, y_train))


for i in range(start, end + 1):
    print("-------device_" + str(i) + "-------")
    save_model(i)
