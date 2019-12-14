from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
from sklearn import metrics

ndevices = 5
start = 1
end = 1


def save_model(device_no):
    # 导入数据
    feature_matrix = np.load('feature_matrixs/feature_matrix' + str(device_no) + '.npy')
    label_matrix = np.load('feature_matrixs/label_matrix' + str(device_no) + '.npy')
    # print(feature_matrix)

    # 定义训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix, label_matrix, test_size=0.2, random_state=0)

    y_pred = KMeans(n_clusters=2, random_state=0).fit_predict(X_train)

    for index, k in enumerate((2, 3, 4, 5)):
        plt.subplot(2, 2, index + 1)
        y_pred = MiniBatchKMeans(n_clusters=k, batch_size=200, random_state=0).fit_predict(X_train)
        score = metrics.calinski_harabasz_score(X_train, y_pred)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_pred, s=10)
        plt.text(.99, .01, ('k=%d, score: %.2f' % (k, score)),
                 transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')
    plt.show()
    # plt.scatter(X_train[:, 0], X_train[:, 1], c=y_pred)
    # plt.show()

for i in range(start, end + 1):
    print("-------device_" + str(i) + "-------")
    save_model(i)
