from sklearn.tree import DecisionTreeClassifier
import numpy as np

ndevices = 5
start = 1
end = ndevices


def save_model(device_no):
    # 导入数据
    feature_matrix = np.load('feature_matrixs/feature_matrix' + str(device_no) + '.npy')
    label_matrix = np.load('feature_matrixs/label_matrix' + str(device_no) + '.npy')
    # print(feature_matrix)

    # 归一化处理
    from sklearn import preprocessing
    # feature_matrix = preprocessing.scale(feature_matrix)  #z-score标准化
    # z-score标准化
    # feature_matrix = preprocessing.StandardScaler().fit_transform(feature_matrix)
    # min-max标准化
    feature_matrix = preprocessing.MinMaxScaler().fit_transform(feature_matrix)

    # 如果超过三维则降维到三维数据
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    feature_sum = 3
    if(feature_matrix.shape[1]>feature_sum):
        feature_matrix = PCA(n_components=feature_sum).fit_transform(feature_matrix)
        # feature_matrix_reduce = TSNE(n_components=3).fit_transform(feature_matrix_normalizion)

    # 定义训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix, label_matrix, test_size=0.2, random_state=0)

    # 训练和预测
    clf = DecisionTreeClassifier(random_state=0, criterion='entropy')
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print('device_' + str(device_no) + '\'s train score:', train_score)
    print('device_' + str(device_no) +'\'s test score:', test_score)

    # 交叉验证
    from sklearn.model_selection import cross_val_score
    # print("交叉验证分数:", cross_val_score(clf, X_train, y=y_train, cv=5))  # cv表示几倍交叉验证
    print("交叉验证平均分数:", cross_val_score(clf, X_train, y=y_train, cv=10).mean())

for i in range(start, end + 1):
    save_model(i)
