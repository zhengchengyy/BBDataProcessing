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

with open('models_discard/0.966model.pickle', 'rb') as f:
    model = pickle.load(f)
train_score = model.score(trainfea_matrix, trainlab_matrix)
test_score = model.score(test_fea_matrix, test_lab_matrix)
print("train score:", train_score)
print("test score:", test_score)

# 画图
X = trainfea_matrix
y = trainlab_matrix

# 训练模型，限制树的最大深度4
clf = DecisionTreeClassifier(max_depth=4)
#拟合模型
clf.fit(X, y)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
from matplotlib import pyplot as plt
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.show()

# 模型评估
from sklearn.model_selection import cross_val_score


# cross_val_score(model, trainfea_matrix, y=trainlab_matrix, scoring=None, cv=None, n_jobs=1)

# 考察深度对分类决策树的影响
def test_DecisionTreeClassifiter_depth(*data, maxdepth):
    X_train, X_test, y_train, y_test = data
    depths = np.arange(1, maxdepth)
    training_scores = []
    testing_scores = []
    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth)
        clf.fit(X_train, y_train)
        training_scores.append(clf.score(X_train, y_train))
        testing_scores.append(clf.score(X_test, y_test))
    # 绘图
    from matplotlib import pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(depths, training_scores, label='traing score', marker='o')
    ax.plot(depths, testing_scores, label='testing score', marker='*')
    ax.set_xlabel('maxdepth')
    ax.set_ylabel('score')
    ax.set_title('Decision Tree Classification')
    ax.legend(framealpha=0.5, loc='best')
    plt.show()
test_DecisionTreeClassifiter_depth(trainfea_matrix, test_fea_matrix,
                                   trainlab_matrix, test_lab_matrix, maxdepth=20)


#考察评价切分质量的评价标准criterion对于分类性能的影响
def test_DecisionTreeClassifier_criterion(*data):
    X_train, X_test, y_train, y_test = data
    criterions = ['gini','entropy']
    for criterion in criterions:
        clf = DecisionTreeClassifier(criterion=criterion)
        clf.fit(X_train,y_train)
        print('criterion:%s'%criterion)
        print("Traing score:%f" % (clf.score(X_train, y_train)))
        print("Testing score:%f"%(clf.score(X_test,y_test)))
# test_DecisionTreeClassifier_criterion(trainfea_matrix, test_fea_matrix,
#                                    trainlab_matrix, test_lab_matrix)

# 检测随机划分与最优划分的影响
def test_DecisionTreeClassifier_splitter(*data):
    X_train, X_test, y_train, y_test = data
    splitters = ['best','random']
    for splitter in splitters:
        clf = DecisionTreeClassifier(splitter=splitter)
        clf.fit(X_train,y_train)
        print("splitter:%s"%splitter)
        print("Traing score:%f" % (clf.score(X_train, y_train)))
        print("Testing score:%f"%(clf.score(X_test,y_test)))
# test_DecisionTreeClassifier_splitter(trainfea_matrix, test_fea_matrix,
#                                    trainlab_matrix, test_lab_matrix)



