from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

# 导入全局变量
import GlobalVariable as gv

action_names = gv.action_names
feature_names = gv.feature_names
feature_num = len(feature_names)


def loadModel(device_no, test_score):
    # 读取模型
    import pickle
    with open('models/' + 'device_' + str(device_no) + 'Acc_' + str(round(test_score, 3))
              + 'Fea_' + str(feature_num) + '_prune.pickle', 'rb') as f:
        model = pickle.load(f)

    return model


def loadMatrix(device_no):
    # 导入数据
    feature_matrix = np.load('feature_matrixs/feature_matrix' + str(device_no) + '.npy')
    label_matrix = np.load('feature_matrixs/label_matrix' + str(device_no) + '.npy')

    # 定义训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix, label_matrix, test_size=0.25, random_state=0)

    return X_train, X_test, y_train, y_test


model1 = loadModel(1,0.905)
model2 = loadModel(2,0.924)
model3 = loadModel(3,0.973)
X_train1, X_test1, y_train1, y_test1 = loadMatrix(1)
X_train2, X_test2, y_train2, y_test2 = loadMatrix(2)
X_train3, X_test3, y_train3, y_test3 = loadMatrix(3)


for device_no in range(1,3+1):
    X_train, X_test, y_train, y_test = loadMatrix(device_no)
    print("The length of test data",len(y_test))
    train_score = eval("model" + str(device_no) + ".score(X_train, y_train)")
    print("device_" + str(device_no) + "'train score:", train_score)
    test_score = eval("model"+str(device_no)+".score(X_test, y_test)")
    print("device_"+str(device_no)+"'test score:", test_score)

train_pre1 = model1.predict(X_train1)
train_pre2 = model2.predict(X_train2)
train_pre3 = model3.predict(X_train3)
X_train = np.dstack((train_pre1,train_pre2,train_pre3)).squeeze()
print(X_train.shape)

test_pre1 = model1.predict(X_test1)
test_pre2 = model2.predict(X_test2)
test_pre3 = model3.predict(X_test3)
X_test = np.dstack((test_pre1,test_pre2,test_pre3)).squeeze()

predict_names = ["pre1","pre2", "pre3"]
clf = DecisionTreeClassifier(random_state=0, criterion='entropy')
# clf = LogisticRegression(C=0.01)
clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
train_score = clf.score(X_test, y_test)
# test_score = clf.score(X_test2, y_test2)
print('device_all\'s train score:', train_score)
print('device_all\'s test score:', test_score)

# 决策树可视化
from IPython.display import Image
from sklearn import tree
import pydotplus
import os

os.environ["PATH"] += os.pathsep + 'D:/Anaconda3/Library/bin/graphviz'

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=predict_names,
                                class_names=action_names,
                                filled=True, rounded=True,
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('tree_images/' + 'device_all'  + 'Acc_' + str(round(test_score, 3))
           + 'Fea_' + str(feature_num) + '.png')