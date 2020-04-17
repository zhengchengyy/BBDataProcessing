import numpy as np
from sklearn.tree import DecisionTreeClassifier


ndevices = 5
start = 1
end = 1

# 导入全局变量
import GlobalVariable as gv
action_names = gv.action_names
feature_names = gv.feature_names



def compute_accuracy(predict_result, true_result):
    sum = 0
    for i in range(len(action_names)):
        print(action_names[i], end=": ")
        count = 0
        correct_count = 0
        for j in range(len(predict_result)):
            if(i == predict_result[j]):
                count += 1
                if (predict_result[j] == true_result[j]):
                    correct_count += 1
        correct = correct_count / count
        sum += correct
        print(str(correct) +"≈" + str(round(correct, 3)),end="")
        print(str("(") + str(correct_count) + ":" + str(count) + str(")"), end=" ")
        print(str(round(correct * 100, 1)) + "%")

    average_accuracy = sum / len(action_names)
    print("平均识别准确率: " + str(average_accuracy), end=" ")
    print(str(round(average_accuracy * 100, 1)) + "%")


def save_model(device_no):
    # 导入数据
    feature_matrix = np.load('feature_matrixs/feature_matrix' + str(device_no) + '.npy')
    label_matrix = np.load('feature_matrixs/label_matrix' + str(device_no) + '.npy')
    # print(feature_matrix)

    # 定义训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix, label_matrix, test_size=0.2, random_state=0)

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    # train_score = clf.score(X_train, y_train)
    # test_score = clf.score(X_test, y_test)
    # print('device_' + str(device_no) + '\'s train score:', train_score)
    # print('device_' + str(device_no) + '\'s test score:', test_score)
    print("-------train data-------")
    predict_result_train = clf.predict(X_train)
    compute_accuracy(predict_result_train, y_train)
    print("-------test data-------")
    predict_result_test = clf.predict(X_test)
    compute_accuracy(predict_result_test, y_test)



for i in range(start, end + 1):
    print("-------device_" + str(i) + "-------")
    save_model(i)
