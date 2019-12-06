from sklearn.tree import DecisionTreeClassifier
import numpy as np

# ————导入数据————
device_no = 1
feature_matrix = np.load('feature_matrixs/feature_matrix'+str(device_no)+'.npy')
label_matrix = np.load('feature_matrixs/label_matrix'+str(device_no)+'.npy')

# 划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    feature_matrix, label_matrix, test_size=0.25, random_state=0)

def DecisionTreeClassifiter_param(*data, param_name, param_value):
    X_train, X_test, y_train, y_test = data
    if(param_name=="min_samples_split"):
        param_values = np.arange(2, param_value)
    elif(param_name=="min_impurity_decrease"):
        param_values = np.arange(0, 1, 0.1)
    elif(param_name == "max_features"):
        n_features = X_train.shape[1]
        param_values = np.arange(1, n_features+1)
    else:
        param_values = np.arange(1, param_value)
    training_scores = []
    testing_scores = []
    for value in param_values:
        # param = param_name + '=' + str(value)
        # print(param)
        # clf = DecisionTreeClassifier(param)
        if(param_name=="max_depth"):
            clf = DecisionTreeClassifier(max_depth=value,random_state=0)
        elif(param_name=="min_samples_split"):
            clf = DecisionTreeClassifier(min_samples_split=value,random_state=0)
        elif(param_name=="min_samples_leaf"):
            clf = DecisionTreeClassifier(min_samples_leaf=value,random_state=0)
        elif (param_name == "max_leaf_nodes"):
            clf = DecisionTreeClassifier(min_samples_leaf=value,random_state=0)
        elif(param_name == "min_impurity_decrease"):
            clf = DecisionTreeClassifier(min_impurity_decrease=value,random_state=0)
        elif(param_name == "max_features"):
            clf = DecisionTreeClassifier(max_features=value, random_state=0)
        clf.fit(X_train, y_train)
        training_scores.append(clf.score(X_train, y_train))
        testing_scores.append(clf.score(X_test, y_test))
    # 绘图
    from matplotlib import pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(param_values, training_scores, label='traing score', marker='o')
    ax.plot(param_values, testing_scores, label='testing score', marker='*')
    ax.set_xlabel(param_name)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('score')
    ax.set_title('Score of different '+param_name)
    ax.legend(framealpha=0.5, loc='best')

    max_indx = np.argmax(training_scores)  # max value index
    min_indx = np.argmin(training_scores)  # min value index
    plt.plot(max_indx, training_scores[max_indx], 'ks', color="r")
    show_max = '[' + str(max_indx) + ', ' + str(training_scores[max_indx]) + ']'
    plt.annotate(show_max, xytext=(max_indx, training_scores[max_indx]+0.05),
                 xy=(max_indx, training_scores[max_indx]), color="r")

    max_indx = np.argmax(testing_scores)  # max value index
    min_indx = np.argmin(testing_scores)  # min value index
    plt.plot(max_indx, testing_scores[max_indx], 'ks', color="r")
    show_max = '[' + str(max_indx) + ', ' + str(round(testing_scores[max_indx],2)) + ']'
    plt.annotate(show_max, xytext=(max_indx, testing_scores[max_indx] - 0.05),
                 xy=(max_indx, testing_scores[max_indx]), color="r")

    plt.show()
# param_names = ["max_depth","min_samples_split","min_samples_leaf","max_leaf_nodes"]
# param_names = ["min_impurity_split"] #will be removed in 0.25
# param_names = ["min_impurity_decrease"]
# param_names = ["max_features"]
param_names = ["max_depth"]
for param_name in param_names:
    DecisionTreeClassifiter_param(X_train, X_test, y_train, y_test,
                                  param_name=param_name, param_value=50)


# -----单独方法验证------
# 考察深度对分类决策树的影响
def DecisionTreeClassifiter_depth(*data, maxdepth):
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
# DecisionTreeClassifiter_depth(X_train, X_test, y_train, y_test, maxdepth=20)

# 考察min_samples_split对分类决策树的影响
def DecisionTreeClassifiter_split(*data, min_samples_split):
    X_train, X_test, y_train, y_test = data
    splits = np.arange(2, min_samples_split)
    training_scores = []
    testing_scores = []
    for split in splits:
        clf = DecisionTreeClassifier(min_samples_split=split)
        clf.fit(X_train, y_train)
        training_scores.append(clf.score(X_train, y_train))
        testing_scores.append(clf.score(X_test, y_test))
    # 绘图
    from matplotlib import pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(splits, training_scores, label='traing score', marker='o')
    ax.plot(splits, testing_scores, label='testing score', marker='*')
    ax.set_xlabel('maxdepth')
    ax.set_ylabel('score')
    ax.set_title('Decision Tree Classification')
    ax.legend(framealpha=0.5, loc='best')
    plt.show()
# DecisionTreeClassifiter_split(X_train, X_test, y_train, y_test, min_samples_split=20)


# 考察评价切分质量的评价标准criterion对于分类性能的影响
def DecisionTreeClassifier_criterion(*data):
    X_train, X_test, y_train, y_test = data
    criterions = ['gini','entropy']
    for criterion in criterions:
        clf = DecisionTreeClassifier(criterion=criterion)
        clf.fit(X_train,y_train)
        print('criterion:%s'%criterion)
        print("Traing score:%f" % (clf.score(X_train, y_train)))
        print("Testing score:%f"%(clf.score(X_test,y_test)))
# DecisionTreeClassifier_criterion(X_train, X_test, y_train, y_test)


# 检测随机划分与最优划分的影响
def DecisionTreeClassifier_splitter(*data):
    X_train, X_test, y_train, y_test = data
    splitters = ['best','random']
    for splitter in splitters:
        clf = DecisionTreeClassifier(splitter=splitter)
        clf.fit(X_train,y_train)
        print("splitter:%s"%splitter)
        print("Traing score:%f" % (clf.score(X_train, y_train)))
        print("Testing score:%f"%(clf.score(X_test,y_test)))
# DecisionTreeClassifier_splitter(X_train, X_test, y_train, y_test)



