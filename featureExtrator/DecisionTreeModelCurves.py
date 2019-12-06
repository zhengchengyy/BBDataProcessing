import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


# https://blog.csdn.net/qq_36523839/article/details/82556932
# https://www.cnblogs.com/eternallql/p/8718919.html
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=3,
                        n_jobs=1, train_sizes=np.linspace(0.1, 1.0, 5)):
    """
        画出data在某模型上的learning curve.
        参数解释
        ----------
        estimator : 你用的分类器。
        title : 表格的标题。
        X : 输入的feature，numpy类型
        y : 输入的target vector
        ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
        cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
        n_jobs : 并行的的任务数(默认1)
        train_sizes : np.linspace(.1, 1.0, 10))开始为0.1，结束为1，把交叉验证中的训练集分成10份
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    # train_sizes会少于10，
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

device_no = 1
# 导入数据
feature_matrix = np.load('feature_matrixs/feature_matrix' + str(device_no) + '.npy')
label_matrix = np.load('feature_matrixs/label_matrix' + str(device_no) + '.npy')

# 定义训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    feature_matrix, label_matrix, test_size=0.2, random_state=0)

# 把训练数据用于交叉验证
X = X_train
y = y_train

# ★决策树分类
from sklearn.tree import DecisionTreeClassifier
estimator = DecisionTreeClassifier(random_state=0,
                                 max_depth=11,
                                 max_leaf_nodes=78,
                                 min_impurity_decrease=0.00032,
                                 min_samples_leaf=2,
                                 min_samples_split=5,
                                 splitter='best',
                                 criterion='entropy')
# import pickle
# with open('models_discard/0.968Acc_2Fea.pickle', 'rb') as f:
#     model = pickle.load(f)
# plot_learning_curve(estimator, "Learning Curves", X, y)
from sklearn.model_selection import ShuffleSplit
title = "Decision Tree (DT)"
# 自定义交叉验证的生成器，下面是把数据生成10种，每种测试集占0.1，其实这就是10折交叉验证。
# cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
# 画学习曲线，cv=5表示5倍交叉验证，把训练集分成5份，把四份作为训练，一份作为验证，可以组成5种，最后得到平均验证分数
plot_learning_curve(estimator, title, X, y, cv=20, n_jobs=1)
plt.show()

# ★梯度提升决策树
from sklearn.ensemble import GradientBoostingClassifier
title = "Learning Curves (GradientBoosting)"
# cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
estimator = GradientBoostingClassifier(n_estimators=100)  # 建模，速度有点慢
# plot_learning_curve(estimator, title, X, y, cv=10, n_jobs=1)  #到达0.9
plt.show()

# ★K近邻分类算法
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

title = "Learning Curves (KNN)"
cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
estimator = KNeighborsClassifier()  # 建模
# plot_learning_curve(estimator, title, X, y, cv=10, n_jobs=1)
plt.show()

# 线性回归分类
from sklearn.linear_model import LinearRegression
title = "Learning Curves (KNN)"
cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
estimator = LinearRegression()  # 建模
# plot_learning_curve(estimator, title, X, y, cv=10, n_jobs=1)
plt.show()

# ★随机森林
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=196)
title = "Learning Curves (RandomForest)"
cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
estimator = RandomForestClassifier(n_estimators=8)  # 建模，必须初始化n_estimators，否则报错
# plot_learning_curve(estimator, title, X, y, cv=10, n_jobs=1)
plt.show()

# 朴素贝叶斯分类GaussianNB，['BernoulliNB', 'GaussianNB', 'MultinomialNB', 'ComplementNB']
from sklearn.naive_bayes import GaussianNB
title = "Learning Curves (GaussianNB)"
cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
estimator = GaussianNB()  # 建模
# plot_learning_curve(estimator, title, X, y, cv=10, n_jobs=1)
plt.show()

# 朴素贝叶斯分类MultinomialNB
from sklearn.naive_bayes import MultinomialNB
title = "Learning Curves (MultinomialNB)"
# cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
estimator = MultinomialNB()  # 建模
# plot_learning_curve(estimator, title, X, y, cv=10, n_jobs=1)
plt.show()

# 支持向量机分类
from sklearn.svm import SVC
title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
estimator = SVC(gamma=0.001)  # 建模
# plot_learning_curve(estimator, title, X, y, cv=cv, n_jobs=1)
plt.show()
