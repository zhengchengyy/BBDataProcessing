import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

# https://blog.csdn.net/qq_36523839/article/details/82556932
# https://www.cnblogs.com/eternallql/p/8718919.html
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=3,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 20)):
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
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    print("test_scores_mean:",np.mean(test_scores))
    print("test_scores_std:", np.std(test_scores))
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
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")

    return plt


# ★决策树分类画学习曲线
from sklearn.tree import DecisionTreeClassifier
X = np.load('feature_matrixs/feature_matrix2.npy')
y = np.load('feature_matrixs/label_matrix2.npy')
title = "Learning Curves (DecisionTree)"
cv = ShuffleSplit(n_splits=100, test_size=0.25, random_state=0)
estimator = DecisionTreeClassifier(max_depth=9, min_samples_split=10, max_leaf_nodes=15)
# import pickle
# with open('models/0.968Acc_2Fea.pickle', 'rb') as f:
#     model = pickle.load(f)
plot_learning_curve(estimator, title, X, y, cv=cv, n_jobs=1)
plt.show()