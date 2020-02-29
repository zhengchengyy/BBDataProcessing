# 训练完把测试标签和预测分数二值化后画出ROC，后剪枝部分采用的是该方法
# 引入必要的库
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.tree import DecisionTreeClassifier

ndevices = 5
start = 1
end = ndevices

# 读取模型
import pickle
# 导入全局变量
import GlobalVariable as gv
action_names = gv.action_names
feature_names = gv.feature_names

def drawROC(device_no):
    with open('models/' + 'device_' + str(device_no) + '_post_prune.pickle', 'rb') as f:
        clf = pickle.load(f)

    # 导入数据
    feature_matrix = np.load('feature_matrixs/feature_matrix'+str(device_no)+'.npy')
    label_matrix = np.load('feature_matrixs/label_matrix'+str(device_no)+'.npy')

    # 随机化和划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix, label_matrix, test_size=0.2, random_state=0)

    # 使用决策树训练
    # clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print('device_' + str(device_no) + '\'s train score:', train_score, round(train_score,3))
    print('device_' + str(device_no) +'\'s test score:', test_score, round(test_score,3))

    # 不能使用predict，使用结果对应位置为1表示该类，但不代表该类的概率，
    # 这里使用了影响也不大的原因是模型准确率高，predict_proba得到对应位置的概率也是1
    # y_score = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)
    # 将标签类二值化,对应类的位置为1,其它位置为0,如[1,0,0],[0,1,0],[0,0,1]
    n_classes = len(action_names)
    # y_score = label_binarize(y_score, classes=range(n_classes))
    y_test = label_binarize(y_test, classes=range(n_classes))

    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC curve and ROC area（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Compute micro-average ROC curve and ROC area（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot all ROC curves
    lw = 1.5  # the width of line
    title = "ROC" + "_device_" + str(device_no)
    plt.figure(title)
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (AUC = {0:0.3f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (AUC = {0:0.3f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    plot_colors = ['r', 'm', 'c', 'b', 'g', 'lime', 'y', 'peru', 'navy', 'orange']
    for i, color in zip(range(n_classes), plot_colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label=action_names[i] + '(AUC = {0:0.3f})'.format(roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', color='pink', lw=lw, label="y=x(AUC=0.5)")
    plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.ylim([0.7, 1.01])
    plt.ylim([0.75, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves of different sleep movements of device_'+str(device_no))
    plt.legend(loc="lower right")

    # 画单个类的roc曲线
    # plt.figure()
    # lw = 2
    # plt.plot(fpr[2], tpr[2], color=plot_colors[2],
    #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    # # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()

    # 最大化显示图像窗口
    # plt.get_current_fig_manager().window.state('zoomed')
    plt.savefig("ROC_images/" + title + ".png")
    # plt.savefig("ROC_images/" + title + ".png", dpi=200)
    # plt.show()


for i in range(start, end + 1):
    print("---------device_" + str(i) + " write ROC image write begin---------")
    drawROC(i)
    print("---------device_" + str(i) + " write ROC image write end---------")