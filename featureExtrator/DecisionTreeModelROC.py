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

# 导入数据
feature_matrix = np.load('feature_matrixs/feature_matrix1.npy')
label_matrix = np.load('feature_matrixs/label_matrix1.npy')

# 导入全局变量
import GlobalVariable as gv

action_names = gv.action_names
feature_names = gv.feature_names

# 绘制ROC曲线
X = feature_matrix
y = label_matrix
# 设置种类
n_classes = feature_matrix.shape[1]
# 将标签类数值化
# y = label_binarize(y, classes=[0, 1, 2, 3, 4, 5, 6, 7])
y = label_binarize(y, classes=range(n_classes))

# 训练模型并预测
n_samples, n_features = X.shape
# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# 使用决策树训练
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_score = clf.predict(X_test)
score = clf.score(X_test, y_test)
print(score)

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

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

# Plot all ROC curves
lw = 2
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

# colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
plot_colors = ['r', 'm', 'c', 'b', 'g', 'lime', 'y', 'peru', 'navy', 'orange']
for i, color in zip(range(n_classes), plot_colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label=action_names[i] + '(area = {0:0.2f})'.format(roc_auc[i]))

# plt.plot([0, 1], [0, 1], 'k--', color='navy', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves of different sleep movements')
plt.legend(loc="lower right")
plt.show()

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
