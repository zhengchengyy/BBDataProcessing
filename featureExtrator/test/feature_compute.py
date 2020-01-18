from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree.export import export_graphviz
from sklearn.feature_selection import mutual_info_classif

X = [[1,0,0], [0,0,0], [0,0,1], [0,1,0]]

y = [1,0,1,1]

clf = DecisionTreeClassifier()
clf.fit(X, y)
print("方法1得到的重要性：", clf.feature_importances_)  #底层调用compute_feature_importances()

feat_importance = clf.tree_.compute_feature_importances(normalize=False)
print("方法2得到的重要性(归一化前)：" + str(feat_importance))

print("通过归一化计算：",end="")
for i in feat_importance:
    print(i / sum(feat_importance),end=",")

print()
feat_importance = clf.tree_.compute_feature_importances()
print("方法2得到的重要性(归一化后)：" + str(feat_importance))

# out = StringIO()
# out = export_graphviz(clf, out_file='test/tree.dot')