from sklearn.tree import DecisionTreeClassifier
import numpy as np

ndevices = 5
start = 1
end = ndevices
iter_times = 1000

def searchParam(device_no, iter_times):
        # ————导入数据————
        feature_matrix = np.load('feature_matrixs/feature_matrix_bed' + str(device_no) + '.npy')
        label_matrix = np.load('feature_matrixs/label_matrix_bed' + str(device_no) + '.npy')

        # 划分训练集和测试集
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            feature_matrix, label_matrix, test_size=0.2, random_state=0)

        # # 用GridSearchCV寻找最优参数（字典）
        # from sklearn.model_selection import GridSearchCV
        # param_dist = {'criterion': ['gini', 'entropy'],
        #          'splitter': ['best', 'random'],
        #          'max_depth': [8,9,10,11,12,13,14],
        #          'min_samples_split': [2,3,4,5,6,7,8,9],
        #          'min_samples_leaf': [1,2,3,4,5,10],
        #          'max_leaf_nodes': [23,24,25,26,29,30,31],
        #          'min_impurity_decrease':[0.0001,0.0002,0.0003,0.0004,0.00008,0.00009]}
        # # param_dist = {'criterion': ['gini', 'entropy'],
        # #          'splitter': ['best', 'random'],
        # #          'max_depth': range(5, 20),
        # #          'min_samples_split': range(2, 20),
        # #          'min_samples_leaf': range(1, 20),
        # #          'max_leaf_nodes': range(10, 50),
        # #          'min_impurity_decrease':np.arange(0.0001, 0.01, 0.0001)}
        # grid = GridSearchCV(DecisionTreeClassifier(), param_grid=param_dist, cv=3)
        # grid.fit(X_train, y_train)
        # print('最优参数:', grid.best_params_)
        # print('最优分数:', grid.best_score_)
        # reg = grid.best_estimator_
        # print('测试分数: %f'%reg.score(X_test, y_test))

        # #保存结果
        # import pandas as pd
        # cv_result = pd.DataFrame.from_dict(grid.cv_results_)
        # with open('cv_result.csv','w') as f:
        #     cv_result.to_csv(f)

        # 用RandomizedSearchCV寻找最优参数
        from sklearn.model_selection import RandomizedSearchCV
        param_dist = {
                'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random'],
                'max_depth':range(5,50),
                'min_samples_split': range(2,50),
                'min_samples_leaf': range(1,50),
                'max_leaf_nodes': range(2,100),

                }
        # 'min_impurity_decrease':np.arange(0.0001,0.01,0.0001)

        # param_dist = {
        #         'criterion': ['gini', 'entropy'],
        #         'splitter': ['best', 'random'],
        #         'max_depth':range(5,50),
        #         'min_samples_split': range(2,50),
        #         'min_samples_leaf': range(1,50),
        #         'max_leaf_nodes': range(2,50),
        #         'min_impurity_decrease':np.arange(0.0001,0.01,0.0001)
        #         }
        # grid = RandomizedSearchCV(DecisionTreeClassifier(),param_dist, cv = 3,
        #                           scoring = 'neg_log_loss',n_iter=300)
        grid = RandomizedSearchCV(DecisionTreeClassifier(), param_dist, iid=False, cv=10, n_iter=iter_times)
        grid.fit(X_train, y_train)
        print('最优参数:', grid.best_params_)
        print('最优分数:', grid.best_score_)
        reg = grid.best_estimator_
        print('测试分数: %f'%reg.score(X_test, y_test))


for i in range(start, end + 1):
        print("---------device_" + str(i) + "---------")
        searchParam(i, iter_times)