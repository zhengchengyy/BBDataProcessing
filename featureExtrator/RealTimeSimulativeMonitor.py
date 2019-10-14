from feature_modules import *
from feature_extractor import FeatureExtractor
import time

from pymongo import MongoClient
from exceptions import CollectionError
from pymongo import MongoClient

from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np
import os

config = {'db': 'beaglebone',
          'tag_collection': 'tags_411',
          'volt_collection': 'volts_411',
          'offset': 0}


# 导入全局变量
import GlobalVariable as gv
action_names = gv.action_names
feature_names = gv.feature_names


def timeToFormat(t):
    ftime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
    return ftime


def timeToSecond(t):
    stime = time.strftime("%M:%S", time.localtime(t))
    return stime


def draw_features_from_db(action, db, volt_collection, tag_collection, port=27017,
                          host='localhost', ndevices=5, offset=0, action_num=0):
    client = MongoClient(port=port, host=host)
    database = client[db]
    tag_collection = database[tag_collection]
    volt_collection = database[volt_collection]

    try:
        if volt_collection.count_documents({}) + tag_collection.count_documents({}) < 2:
            raise CollectionError('Collection not found！')
    except CollectionError as e:
        print(e.message)

    ntags = tag_collection.count_documents({'tag': action})

    title = config['volt_collection'][6:] + "" + action + "_features"
    fig = plt.figure(title, figsize=(6, 8))

    # 根据时间采集数据，基本单位为s，比如1s、10s、30s、60s
    # interval表示每次分析的时间跨度，rate表示间隔多长时间进行一次分析
    interval = 1
    rate = 1
    fig.suptitle(action + " (" + "interval:" + str(interval) + "s, " + "stepsize:" + str(rate) + "s)")

    # 定义特征提取器
    extractor = FeatureExtractor()

    for feature in feature_names:
        # 定义特征提取模块
        module = eval(feature + "(" + str(interval) + "," + str(rate) + ")")
        # 注册特征提取模块
        extractor.register(module)

    # 定义画布左右位置的计数：标签累加，即人数累加
    tag_acc = 1

    # 读取模型
    import pickle
    with open('models/0.973Acc_6Fea.pickle', 'rb') as f:
        model = pickle.load(f)

    # read the data that is of a certain action one by one
    for tag in tag_collection.find({'tag': action}):
        inittime, termtime = tag['inittime'], tag['termtime']
        # 定义总特征计数
        feature_acc = 0
        # 定义精确率计数
        accuracy_acc = 0

        # get the arrays according to which we will plot later
        times, volts = {}, {}
        for i in range(1, ndevices + 1):
            times[i] = []
            volts[i] = []

        for volt in volt_collection.find({'time': {'$gt': inittime, '$lt': termtime}}):
            device_no = int(volt['device_no'])
            v = volt['voltage']
            t = volt['time']
            times[device_no].append(t)
            volts[device_no].append(v)

        # 定义存储时间、特征列表
        feature_times, feature_values = {}, {}
        for i in range(1, ndevices + 1):
            feature_times[i] = []
            from collections import defaultdict
            feature_values[i] = defaultdict(list)
            for feature in feature_names:
                feature_values[i][feature[:-6]] = []

        # 提取第几个设备的特征
        start = 2

        # 对每个采集设备进行特征提取 ndevices
        for i in range(start, start + 1):
            for j in range(len(volts[i])):
                value = {
                    "time": times[i][j],
                    "volt": volts[i][j]
                }
                output = extractor.process(value)
                if (output):
                    feature_acc += 1
                    features = {
                        "device_no": i,
                        "feature_time": times[i][j],
                        "feature_value": output,
                        "interval": interval,
                        "rate": rate
                    }
                    feature_times[i].append(features['feature_time'])
                    feature_temp = []  #存储实时计算的一条特征数据
                    for feature_type in feature_values[i].keys():
                        # print(feature_type, features['feature_value'][feature_type])
                        feature_temp.append(features['feature_value'][feature_type])
                        feature_values[i][feature_type].append(features['feature_value'][feature_type])

                    # print(feature_temp)

                    predict_result = model.predict([feature_temp])
                    format_result = "person" + str(tag_acc) + " " + "device_" + str(i)
                    # print(format_result, action_names[predict_result[0]])
                    if(predict_result[0]==action_num):
                        accuracy_acc += 1

            # 清理所有模块，防止过期数据
            extractor.clear()

        print("action:", action_names[action_num])
        print("person" + str(tag_acc) + "的精确率：", accuracy_acc/feature_acc)

        tag_acc += 1


if __name__ == '__main__':

    for action_num in range(len(action_names)):
    # action_num = 0
        draw_features_from_db(action=action_names[action_num],
                                  db=config['db'],
                                  tag_collection=config['tag_collection'],
                                  volt_collection=config['volt_collection'],
                                  offset=config['offset'],
                                  action_num=action_num)
