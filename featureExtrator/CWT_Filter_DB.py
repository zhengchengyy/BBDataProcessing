from pymongo import MongoClient
from matplotlib import pyplot as plt
from matplotlib import style
from exceptions import CollectionError
import time

import numpy as np
import pywt

action = ["still", "turn_over", "legs_stretch", "hands_stretch",
          "legs_twitch", "hands_twitch", "head_move", "grasp", "kick"]

config = {'action': action[1],
          'db': 'beaglebone',
          'tag_collection': 'tags_411',
          'volt_collection': 'volts_411',
          'ndevices': 3,
          'offset': 0
          }

def timeToFormat(t):
    ftime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
    return ftime

def timeToSecond(t):
    stime = time.strftime("%M:%S", time.localtime(t))
    return stime

def plot_from_db(action, db, volt_collection, tag_collection,port=27017, host='localhost', ndevices=3, offset=0):
    client = MongoClient(port=port, host=host)
    database = client[db]
    tag_collection = database[tag_collection]
    volt_collection = database[volt_collection]

    try:
        if volt_collection.count_documents({}) + tag_collection.count_documents({}) < 2:
            raise CollectionError('Collection not found, please check names of the collection and database')
    except CollectionError as e:
        print(e.message)

    ntags = tag_collection.count_documents({'tag':action})
    n = 1
    # 用于查看几号设备的图
    start = 2

    title =config['volt_collection'][6:] + "" + action +"_cwt"
    fig = plt.figure(title, figsize=(6,8))
    fig.suptitle(action+"_cwt")

    # plot the data that is of a certain action one by one
    for tag in tag_collection.find({'tag': action}):
        # inittime termtime
        inittime, termtime = tag['inittime'] - offset, tag['termtime'] - offset
        # get the arrays according to which we will plot later
        times, volts = {}, {}
        for i in range(1, ndevices + 1):
            times[i] = []
            volts[i] = []

        for volt in volt_collection.find({'time': {'$gt': inittime,'$lt': termtime}}):
            device_no = int(volt['device_no'])
            v = volt['voltage']
            t = volt['time']
            times[device_no].append(t)
            volts[device_no].append(v)

        style.use('default')
        colors = ['r', 'b', 'g', 'c', 'm']
        subtitle = ['A', 'B', 'C' ,'D', 'E', 'F', 'G']
        base = ntags*100+10

        # plot, add_subplot(211)将画布分割成2行1列，图像画在从左到右从上到下的第1块
        ax = fig.add_subplot(base+n)
        plt.subplots_adjust(hspace=0.5)  # 函数中的wspace是子图之间的垂直间距，hspace是子图的上下间距
        ax.set_title("Person" + subtitle[n - 1] + ": " + timeToFormat(inittime + offset) + " ~ " + timeToFormat(termtime + offset))

        # 自定义y轴的区间范围，可以使图放大或者缩小
        # ax.set_ylim(0, 4)
        ax.set_ylabel('Frequency')

        for i in range(start, start + 1):
            # ax.plot(times[i], volts[i], label='device_' + str(i), color=colors[i - 1], alpha=0.9)
            x = times[i]
            y = volts[i]
            ca, cd = pywt.dwt(y, 'db4')
            ya = pywt.idwt(ca, None, 'db4')  # approximated component
            yd = pywt.idwt(None, cd, 'db4')  # detailed component
            plt.figure(figsize=(12, 9))
            plt.subplot(311)
            plt.plot(x, y)
            plt.title('original signal')
            plt.subplot(312)
            plt.plot(x, ya)
            plt.title('approximated component')
            plt.subplot(313)
            plt.plot(x, yd)
            plt.title('detailed component')
            plt.tight_layout()
            plt.show()



        if n  == 1:
            ax.legend(loc='upper right')
        if n == ntags:
            ax.set_xlabel('Time')
        n += 1

        # 以第一个设备的时间数据为准，数据的每1/10添加一个x轴标签
        xticks = []
        xticklabels = []
        length = len(times[1])
        interval = length // 10 - 1
        for i in range(0, length, interval):
            xticks.append(times[1][i])
            xticklabels.append(timeToSecond(times[1][i] + offset))
        ax.set_xticks(xticks)  # 设定标签的实际数字，数据类型必须和原数据一致
        ax.set_xticklabels(xticklabels, rotation=15)  # 设定我们希望它显示的结果，xticks和xticklabels的元素一一对应

    plt.show()


if __name__=='__main__':
    plot_from_db(action=config['action'],
                 db=config['db'],
                 tag_collection=config['tag_collection'],
                 volt_collection=config['volt_collection'],
                 ndevices=config['ndevices'],
                 offset=config['offset'])