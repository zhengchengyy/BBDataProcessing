from pymongo import MongoClient
from matplotlib import pyplot as plt
from matplotlib import style
from exceptions import CollectionError
from scipy import signal
import time

config = {'action': 'turn_over',
          'db': 'beaglebone',
          'tag_collection': 'tags_411',
          'volt_collection': 'volts_411',
          'offset': 0}


def timeToFormat(t):
    ftime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
    return ftime


def timeToSecond(t):
    stime = time.strftime("%M:%S", time.localtime(t))
    return stime


def plot_from_db(action, db, volt_collection, tag_collection, port=27017, host='localhost', ndevices=3, offset=0):
    client = MongoClient(port=port, host=host)
    database = client[db]
    tag_collection = database[tag_collection]
    volt_collection = database[volt_collection]

    try:
        if volt_collection.count_documents({}) + tag_collection.count_documents({}) < 2:
            raise CollectionError('Collection not found, please check names of the collection and database')
    except CollectionError as e:
        print(e.message)

    ntags = tag_collection.count_documents({'tag': action})
    n = 1

    title = config['volt_collection'][6:] + "" + action + "_filter"
    fig = plt.figure(title, figsize=(6, 8))
    fig.suptitle(action + "_filter")

    # plot the data that is of a certain action one by one
    for tag in tag_collection.find({'tag': action}):
        # inittime
        inittime, termtime = tag['inittime'] - offset, tag['termtime'] - offset
        # get the arrays according to which we will plot later
        times, volts, filter_volts = {}, {}, {}
        for i in range(1, ndevices + 1):
            times[i] = []
            volts[i] = []
            filter_volts[i] = []

        for volt in volt_collection.find({'time': {'$gt': inittime, '$lt': termtime}}):
            device_no = int(volt['device_no'])
            v = volt['voltage']
            t = volt['time']
            times[device_no].append(t)
            volts[device_no].append(v)

        style.use('default')
        colors = ['r', 'b', 'g', 'c', 'm']  # m c
        subtitle = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        base = ntags * 100 + 10

        # plot, add_subplot(211)将画布分割成2行1列，图像画在从左到右从上到下的第1块
        ax = fig.add_subplot(base + n)
        plt.subplots_adjust(hspace=0.5)  # 函数中的wspace是子图之间的垂直间距，hspace是子图的上下间距
        ax.set_title("Person" + subtitle[n - 1] + ": " + timeToFormat(inittime + offset) + " ~ " + timeToFormat(
            termtime + offset))
        ax.set_xlim(inittime, termtime)

        # 自定义y轴的区间范围，可以使图放大或者缩小
        # ax.set_ylim([0.8,1.8])
        ax.set_ylim([0.75, 0.90])
        # ax.set_ylim([0.82, 0.83])
        # ax.set_ylim(-0.1,0.1)
        ax.set_ylabel('voltage')

        for i in range(1, ndevices + 1):
            # [v + i*0.2 for v in volts[i]]为了把多个设备的数据隔离开
            b, a = signal.butter(8, 4 / 7, 'lowpass')  # 配置滤波器，8表示滤波器的阶数
            filter_volts[i] = signal.lfilter(b, a, volts[i])
            # b, a = signal.butter(8, [1/7,2/7], 'bandpass')  # 带通滤波
            print(len(volts[i]))
            import numpy as np
            filter_volts[i] = signal.filtfilt(b, a, np.asarray(volts[i]))
            ax.plot(times[i], filter_volts[i], label='device_' + str(i), color=colors[i - 1], alpha=0.9)

        # print(str(volts[1][3])+"&&"+str(filter_volts[1][3]))
        if n == 1:
            ax.legend(loc='upper right')
        if n == ntags:
            ax.set_xlabel('time')
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

    # 最大化显示图像窗口
    plt.get_current_fig_manager().window.showMaximized()
    plt.show()


if __name__ == '__main__':
    plot_from_db(action=config['action'],
                 db=config['db'],
                 tag_collection=config['tag_collection'],
                 volt_collection=config['volt_collection'],
                 offset=config['offset'])
