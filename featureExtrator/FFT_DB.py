from pymongo import MongoClient
from matplotlib import pyplot as plt
from matplotlib import style
from exceptions import CollectionError
import time

import numpy as np

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


def plot_from_db(action, db, volt_collection, tag_collection, port=27017, host='localhost',
                 ndevices=5, offset=0):
    client = MongoClient(port=port, host=host)
    database = client[db]
    tag_collection = database[tag_collection]
    volt_collection = database[volt_collection]

    try:
        if volt_collection.count_documents({}) + tag_collection.count_documents({}) < 2:
            raise CollectionError('Collection not found!')
    except CollectionError as e:
        print(e.message)

    ntags = tag_collection.count_documents({'tag': action})
    n = 1
    # 用于查看几号设备的图
    start = 3

    title = config['volt_collection'][6:] + "" + action + "_fft_" + str(start)
    fig = plt.figure(title, figsize=(6, 8))
    fig.suptitle(action + "_fft_" + str(start))

    # plot the data that is of a certain action one by one
    for tag in tag_collection.find({'tag': action}):
        # inittime
        inittime, termtime = tag['inittime'] - offset, tag['termtime'] - offset
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

        style.use('default')
        colors = ['r', 'b', 'g', 'c', 'm']
        subtitle = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        base = ntags * 100 + 10

        # plot, add_subplot(211)将画布分割成2行1列，图像画在从左到右从上到下的第1块
        ax = fig.add_subplot(base + n)
        plt.subplots_adjust(hspace=0.5)  # 函数中的wspace是子图之间的垂直间距，hspace是子图的上下间距
        ax.set_title("Person" + subtitle[n - 1] + ": " + timeToFormat(inittime + offset)
                     + " ~ " + timeToFormat(termtime + offset))

        # 自定义y轴的区间范围，可以使图放大或者缩小
        ax.set_ylim(0, 0.001)
        # ax.set_ylim(0, 0.0003)
        # ax.set_ylim(0, 1)
        ax.set_ylabel('Amplitude')

        for i in range(start, start + 1):
            # fft返回值实部表示
            result = np.fft.fft(volts[i])  # 除以长度表示归一化处理
            # fftfreq第一个参数n是FFT的点数，一般取FFT之后的数据的长度（size）
            # 第二个参数d是采样周期，其倒数就是采样频率Fs，即d=1/Fs
            freq = np.fft.fftfreq(len(result), d=1 / 70)
            amplitude = np.sqrt(result.real ** 2 + result.imag ** 2) / (len(volts[i]) / 2)
            ax.plot(abs(freq), amplitude, label='device_' + str(i),
                    color=colors[i - 1], alpha=0.9)

        if n == 1:
            ax.legend(loc='upper right')
        if n == ntags:
            ax.set_xlabel('Frequency')
        n += 1

    # 最大化显示图像窗口
    plt.get_current_fig_manager().window.showMaximized()
    plt.show()


if __name__ == '__main__':
    plot_from_db(action=config['action'],
                 db=config['db'],
                 tag_collection=config['tag_collection'],
                 volt_collection=config['volt_collection'],
                 offset=config['offset'])
