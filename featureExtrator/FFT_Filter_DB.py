from pymongo import MongoClient
from matplotlib import pyplot as plt
from matplotlib import style
from exceptions import CollectionError
import time
import numpy as np

action = ["still", "turn_over", "legs_stretch", "hands_stretch",
          "legs_twitch", "hands_twitch", "head_move", "grasp", "kick"]

action = ["get_up", "go_to_bed",
          "turn_over", "legs_stretch", "hands_stretch",
          "legs_tremble", "hands_tremble", "body_tremble",
          "head_move", "legs_move", "hands_move",
          "hands_rising", "kick"]

config = {'action': 'turn_over',
          'db': 'beaglebone',
          'tag_collection': 'tags_411',
          'volt_collection': 'volts_411',
          'offset': 0}


# config = {'action': "still",
#           'db': 'beaglebone',
#           'tag_collection': 'tags_424',
#           'volt_collection': 'volts_424',
#           'ndevices': 5,
#           'offset': 0
#           }


# config = {'action': "still",
#           'db': 'beaglebone',
#           'tag_collection': 'tags_1105',
#           'volt_collection': 'volts_1105',
#           'ndevices': 5,
#           'offset': 0
#           }


def timeToFormat(t):
    ftime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
    return ftime


def timeToSecond(t):
    stime = time.strftime("%M:%S", time.localtime(t))
    return stime


def fft_filter(data, sampling_frequency, threshold_frequency):
    fft_result = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(fft_result), d=sampling_frequency)
    begin = int(len(data) * threshold_frequency * sampling_frequency)
    fft_result[begin:] = 0  # 高通滤波
    filter_data = np.fft.ifft(fft_result)
    return abs(filter_data)


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

    # ntags表示总标签数，即人数；tag_acc表示累加计数
    ntags = tag_collection.count_documents({'tag': action})
    # ntags = 1
    tag_acc = 0

    # 用于查看几号设备的图
    start = 3
    end = start

    title = config['volt_collection'][6:] + "" + action + "_fft_" + str(start)
    fig = plt.figure(title, figsize=(6, 8))
    fig.suptitle(action + "_fft_" + str(start))

    # plot the data that is of a certain action one by one
    for tag in tag_collection.find({'tag': action}):
        tag_acc += 1
        if (tag_acc > ntags):
            break
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
        subtitle = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'M', 'N']

        ax = fig.add_subplot(ntags, 1, tag_acc)

        plt.subplots_adjust(hspace=0.5)  # 函数中的wspace是子图之间的垂直间距，hspace是子图的上下间距
        ax.set_title("Person" + subtitle[tag_acc - 1] + ": " + timeToFormat(inittime + offset)
                     + " ~ " + timeToFormat(termtime + offset))

        # 自定义y轴的区间范围，可以使图放大或者缩小
        ax.set_ylim(0, 0.001)
        # ax.set_ylim(0, 0.0003)
        # ax.set_ylim(0, 1)
        ax.set_ylabel('Amplitude')

        for i in range(start, start + 1):
            volts[i] = fft_filter(volts[i], 1/70, 15)
            # fft返回值实部表示
            result = np.fft.fft(volts[i])
            # 实数fft后会使原信号幅值翻N/2倍，直流分量即第一点翻N倍
            amplitudes = abs(result) / (len(result) / 2)  # 复数的绝对值其实就是它的模长
            amplitudes[0] /= 2
            # fftfreq第一个参数n是FFT的点数，一般取FFT之后的数据的长度（size）
            # 第二个参数d是采样周期，其倒数就是采样频率Fs，即d=1/Fs
            freqs = np.fft.fftfreq(len(result), d=1 / 70)
            ax.plot(abs(freqs), amplitudes, label='device_' + str(i), color=colors[i - 1], alpha=0.9)

        ax.grid(linestyle=':')
        if tag_acc == 1:
            ax.legend(loc='upper right')
        if tag_acc == ntags:
            ax.set_xlabel('Frequency')

    # 最大化显示图像窗口
    plt.get_current_fig_manager().window.state('zoomed')
    plt.show()


if __name__ == '__main__':
    plot_from_db(action=config['action'],
                 db=config['db'],
                 tag_collection=config['tag_collection'],
                 volt_collection=config['volt_collection'],
                 offset=config['offset'])
