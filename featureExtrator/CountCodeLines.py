# 统计python代码行数
import os
import time

# 需要统计的文件夹或者文件，这是在windows下运行的，如果使用Linux系统可以使用basedir = '/app/log'
test = 'D:\Offer\BBDataProcessing\\featureExtrator\\test\CodeLines'
data_processing_dir = 'D:\Offer\BBDataProcessing\\featureExtrator'
test_dir = 'D:\Offer\BBDataProcessing\\featureExtrator\\test'
expired_codes_dir = 'D:\Offer\BBDataProcessing\\featureExtrator\\expired_codes'
data_collect_dir = 'D:\Offer\BBDetection'

dir_list = [test, data_processing_dir, test_dir, expired_codes_dir, data_collect_dir]

# testdir2 = 'D:\Offer\BBDataProcessing\\synchronizeData\\test'

filelists = []
# 指定想要统计的文件类型
whitelist = ['py']


# 遍历文件, 递归遍历文件夹中的所有
def getFile(basedir):
    global filelists
    for parent, dirnames, filenames in os.walk(basedir):
        for filename in filenames:
            ext = filename.split('.')[-1]
            # 只统计指定的文件类型，略过一些log和cache文件
            if ext in whitelist:
                filelists.append(os.path.join(parent, filename))


# 统计一个文件的行数
def countLine(fname):
    count = 0
    for file_line in open(fname, encoding='UTF-8').readlines():
        if file_line != '' and file_line != '\n' and file_line.lstrip()[0]!='#':  # 过滤掉空行、注释
            count += 1
    # print(fname + '----' , count)
    return count


if __name__ == '__main__':
    for dir in dir_list:
        filelists.clear()
        startTime = time.clock()
        getFile(dir)
        totalline = 0
        for filelist in filelists:
            totalline = totalline + countLine(filelist)
        print('------------'+dir)
        print('total files:', len(filelists))
        print('total lines:', totalline)
        print('Done! Cost Time: %0.2f second' % (time.clock() - startTime))