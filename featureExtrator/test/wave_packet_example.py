import pywt


x = [1, 2, 3, 4, 5, 6, 7, 8]
wp = pywt.WaveletPacket(data=x, wavelet='db1', mode='symmetric')

# 输入数据和分解系数（细节系数和逼近系数）都可以通过WaveletPacket.data得到
print(wp.data)
# 根节点的路径
print(repr(wp.path))
print(wp.level)
# 最大分解层
print(wp['ad'].maxlevel)
# 检查最大分解层数
print(wp.maxlevel)
# 获取小波包树的子节点
print(wp['a'].data)
# print(wp['a'].path)
print(wp['aa'].data)
# print(wp['aa'].path)
print(wp['aaa'].data)
# print(wp['aaa'].path)