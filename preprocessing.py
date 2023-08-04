import os
#更改工作路径 大家自己注意路径中的斜杠问题
os.chdir("F:/64 VEEG")
#引入mne 和 numpy
import mne
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle

#导入一例原始数据
#读取原始数据 mne区分是否是原始数据的方式是看数据的维度 二维的数据(连续数据)都是raw
#分段过的数据都用epoch的形式读
raw = mne.io.read_raw_edf('DingXiaoyao.edf',preload=True)
#eeg1 = mne.io.read_epochs_eeglab('1_LH.set')
#SET TYPES
# ch_types_map = {}
# for i in raw.ch_names:
# 	if i.isdigit():
# 		ch_types_map[i] = "seeg"
# 	if "ECG" in i:
# 		ch_types_map[i] = "ecg"
#     if "EEG" in i:
# 		ch_types_map[i] = "eeg"

#查看数据的基本信息
raw.info
print(raw)
dir(raw)
#查看采样率信息
sampling_rate = raw.info['sfreq']
#采样点信息
n_time_samps = raw.n_times
#每个采样点 对应的时间信息 单位是s
time_secs = raw.times
#查看通道名称
ch_names = raw.ch_names
#查看通道的数量
n_ch = len(ch_names)
#绘制数据
raw.plot()
#定义一次显示多少个通道 多长时间的数据 以及绘图的尺度
raw.plot(n_channels= 64, duration=5, scalings = 30e-6)


# #通道定位 绘制地形图
# #查看数据通道信息
# print(raw.ch_names)
# montage = mne.channels.read_custom_montage('standard-10-10-cap385.elp')
# raw.set_montage(montage)
#由于数据中的原始通道名称 没有办法被成功定位
#所以我们需要修改通道的名称信息
mapping = {'FP1':'Fp1', 'FPZ':'Fpz','FP2':'Fp2', 'FZ':'Fz',
          'CZ':'Cz',  'PZ':'Pz','POZ':'POz',  'OZ':'Oz',}
#修改通道的名称
raw_rename_ch = raw.copy().rename_channels(mapping)
ch_types_map = {}
for i in raw.ch_names:
    if i.isdigit():
        ch_types_map[i] = "seeg"
    if "ECG" in i:
        ch_types_map[i] = "ecg"
    if "EMG" in i:
        ch_types_map[i] = "emg"
    if re.match("[LR][0-9]", i):
        ch_types_map[i] = "dbs"
    if re.match("A[0-9]", i):
        ch_types_map[i] = "syst"
    if i.startswith('SPH'):
        ch_types_map[i] = "exci"
raw.set_channel_types(ch_types_map)
#读取通道定位文件
# montage = mne.channels.read_custom_montage('standard-10-5-cap385.elp')
#进行通道定位
raw_rename_ch.set_montage('standard_1005',on_missing='warn')
#绘制通道排布图
raw_rename_ch.plot_sensors()
#设定绘制的参数 通道的类型 通道的名称 和脑子轮廓的大小
raw_rename_ch.plot_sensors(ch_type = 'eeg', show_names = True, sphere = 0.075)
#绘制频谱响应 规定绘制的频率范围
raw_rename_ch.plot_psd(fmin = 1, fmax = 70,spatial_colors = True)
#去除无用电极 （更改并选择通道类型）
raw_select_ch = raw_rename_ch.copy()
#定义一些通道的类型 将HEOG 和 VEOG 定义为眼电电极
raw_select_ch.set_channel_types({'SPH1':'exci', 'SPH2':'exci', 'ECG':'ecg', 'LEMG':'emg', 'REMG':'emg','EOG1':'eog','EOG2':'eog','EOG3':'eog','EOG4':'eog'})
#查看对象的信息
raw_select_ch.info
#只保留eeg类型的数据
raw_select_ch.pick(['eeg'])
raw_select_ch.info
#ICA
from mne.preprocessing import (ICA)
ica_data = raw_select_ch.copy()
#定义ICA 独立成分分析的数量 数量 =  实际的电极数量 - 坏通道的数量
#但是当电极很多的时候 我们可以只进行64个独立成分的分解
ica = ICA(n_components= 50)
#进行ICA分析
ica.fit(ica_data)
#绘制ICA的独立成分图 只有地形图
ica.plot_components()
#range(0,50) [0,50)
#绘制一个一个独立的成分属性图
# ica.plot_properties(raw_resampled, picks= [0,1])
# ica.plot_properties(raw_resampled, picks= np.array(range(0,50)))
#查看某个特定成分去除前后对数据的整体影响 方便我们判断成分是否选择对了
ica.plot_overlay(ica_data, exclude=[0],picks='eeg')
# output = open('ica.pkl','wb')
# pickle.dump(ica, output)
# output.close()

# pkl_file = open('ica.pkl','rb')
# my_epochs = pickle.load(pkl_file)
# pkl_file.close()
#去除伪迹成分了 成分是从第0个开始数的
#指定要去除的成分是哪个成分
ica.exclude = [0,12]
#再次确认去除成分的地形图
ica.plot_components([1,12])
ica_clean = ica_data.copy()
#导入数据
ica_clean.load_data()
#根据刚才指定的去除的编号 进行伪迹成分去除
ica.apply(ica_clean)
#分别看一下去除伪迹成分前后的数据 方便对比去除成分的效果
# ica_clean.plot(n_epochs = 5, scalings = 30e-6)
# ica_data.plot(n_epochs = 5, scalings = 30e-6)

output = open('ica_clean.pkl','wb')
pickle.dump(ica_clean, output)
output.close()

pkl_file = open('ica_clean.pkl','rb')
raw_resampled = pickle.load(pkl_file)
pkl_file.close()
ica.plot_sources(raw_resampled, show_scrollbars=False)


#滤波
#对数据进行1到45的带通滤波
raw_band = raw_select_ch.copy().filter(1,45,picks = 'eeg')
#绘制频谱响应图 方便观察滤波前后数据在频域响应上的变化
raw_band.plot_psd(fmin = 1, fmax = 70,spatial_colors = True)

#raw_band.plot()
raw_band.info
#凹陷(带阻)滤波 去除工频干扰 中国是50Hz 国外是60Hz
#以50Hz为中心频率点 带阻的宽度为4Hz --> 48 - 52 的带阻
raw_band_notch = raw_band.notch_filter(50, notch_widths = 4)
raw_band_notch.plot_psd(fmin = 1, fmax = 70,spatial_colors = True)
#我们能够分析到的最大频率是采样率的一般
#但是实际上我们能够准确分析到的频率 是采样率 1/3 ~ 1/4
#降采样的好处是缩减数据量 节约计算和储存成本
#但是要注意降采样后的数据对频域分析的影响
#建议先滤波再降采样
#对数据进行将采样到500Hz(250HZ)
raw_resampled = raw_band_notch.copy().resample(sfreq = 250)
raw_resampled.info
#
# # #分段
# # #读取数据的marker信息
# # events_from_anno, event_dict = mne.events_from_annotations(raw_resampled)
# # #event_dict中是marker的类型和标记名称之间的关系
# # print(event_dict)
# # #每个标记出现的 维度个数*3（ 时间点 持续时间（0） 类型信息）
# # print(events_from_anno)
# # #定义想要对什么marker的数据进行分段
# # custom_mapping = {'Stimulus/10': 10, 'Stimulus/11': 11}
# # #更改数据中的marker信息
# # (events_from_anno,event_dict) = mne.events_from_annotations(raw_resampled,event_id = custom_mapping)
# # print(event_dict)
# # print(events_from_anno)
# # #对数据进行分段 定义分段的长度 和基线的范围
# # my_epochs = mne.Epochs(raw_resampled, events_from_anno, tmin = -0.2, tmax = 0.8,baseline = (-0.2,0))
# # #基线校正
# # my_epochs.apply_baseline()
# # my_epochs.info
# #
# #
# # #保存刚分段好的数据
import pickle
# # #用pickle 把 my_epohcs 变量存成 my_epochs.pkl 文件
# # output = open('my_epochs.pkl','wb')
# # pickle.dump(my_epochs, output)
# # output.close()
# #
# # del my_epochs
# # #将my_epochs.pkl文件读进来 读取在my_epochs变量中
# # pkl_file = open('my_epochs.pkl','rb')
# # my_epochs = pickle.load(pkl_file)
# # pkl_file.close()
# # my_epochs.info
# # #去除坏段
# # my_epochs_good = my_epochs.copy()
# # #绘制数据 分段前的数据 定义绘制数据的时间长度 分段后的数据 定义绘制的分段个数
# # #将认为是坏段的数据手动将分段标红
# # my_epochs_good.plot(n_epochs = 5,n_channels= 64)
# # #去除标记过的坏段
# # my_epochs_good.drop_bad
# # my_epochs_good.info
# # #查看剩余分段的信息
# # print(len(my_epochs_good.events))
# # #插值坏电极
# # #标记坏通道 鼠标电极通道的名称 通道变为灰色 即标记成功
# # my_epochs_good.plot(n_epochs = 5,n_channels= 64)
# # #进行坏通道插值
# # good_ch = my_epochs_good.load_data().copy().interpolate_bads(reset_bads=True)
# # good_ch.info
# # good_ch.plot(n_epochs = 5,n_channels= 64)
# #
# #
# # output = open('good_ch.pkl','wb')
# # pickle.dump(good_ch, output)
# # output.close()
# #
# # pkl_file = open('good_ch.pkl','rb')
# # good_ch = pickle.load(pkl_file)
# # pkl_file.close()
#
# #ICA独立成分分析 我们在做ICA的时候 意图是去伪迹
# #但是ICA不是万能的 比较好算出来的独立成分 是类似于眼电 肌电
# #在数据中对数据影响并不是非常大 而且规律出现的
# #建议大家在做ICA之前 先对数据进行 坏段去除 坏导插值
# #将规律的 眼电信息保留
# #引入ICA
f
#
# #极端值去伪迹
# extreme_values = ica_clean.copy()
# #建立一个伪迹去除的标准 100uv 60%
# stronger_reject_criteria = dict(eeg=100e-6)
# #stronger_reject_criteria = dict(eeg=70e-6)
# #按照上述设定的标准 进行自动的伪迹去除
# extreme_values.drop_bad(reject=stronger_reject_criteria)
# #查看去除的各个分段是为什么被拒绝的（人为标记 以及由于哪个通道超过了自动拒绝标准）
# print(extreme_values.drop_log)
# #绘制分段拒绝的情况图
# extreme_values.plot_drop_log()
#
# output = open('extreme_values.pkl','wb')
# pickle.dump(extreme_values, output)
# output.close()
#
# pkl_file = open('extreme_values.pkl','rb')
# extreme_values = pickle.load(pkl_file)
# pkl_file.close()
# #重参考 在线记录的参考电极 > 重参考电极 > 要分析的电极 > 剩余电极
# refered = extreme_values.copy().set_eeg_reference(ref_channels=['TP9', 'TP10'])
# #如果你需要做的是平均参考 那么请先去除双侧乳突等非脑电电极 再进行平均参考
# #refered = extreme_values.copy().set_eeg_reference(ref_channels='average')
#
# output = open('refered.pkl','wb')
# pickle.dump(refered, output)
# output.close()
getclone

