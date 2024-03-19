import wfdb
import numpy as np
import matplotlib.pyplot as plt
from function import fun


# Load the WFDB record and annotation
record = wfdb.rdrecord('sampledata/100', sampto=1500) #实际数据
annotation = wfdb.rdann('sampledata/100', 'atr', sampto=1500) #专家标记

# 提取心电图信号数据
ecg_signal = record.p_signal
annotation_indices = annotation.sample
# 创建一个新的图形
plt.figure()

# 在两个子图上分别绘制心电图信号和注释标签
for i in range(ecg_signal.shape[1]):
    plt.subplot(2, 1, i+1)
    plt.plot(ecg_signal[:, i], label='ECG signal')
    #plt.scatter(annotation_indices, ecg_signal[annotation_indices, i], color='red')
    plt.title('ECG signal with annotations')
# 显示图形
plt.show(block=False)

plt.figure()
for i in range(ecg_signal.shape[1]):
    plt.subplot(2, 1, i+1)
    plt.plot(fun.wavelet_denoising(fun.elimate_base(ecg_signal[:, i])), label='ECG signal')
    #plt.scatter(annotation_indices, ecg_signal[annotation_indices, i], color='red')
    plt.title('ECG signal with annotations')
plt.show(block = False) 
#进行Music
music1=fun.wavelet_denoising(fun.elimate_base(ecg_signal[:, 0]))
music2=fun.wavelet_denoising(fun.elimate_base(ecg_signal[:, 1]))
music = np.vstack((music1, music2))
plt.figure()
for i in range(ecg_signal.shape[1]):
    plt.subplot(2, 1, i+1)
    plt.plot(fun.Music((music[i,:]),3,1), label='ECG signal')
    #plt.scatter(annotation_indices, ecg_signal[annotation_indices, i], color='red')
    plt.title('ECG signal with annotations')
plt.show() 