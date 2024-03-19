import numpy as np
from scipy.signal import butter, filtfilt
from scipy import linalg
import pywt # 导入PyWavelets库
import matplotlib.pyplot as plt

def elimate_base(ecg_signal):
    cutoff = 0.5  # 截止频率为0.5Hz
    fs = 1000.0  # 假设采样频率为1000Hz
    # 计算归一化的截止频率
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    # 使用butter函数创建一个高通滤波器
    b, a = butter(1, normal_cutoff, btype='high', analog=False)
    ecg_signal= filtfilt(b, a, ecg_signal)
    return ecg_signal

def  wavelet_denoising(data):
    # 使用db8小波进行多级分解
    coeffs = pywt.wavedec(data, 'db8', level=5)
    # 将高频系数阈值化
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], value=np.std(coeffs[i])/2, mode='soft')
    # 使用更新的系数重构信号
    return pywt.waverec(coeffs, 'db8')

def Music(ecg,num_sources,fs):
    #将ecg转换为行向量
    ecgcol=ecg.reshape(-1, 1)
    ecgrow=ecg.reshape(1, -1)
    print(ecgcol.shape,ecgrow.shape)

    R=np.dot(ecgcol,ecgrow)/ len(ecg)
    #绘制自相关矩阵的特征值
    w,v = linalg.eig(R)
    indices = np.argsort(w)

    # 获取噪声子空间（即最小的特征值对应的特征向量）
    noise_subspace = v[:, indices[:len(indices)-num_sources]]
    f = np.linspace(0, fs/2, len(ecg))
    music_spectrum = np.zeros(len(f))
    for i in range(len(f)):
        e = np.exp(-1j*2*np.pi*f[i]*np.arange(len(ecg)))
        music_spectrum[i] = 1 / np.linalg.norm(np.dot(e, noise_subspace),2)

    return f, music_spectrum
    
