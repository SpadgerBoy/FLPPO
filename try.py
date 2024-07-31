
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

def relu(x):
    return np.where(x < 0, 0, x)

x = np.arange(-4, 4, 0.01)
y1 = 1/(1+np.exp(-x))
y2 = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

plt.figure()
ax = plt.gca()  # 得到图像的Axes对象
ax.spines['right'].set_color('none')  # 将图像右边的轴设为透明
ax.spines['top'].set_color('none')  # 将图像上面的轴设为透明
ax.xaxis.set_ticks_position('bottom')  # 将x轴刻度设在下面的坐标轴上
ax.yaxis.set_ticks_position('left')  # 将y轴刻度设在左边的坐标轴上
ax.spines['bottom'].set_position(('data', 0))  # 将两个坐标轴的位置设在数据点原点
ax.spines['left'].set_position(('data', 0))
plt.plot(x, y1, label='Sigmoid')
plt.plot(x, y2, label='tanh')
plt.plot(x, relu(x), label='ReLU')
plt.legend()
plt.savefig(f"a.svg")
plt.show()