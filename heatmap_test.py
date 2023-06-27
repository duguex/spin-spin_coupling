import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

# 创建一个字典，指定红色，绿色和蓝色通道的变化规律
cdict = {
    'red': (
        (0.0, 1.0, 1.0),  # x = 0时，y = 1，白色
        (0.01, 1.0, 1.0),  # x = 0.01时，y = 1，白色
        (0.5, 1.0, 0.0),  # x = 0.5时，y = 1 -> 0，红色 -> 蓝色
        (0.99, 0.0, 0.0),  # x = 0.99时，y = 0，蓝色
        (1.0, 0.0, 0.0),  # x = 1时，y = 0，黑色
    ),
    'green': (
        (0.0, 1.0, 1.0),  # x = 0时，y = 1，白色
        (0.01, 1.0, 1.0),  # x = 0.01时，y = 1，白色
        (0.5, 1.0, 1.0),  # x = 0.5时，y = 1，无变化
        (0.99, 1.0, 1.0),  # x = 0.99时，y = 1，无变化
        (1.0, 0.0, 0.0),  # x = 1时，y = 0，黑色
    ),
    'blue': (
        (0.0, 1.0, 1.0),  # x = 0时，y = 1，白色
        (0.01, 1.0, 1.0),  # x = 0.01时，y = 1，白色
        (0.5, 1.0, 1.0),  # x = 0.5时，y = 1 -> 1，无变化
        (0.99, 1.0, 0.5),  # x = 99时，y = 1 -> .5 蓝色 -> 紫红色
        (1., .5, .5)  # x=100 y=.5 紫红色 -> 黑色
    )
}

# 创建一个颜色映射对象
cmap = LinearSegmentedColormap('my_cmap', cdict)

# 创建一个归一化对象，并设置阈值为[10^-2 ,10]
norm = Normalize(vmin=10 ** -2, vmax=10)

# 创建一些随机数据，在[10^-3 ,10^2]之间
data = np.random.uniform(10 ** -3, 10 ** 2, (10, 10))
# print(data)

# 显示热图和颜色条
plt.imshow(data, cmap=cmap, norm=norm)
plt.colorbar()
plt.show()
