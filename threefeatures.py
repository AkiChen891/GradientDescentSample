import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

gd = pd.read_csv('dataset_3.csv')


cols=gd.shape[1]
Features=gd.iloc[ : ,    : cols-1]
Target=gd.iloc[ : , cols-1 : cols]

Features=np.matrix(Features.values)
Target=np.matrix(Target.values)
theta=np.matrix(np.array([0,0,0]))

def computecost(Features,Target,theta):
    inner=np.power(((Features * theta.T) - Target),2)
    return np.sum(inner)/(2 * Features.shape[0])

def batch_grandientdescent(Features,Target,theta,alpha,iterations):
    temp=np.matrix(np.zeros(theta.shape))
    parameters=int(theta.ravel().shape[1])
    cost=np.zeros(iterations)
    for i in range(iterations):
        error=(Features * theta.T) - Target
        for j in range(parameters):
            term=np.multiply(error,Features[ : , j])
            temp[0,j]=theta[0,j]-(alpha/len(Features) * np.sum(term))
        theta=temp
        cost[i]=computecost(Features,Target,theta)
    return theta,cost

alpha=0.02
iterations=1500

g,cost=batch_grandientdescent(Features,Target,theta,alpha,iterations)

# Print final weights
print("Final theta values:\n", g)
print("Final cost:", computecost(Features, Target, g))

# Use the fitted hypothesis function to predict the new Target value by inputting a new eigenvalue matrix.
New_Feature_one = 0.2
New_Feature_two = 0.4
New_Feature_three = 0.6

New_Feature=np.array([New_Feature_one,New_Feature_two,New_Feature_three])

New_Target=np.dot(g,New_Feature)

print("New Target is:",New_Target) 

# 绘制三维散点图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 获取三个特征（无需偏置量）
x = np.array(Features[:, 0])  # 第一特征
y = np.array(Features[:, 1])  # 第二特征
z = np.array(Features[:, 2])  # 第三特征

# 目标值
target = np.array(Target)

# 绘制目标值的散点图
scat = ax.scatter(x, y, z, c=target, cmap='viridis', label='Actual Target Values', alpha=0.6)

# 添加颜色条
cbar = fig.colorbar(scat, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('Target Values')

# 在三维图中绘制拟合曲面
# 计算预测值的曲面
x_surf, y_surf = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))
z_surf = g[0, 0] * x_surf + g[0, 1] * y_surf  # 拟合平面公式

# 绘制拟合的平面
ax.plot_surface(x_surf, y_surf, z_surf, cmap='plasma', alpha=0.5, rstride=100, cstride=100)

# 设置轴标签
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.set_title('3D Scatter Plot with Fitted Surface')

plt.show()





