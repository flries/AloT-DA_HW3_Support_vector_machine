import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st

# 設定 Streamlit 頁面
st.title("3D Scatter Plot with Separating Hyperplane")
st.write("Use the slider to adjust the distance threshold for classification.")

# 步驟 1：生成隨機點
np.random.seed(0)
points = np.random.multivariate_normal(mean=[0, 0], cov=[[10, 0], [0, 10]], size=600)

# 使用者可調整的距離閾值
distance_threshold = st.slider("Distance Threshold", min_value=1.0, max_value=10.0, value=4.0, step=0.1)

# 計算每個點到原點的距離，並根據距離進行分類
distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
labels = np.where(distances < distance_threshold, 0, 1)

# 定義高斯函數
def gaussian_function(x1, x2):
    return np.exp(-(x1**2 + x2**2) / 20)

# 計算 x3（第三個維度）
x3 = gaussian_function(points[:, 0], points[:, 1])

# 合併成三維特徵矩陣 X
X = np.column_stack((points[:, 0], points[:, 1], x3))

# 使用 LinearSVC 進行分類
model = LinearSVC()
model.fit(X, labels)

# 獲取模型係數和截距
coef = model.coef_
intercept = model.intercept_

# 繪製 3D 散點圖和超平面
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title("3D Scatter Plot with Y Color and Separating Hyperplane")

# 設置 x 和 y 軸的間距
ax.set_xticks(np.arange(-10, 11, 2.5))
ax.set_yticks(np.arange(-10, 11, 2.5))

# 畫出不同類別的點
ax.scatter(X[labels == 0, 0], X[labels == 0, 1], X[labels == 0, 2], color='blue', label='Y=0')
ax.scatter(X[labels == 1, 0], X[labels == 1, 1], X[labels == 1, 2], color='red', label='Y=1')

# 計算超平面
xx, yy = np.meshgrid(np.linspace(-10, 10, 10), np.linspace(-10, 10, 10))
zz = (-coef[0][0] * xx - coef[0][1] * yy - intercept) / coef[0][2]

# 繪製超平面
ax.plot_surface(xx, yy, zz, color='gray', alpha=0.5, rstride=100, cstride=100)

# 設定標籤和圖例
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
ax.legend()

# 在 Streamlit 上顯示圖表
st.pyplot(fig)
