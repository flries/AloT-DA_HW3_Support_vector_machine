## Prompt:
Write a Python program using Streamlit, numpy, matplotlib, and scikit-learn libraries, following below step:  
Step 1:　　
1. Generate 600 random points with coordinates following a Gaussian distribution centered at (0, 0) and with a variance of 10.
2. Calculate the distance of each point from the origin.
3. Based on the distance, categorize these points into two classes (0 and 1): if the distance is less than 4, classify as 0; otherwise, classify as 1.

Step 2:　　
1. Define a Gaussian function gaussian_function(x1, x2), which takes two inputs, x1 and x2, and returns a Gaussian-distributed value. This value is an exponential function with x1 and x2 as coordinates, representing a third-dimensional feature.

Step 3:　　
1. Combine x1, x2, and the previously defined x3 (computed from the Gaussian function) into a three-dimensional feature matrix X.
2. Use LinearSVC (a linear support vector classifier) to train a model aimed at finding a hyperplane that can separate the two classes.
3. Obtain the model's coefficients (coef) and intercept (intercept) after training.
## Response:
以下是使用 `numpy` 和 `matplotlib` 實現該任務的程式碼：

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from mpl_toolkits.mplot3d import Axes3D

# 步驟 1
# 產生 600 個隨機點
np.random.seed(0)
points = np.random.multivariate_normal(mean=[0, 0], cov=[[10, 0], [0, 10]], size=600)

# 計算每個點到原點的距離
distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)

# 分類
labels = np.where(distances < 4, 0, 1)

# 步驟 2
# 定義高斯函數
def gaussian_function(x1, x2):
    return np.exp(-(x1**2 + x2**2) / 20)

# 計算 x3（第三個維度）
x3 = gaussian_function(points[:, 0], points[:, 1])

# 步驟 3
# 合併成三維的特徵矩陣 X
X = np.column_stack((points[:, 0], points[:, 1], x3))

# 使用 LinearSVC 進行分類
model = LinearSVC()
model.fit(X, labels)

# 獲取模型係數和截距
coef = model.coef_
intercept = model.intercept_

# 可視化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 畫出不同類別的點
ax.scatter(X[labels == 0, 0], X[labels == 0, 1], X[labels == 0, 2], color='blue', label='Class 0')
ax.scatter(X[labels == 1, 0], X[labels == 1, 1], X[labels == 1, 2], color='red', label='Class 1')

# 計算超平面
xx, yy = np.meshgrid(np.linspace(-10, 10, 10), np.linspace(-10, 10, 10))
zz = (-coef[0][0] * xx - coef[0][1] * yy - intercept) / coef[0][2]

# 繪製超平面
ax.plot_surface(xx, yy, zz, color='green', alpha=0.5, rstride=100, cstride=100)

# 圖例和標籤
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
ax.legend()
plt.show()
```

### 說明

1. **步驟 1**：產生 600 個隨機點，並將這些點分類為 0 或 1。
2. **步驟 2**：定義 `gaussian_function`，用來計算每個點的第三維度（`x3`）。
3. **步驟 3**：訓練 `LinearSVC` 模型，並繪製分類結果和決策平面。

## Prompt:
Perform the following visualization operations:  
Step 4:  
1. Create a 3D scatter plot to display the two classes of points (Y=0 and Y=1) with different colors in three-dimensional space.
2. On the same plot, draw the hyperplane that separates the two classes, represented by a gray plane.

Step 5:  
Based on this code, modify the "distance less than 4, classify as 0; otherwise, classify as 1" rule so that the distance becomes an adjustable variable controlled by a slider. Develop a Streamlit application where users can instantly visualize:
1. A 3D scatter plot showing the two classes of points (Y=0 and Y=1) in different colors.
2. A gray plane representing the hyperplane that separates the two classes, displayed on the same plot.

## Response:
為了將距離閾值設定為可調整的變數並在 Streamlit 中實現，即時視覺化更新效果，可以使用 Streamlit 的滑桿（slider）元件。這樣，使用者可以通過滑桿調整距離閾值，而 3D 散點圖和超平面會根據新的分類結果自動更新。

以下是完整的 Streamlit 程式碼：

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st

# 設定 Streamlit 頁面
st.title("3D Scatter Plot with Adjustable Distance Threshold")
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

# 畫出不同類別的點
ax.scatter(X[labels == 0, 0], X[labels == 0, 1], X[labels == 0, 2], color='blue', label='Class 0')
ax.scatter(X[labels == 1, 0], X[labels == 1, 1], X[labels == 1, 2], color='red', label='Class 1')

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
```

### 程式說明

1. **距離閾值滑桿**：使用 `st.slider` 來創建可調整的距離閾值（`distance_threshold`），初始值為 4.0，範圍在 1.0 到 10.0 之間。
2. **分類**：每次調整滑桿時，`distance_threshold` 會更新，隨之更新分類結果（`labels`）。
3. **重新訓練模型**：根據新的 `labels` 訓練 `LinearSVC` 模型，並重新計算分隔兩類的超平面。
4. **即時更新圖形**：使用 `st.pyplot(fig)` 將 3D 散點圖和超平面顯示在 Streamlit 畫面上，並在滑桿調整時自動更新視覺化。

### 執行程式

將此程式碼存為 `app.py`，並在終端執行以下指令啟動 Streamlit 應用：

```bash
bash複製程式碼streamlit run app.py
```

調整滑桿後，3D 散點圖和超平面將即時更新，視覺化展示不同距離閾值下的分類結果。


## Prompt:
modify the code with the following requirements:　　

1. Set the main page title to "3D Scatter Plot with Separating Hyperplane."
2. Set the chart title to "3D Scatter Plot with Y Color and Separating Hyperplane."
3. In the chart, set the spacing for the x1 and x2 axes to show a grid line every 2.5 units.
4. Set the label for the scatter plot of class-1 to "Y=0."
5. Set the label for the scatter plot of class-2 to "Y=1."

## Response:
好的，以下是依照要求修改的程式碼：

```python
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
```

### 修改說明

1. **頁面標題**：使用 `st.title("3D Scatter Plot with Separating Hyperplane")` 設置頁面大標題。
2. **圖表標題**：使用 `ax.set_title("3D Scatter Plot with Y Color and Separating Hyperplane")` 設置圖表的標題。
3. **軸間距**：使用 `ax.set_xticks(np.arange(-10, 11, 2.5))` 和 `ax.set_yticks(np.arange(-10, 11, 2.5))` 設置 `X1` 和 `X2` 軸的間距為每 2.5 顯示一條格線。
4. **分類標籤**：將 `Y=0` 設為藍色點的標籤，`Y=1` 設為紅色點的標籤。

## Prompt:
Add two more sliders: one to adjust the semi-major axis and another to adjust the semi-minor axis. Please provide the modified code.

## Response:
好的，以下是添加了調整橢圓的「半長軸」(semi-major axis)和「半短軸」(semi-minor axis)的滑桿的程式碼。這些滑桿會動態控制橢圓形的分佈範圍，並即時更新 3D 散點圖。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st

# 設定 Streamlit 頁面
st.title("3D Scatter Plot with Separating Hyperplane")
st.write("Use the sliders to adjust the distance threshold, semi-major axis, and semi-minor axis.")

# 使用者可調整的橢圓分佈參數
semi_major_axis = st.slider("Semi-Major Axis", min_value=5.0, max_value=20.0, value=10.0, step=0.5)
semi_minor_axis = st.slider("Semi-Minor Axis", min_value=5.0, max_value=20.0, value=10.0, step=0.5)

# 使用者可調整的距離閾值
distance_threshold = st.slider("Distance Threshold", min_value=1.0, max_value=10.0, value=4.0, step=0.1)

# 步驟 1：生成隨機點
np.random.seed(0)
cov_matrix = [[semi_major_axis, 0], [0, semi_minor_axis]]
points = np.random.multivariate_normal(mean=[0, 0], cov=cov_matrix, size=600)

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
ax.set_title("3D Scatter Plot with Y Color and Separating Hyperplane", fontsize=10)

# 設置 x 和 y 軸的間距
ax.set_xticks(np.arange(-20, 21, 2.5))
ax.set_yticks(np.arange(-20, 21, 2.5))

# 畫出不同類別的點
ax.scatter(X[labels == 0, 0], X[labels == 0, 1], X[labels == 0, 2], color='blue', label='Y=0', s=10)
ax.scatter(X[labels == 1, 0], X[labels == 1, 1], X[labels == 1, 2], color='red', label='Y=1', s=10)

# 計算超平面
xx, yy = np.meshgrid(np.linspace(-20, 20, 10), np.linspace(-20, 20, 10))
zz = (-coef[0][0] * xx - coef[0][1] * yy - intercept) / coef[0][2]

# 繪製超平面
ax.plot_surface(xx, yy, zz, color='gray', alpha=0.5, rstride=100, cstride=100)

# 設定標籤和圖例
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
ax.legend(fontsize=8, markerscale=0.5)  # 調整圖例字體和圖示大小

# 在 Streamlit 上顯示圖表
st.pyplot(fig)
```

### 修改說明

1. **新增滑桿**：`semi_major_axis` 和 `semi_minor_axis` 滑桿，用於控制橢圓分佈的「半長軸」和「半短軸」，範圍設定為 5.0 到 20.0，預設值為 10.0。
2. **動態調整橢圓分佈**：根據滑桿的輸入，設置隨機點的協方差矩陣 (`cov_matrix`)。
3. **可視化設定**：
   * 設置標題字體大小和圖例的標籤和圖示縮小為適合的大小（字體為 8，圖示縮放至 0.5）。
   * 使用 `s=10` 縮小散點的大小，以符合整體視覺效果。

