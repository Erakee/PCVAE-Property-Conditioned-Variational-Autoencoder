import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# --------- 数据准备 ----------
# Group 1: 中高焓区间
group1_pred = np.array([-2.6354, 69.1529, 27.2581, 59.0857, -22.0725, 10.2144, 17.4864, 34.5382, 54.2802])
group1_dft = np.array([13.3300, 75.6200, 45.8300, 68.3400, 3.0200, 33.1200, 35.9400, 56.9000, 78.4100])

# --------- 图1：Group 1 ----------
plt.figure(figsize=(6, 6))

# 拟合直线
model = LinearRegression()
model.fit(group1_pred.reshape(-1, 1), group1_dft)
x_fit = np.linspace(min(group1_pred), max(group1_pred), 100)
y_fit = model.predict(x_fit.reshape(-1, 1))

# 绘图
plt.plot([min(group1_pred.min(), group1_dft.min()), max(group1_pred.max(), group1_dft.max())],
         [min(group1_pred.min(), group1_dft.min()), max(group1_pred.max(), group1_dft.max())],
         'k--', linewidth=1, label='Ideal')

plt.scatter(group1_pred, group1_dft, color='#9AC9DB', s=60, label='NTO-Targeted')
plt.plot(x_fit, y_fit, color='#14517C', linewidth=2.0, label='Linear Fit')

# 坐标轴范围设置
plt.xlim(-30, 80)
plt.ylim(-20, 100)

# 图设置
plt.xlabel('Predicted EoF (kcal/mol)', fontsize=20)
plt.ylabel('DFT-computed EoF (kcal/mol)', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.title('Group 1: Predicted vs DFT-computed EoF', fontsize=14)
plt.legend(fontsize=16)
plt.grid(False)  # 去掉网格线
plt.tight_layout()
plt.savefig('group1_eof_plot.png', dpi=450, bbox_inches='tight')
plt.show()
