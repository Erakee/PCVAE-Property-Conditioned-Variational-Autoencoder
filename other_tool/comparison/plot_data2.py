import matplotlib.pyplot as plt
import numpy as np

# --------- Group 2 数据 ----------
group2_pred = np.array([-96.6014, -89.5253, -102.6211, -91.5400, -98.1022, -96.2815, -88.9087, -103.9004])
group2_dft = np.array([-67.7400, -78.4600, -64.9100, -63.5200, -72.9700, -70.7600, -71.0600, -80.6000])

# --------- 图2 ----------
plt.figure(figsize=(6, 6))

# 散点图
plt.scatter(group2_pred, group2_dft, color='#9AC9DB', s=60, label='EoF-Targeted')

center_x = np.mean(group2_pred)
center_y = np.mean(group2_dft)
# Targeted Center 引线：从 (center_x, center_y) 向下、向左
plt.plot([center_x, center_x], [center_y, -110], linestyle='--', color='gray', linewidth=1.5)  # 向下引线
plt.plot([center_x, -110], [center_y, center_y], linestyle='--', color='gray', linewidth=1.5)  # 向左引线


# 添加中心点标记（空心圆）
plt.scatter(center_x, center_y, s=100, facecolors='none', edgecolors='#14517C', linewidths=2,
            label='EoF-Targeted Center', zorder=5)

# 理想圆圈（以 -100, -100 为圆心，半径设为 10 kcal/mol，可根据需要调整）
# ideal_center = (-100, -100)
# radius = 10
# circle = plt.Circle(ideal_center, radius, color='gray', linestyle='--', linewidth=1.5, fill=False, label='Ideal Zone')
# plt.gca().add_patch(circle)

# 标出圆心位置
plt.scatter(-100, -100, color='C82423', s=100, label='Ideal Center', zorder=5)
# Ideal Center 引线（从 -100, -100 出发）
plt.plot([-100, -100], [-100, -110], linestyle='--', color='gray', linewidth=1.5)  # 垂直线向下
plt.plot([-100, -110], [-100, -100], linestyle='--', color='gray', linewidth=1.5)  # 水平线向左


# 坐标轴范围设置
plt.xlim(-110, -50)
plt.ylim(-110, -50)

# 图设置
plt.xlabel('Predicted EoF (kcal/mol)', fontsize=20)
plt.ylabel('DFT-computed EoF (kcal/mol)', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.title('Group 2: Predicted vs DFT-computed EoF', fontsize=14)
plt.legend(fontsize=15)
plt.grid(False)  # 无网格
# plt.axis('equal')  # 保持比例，圆不会变形
plt.tight_layout()
plt.savefig('group2_eof_plot.png', dpi=450, bbox_inches='tight')
plt.show()
