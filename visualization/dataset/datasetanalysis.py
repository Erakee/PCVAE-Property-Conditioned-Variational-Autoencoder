import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取 CSV 文件
data = pd.read_csv(r'D:\Project\EVAE_paper\data\em_train.csv')

# 提取 'heat_of_formation' 列
heat_of_formation = data['heat_of_formation']

# 设置 Seaborn 样式，去掉网格
sns.set(style="white", palette="muted")

# 创建密度分布图
plt.figure(figsize=(6, 4))
sns.kdeplot(heat_of_formation, fill=True, color="b", alpha=0.6)

# 设置标题和标签
# plt.title('Density Distribution of Enthalpy of Formation', fontsize=18)
plt.xlabel('Heat of Formation', fontsize=16)
plt.ylabel('Density', fontsize=16)

# 调整布局，避免标签被截断
plt.tight_layout()

# 保存图片到当前文件夹并设置分辨率
plt.savefig('heat_of_formation_density.png', dpi=300)

# 显示图表
plt.show()
