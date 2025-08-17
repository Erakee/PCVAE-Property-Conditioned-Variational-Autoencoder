import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ======================
# 1. 设置学术绘图风格（增加字体大小）
# ======================
plt.style.use('seaborn-v0_8-poster')  # 基础风格
mpl.rcParams.update({
    'font.family': 'Arial',
    'font.size': 20,  # 全局字体大小
    'axes.labelsize': 20,  # 坐标轴标签大小
    'axes.titlesize': 22,  # 标题大小
    'xtick.labelsize': 20,  # x轴刻度标签大小
    'ytick.labelsize': 20,  # y轴刻度标签大小
    'legend.fontsize': 20,  # 图例大小
    'axes.linewidth': 1.5,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.5,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.transparent': False
})

# ======================
# 2. 读取数据
# ======================
seed = 42
df = pd.read_excel(fr"D:\Project\VAE_Related\ConVAE\Results4Paper\training_file\outputs_data\CVAE_DHR\experiments\seed_42\training_log.xlsx")

# 设置 epoch 区间
start_epoch = 0
end_epoch = 110
df = df[(df['epoch'] >= start_epoch) & (df['epoch'] <= end_epoch)]

# ======================
# 3. 创建画布（增大高度以适应更大字体）
# ======================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 14))  # 增大画布尺寸

# ======================
# 4. 绘制第一个子图（损失曲线）
# ======================
colors = plt.cm.tab10(np.linspace(0, 1, 3))

ax1.semilogy(df['epoch'], df['recon_loss'], color=colors[0], linestyle='-', label='Reconstruction Loss')
ax1.semilogy(df['epoch'], df['kld_loss'], color=colors[1], linestyle='--', label='KL Loss')
ax1.semilogy(df['epoch'], df['cond_loss'], color=colors[2], linestyle='-.', label='Property Loss')

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (log scale)')
ax1.grid(True, which='both', linestyle=':', alpha=0.6)
ax1.legend(loc='upper right', frameon=True, edgecolor='black')

# ======================
# 5. 绘制第二个子图（总损失和有效率）
# ======================
ax2a = ax2.twinx()

# 绘制总损失
line1 = ax2.semilogy(df['epoch'], df['total_loss'], color='#2ca02c', linestyle='-', label='Total Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Total Loss (log scale)', color='#2ca02c')
ax2.tick_params(axis='y', colors='#2ca02c')

# 绘制有效率
line2 = ax2a.plot(df['epoch'], df['valid_rate'], color='#d62728', linestyle='--', linewidth=2.5, label='Validity Rate')

# 标记最大有效率
max_valid_rate = df['valid_rate'].max()
max_epoch = df[df['valid_rate'] == max_valid_rate]['epoch'].values[0]

ax2a.plot(max_epoch, max_valid_rate, 'ro', markersize=10)  # 增大标记点尺寸
ax2a.annotate(
    f'Max: {max_valid_rate:.2f}',
    xy=(max_epoch, max_valid_rate),
    xytext=(max_epoch + 3, max_valid_rate - 0.2),  # 调整标注位置
    arrowprops=dict(facecolor='black', shrink=0.02, width=2, headwidth=8),  # 优化箭头样式
    fontsize=20  # 增大标注字体
)

# 绘制垂直虚线
ax2a.axvline(x=max_epoch, linestyle='--', color='gray', alpha=0.7)

# 在x轴标注对应的epoch数
ax2a.annotate(
    f'{max_epoch}',
    xy=(max_epoch, 0),
    xytext=(max_epoch, -0.02),  # 调整标注位置
    ha='center', va='top',
    fontweight='bold',
    fontsize=20  # 增大标注字体
)

ax2a.set_ylabel('Validity Rate', color='#d62728', fontsize=20)  # 增大y轴标签字体
ax2a.tick_params(axis='y', colors='#d62728')
ax2a.set_ylim(0, 1)

# 合并图例
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.98, 0.8),
           frameon=True, edgecolor='black', fontsize=20)  # 增大图例字体

# ======================
# 6. 优化布局并保存
# ======================
plt.tight_layout(pad=3)  # 增加子图间距
plt.savefig(fr'D:\Project\VAE_Related\ConVAE\Results4Paper\training_file\outputs_data\CVAE_DHR\experiments\seed_42\loss.png')
plt.show()