import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ---------------------------
# 参数区
# ---------------------------
model1_path = r'D:\Project\EVAE_paper\generate_smi\collection\CVAE_DHR_uniq.xlsx'
model2_path = r'D:\Project\EVAE_paper\generate_smi\collection\CVAE_HC_uniq.xlsx'
random_data_path = r'D:\Project\EVAE_paper\generate_smi\collection\vae_enthalpy.txt'
save_path = 'enthalpy_distribution_7_4.png'
selected_enthalpies = [-300, -200, -50, 50, 100, 200]
offsets = [-15, -10, 10, 15]
alpha_fill = 0.3
color_model1 = '#1f77b4'  # 蓝色
color_model2 = '#ff7f0e'  # 橙色
color_model3 = '#2ca02c'  # 绿色
vline_color = '#888888'  # 灰色
kde_bw_adjust = 1
cols = 3
rows = int(np.ceil(len(selected_enthalpies) / cols))

# ---------------------------
# 加载数据
# ---------------------------
model1 = pd.read_excel(model1_path)
model2 = pd.read_excel(model2_path)
random_data = np.loadtxt(random_data_path)
model1_cols_float = model1.columns.astype(float)
model2_cols_float = model2.columns.astype(float)

# ---------------------------
# 计算全局y轴最大值
# ---------------------------
global_max_y = 0
for center in selected_enthalpies:
    model1_col = model1.columns[np.abs(model1_cols_float - center).argmin()]
    model2_col = model2.columns[np.abs(model2_cols_float - center).argmin()]
    data1 = model1[model1_col].dropna()
    data2 = model2[model2_col].dropna()

    kde1 = sns.kdeplot(data1, bw_adjust=kde_bw_adjust, fill=False, label='ECVAE')
    kde2 = sns.kdeplot(data2, bw_adjust=kde_bw_adjust, fill=False, label='CVAE')
    kde3 = sns.kdeplot(random_data, bw_adjust=kde_bw_adjust, fill=False, label='VAE')

    current_max = max(
        np.max(kde1.lines[0].get_ydata()),
        np.max(kde2.lines[0].get_ydata()),
        np.max(kde3.lines[0].get_ydata())
    )
    if current_max > global_max_y:
        global_max_y = current_max
    kde1.clear()
    kde2.clear()
    kde3.clear()

# ---------------------------
# 绘图
# ---------------------------
sns.set_style("white")
sns.set_context("paper", font_scale=1.2)
fig, axes = plt.subplots(rows, cols, figsize=(5.3 * cols, 4 * rows))
sns.set_style("whitegrid")  # 替代 white，显示网格和刻度线

axes = axes.flatten()

for idx, center in enumerate(selected_enthalpies):
    ax = axes[idx]
    model1_col = model1.columns[np.abs(model1_cols_float - center).argmin()]
    model2_col = model2.columns[np.abs(model2_cols_float - center).argmin()]
    data1 = model1[model1_col].dropna()
    data2 = model2[model2_col].dropna()

    # 绘制KDE曲线
    sns.kdeplot(data1, ax=ax, color=color_model1, fill=True, alpha=alpha_fill,
                bw_adjust=kde_bw_adjust, label='PCVAE')
    sns.kdeplot(data2, ax=ax, color=color_model2, fill=True, alpha=alpha_fill,
                bw_adjust=kde_bw_adjust, label='CVAE')
    sns.kdeplot(random_data, ax=ax, color=color_model3, fill=True, alpha=alpha_fill,
                bw_adjust=kde_bw_adjust, label='VAE')

    # 仅绘制±15的虚线
    ax.axvline(center - 15, linestyle='--', color=vline_color, linewidth=2)
    ax.axvline(center + 15, linestyle='--', color=vline_color, linewidth=2)

    # 添加目标值的实线
    # ax.axvline(center, linestyle='-', color='black', linewidth=1.2)
    # ax.text(center, global_max_y * 0.95, f"{center}", ha='center', va='top',
    #         fontsize=16, color='black', weight='bold')

    # 关闭子图轴标签
    ax.set_xlabel('')
    ax.set_ylabel('')

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    # 设置刻度线显示
    ax.tick_params(
        axis='both',
        which='major',
        direction='out',
        length=6,
        width=2,
        color='black',
        bottom=True, top=False, left=True, right=False,
        labelsize=22   # 标签字体大小（加大）
    )
    # 设置刻度间隔（可根据需要自定义）
    ax.xaxis.set_major_locator(plt.MaxNLocator(7))  # x轴最多 5 个主刻度
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))  # y轴最多 4 个主刻度

    # 设置标题和刻度
    ax.set_title(f"Target: {center}", fontsize=22, weight='bold')
    # ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylim(0, global_max_y)
    ax.set_xlim(-450, 300)

    if idx == 0:
        ax.legend(
            fontsize=20,
            loc='upper right',
            bbox_to_anchor=(1.02, 1),
            frameon=False
        )


# ---------------------------
# 添加全局坐标轴标签（带数学上标）
# ---------------------------
# y轴名称使用LaTeX上标表示
fig.text(
    0.005, 0.5,  # 调整x坐标避免重叠
    r'Density',  # $\times 10^3$',
    va='center',
    rotation='vertical',
    fontsize=22,
    fontweight='bold'
)
11
fig.text(
    0.5, 0.02,
    'Generated Enthalpy (kcal/mol)',
    ha='center',
    fontsize=22,
    weight='bold'
)

# 移除多余子图
for j in range(len(selected_enthalpies), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(pad=3, w_pad=2)  # 增加横向间距
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"图保存到：{os.path.abspath(save_path)}")