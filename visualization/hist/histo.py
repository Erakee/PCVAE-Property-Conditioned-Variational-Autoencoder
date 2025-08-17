# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import os
#
# # ---------------------------
# # 参数区
# # ---------------------------
# model1_path = r'D:\Project\EVAE_paper\generate_smi\collection\CVAE_DHR_uniq.xlsx'
# model2_path = r'D:\Project\EVAE_paper\generate_smi\collection\CVAE_HC_uniq.xlsx'
# random_data_path = r'D:\Project\EVAE_paper\generate_smi\collection\vae_enthalpy.txt'  # 随机数据路径
#
# save_path = 'enthalpy_distribution.png'
#
# selected_enthalpies = [-300, -200, -50, 50, 100, 200]  # 设定target值
#
# # 虚线偏移
# offsets = [-15, -10, 10, 15]
#
# # 曲线透明度
# alpha_fill = 0.3
#
# # 曲线颜色
# color_model1 = '#1f77b4'  # 蓝色
# color_model2 = '#ff7f0e'  # 橙色
# color_model3 = '#2ca02c'  # 绿色
#
# # 虚线颜色
# vline_color = '#888888'  # 灰色
#
# # KDE平滑系数
# kde_bw_adjust = 1
#
# # 宫格布局
# cols = 3
# rows = int(np.ceil(len(selected_enthalpies) / cols))
#
# # ---------------------------
# # 加载数据
# # ---------------------------
# model1 = pd.read_excel(model1_path)
# model2 = pd.read_excel(model2_path)
#
# # 加载随机生成的焓值数据
# random_data = np.loadtxt(random_data_path)
#
# model1_cols_float = model1.columns.astype(float)
# model2_cols_float = model2.columns.astype(float)
#
# # ---------------------------
# # 绘图
# # ---------------------------
# sns.set_style("white")  # 使用纯白色背景
# sns.set_context("paper")  # 适合论文的整体大小
# fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
# axes = axes.flatten()
#
# for idx, center in enumerate(selected_enthalpies):
#     ax = axes[idx]
#
#     # 找到最接近target的列
#     model1_col = model1.columns[np.abs(model1_cols_float - center).argmin()]
#     model2_col = model2.columns[np.abs(model2_cols_float - center).argmin()]
#
#     data1 = model1[model1_col].dropna()
#     data2 = model2[model2_col].dropna()
#
#     # 画KDE曲线（曲线 + 填充）
#     sns.kdeplot(data1, ax=ax, color=color_model1, fill=True, alpha=alpha_fill, bw_adjust=kde_bw_adjust, label='CVAE_DHR')
#     sns.kdeplot(data2, ax=ax, color=color_model2, fill=True, alpha=alpha_fill, bw_adjust=kde_bw_adjust, label='CVAE_HC')
#
#     # 绘制随机生成的数据的KDE曲线
#     sns.kdeplot(random_data, ax=ax, color=color_model3, fill=True, alpha=alpha_fill, bw_adjust=kde_bw_adjust, label='VAE')
#
#     # target ±10, ±15 位置画虚线
#     for offset in offsets:
#         ax.axvline(center + offset, linestyle='--', color=vline_color, linewidth=1)
#
#     # x、y标签
#     ax.set_xlabel('Generated Enthalpy', fontsize=20)
#     ax.set_ylabel('Density', fontsize=20)
#
#     # 标题
#     ax.set_title(f"Target: {center}", fontsize=22, weight='bold')
#     # 设置坐标轴刻度字体大小
#     ax.tick_params(axis='both', which='major', labelsize=20)
#
#     # 图例只在第一张显示
#     if idx == 0:
#         ax.legend(fontsize=20, loc='upper right')
#     else:
#         legend = ax.get_legend()
#         if legend is not None:
#             legend.remove()
#
# # 去除多余子图
# for j in range(idx + 1, len(axes)):
#     fig.delaxes(axes[j])
#
# plt.tight_layout()
# plt.savefig(save_path, dpi=300)
# print(f"图保存到：{os.path.abspath(save_path)}")
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
save_path = 'enthalpy_distribution.png'
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
axes = axes.flatten()

for idx, center in enumerate(selected_enthalpies):
    ax = axes[idx]
    model1_col = model1.columns[np.abs(model1_cols_float - center).argmin()]
    model2_col = model2.columns[np.abs(model2_cols_float - center).argmin()]
    data1 = model1[model1_col].dropna()
    data2 = model2[model2_col].dropna()

    # 绘制带标签的KDE曲线
    sns.kdeplot(data1, ax=ax, color=color_model1, fill=True, alpha=alpha_fill,
                bw_adjust=kde_bw_adjust, label='ECVAE')
    sns.kdeplot(data2, ax=ax, color=color_model2, fill=True, alpha=alpha_fill,
                bw_adjust=kde_bw_adjust, label='CVAE')
    sns.kdeplot(random_data, ax=ax, color=color_model3, fill=True, alpha=alpha_fill,
                bw_adjust=kde_bw_adjust, label='VAE')

    # 绘制虚线
    for offset in offsets:
        ax.axvline(center + offset, linestyle='--', color=vline_color, linewidth=1)

    # 关闭子图轴标签
    ax.set_xlabel('')
    ax.set_ylabel('')

    # 设置标题和刻度
    ax.set_title(f"Target: {center}", fontsize=22, weight='bold')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylim(0, global_max_y)
    # ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y * 1000)}'))
    # ax.set_xlim(-500, 500)
    # ax.xaxis.set_ticks(np.arange(-500, 501, 100))

    # 仅在第一张图显示图例
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