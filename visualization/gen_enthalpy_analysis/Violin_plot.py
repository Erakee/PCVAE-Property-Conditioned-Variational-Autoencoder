import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置期刊风格（更新为正确参数）
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'figure.figsize': (6, 4),
    'axes.spines.right': False,
    'axes.spines.top': False
})

# 设置seaborn样式（修正弃用警告）
sns.set_style("whitegrid")

# 数据准备
df = pd.read_excel(r"D:\Project\VAE_Related\ConVAE\visualization\gen_enthalpy_analysis\set_enthalpy.xlsx",
                   sheet_name="Sheet1", index_col=0)
melt_df = df.reset_index().melt(id_vars='ENTHALPY', var_name='Molecule', value_name='Value')

# 创建箱型图（颜色参数移至boxplot内部）
plt.figure(figsize=(6, 4))
ax = sns.boxplot(
    x='ENTHALPY',
    y='Value',
    data=melt_df,
    color='#4C72B0',  # 颜色参数在此设置
    width=0.6,
    linewidth=0.7,
    flierprops=dict(marker='o', markersize=3,
                    markerfacecolor='none',
                    markeredgecolor='#2F4F4F'),
    boxprops=dict(facecolor='#4C72B0')  # 新增箱体填充色设置
)
# 格式优化
ax.set_xticklabels([f"{x:.0f}" for x in sorted(melt_df['ENTHALPY'].unique())], rotation=45)
plt.xlabel("Set Enthalpy (kJ/mol)", labelpad=8)
plt.ylabel("Actual Enthalpy (kJ/mol)", labelpad=8)
plt.axhline(0, color='grey', linestyle='--', linewidth=0.5, alpha=0.7)

# 添加统计标注
medians = melt_df.groupby('ENTHALPY')['Value'].median()
x_offset = 0.35  # 定义 x 轴偏移量，可按需调整
for xtick in ax.get_xticks():
    # 调整标注的 x 轴位置
    ax.text(xtick + x_offset, medians.iloc[xtick] + 19, f'{medians.iloc[xtick]:.1f}',
            horizontalalignment='center', size=6, color='black')

plt.tight_layout()
plt.savefig('boxplot.png', bbox_inches='tight')
plt.show()

# 设置小提琴图参数（移除无效rc设置）
plt.rcParams.update({
    'font.family': 'Arial',
    'figure.dpi': 300,
    'axes.spines.right': False,
    'axes.spines.top': False
})

sns.set_style("whitegrid")

# 创建小提琴图
plt.figure(figsize=(6, 4))
ax = sns.violinplot(
    x='ENTHALPY',
    y='Value',
    data=melt_df,
    palette='viridis',
    inner='quartile',
    bw=0.2,
    cut=0,
    linewidth=0.5,
    saturation=0.8
)

# 格式优化
ax.set_xticklabels([f"{x:.0f}" for x in sorted(melt_df['ENTHALPY'].unique())], rotation=45)
plt.xlabel("Set Enthalpy (kJ/mol)", labelpad=8)
plt.ylabel("Actual Enthalpy (kJ/mol)", labelpad=8)
plt.axhline(0, color='grey', linestyle='--', linewidth=0.5, alpha=0.7)

# 添加密度曲线标注
for violin in ax.collections[::2]:
    violin.set_alpha(0.8)

plt.tight_layout()
plt.savefig('violinplot.png', bbox_inches='tight')
plt.show()