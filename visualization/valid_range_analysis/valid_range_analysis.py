import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.interpolate import make_interp_spline

# ========== 专业科研风格设置 ==========
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'figure.dpi': 300,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.linewidth': 1.2,
    'lines.linewidth': 2.5,
})

# ========== 随机生成数据分析 ==========
def analyze_random_data(filepath):
    # 读取数据
    with open(filepath, 'r') as f:
        data = [float(line.strip()) for line in f if line.strip()]

    # 分箱统计（每50划分区间）
    bins = np.arange(-400, 301, 50)
    counts, bin_edges = np.histogram(data, bins=bins)
    percentages = counts / len(data)

    # 生成平滑折线图数据
    x_center = (bin_edges[:-1] + bin_edges[1:]) / 2
    x_smooth = np.linspace(x_center.min(), x_center.max(), 300)
    spl = make_interp_spline(x_center, percentages, k=3)
    y_smooth = spl(x_smooth)

    return x_smooth, y_smooth


# ========== 条件生成数据分析 ==========
def analyze_conditioned_data(filepath):
    # 读取数据并修复列名类型
    df = pd.read_excel(filepath, header=0)
    df.columns = df.columns.astype(float)
    set_values = df.columns

    # 修正上下界计算逻辑
    valid_ratios = []
    for set_val in set_values:
        if set_val < 0:
            lower = set_val * 1.15  # 负值需要扩大绝对值范围
            upper = set_val * 0.85
        else:
            lower = set_val * 0.85
            upper = set_val * 1.15

        col_data = df[set_val].dropna()
        valid_count = ((col_data >= lower) & (col_data <= upper)).sum()
        valid_ratios.append(valid_count / len(col_data))

    return set_values, valid_ratios
# ========== 组合绘图优化版本 ==========
def plot_combined_enhanced(random_file, conditioned_file):
    # 生成数据
    rand_x, rand_y = analyze_random_data(random_file)
    cond_x, cond_y = analyze_conditioned_data(conditioned_file)

    # 创建画布
    fig, ax1 = plt.subplots(figsize=(12, 6.5))

    # ===== 左侧坐标轴设置 =====
    ax1.plot(rand_x, rand_y, color='#1f77b4', label='Random Distribution')
    ax1.fill_between(rand_x, rand_y, color='#1f77b4', alpha=0.15)
    ax1.set_xlabel('Enthalpy (kcal/mol)', fontweight='bold')
    ax1.set_ylabel('Distribution Density', color='#1f77b4', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#1f77b4', width=1.5)
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))

    # 设置轴范围（重点调整）
    ax1.set_ylim(0, rand_y.max())  # 上方留出25%空白
    ax1.set_xlim(-350, 350)
    ax1.set_xticks(np.arange(-400, 351, 50))
    ax1.grid(True, linestyle='--', alpha=0.4)

    # ===== 右侧坐标轴设置 =====
    ax2 = ax1.twinx()

    # 颜色映射设置（蓝→红渐变）
    norm = plt.Normalize(min(cond_y), max(cond_y))
    colors = plt.cm.coolwarm(norm(cond_y))

    bars = ax2.bar(cond_x, cond_y, width=20,
                   color=colors, edgecolor='k', linewidth=0.8,
                   alpha=0.9, label='Success Rate')

    # 标注成功率
    for bar, ratio in zip(bars, cond_y):
        ax2.text(bar.get_x() + bar.get_width() / 2., ratio + 0.02,
                 f'{ratio:.0%}', ha='center', va='bottom',
                 fontsize=10, color='#2F4F4F')

    # 坐标轴样式增强
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color('#B22222')
    ax2.spines['right'].set_linewidth(1.5)
    ax2.set_ylabel('Success Rate (±15%)',
                   color='#B22222', fontweight='bold')
    ax2.tick_params(axis='y', colors='#B22222', width=1.5)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.set_ylim(0, 1.3)  # 上方留出10%空间

    # ===== 添加科研箭头 =====
    # 左侧箭头
    ax1.annotate('', xy=(0, 1), xycoords='axes fraction',
                 xytext=(0, 1.1), textcoords='axes fraction',
                 arrowprops=dict(arrowstyle="->", color='#1f77b4', lw=2))
    # 右侧箭头
    ax2.annotate('', xy=(1, 1), xycoords='axes fraction',
                 xytext=(1, 1.1), textcoords='axes fraction',
                 arrowprops=dict(arrowstyle="->", color='#B22222', lw=2))

    # ===== 专业图例设置 =====
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#1f77b4', lw=4, label='Random Distribution'),
        Line2D([0], [0], marker='s', color='w',
               markerfacecolor='#B22222', markersize=12,
               label='Success Rate (±15%)')
    ]

    ax1.legend(handles=legend_elements, loc='upper left',
               bbox_to_anchor=(0.02, 0.98), frameon=False,
               fontsize=12)

    # ===== 保存输出 =====
    plt.title('Molecular Generation Performance Analysis\n',
              fontsize=18, pad=20)
    plt.tight_layout()
    plt.savefig('enhanced_combined_analysis.pdf',
                bbox_inches='tight', transparent=True)
    plt.show()


# ========== 主程序 ==========
if __name__ == "__main__":
    random_file = r"D:\Project\VAE_Related\ConVAE\visualization\valid_range_analysis\rand_generated_enthalpy.txt"
    conditioned_file = r"D:\Project\VAE_Related\ConVAE\visualization\valid_range_analysis\set_enthalpy_analysis.xlsx"
    plot_combined_enhanced(random_file, conditioned_file)