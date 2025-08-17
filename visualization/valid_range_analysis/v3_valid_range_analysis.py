import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.interpolate import make_interp_spline

# ========== 专业科研风格设置 ==========
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'figure.dpi': 300,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.linewidth': 1.2,
    'lines.linewidth': 3.5,
})


def analyze_random_data(filepath, smooth_method='raw', **smooth_params):
    # 保持原有逻辑不变
    with open(filepath, 'r') as f:
        data = [float(line.strip()) for line in f if line.strip()]
    bins = np.arange(-450, 351, 50)
    counts, bin_edges = np.histogram(data, bins=bins)
    percentages = counts / len(data)
    x_center = (bin_edges[:-1] + bin_edges[1:]) / 2
    if smooth_method == 'spline':
        k = smooth_params.get('k', 3)
        x_smooth = np.linspace(x_center.min(), x_center.max(), 300)
        spl = make_interp_spline(x_center, percentages, k=k)
        y_smooth = spl(x_smooth)
        return x_smooth, y_smooth
    else:
        return x_center, percentages


def analyze_conditioned_data(conditioned_file):
    # 保持原有逻辑不变
    nSample = 200
    df = pd.read_excel(conditioned_file, header=0)
    df.columns = df.columns.astype(float)
    results = {}
    for set_val in df.columns:
        values = df[set_val].dropna()
        count_10 = 0
        count_15 = 0
        count_over = 0
        for val in values:
            delta = abs(val - set_val)
            if delta <= 10:
                count_10 += 1
            elif 10 < delta <= 15:
                count_15 += 1
            else:
                count_over += 1
        total_generated = len(values)
        results[set_val] = {
            '±10': count_10 / nSample,
            '±(10-15)': count_15 / nSample,
            '>±15': count_over / nSample,
            'Invalid': (nSample - total_generated) / nSample
        }
    return pd.DataFrame(results).T


def plot_stacked_bar(results, random_file=None, show_random_curve=True):
    # 定义颜色时指定亮度（深色/浅色区分）
    colors = {
        '±10': '#4995c6',  # 深色湖蓝
        '±(10-15)': '#369f2d',  # 深绿色
        '>±15': '#fabb6e',  # 浅色珊瑚粉
        'Invalid': '#d3d3d3'  # 浅灰色
    }

    # 计算每个颜色的亮度，用于判断文本颜色（亮度公式：L = 0.2126*R + 0.7152*G + 0.0722*B）
    def get_text_color(rgb_hex):
        rgb = tuple(int(rgb_hex.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
        L = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255
        return 'white' if L < 0.6 else 'black'  # 亮度低于0.5用白色，否则用黑色

    fig, ax1 = plt.subplots(figsize=(13, 7))

    categories = ['±10', '±(10-15)', '>±15', 'Invalid']
    bottom = np.zeros(len(results))

    for cat in categories:
        color = colors[cat]
        text_color = get_text_color(color)  # 获取文本颜色
        values = results[cat].values
        bars = ax1.bar(results.index, values,
                       width=35,
                       label=cat,
                       color=color,
                       bottom=bottom,
                       edgecolor='black',
                       linewidth=0.5,
                       alpha=0.85,
                       zorder=3)

        for bar, val in zip(bars, values):
            if val > 0.05:  # 仅显示比例>5%的标签
            # if val > 0:  # 显示全标签
                ax1.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_y() + bar.get_height() / 2,
                         f"{val * 100:.0f}%",
                         ha='center', va='center',
                         fontsize=14, color=text_color, fontweight='bold')
        bottom += values

    ax1.set_xlabel("Target EoF (kcal/mol)", fontweight='bold')
    ax1.set_ylabel("Proportion in Error Intervals", fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.set_xticks(np.arange(-450, 351, 50))
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax1.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)

    # 次轴：随机采样分布曲线（可选）
    if show_random_curve and random_file is not None:
        rand_x, rand_y = analyze_random_data(random_file, smooth_method='spline')
        ax2 = ax1.twinx()
        ax2.plot(rand_x, rand_y, color='dimgray', linewidth=2.2, linestyle='--', label='Unconditional Distribution',
                 zorder=2)
        ax2.fill_between(rand_x, rand_y, color='gray', alpha=0.15, zorder=1)
        ax2.set_ylabel("Random Sampling Distribution Density", fontweight='bold', color='dimgray')
        ax2.tick_params(axis='y', labelcolor='dimgray', labelsize=14)
        ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax2.set_ylim(0, rand_y.max() * 1.2)
        ax2.spines['right'].set_visible(True)

    ax1.legend(loc='upper right', ncol=2, frameon=False, fontsize=14, bbox_to_anchor=(1.0, 1.05))
    plt.tight_layout(pad=1.5)
    plt.savefig('plot.png', bbox_inches='tight', dpi=600)
    plt.show()


# 执行主流程
random_file = r"vae_enthalpy.txt"
conditioned_file = "CVAE_DHR_uniq.xlsx"
cond_results = analyze_conditioned_data(conditioned_file)
plot_stacked_bar(cond_results, random_file=random_file, show_random_curve=True)