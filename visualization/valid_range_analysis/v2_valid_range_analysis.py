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
def analyze_random_data(filepath, smooth_method='raw', **smooth_params):
    """
    参数说明：
    - smooth_method: 可选 'spline'(默认)/'raw'/
    - smooth_params: 各方法专用参数
        - spline: k (样条阶数，默认3)
        - moving_avg: window_size (窗口大小，默认3)
        - polyfit: degree (多项式阶数，默认2)
        - kde: bw_method (带宽方法，默认'scott')
    """
    # 读取数据
    with open(filepath, 'r') as f:
        data = [float(line.strip()) for line in f if line.strip()]

    # 分箱统计（每50划分区间）
    bins = np.arange(-400, 301, 100)
    counts, bin_edges = np.histogram(data, bins=bins)
    percentages = counts / len(data)
    x_center = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 处理不同平滑方法
    if smooth_method == 'raw':
        # 原始折线图（不进行平滑）
        return x_center, percentages

    elif smooth_method == 'spline':
        # 样条插值（默认方法）
        k = smooth_params.get('k', 3)
        x_smooth = np.linspace(x_center.min(), x_center.max(), 300)
        spl = make_interp_spline(x_center, percentages, k=k)
        y_smooth = spl(x_smooth)
        return x_smooth, y_smooth

    else:
        raise ValueError(f"不支持的平滑方法: {smooth_method}")


# ========== 条件生成数据分析 ==========
def analyze_conditioned_data(conditioned_file):
    nSample = 200
    # 读取数据并修复列名类型
    df = pd.read_excel(conditioned_file, header=0)
    df.columns = df.columns.astype(float)

    results = {}
    for set_val in df.columns:
        # 获取当前列所有有效值
        values = df[set_val].dropna()

        # 初始化统计计数器
        count_10 = 0  # ±10区间
        count_15 = 0  # ±10-15区间
        count_over = 0  # 超过±15区间

        # 计算偏差
        for val in values:
            delta = abs(val - set_val)

            if delta <= 10:
                count_10 += 1
            elif 10 < delta <= 15:
                count_15 += 1
            else:
                count_over += 1

        # 计算比例（基于目标100个）
        total_generated = len(values)
        results[set_val] = {
            '±10': count_10 / nSample,
            '±(10-15)': count_15 / nSample,
            '>±15': count_over / nSample,
            'Invalid': (100 - total_generated) / nSample  # 未生成部分
        }

    return pd.DataFrame(results).T


# ========== 可视化模块 ==========
def plot_stacked_bar(results, random_file):
    # 新的颜色配置 (Nature 风格)
    colors = {
        '±10': '#1f78b4',  # 深蓝 - 高准确率
        '±(10-15)': '#33a02c',  # 草绿 - 中等偏差
        '>±15': '#e31a1c',  # 深红 - 偏差较大
        'Invalid': '#bdbdbd'  # 灰色 - 未生成（低亮度）
    }

    # 背景图数据
    rand_x, rand_y = analyze_random_data(random_file)

    # 创建图像和主轴
    fig, ax1 = plt.subplots(figsize=(12, 6.5))

    # 背景分布图
    ax1.plot(rand_x, rand_y, color='black', alpha=0.6,
             linewidth=2.0, label='Unconditional Distribution')
    ax1.fill_between(rand_x, rand_y, color='#4878CF', alpha=0.1, linewidth=2.0)
    ax1.set_xlabel('Enthalpy (kJ/mol)', fontweight='bold')
    ax1.set_ylabel('Random Sample Distribution Density', color='black', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='black', width=1.5)
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.set_xlim(-400, 300)
    ax1.set_ylim(0, rand_y.max() * 1.1)
    ax1.set_xticks(np.arange(-400, 351, 50))

    # 前景堆叠柱状图
    ax2 = ax1.twinx()
    categories = ['±10', '±(10-15)', '>±15', 'Invalid']
    bottom = np.zeros(len(results))

    # 遍历绘制每一类
    for cat in categories:
        values = results[cat].values
        bars = ax2.bar(results.index, values,
                       width=25,
                       label=cat,
                       color=colors[cat],
                       bottom=bottom,
                       edgecolor='black',
                       linewidth=0.4,
                       alpha=0.6,  # 更通透，背景可见
                       zorder=2)  # 保证图层在背景曲线之上

        # 添加百分比标签
        for bar, val in zip(bars, values):
            if val > 0.05:  # 避免标注太小的
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() / 2 + bar.get_y(),
                         f"{val * 100:.0f}%",
                         ha='center', va='center',
                         fontsize=10, color='black',
                         fontweight='bold', zorder=3)

        bottom += values

    ax2.set_xlabel("Set Enthalpy (kJ/mol)", fontweight='bold', fontname='Arial')
    ax2.set_ylabel("Proportion of generated  in Each Enthalpy Setting", color='#B22222', fontweight='bold', fontname='Arial')
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.spines['right'].set_color('#B22222')
    ax2.tick_params(axis='y', colors='#B22222', width=1.5)
    ax2.spines['right'].set_visible(True)
    ax2.set_ylim(0, 1.1)  # 上方留出10%空间

    # 图例（自定义手动设置）
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=colors['±10']),
        plt.Rectangle((0, 0), 1, 1, fc=colors['±(10-15)']),
        plt.Rectangle((0, 0), 1, 1, fc=colors['>±15']),
        plt.Rectangle((0, 0), 1, 1, fc=colors['Invalid'])
    ]
    labels = ['Within ±10', 'Within ±(10-15)', 'Beyond ±15', 'Invalid']
    ax2.legend(handles, labels, loc='upper center',
               bbox_to_anchor=(0.8, 1.05), ncol=2, frameon=True,
               fontsize=12)

    # plt.title("Molecular Generation Precision Analysis", fontsize=16, fontweight='bold', fontname='Arial', pad=20)
    plt.tight_layout()
    plt.savefig('delete_repeat_stacked_bar_analysis.png', bbox_inches='tight')
    plt.show()


# ========== 主程序 ==========
if __name__ == "__main__":
    random_file = r"rand_generated_enthalpy.txt"
    conditioned_file = r"CVAE_DHR_uniq.xlsx"

    # 分析数据
    cond_results = analyze_conditioned_data(conditioned_file)

    # 可视化
    plot_stacked_bar(cond_results, random_file)