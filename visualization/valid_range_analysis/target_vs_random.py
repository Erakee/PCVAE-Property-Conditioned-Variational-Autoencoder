import numpy as np

# 读取数据
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = [float(line.strip()) for line in f if line.strip()]
    return np.array(data)

# 统计函数（以总数为基准）
def compute_distribution(data, target_points):
    total = len(data)
    results = []

    for target in target_points:
        in_10 = ((data >= target - 10) & (data <= target + 10)).sum()
        in_15 = ((data >= target - 15) & (data <= target + 15)).sum()

        results.append({
            'Target': target,
            '±10 Count': in_10,
            '±10 Ratio': round(in_10 / total, 4),
            '±15 Count': in_15,
            '±15 Ratio': round(in_15 / total, 4)
        })

    return results

# 打印结果表格
def print_table(results):
    print(f"{'Target':>7} | {'±10 C':>7} | {'±10 %':>7} | {'±15 C':>7} | {'±15 %':>7}")
    print("-" * 45)
    for row in results:
        print(f"{row['Target']:>7} | {row['±10 Count']:>7} | {row['±10 Ratio']:>7.2%} | "
              f"{row['±15 Count']:>7} | {row['±15 Ratio']:>7.2%}")

# # 主函数
# if __name__ == "__main__":
#     file_path = 'your_data.txt'  # 替换成你的文件路径
#     data = load_data(file_path)
#
#     target_values = list(range(-450, 301, 50))  # 从 -450 到 300，步长为 50
#     results = compute_distribution(data, target_values)
#     print_table(results)
#

# 主函数
if __name__ == "__main__":
    # 修改这里为你的实际路径
    file_path = r'D:\Project\EVAE_paper\visualization\valid_range_analysis\vae_enthalpy.txt'  # 请替换为你的文件名
    data = load_data(file_path)

    target_values = list(range(-450, 301, 50))
    results = compute_distribution(data, target_values)
    print_table(results)
