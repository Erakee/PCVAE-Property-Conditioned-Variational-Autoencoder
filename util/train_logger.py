import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


class TrainingLogger:
    def __init__(self, base_dir="experiments"):
        # 创建唯一实验文件夹
        self.exp_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = os.path.join(base_dir, self.exp_time)
        os.makedirs(self.log_dir, exist_ok=True)

        # 初始化数据存储
        self.log_data = pd.DataFrame(columns=[
            'epoch', 'recon_loss', 'kld_loss',
            'cond_loss', 'total_loss', 'valid_rate'
        ])

        # 图表样式设置
        plt.style.use('seaborn')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    def log_metrics(self, epoch, metrics_dict):
        """记录单epoch指标"""
        new_row = pd.DataFrame([{
            'epoch': epoch,
            **metrics_dict
        }])
        self.log_data = pd.concat([self.log_data, new_row], ignore_index=True)

        # 实时保存到Excel
        excel_path = os.path.join(self.log_dir, 'training_log.xlsx')
        self.log_data.to_excel(excel_path, index=False)

    def plot_losses(self, epoch_interval=50):
        """绘制损失曲线并保存（对数坐标优化版）"""
        if len(self.log_data) == 0:
            return

        plt.figure(figsize=(15, 6), facecolor='white', dpi=300)
        plt.suptitle(f"Training Progress @ Epoch {self.log_data['epoch'].max()}", y=1.02, fontsize=14)

        # ================= 左侧损失曲线 =================
        ax1 = plt.subplot(1, 2, 1)

        # 绘制主损失曲线（对数坐标）
        lines = []
        labels = []
        for i, col in enumerate(['recon_loss', 'kld_loss', 'cond_loss']):
            line, = ax1.semilogy(self.log_data['epoch'],  # 使用semilogy直接创建对数坐标
                                 self.log_data[col],
                                 color=self.colors[i],
                                 linewidth=2.5,
                                 marker='o' if i == 0 else None,
                                 markersize=4,
                                 markevery=20)
            lines.append(line)
            labels.append(col.replace('_', ' ').title())

        # 格式设置
        ax1.set_xlabel('Epoch', fontsize=12, labelpad=10)
        ax1.set_ylabel('Loss Value (log scale)', fontsize=12, labelpad=10)
        ax1.grid(True, which="both", ls="--", alpha=0.6)
        ax1.legend(lines, labels, loc='upper right', fontsize=10,
                   frameon=True, shadow=True, borderpad=1)

        # ================= 右侧综合视图 =================
        ax2 = plt.subplot(1, 2, 2)

        # 总损失曲线（对数坐标）
        line_total, = ax2.semilogy(self.log_data['epoch'], self.log_data['total_loss'],
                                   color=self.colors[3], linewidth=2.5,
                                   label='Total Loss')

        # 验证率曲线（线性坐标）
        ax3 = ax2.twinx()
        line_valid, = ax3.plot(self.log_data['epoch'], self.log_data['valid_rate'] * 100,
                               color='#9467bd', linewidth=2.5, linestyle='--',
                               marker='s', markersize=5, markevery=20,
                               label='Validation Rate (%)')

        # 格式设置
        ax2.set_xlabel('Epoch', fontsize=12, labelpad=10)
        ax2.set_ylabel('Total Loss (log scale)', fontsize=12, labelpad=15)
        ax3.set_ylabel('Validation Rate (%)', fontsize=12, labelpad=15, rotation=-90, va='bottom')
        ax2.grid(True, which="both", ls="--", alpha=0.6)

        # 合并图例
        lines = [line_total, line_valid]
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper left', fontsize=10,
                   frameon=True, shadow=True, borderpad=1)

        # ================= 通用设置 =================
        plt.tight_layout(pad=3)

        # 保存图片
        plot_path = os.path.join(self.log_dir,
                                 f"loss_plot_epoch_{self.log_data['epoch'].max()}.png")
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.2)
        plt.close()