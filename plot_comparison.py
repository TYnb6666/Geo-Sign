import json
import matplotlib.pyplot as plt
import numpy as np

# 读取三个log文件
log_files = {
    'Hand Only': '/data/taoye/Geo-Sign/out/train_pure_hand_4D/log.txt',
    'Hand + Body': '/data/taoye/Geo-Sign/out/train_hand_body_4D/log.txt',
    'Hand + Body + Face': '/data/taoye/Geo-Sign/out/train_hand_body_face_4D/log.txt'
}

# 定义要提取的指标
metrics = ['test_loss', 'test_rouge', 'test_bleu1', 'test_bleu2', 'test_bleu3', 'test_bleu4']
metric_names = ['Test Loss', 'Test ROUGE', 'Test BLEU-1', 'Test BLEU-2', 'Test BLEU-3', 'Test BLEU-4']

# 存储数据
data = {label: {metric: [] for metric in metrics} for label in log_files.keys()}
epochs = {label: [] for label in log_files.keys()}

# 解析每个log文件
for label, filepath in log_files.items():
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                epochs[label].append(entry['epoch'])
                for metric in metrics:
                    data[label][metric].append(entry[metric])
            except json.JSONDecodeError:
                continue

# 设置图表样式
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
axes = axes.flatten()

# 颜色和线型
colors = {'Hand Only': '#1f77b4', 'Hand + Body': '#ff7f0e', 'Hand + Body + Face': '#2ca02c'}
markers = {'Hand Only': 'o', 'Hand + Body': 's', 'Hand + Body + Face': '^'}

# 绘制每个指标的折线图
for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
    ax = axes[idx]
    for label in log_files.keys():
        ax.plot(epochs[label], data[label][metric],
                label=label, color=colors[label],
                marker=markers[label], markersize=3, linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(metric_name, fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Comparison of Test Metrics: Hand Only vs Hand+Body vs Hand+Body+Face',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/data/taoye/Geo-Sign/out/metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("图片已保存到: /data/taoye/Geo-Sign/out/metrics_comparison.png")

# 打印最终结果对比
print("\n=== 最终Epoch (Epoch 49) 结果对比 ===")
for metric, metric_name in zip(metrics, metric_names):
    print(f"\n{metric_name}:")
    for label in log_files.keys():
        final_value = data[label][metric][-1]
        print(f"  {label}: {final_value:.4f}")

# 计算提升百分比
print("\n=== 提升效果分析 ===")
base_label = 'Hand Only'
for label in ['Hand + Body', 'Hand + Body + Face']:
    print(f"\n{label} 相对于 {base_label} 的提升:")
    for metric, metric_name in zip(metrics, metric_names):
        base_value = data[base_label][metric][-1]
        current_value = data[label][metric][-1]
        if metric == 'test_loss':
            # loss越低越好
            improvement = (base_value - current_value) / base_value * 100
            print(f"  {metric_name}: {improvement:+.2f}% (越低越好)")
        else:
            # 其他指标越高越好
            improvement = (current_value - base_value) / base_value * 100
            print(f"  {metric_name}: {improvement:+.2f}%")
