import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

quant_schemes = ['FP32', 'INT8', 'INT4']
x = np.arange(3)
width = 0.35

# Clean Accuracy
ax = axes[0]
cifar_acc = [90.8, 90.8, 52.7]
gtsrb_acc = [96.0, 96.0, 95.6]

bars1 = ax.bar(x - width/2, cifar_acc, width, label='CIFAR-10', color='#2ecc71', edgecolor='black')
bars2 = ax.bar(x + width/2, gtsrb_acc, width, label='GTSRB', color='#3498db', edgecolor='black')

ax.set_xlabel('Quantization Scheme', fontweight='bold')
ax.set_ylabel('Clean Accuracy (%)', fontweight='bold')
ax.set_title('Model Accuracy Under Quantization', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(quant_schemes)
ax.set_ylim(0, 110)
ax.legend()

for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

# Attack Success Rate
ax = axes[1]
cifar_asr = [99.8, 99.8, 91.0]
gtsrb_asr = [99.4, 99.4, 99.3]

bars1 = ax.bar(x - width/2, cifar_asr, width, label='CIFAR-10', color='#e74c3c', edgecolor='black')
bars2 = ax.bar(x + width/2, gtsrb_asr, width, label='GTSRB', color='#9b59b6', edgecolor='black')

ax.set_xlabel('Quantization Scheme', fontweight='bold')
ax.set_ylabel('Attack Success Rate (%)', fontweight='bold')
ax.set_title('Backdoor Persistence Under Quantization', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(quant_schemes)
ax.set_ylim(0, 110)
ax.legend()

for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('fig3_model_quality.pdf')
plt.savefig('fig3_model_quality.png')
plt.close()
print("Saved: fig3_model_quality.pdf and fig3_model_quality.png")