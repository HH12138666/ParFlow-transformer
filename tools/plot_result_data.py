import pandas as pd
import matplotlib.pyplot as plt

# Ensure English text rendering
plt.rcParams['font.family'] = 'DejaVu Sans'  # or any sans-serif font that supports English well
plt.rcParams['axes.unicode_minus'] = False

# 1. Load data
file_path = '/data/huanghui-data/ParFlow-transformer/result_analyze'
file_name = 'training_summary_2025-11-28_000.xlsx'
sheet_name = 'Epoch_Metrics'
file_path = f"{file_path}/{file_name}"
try:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
except Exception as e:
    print("Error reading file:", e)
    exit()

# 2. Check required columns
required_cols = {'Epoch', 'Train Loss', 'Validation Loss', 'RMSE'}
missing = required_cols - set(df.columns)
if missing:
    print("Missing columns:", missing)
    exit()

# 3. Plot RMSE (matching your first image)
plt.figure(figsize=(12, 6))
plt.plot(df['Epoch'], df['RMSE'], color='black', marker='.', linewidth=2, label='RMSE Press (m)')
plt.title('RMSE Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('RMSE Press (m)')
plt.ylim(1.0, 5.0)      # Fixed y-range as in image
plt.xlim(0, 100)        # Fixed x-range
plt.grid(False)
plt.legend(loc='upper right')  
plt.xticks(range(0, 101, 10))  # ticks at 0, 10, 20, ..., 100
plt.tight_layout()
plt.savefig('rmse_plot_reference_style.png')


# 4. Plot Train Loss and Validation Loss on dual Y-axis (matching your second image)
fig, ax1 = plt.subplots(figsize=(12, 6))

# Left Y-axis: Train Loss
color_left = 'black'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Train Loss', color=color_left)
line_train = ax1.plot(df['Epoch'], df['Train Loss'], color=color_left, marker='o', linewidth=2, label='Train Loss')
ax1.tick_params(axis='y', labelcolor=color_left)
ax1.set_ylim(0, 0.25)     # As in image
ax1.set_xlim(0, 100)
ax1.grid(True)

# Right Y-axis: Validation Loss
ax2 = ax1.twinx()
color_right = 'red'
ax2.set_ylabel('Val Loss', color=color_right)
line_val = ax2.plot(df['Epoch'], df['Validation Loss'], color=color_right, marker='s', linewidth=2, label='Val Loss')
ax2.tick_params(axis='y', labelcolor=color_right)
ax2.set_ylim(0.38, 0.43)  # As in image

# Combine legends
lines = line_train + line_val
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right')

plt.title('Training and Validation Loss Over Epochs')
plt.xticks(range(0, 101, 20))  # ticks at 0, 20, 40, 60, 80, 100
plt.tight_layout()
plt.savefig('loss_comparison_dual_axis.png')
plt.show()

print("Plots generated:")
print("  - rmse_plot_reference_style.png")
print("  - loss_comparison_dual_axis.png")