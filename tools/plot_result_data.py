import datetime
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def find_latest_xlsx(base_dir: str) -> str:
    """Pick the most recent Excel file under base_dir."""
    candidates = glob.glob(os.path.join(base_dir, '*.xlsx'))
    if not candidates:
        raise FileNotFoundError(f'No .xlsx files under {base_dir}')
    return max(candidates, key=os.path.getmtime)


# ----------------------------
# 手动配置路径，后续自行修改
# ----------------------------
# dir_path: 你的 Excel 所在目录（collect_analyze_data.py 输出的 .xlsx）
# excel_file: 如果想指定具体文件名，填入文件名；保持 None 则自动取目录下最新的 .xlsx
# save_path: 图像保存目录
# sheet_name: Excel 中的 Sheet 名
dir_path = '/data/huanghui-data/ParFlow-transformer/result_analyze'
excel_file = 'training_summary_2025-12-08_000.xlsx'  # e.g., 'training_summary_2025-12-05_000.xlsx'
sheet_name = 'Epoch_Metrics'
save_path = '/data/huanghui-data/ParFlow-transformer/result_analyze/data_plots'


def main():
    # 决定 Excel 路径
    if excel_file:
        excel_path = os.path.join(dir_path, excel_file)
    elif os.path.isfile(dir_path):
        excel_path = dir_path  # dir_path 指向具体文件时直接使用
    else:
        excel_path = find_latest_xlsx(dir_path)

    # 输出目录：按日期建子目录便于区分
    out_root = os.path.join(save_path, datetime.datetime.now().strftime('%Y-%m-%d'))
    os.makedirs(out_root, exist_ok=True)

    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    required_cols = {'Epoch', 'Train Loss', 'Validation Loss', 'RMSE'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns in sheet \"{sheet_name}\": {missing}')

    # 1) Train vs Validation Loss (同轴两条线)
    plt.figure(figsize=(10, 5))
    plt.plot(df['Epoch'], df['Train Loss'], label='Train Loss', color='steelblue', linewidth=2)
    plt.plot(df['Epoch'], df['Validation Loss'], label='Validation Loss', color='tomato', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    loss_path = os.path.join(out_root, excel_file + '_loss_epoch.png')
    plt.savefig(loss_path, dpi=300)

    # 2) RMSE 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(df['Epoch'], df['RMSE'], label='RMSE', color='black', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Press RMSE (m)')
    plt.title('RMSE Over Epochs')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    rmse_path = os.path.join(out_root, excel_file + '_rmse_epoch.png')
    plt.savefig(rmse_path, dpi=300)

    print('Plots saved:')
    print(f'  Loss: {loss_path}')
    print(f'  RMSE: {rmse_path}')


if __name__ == '__main__':
    main()

# python /data/huanghui-data/ParFlow-transformer/tools/plot_result_data.py