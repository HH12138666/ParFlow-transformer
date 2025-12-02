import os
import re
import pandas as pd
from typing import Dict, List, Optional
import datetime
import ast  

# -------------------------------
# 1. 从文件读取日志内容
# -------------------------------
log_file_path = 'work_dirs/ParFlow_press'  
# 请根据实际情况修改log_file_result和log_file_name
log_file_result = '2025-11-28-18-07_PredFormer_depth4_Quadruplet_FACTS_sd0.25_dp0.1_ps16_bs10_256_8_32_5e-4_Adamw_cosine_50ep'
log_file_name = f'train_20251128_180728.log'


log_file_path_final = os.path.join(log_file_path, log_file_result, log_file_name)
if not os.path.isfile(log_file_path_final):
    raise FileNotFoundError(f"日志文件不存在，请检查路径：{log_file_path_final}")
with open(log_file_path_final, 'r', encoding='utf-8') as f:
    log_data = f.read()
print(f"正在读取日志文件：{log_file_path_final}")   




# -------------------------------
# 2. 提取关键配置参数
# -------------------------------
def extract_config_params(log_text: str) -> Dict[str, object]:
    config = {}
    patterns = {
        'batch_size': r'batch_size:\s*(\d+)',
        'val_batch_size': r'val_batch_size:\s*(\d+)',
        'num_workers': r'num_workers:\s*(\d+)',
        'pre_seq_length': r'pre_seq_length:\s*(\d+)',
        'aft_seq_length': r'aft_seq_length:\s*(\d+)',
        'total_length': r'total_length:\s*(\d+)',
        'use_augment': r'use_augment:\s*(True|False)',
        'epoch': r'epoch:\s*(\d+)',
        #'patch_size': r'patch_size:\s*(\d+)',
        
        #'data_root': r'data_root:\s*(\S+)',
        #'dataname': r'dataname:\s*(\S+)',
        #'metrics': r'metrics:\s*\[([^\]]+)\]',
        #'model_type': r'model_type:\s*(\S+)',
        #'method': r'method:\s*(\S+)',
        #'opt': r'opt:\s*(\S+)',
        #'lr': r'lr:\s*([\d\.eE+-]+)',
        #'weight_decay': r'weight_decay:\s*([\d\.eE+-]+)',
        #'sched': r'sched:\s*(\S+)',
        #'multi_decay_epoch': r'multi_decay_epoch:\s*\[([^\]]+)\]',
        #'decay_rate': r'decay_rate:\s*([\d\.eE+-]+)',
        #'min_lr': r'min_lr:\s*([\d\.eE+-]+)',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, log_text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if key == 'metrics':
                value = [x.strip().strip("'\"") for x in value.strip('[]').split(',')]
            config[key] = value


    model_config_lines = [line for line in log_text.split('\n') if 'model_config:' in line]
    if model_config_lines:
        model_config_line = model_config_lines[0].strip()  # 取第一个匹配行
        
        try:
            
            start_idx = model_config_line.find('{')
            end_idx = model_config_line.rfind('}')
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                dict_str = model_config_line[start_idx:end_idx + 1].strip()
                
                model_config_dict = ast.literal_eval(dict_str)
                if isinstance(model_config_dict, dict):
                    for k, v in model_config_dict.items():
                        config_key = f'model_{k}'  
                        config[config_key] = v
        except (SyntaxError, ValueError) as e:
            print(f"⚠️  提取 model_config 时发生错误（可能格式不一致）: {e}")


    model_params_pattern = re.compile(
        r'\|\s*model\s*\|\s*([\d\.]+[MmGg])\s*\|\s*([\d\.]+[MmGg])\s*\|', re.IGNORECASE)
    model_params_match = model_params_pattern.search(log_text)
    if model_params_match:
        model_parameters = model_params_match.group(1).strip()  
        model_flops = model_params_match.group(2).strip()        
        config['model_parameters'] = model_parameters
        config['model_flops'] = model_flops
    else:
        for line in log_text.split('\n'):
            line = line.strip()
            if '| model' in line and '|' in line:
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(parts) >= 3:
                    model_parameters = parts[1]  
                    model_flops = parts[2]       
                    config['model_parameters'] = model_parameters
                    config['model_flops'] = model_flops
                    break

    return config


# -------------------------------
# 3. 提取每个 epoch 的指标 + 时间 + 训练使用时间 + 累计训练时间
# -------------------------------
def extract_epoch_metrics_triplets(log_text: str) -> list:
    lines = log_text.split('\n')
    all_data = []
    training_start_time = None
    previous_end_time = None
    current_epoch = 1  # 从1开始计数

    epoch_pattern = re.compile(r'Epoch:\s*(\d+)')
    val_pattern = re.compile(r'val\s+mse:([\d\.eE+-]+)\s*,\s*mae:([\d\.eE+-]+)\s*,\s*rmse:([\d\.eE+-]+)\s*,\s*mape:([\d\.eE+-]+)')
    timestamp_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3}')
    #提取训练开始时间
    match = re.match(timestamp_pattern, lines[0])
    if match:
        timestamp_without_ms = match.group(1)  # '2025-11-07 12:56:33'
        training_start_time = datetime.datetime.strptime(timestamp_without_ms, '%Y-%m-%d %H:%M:%S')
    else:
        print("第一行未匹配到时间戳")
        
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 检查是否是验证指标行
        val_match = val_pattern.search(line)
        timestamp_match = timestamp_pattern.match(line)
        if val_match and timestamp_match:
            val_timestamp_str = timestamp_match.group(1)
            val_timestamp = datetime.datetime.strptime(val_timestamp_str, '%Y-%m-%d %H:%M:%S')
            mse = float(val_match.group(1))
            mae = float(val_match.group(2))
            rmse = float(val_match.group(3))
            mape = float(val_match.group(4))
            
            # 确定当前 epoch
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
            else:
                # 如果当前行没有 Epoch 信息，尝试从后续行获取
                # 寻找下一个包含 Epoch 信息的行
                epoch_info = None
                for next_line in lines[lines.index(line)+1:]:
                    epoch_match_next = epoch_pattern.search(next_line)
                    if epoch_match_next:
                        epoch_info = epoch_match_next.group(1)
                        break
                if epoch_info:
                    current_epoch = int(epoch_info)
                else:
                    continue  
                
            epoch_start_time = previous_end_time if previous_end_time else training_start_time

            # 计算训练使用时间（当前epoch的持续时间）
            if previous_end_time:
                epoch_duration = (val_timestamp - previous_end_time).total_seconds()
            else:
                epoch_duration = (val_timestamp - training_start_time).total_seconds()  # 第一个epoch的持续时间

            # 计算累积训练时间（从训练开始到当前epoch结束）
            cumulative_duration = (val_timestamp - training_start_time).total_seconds()

            # 存储当前epoch的指标
            epoch_metrics = {
                'Epoch': current_epoch,
                'Epoch_Start_Time': epoch_start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'Epoch_End_Time': val_timestamp_str,
                'Epoch_Duration_Seconds': round(epoch_duration, 2),
                'Cumulative_Training_Time_Seconds': round(cumulative_duration, 2),
                'MSE': mse,
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape
            }

            # 提取其他信息：Lr, Train Loss, Vali Loss
            # 寻找包含 Epoch 信息的行，通常在验证指标行之后
            epoch_info_line = None
            for next_line in lines[lines.index(line)+1:]:
                epoch_match_next = epoch_pattern.search(next_line)
                if epoch_match_next:
                    epoch_info_line = next_line
                    break

            if epoch_info_line:
                # 提取 Lr, Train Loss, Vali Loss
                lr_match = re.search(r'Lr:\s*([\d\.eE+-]+)', epoch_info_line)
                train_loss_match = re.search(r'Train Loss:\s*([\d\.eE+-]+)', epoch_info_line)
                vali_loss_match = re.search(r'Vali Loss:\s*([\d\.eE+-]+)', epoch_info_line)
                lr = float(lr_match.group(1)) if lr_match else None
                train_loss = float(train_loss_match.group(1)) if train_loss_match else None
                vali_loss = float(vali_loss_match.group(1)) if vali_loss_match else None

                epoch_metrics.update({
                    'Learning Rate': lr,
                    'Train Loss': train_loss,
                    'Validation Loss': vali_loss
                })

            # 更新 previous_end_time 为当前 epoch 的结束时间
            previous_end_time = val_timestamp

            # 添加到结果列表
            all_data.append(epoch_metrics)

    return all_data


# -------------------------------
# 4. 提取最终测试指标
# -------------------------------
def extract_final_test_metrics(log_text: str) -> Optional[Dict[str, float]]:
    test_mse_mae_rmse_mape = re.search(
        r'-\s*mse:([\d\.]+),\s*mae:([\d\.]+),\s*rmse:([\d\.]+),\s*mape:([\d\.]+)', log_text)
    final_result = re.search(r'-\s*Final result:\s*([\d\.]+)', log_text)

    if test_mse_mae_rmse_mape and final_result:
        mse, mae, rmse, mape = test_mse_mae_rmse_mape.groups()
        final_score = final_result.group(1)
        return {
            'Test MSE': float(mse),
            'Test MAE': float(mae),
            'Test RMSE': float(rmse),
            'Test MAPE': float(mape),
            'Test Final Score': float(final_score)
        }
    return None

# -------------------------------
# 5. 主函数：整合所有信息并导出 Excel
# -------------------------------
def main():
    # 1. 提取配置参数
    config = extract_config_params(log_data)

    # 2. 提取每个 epoch 的指标 + 时间 + 训练使用时间 + 累计训练时间
    epoch_metrics = extract_epoch_metrics_triplets(log_data)

    # 3. 提取测试指标
    test_metrics = extract_final_test_metrics(log_data)

    # 4. 构造 DataFrame
    df_epochs = pd.DataFrame(epoch_metrics)
    df_config = pd.DataFrame([config])
    df_test = pd.DataFrame([test_metrics]) if test_metrics else pd.DataFrame()

    # 5. 保存到 Excel
    # 获取当前日期
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    # 目标文件夹路径
    output_dir = 'result_analyze/'
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    pattern = re.compile(r'^training_summary_' + re.escape(current_date) + r'_([0-9]{3})\.xlsx$', re.IGNORECASE)

    max_num = -1
    for fname in os.listdir(output_dir):
        m = pattern.match(fname)
        if m:
            num = int(m.group(1))
            if num > max_num:
                max_num = num

    # 下一个编号 = max_num + 1
    next_num = max_num + 1
    run_id = f"{next_num:03d}"  

    # 拼接最终文件名
    output_filename = f'{output_dir}training_summary_{current_date}_{run_id}.xlsx'
    
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        df_config.to_excel(writer, sheet_name='Config_Parameters', index=False)
        df_epochs.to_excel(writer, sheet_name='Epoch_Metrics', index=False)
        if not df_test.empty:
            df_test.to_excel(writer, sheet_name='Test_Evaluation', index=False)

    print(f"✅ 所有数据已成功保存至 Excel 文件：{output_filename}")

# -------------------------------
# 6. 运行
# -------------------------------
if __name__ == '__main__':
    main()