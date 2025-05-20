#!/usr/bin/env python3
import pandas as pd
import numpy as np
import h3
import json
import os
from pathlib import Path
import argparse
import datetime

def convert_to_h3(lat, lng, resolution=7):
    """将经纬度坐标转换为H3编码"""
    return h3.latlng_to_cell(lat, lng, resolution)

def create_trajectory(df, device_id, resolution=7):
    """为单个设备创建轨迹数据"""
    # 按时间排序设备数据
    device_df = df[df['deviceid'] == device_id].sort_values('eventtime')
    
    # 提取经纬度并转换为H3编码
    h3_cells = []
    for _, row in device_df.iterrows():
        try:
            h3_cell = convert_to_h3(row['lat'], row['lng'], resolution)
            h3_cells.append(h3_cell)
        except Exception as e:
            print(f"转换H3失败: {e} - 经度: {row['lng']}, 纬度: {row['lat']}")
            continue
    
    return h3_cells

def process_data(input_csv, output_dir, resolution=7, min_trajectory_length=5):
    """处理数据并转换为TrajLearn格式"""
    print(f"加载数据: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # 确保时间列是datetime格式
    df['eventtime'] = pd.to_datetime(df['eventtime'])
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取唯一设备ID列表
    device_ids = df['deviceid'].unique()
    print(f"发现 {len(device_ids)} 个设备")
    
    # 为每个设备创建轨迹
    all_trajectories = []
    device_trajectories = {}
    
    for i, device_id in enumerate(device_ids):
        print(f"处理设备 {i+1}/{len(device_ids)}: {device_id}")
        trajectory = create_trajectory(df, device_id, resolution)
        
        if len(trajectory) >= min_trajectory_length:
            device_trajectories[device_id] = trajectory
            all_trajectories.append(' '.join(trajectory))
    
    print(f"生成了 {len(device_trajectories)} 个有效轨迹")
    
    # 创建数据集名称
    dataset_name = f"chinese_devices_h3_res{resolution}"
    dataset_dir = Path(output_dir) / dataset_name
    os.makedirs(dataset_dir, exist_ok=True)
    
    # 1. 写入轨迹数据
    data_file = dataset_dir / 'higher_order_trajectory.csv'
    dates = [(datetime.datetime.now() - datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(len(all_trajectories))]
    
    trajectory_df = pd.DataFrame({
        'higher_order_trajectory': all_trajectories,
        'date': dates
    })
    trajectory_df.to_csv(data_file, index=False)
    print(f"轨迹数据已保存到: {data_file}")
    
    # 生成h3格式的预处理数据
    preprocess_dir = Path(output_dir) / f"{dataset_name}_res{resolution}"
    os.makedirs(preprocess_dir, exist_ok=True)
    
    # 2. 创建词汇表
    all_h3_cells = []
    for trajectory in device_trajectories.values():
        all_h3_cells.extend(trajectory)
    
    unique_h3_cells = ["EOT"] + list(set(all_h3_cells))
    
    vocab_file = preprocess_dir / 'vocab.txt'
    with open(vocab_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(unique_h3_cells) + '\n')
    print(f"词汇表已保存到: {vocab_file}")
    
    # 3. 创建映射文件
    mapping = {cell: i for i, cell in enumerate(unique_h3_cells)}
    mapping_file = preprocess_dir / 'mapping.json'
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False)
    print(f"映射文件已保存到: {mapping_file}")
    
    # 4. 创建邻居文件
    neighbors = {}
    for i, h3_cell in enumerate(unique_h3_cells[1:], 1):  # 跳过EOT
        neighbor_cells = h3.grid_ring(h3_cell, 1)  # 获取1环邻居
        neighbor_indices = [mapping.get(cell, 0) for cell in neighbor_cells if cell in mapping]
        neighbors[i] = neighbor_indices
    
    neighbors_file = preprocess_dir / 'neighbors.json'
    with open(neighbors_file, 'w', encoding='utf-8') as f:
        json.dump(neighbors, f, ensure_ascii=False)
    print(f"邻居文件已保存到: {neighbors_file}")
    
    # 5. 创建数据文件 (转换为索引)
    data_file = preprocess_dir / 'data.txt'
    with open(data_file, 'w', encoding='utf-8') as f:
        for trajectory in device_trajectories.values():
            trajectory_indices = [str(mapping[cell]) for cell in trajectory]
            f.write(' '.join(trajectory_indices) + f" {mapping['EOT']}\n")
    print(f"数据文件已保存到: {data_file}")
    
    # 6. 创建配置文件
    create_config_file(dataset_name, resolution)
    
    return dataset_name, resolution

def create_config_file(dataset_name, resolution):
    """创建TrajLearn模型配置文件"""
    config = {
        "data_dir": "./data",
        "dataset": f"{dataset_name}_res{resolution}",
        "model_checkpoint_directory": f"./models/{dataset_name}_res{resolution}",
        "test_ratio": 0.2,
        "validation_ratio": 0.1,
        "delimiter": " ",
        "batch_size": 32,
        "device": "cuda",
        "max_epochs": 100,
        "block_size": 128,
        "learning_rate": 6e-4,
        "min_input_length": 5,
        "max_input_length": 30,
        "test_input_length": 5,
        "test_prediction_length": 5,
        "weight_decay": 1e-1,
        "beta1": 0.9,
        "beta2": 0.95,
        "grad_clip": 1.0,
        "decay_lr": True,
        "warmup_iters": 20,
        "lr_decay_iters": 1000,
        "min_lr": 6e-5,
        "seed": 10, 
        "n_layer": 10,
        "n_head": 16,
        "n_embd": 512,
        "bias": False,
        "dropout": 0.1,
        "custom_initialization": False,
        "train_from_checkpoint_if_exist": True,
        "patience": 10,
        "continuity": True,
        "beam_width": 64,
        "store_predictions": True
    }
    
    config_file = f"chinese_devices_config.yaml"
    with open(config_file, 'w') as f:
        f.write(f"{dataset_name}_res{resolution}:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
    
    print(f"配置文件已保存到: {config_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将GPS数据转换为TrajLearn格式')
    parser.add_argument('--input', type=str, default='data/top_20_devices_indoor_outdoor_data.csv',
                        help='输入CSV文件路径')
    parser.add_argument('--output', type=str, default='data',
                        help='输出目录路径')
    parser.add_argument('--resolution', type=int, default=7,
                        help='H3网格分辨率 (0-15), 默认值为7')
    parser.add_argument('--min_length', type=int, default=5,
                        help='最小轨迹长度, 默认值为5')
    
    args = parser.parse_args()
    
    dataset_name, resolution = process_data(
        args.input, 
        args.output, 
        args.resolution, 
        args.min_length
    )
    
    print(f"\n转换完成!")
    print(f"数据集名称: {dataset_name}_res{resolution}")
    print(f"要运行TrajLearn模型，请执行:")
    print(f"cd Trajectory-prediction")
    print(f"python main.py ../chinese_devices_config.yaml") 