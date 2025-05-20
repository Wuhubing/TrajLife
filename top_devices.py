import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('data/小哥1120优质500100_20250219164014_wangweibing11_保密信息_请勿外传.csv')

# 打印数据基本信息
print(f"数据集总记录数: {len(df)}")
print(f"数据集总设备数: {df['deviceid'].nunique()}")
print("\n数据样例:")
print(df.head())

# 排除特定设备
exclude_device = 'PHONE3E0DCFFC4809D'
df = df[df['deviceid'] != exclude_device]
print(f"\n排除设备 {exclude_device} 后的记录数: {len(df)}")

# 只保留indoor和outdoor事件
df_filtered = df[df['eventname'].isin(['indoor', 'outdoor'])]
print(f"\n包含indoor和outdoor事件的记录数: {len(df_filtered)}")
print(f"包含indoor事件的记录数: {len(df[df['eventname'] == 'indoor'])}")
print(f"包含outdoor事件的记录数: {len(df[df['eventname'] == 'outdoor'])}")

# 检查GPS数据是否有效 (非空值且在合理范围内)
df_filtered['has_valid_gps'] = (~df_filtered['lng'].isna() & ~df_filtered['lat'].isna() & 
                               (df_filtered['lng'] >= -180) & (df_filtered['lng'] <= 180) &
                               (df_filtered['lat'] >= -90) & (df_filtered['lat'] <= 90))

print(f"\n有效GPS记录数: {df_filtered['has_valid_gps'].sum()}")
print(f"无效GPS记录数: {(~df_filtered['has_valid_gps']).sum()}")

# 只保留有效的GPS数据点进行统计
df_valid = df_filtered[df_filtered['has_valid_gps']]

# 统计每个设备的indoor和outdoor有效GPS数据点数量
device_counts = df_valid['deviceid'].value_counts().reset_index()
device_counts.columns = ['deviceid', 'gps_count']

# 获取前20个GPS数据最多的设备
top_20_devices = device_counts.head(20)
print("\n前20个indoor和outdoor有效GPS数据最多的设备:")
print(top_20_devices)

# 创建一个更详细的设备数据统计表
detailed_stats = []
for device_id in top_20_devices['deviceid']:
    device_data = df[df['deviceid'] == device_id]
    indoor_outdoor_data = device_data[device_data['eventname'].isin(['indoor', 'outdoor'])]
    indoor_count = len(indoor_outdoor_data[indoor_outdoor_data['eventname'] == 'indoor'])
    outdoor_count = len(indoor_outdoor_data[indoor_outdoor_data['eventname'] == 'outdoor'])
    
    # 获取用户ID（每个设备对应一个用户ID）
    user_id = device_data['userid'].iloc[0] if len(device_data) > 0 else None
    
    detailed_stats.append({
        'deviceid': device_id,
        'userid': user_id,
        'total_io_count': indoor_count + outdoor_count,
        'indoor_count': indoor_count,
        'outdoor_count': outdoor_count,
        'indoor_ratio': round(indoor_count / (indoor_count + outdoor_count) * 100, 1) if (indoor_count + outdoor_count) > 0 else 0
    })

# 创建详细统计数据的DataFrame并保存
detailed_df = pd.DataFrame(detailed_stats)
detailed_df.to_csv('top_20_devices_detailed.csv', index=False)
print("\n详细设备统计数据已保存到 top_20_devices_detailed.csv")

# 获取这20个设备的所有数据并保存
all_top_device_data = df[df['deviceid'].isin(top_20_devices['deviceid'])]
all_top_device_data.to_csv('top_20_devices_all_data.csv', index=False)
print(f"\n这20个设备的所有数据({len(all_top_device_data)}条记录)已保存到 top_20_devices_all_data.csv")

# 只获取这20个设备的indoor和outdoor数据并保存
io_top_device_data = df_valid[df_valid['deviceid'].isin(top_20_devices['deviceid'])]
io_top_device_data.to_csv('top_20_devices_indoor_outdoor_data.csv', index=False) 
print(f"\n这20个设备的indoor和outdoor数据({len(io_top_device_data)}条记录)已保存到 top_20_devices_indoor_outdoor_data.csv")
