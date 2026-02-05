import pandas as pd
import requests
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ================= 配置区 =================
CSV_PATH = 'annotation.csv'  # 你的csv文件名
SAVE_DIR = './bdd_videos'    # 下载保存的文件夹
URL_COL = 'Input.Video'              # CSV中链接那一列的表头名称（请修改！）
NAME_COL = 'video_name'      # 文件名那一列的表头（如果有的话）
# =========================================

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def download_file(row):
    url = row[URL_COL]
    
    # 确定文件名：如果CSV里有文件名列就用，没有就从URL里截取
    if NAME_COL in row:
        filename = row[NAME_COL]
        # 如果文件名没后缀，尝试加上 .mp4 (视具体情况而定)
        if not filename.endswith('.mp4') and not filename.endswith('.mov'):
            filename += '.mp4'
    else:
        filename = url.split('/')[-1].split('?')[0] # 简单处理URL获取文件名

    file_path = os.path.join(SAVE_DIR, filename)

    # 断点续传/跳过已下载：如果文件存在且大小不为0，就跳过
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        return "Skipped"

    try:
        # 设置超时，防止卡死
        response = requests.get(url, stream=True, timeout=15)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return "Success"
        else:
            return f"Failed: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

# 读取CSV
df = pd.read_csv(CSV_PATH)
print(f"总共有 {len(df)} 个文件需要下载...")

# 使用多线程下载 (max_workers 可以根据你的网速调整，学校网好可以开 8-16)
with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(tqdm(executor.map(download_file, [row for _, row in df.iterrows()]), total=len(df)))

print("下载完成！")