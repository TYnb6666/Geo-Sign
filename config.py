import os
from pathlib import Path

# 获取当前 config.py 文件所在的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. mT5 模型路径
mt5_path = os.path.join(BASE_DIR, "pretrained_weight/mt5-base")

# 2. 标签路径
train_label_paths = {
    "CSL_News": os.path.join(BASE_DIR, "Data/CSL_News_Labels.json"),
    "CSL_Daily": os.path.join(BASE_DIR, "Data/CSL_Daily/labels.train")
}

dev_label_paths = {
    "CSL_News": os.path.join(BASE_DIR, "Data/CSL_News_Labels.json"),
    "CSL_Daily": os.path.join(BASE_DIR, "Data/CSL_Daily/labels.dev")
}

test_label_paths = {
    "CSL_News": os.path.join(BASE_DIR, "Data/CSL_News_Labels.json"),
    "CSL_Daily": os.path.join(BASE_DIR, "Data/CSL_Daily/labels.test")
}

# 3. 骨骼数据路径 (Pose Data)
pose_dirs = {
    "CSL_News": os.path.join(BASE_DIR, "Data/cslnews/pose_format"),
    
    # [修改] 指向 Step 2 清洗后的 Label_Corrected 目录
    # 注意：datasets.py 会在此目录下寻找 processed/S000.../data/keypoints.pkl
    "CSL_Daily": "/data/taoye/Label_Corrected_v2/processed"
}

# 4. RGB 数据路径 (占位符，保持兼容性)
rgb_dirs = {
    "CSL_News": os.path.join(BASE_DIR, "Data/cslnews/rgb"),
    "CSL_Daily": os.path.join(BASE_DIR, "Data/CSL_Daily") 
}