import torch
import utils as utils
import torch.utils.data.dataset as Dataset
from torch.nn.utils.rnn import pad_sequence
import os
import random
import numpy as np
import pickle
import json
from pathlib import Path
from config import rgb_dirs, pose_dirs
from transformers import T5Tokenizer
import gzip
import pandas as pd  # [新增] 引入 Pandas 处理插值
import types
import sys


# ============ Pickle compatibility patch for numpy >= 2.0 files ============
def load_pickle_compatible(path):
    """
    Load pickle file that may be created with a different numpy version.
    Handles numpy._core.numeric module mismatch when loading pickles from numpy >= 2.0
    with numpy < 2.0 environment.
    """
    # Create the numpy._core module structure if it doesn't exist
    if not hasattr(np, '_core'):
        # Create the _core module structure
        np._core = types.ModuleType('numpy._core')

        # Create _core.numeric module with necessary functions
        numeric_module = types.ModuleType('numpy._core.numeric')

        # Add necessary functions that pickle might reference
        def _frombuffer(*args, **kwargs):
            """Compatibility placeholder for numpy._core.numeric._frombuffer"""
            # This is a simplified version - just handle basic cases
            if len(args) >= 2:
                buffer, dtype = args[0], args[1]
                if isinstance(dtype, np.dtype):
                    return np.frombuffer(buffer, dtype=dtype)
            return np.frombuffer(*args, **kwargs)

        numeric_module._frombuffer = _frombuffer
        numeric_module._ufunc_doc_signature_formatter = lambda *args, **kwargs: ""

        # Add more functions as needed
        numeric_module.normalize_axis_index = np.core.numeric.normalize_axis_index if hasattr(np.core, 'numeric') else lambda axis, ndim: axis

        np._core.numeric = numeric_module

        # Create _core._multiarray_umath if needed
        np._core._multiarray_umath = types.ModuleType('numpy._core._multiarray_umath')
        np._core._multiarray_umath._frombuffer = _frombuffer

        # Register the modules in sys.modules so pickle can find them
        sys.modules['numpy._core'] = np._core
        sys.modules['numpy._core.numeric'] = np._core.numeric
        sys.modules['numpy._core._multiarray_umath'] = np._core._multiarray_umath

    # Now try to load the pickle
    with open(path, 'rb') as f:
        data = pickle.load(f)

    # Fix array shapes if they were flattened
    # Mediapipe format shapes:
    # - face: (T, 20, 3)
    # - pose: (T, 17, 4)
    # - left_hand: (T, 21, 7)
    # - right_hand: (T, 21, 7)
    if 'face' in data and data['face'].ndim == 1:
        T = len(data['frame_ids']) if 'frame_ids' in data else (data['face'].shape[0] // 60)
        data['face'] = data['face'].reshape(T, 20, 3)
    if 'pose' in data and data['pose'].ndim == 1:
        T = len(data['frame_ids']) if 'frame_ids' in data else (data['pose'].shape[0] // 68)
        data['pose'] = data['pose'].reshape(T, 17, 4)
    if 'left_hand' in data and data['left_hand'].ndim == 1:
        T = len(data['frame_ids']) if 'frame_ids' in data else (data['left_hand'].shape[0] // 147)
        data['left_hand'] = data['left_hand'].reshape(T, 21, 7)
    if 'right_hand' in data and data['right_hand'].ndim == 1:
        T = len(data['frame_ids']) if 'frame_ids' in data else (data['right_hand'].shape[0] // 147)
        data['right_hand'] = data['right_hand'].reshape(T, 21, 7)

    return data


# ------------------------------------------------------------------------------
# 1. [新增] 混合策略清洗函数 (Hybrid Strategy)
# ------------------------------------------------------------------------------
def hybrid_fill_nan(data_array, threshold=10):
    """
    混合清洗策略:
    - 连续 NaN <= threshold (0.4s): 视为遮挡 -> 线性插值修补
    - 连续 NaN > threshold: 视为不存在 -> 填 0.0
    
    Args:
        data_array: np.array (T, N, C)
    Returns:
        cleaned_array: np.array (T, N, C)
    """
    T, N, C = data_array.shape
    # 展平为 (T, N*C) 以便 Pandas 按列处理
    flat_data = data_array.reshape(T, N * C)
    df = pd.DataFrame(flat_data)
    
    # 1. 线性插值 (只修补短空洞)
    # limit=10 意味着只有连续缺失 <=10 的部分会被填上，超过的不填
    df_interp = df.interpolate(method='linear', limit=threshold, limit_direction='both')
    
    # 2. 剩余的 NaN (长空洞) 视为无手，填 0.0
    df_filled = df_interp.fillna(0.0)
    
    return df_filled.values.reshape(T, N, C).astype(np.float32)

# ------------------------------------------------------------------------------
# 2. [新增] 4D 相对坐标提取函数
# ------------------------------------------------------------------------------
def load_part_kp_4d(data_dict, indices):
    """
    提取 4维特征 [Norm_X, Norm_Y, Norm_Z, Score] 并执行相对归一化

    数据格式:
    - face: (T, 20, 3) -> x, y, z (无 confidence)
    - pose: (T, 17, 4) -> [confidence, x, y, z]
    - left_hand: (T, 21, 7) -> [Score, Nx, Ny, Nz, Wx, Wy, Wz]
    - right_hand: (T, 21, 7) -> [Score, Nx, Ny, Nz, Wx, Wy, Wz]

    输出:
    - body: (T, 9, 4)
    - left: (T, 21, 4)
    - right: (T, 21, 4)
    - face_all: (T, 18, 4)
    """
    kps_with_scores = {}

    # ========== Body (pose) ==========
    # 从 17 个关键点中选择 9 个，与 Uni-Sign 保持一致
    # MediaPipe indices: 0=nose, 7=left_ear, 8=right_ear,
    #                    11=left_shoulder, 12=right_shoulder,
    #                    13=left_elbow, 14=right_elbow,
    #                    15=left_wrist, 16=right_wrist
    # 这里简化为取前9个点
    BODY_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 取前9个点

    if 'pose' in data_dict:
        pose_data = data_dict['pose'][indices]  # (T', 17, 4)
        # pose 格式: [confidence, x, y, z]
        body_coords = pose_data[:, BODY_INDICES, 1:4]  # (T, 9, 3) -> x, y, z
        body_conf = pose_data[:, BODY_INDICES, 0:1]    # (T, 9, 1) -> confidence

        # 处理 NaN
        body_coords = np.nan_to_num(body_coords, nan=0.0)
        body_conf = np.nan_to_num(body_conf, nan=0.0)

        # 相对归一化: 减去鼻子坐标 (第0个点)
        # 这样 body 也变成相对坐标，与 face/hand 处理方式一致
        nose = body_coords[:, 0:1, :]  # (T, 1, 3)
        relative_body_coords = body_coords - nose

        # 拼接: [Rx, Ry, Rz, confidence] -> (T, 9, 4)
        body_feat = np.concatenate([relative_body_coords, body_conf], axis=-1).astype(np.float32)
        kps_with_scores['body'] = body_feat

    # ========== Hands ==========
    # 遍历左右手
    for part_key, part_name in [('left_hand', 'left'), ('right_hand', 'right')]:
        # data_dict[part_key] 已经是清洗过(无NaN)的全量数据
        # 根据 indices 进行采样/截断
        full_data = data_dict[part_key][indices] # shape: (T_sample, 21, 7)

        # 提取列:
        # c1=Nx, c2=Ny, c3=Nz (Coords)
        # c0=Score (Confidence)
        # 原始数据列顺序: [Score, Nx, Ny, Nz, Wx, Wy, Wz]
        # 索引对应:        0      1   2   3   4   5   6

        coords = full_data[:, :, 1:4] # (T, 21, 3) -> Nx, Ny, Nz
        scores = full_data[:, :, 0:1] # (T, 21, 1) -> Score

        # 相对归一化: 减去手腕坐标 (第0个点)
        # 这一步消除了手在屏幕的绝对位置，保留了手势形状和相对深度
        wrist = coords[:, 0:1, :] # (T, 1, 3)

        # 广播减法 (如果本来是0, 0-0=0, 逻辑自洽)
        relative_coords = coords - wrist

        # 拼接: [Rx, Ry, Rz, Score] -> (T, 21, 4)
        feat_4d = np.concatenate([relative_coords, scores], axis=-1)

        kps_with_scores[part_name] = feat_4d

    # ========== Face ==========
    # face: (T, 20, 3) -> select 18 keypoints matching Uni-Sign format
    # Order: 9 eyebrow/edge (chain) + 8 mouth (ring) + 1 nose tip (center)
    # Face keypoint selection (collected index -> output index):
    #   Edge chain: [19, 13, 16, 17, 2, 10, 9, 6, 12]
    #   Mouth ring: [7, 8, 4, 15, 14, 18, 5, 11]
    #   Nose tip center: [0]
    FACE_EDGE_INDICES = [19, 13, 16, 17, 2, 10, 9, 6, 12]  # 9 points
    FACE_MOUTH_INDICES = [7, 8, 4, 15, 14, 18, 5, 11]       # 8 points
    FACE_NOSE_INDEX = [0]                                    # 1 point
    FACE_INDICES = FACE_EDGE_INDICES + FACE_MOUTH_INDICES + FACE_NOSE_INDEX  # 18 points total

    if 'face' in data_dict:
        face_data = data_dict['face'][indices]  # (T', 20, 3)
        face_coords = face_data[:, FACE_INDICES, :]  # (T, 18, 3) -> x, y, z

        # 处理 NaN
        face_coords = np.nan_to_num(face_coords, nan=0.0)

        # Face 没有 confidence，设置全为 1.0
        face_conf = np.ones((face_coords.shape[0], face_coords.shape[1], 1), dtype=np.float32)

        # 相对归一化: 减去鼻尖坐标 (最后一个点，即原始索引 0)
        # 这与 Uni-Sign 的处理方式一致
        nose_tip = face_coords[:, -1:, :]  # (T, 1, 3) 这个相对归一化可以尝试一下
        relative_face_coords = face_coords - nose_tip

        # 拼接: [Rx, Ry, Rz, confidence] -> (T, 18, 4)
        face_feat = np.concatenate([relative_face_coords, face_conf], axis=-1)
        kps_with_scores['face_all'] = face_feat

    return kps_with_scores

# ------------------------------------------------------------------------------
# 3. 数据集类
# ------------------------------------------------------------------------------
class S2T_Dataset(Dataset.Dataset):
    def __init__(self, path, args, phase):
        super(S2T_Dataset, self).__init__()
        self.args = args
        self.phase = phase
        self.max_length = args.max_length
        
        # 加载数据列表 (逻辑保持不变)
        def load_dataset_file(file_path):
            try:
                with gzip.open(file_path, 'rb') as f:
                    return pickle.load(f)
            except (OSError, pickle.UnpicklingError):
                try:
                    with open(file_path, 'rb') as f:
                        return pickle.load(f)
                except Exception:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            return json.load(f)
                    except Exception as e:
                        print(f"❌ Failed to load dataset file {file_path}: {e}")
                        return {}

        if isinstance(path, list):
            self.raw_data = {}
            for p in path:
                self.raw_data.update(load_dataset_file(p))
        else:
            self.raw_data = load_dataset_file(path)
            
        self.list = list(self.raw_data.keys())
        
        # 设置 pose_dir
        if self.args.dataset == "CSL_Daily":
            self.pose_dir = pose_dirs[args.dataset]
        else:
             self.pose_dir = pose_dirs.get(args.dataset, "")

        print(f"[{phase}] Loaded {len(self.list)} samples.")

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        # 增加重试机制，防止单个损坏文件中断训练
        for _ in range(3):
            try:
                key = self.list[index]
                sample = self.raw_data[key]
                text = sample.get('text', '')
                name_sample = sample.get('name', key)
                video_rel_path = sample.get('video_path', f"{name_sample}.mp4")
                
                # 加载骨骼数据 (核心修改在 load_pose 内部)
                pose_sample = self.load_pose(video_rel_path)
                
                return {
                    "name": name_sample, 
                    "pose": pose_sample, 
                    "text": text
                }
            except Exception as e:
                # print(f"⚠️ Error loading index {index} ({self.list[index]}): {e}. Retrying random sample...")
                # 生产环境建议注释掉 print 以免刷屏，或者保留以排查问题
                index = random.randint(0, len(self.list) - 1)
                continue
        
        raise RuntimeError("Failed to load sample after retries.")
    
    def load_pose(self, path):
        # 1. 路径构建 (适配 processed/视频名/data/keypoints.pkl 结构)
        video_name = Path(path).stem
        
        # 尝试寻找文件
        # 结构 A: Label_Corrected/processed/S000000_P0000_T00/data/keypoints.pkl (Step 2 输出的标准结构)
        pkl_path = Path(self.pose_dir) / "processed" / video_name / "data" / "keypoints.pkl"
        
        if not pkl_path.exists():
            # 结构 B: Label_Corrected/S000000_P0000_T00/data/keypoints.pkl (兼容备用)
            pkl_path = Path(self.pose_dir) / video_name / "data" / "keypoints.pkl"
            if not pkl_path.exists():
                raise FileNotFoundError(f"PKL not found: {pkl_path}")
        
        # 加载 Step 2 的数据 (含 NaN), 使用兼容性函数处理 numpy 版本差异
        data = load_pickle_compatible(str(pkl_path))
            
        # 2. [核心] 执行混合清洗策略
        # 必须在采样(Sampling)之前做，以保证时间连续性判断准确
        if 'left_hand' in data and 'right_hand' in data:
            data['left_hand'] = hybrid_fill_nan(data['left_hand'])
            data['right_hand'] = hybrid_fill_nan(data['right_hand'])
        else:
            raise ValueError(f"Invalid data format in {pkl_path}")

        # 清洗 pose 数据 (body)
        if 'pose' in data:
            data['pose'] = hybrid_fill_nan(data['pose'])

        # 清洗 face 数据
        if 'face' in data:
            # face 数据是 (T, 20, 3)，没有 confidence，使用 NaN 填充
            # 对于 face，简单的 NaN 替换为 0.0
            data['face'] = np.nan_to_num(data['face'], nan=0.0).astype(np.float32)
            
        # 3. 采样与截断 (逻辑保持原作者不变)
        duration = len(data['left_hand'])
        
        if duration > self.max_length:
            if self.phase == 'train':
                tmp = sorted(random.sample(range(duration), k=self.max_length))
            else:
                tmp = list(range(duration))[:self.max_length]
        else:
            tmp = list(range(duration))
            
        # 4. 调用新版提取函数 (提取 4D 相对坐标)
        # 传入清洗后的字典和采样索引
        kps_with_scores = load_part_kp_4d(data, tmp)
        
        return kps_with_scores

    def collate_fn(self, batch):
        src_input = {}
        tgt_input = {}

        # 1. 处理 Pose Data (包括 body, left, right, face_all)
        for mode in ['body', 'left', 'right', 'face_all']:
            # 检查该 key 是否存在于所有样本中
            if mode in batch[0]['pose']:
                features = [torch.from_numpy(b['pose'][mode]).float() for b in batch]
                padded = pad_sequence(features, batch_first=True, padding_value=0.0)
                src_input[mode] = padded

        # 2. 处理 Attention Mask
        lengths = [len(b['pose']['left']) for b in batch]
        max_len = max(lengths)
        bs = len(batch)

        attention_mask = torch.zeros((bs, max_len), dtype=torch.long)
        for i, l in enumerate(lengths):
            attention_mask[i, :l] = 1
        src_input['attention_mask'] = attention_mask
        
        # 3. 处理文本
        from config import mt5_path
        global _SHARED_TOKENIZER
        if '_SHARED_TOKENIZER' not in globals():
            _SHARED_TOKENIZER = T5Tokenizer.from_pretrained(mt5_path, legacy=False)
        tokenizer = _SHARED_TOKENIZER
        
        texts = [b['text'] for b in batch]
        
        prefix = "translate sign to text: "
        prefix_inputs = tokenizer([prefix] * bs, return_tensors="pt", padding=True)
        src_input['prefix_ids'] = prefix_inputs.input_ids
        src_input['prefix_mask'] = prefix_inputs.attention_mask
        
        # [修改] 使用 text_target 参数，修复 transformers 的 DeprecationWarning
        labels = tokenizer(
            text_target=texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        )
            
        tgt_input['labels_ids'] = labels.input_ids
        tgt_input['gt_sentence'] = texts 
        
        return src_input, tgt_input