import torch
import numpy as np
import os
import sys
from torch.utils.data import DataLoader

# 引入项目模块
from datasets import S2T_Dataset
from models import Uni_Sign
from config import dev_label_paths

# ==========================================
# 1. 模拟参数 (Mock Arguments)
# ==========================================
class DummyArgs:
    def __init__(self):
        self.dataset = 'CSL_Daily'
        self.max_length = 250   # 模拟序列长度
        self.rgb_support = False
        self.label_smoothing = 0.1
        # 其他可能需要的参数
        self.input_channels = 3 # x, y, conf

args = DummyArgs()

def test_data_pipeline():
    print("\n" + "="*40)
    print("🧪 测试 1: 数据加载与归一化检查")
    print("="*40)

    # 1. 加载数据集
    # 使用 dev 集，因为它通常比 train 小，加载快
    label_path = dev_label_paths[args.dataset]
    if not os.path.exists(label_path):
        print(f"❌ 错误: 找不到标签文件 {label_path}")
        return None

    dataset = S2T_Dataset(path=label_path, args=args, phase='dev')
    print(f"✅ 数据集加载成功，样本数: {len(dataset)}")

    # 2. 获取一个样本
    sample = dataset[0]
    pose = sample['pose'] # 应该是一个 dict: {'left': ..., 'right': ...}

    # 3. 检查键值
    if 'body' in pose or 'face_all' in pose:
        print("❌ 失败: 数据中仍包含 body 或 face_all，请检查 datasets.py")
        return None
    if 'left' not in pose or 'right' not in pose:
        print("❌ 失败: 数据中缺少 left 或 right")
        return None
    
    print("✅ 键值检查通过: 仅包含 ['left', 'right']")

    # 4. 检查归一化 (核心!)
    # 检查左手第一帧的手腕坐标 (索引0) 是否为 0
    left_hand = pose['left'] # shape: (T, 21, 3)
    wrist_coord = left_hand[0, 0, :2] # 取第一帧，第0个点，x和y
    
    print(f"🔍 左手手腕坐标 (应接近 0): {wrist_coord}")
    
    if np.allclose(wrist_coord, 0, atol=1e-5):
        print("✅ 归一化检查通过: 手腕已对齐到原点 (0,0)")
    else:
        print("❌ 失败: 手腕坐标不是 (0,0)，datasets.py 归一化逻辑有误！")
        
    return dataset

def test_model_forward(dataset):
    print("\n" + "="*40)
    print("🧪 测试 2: 模型构建与前向传播")
    print("="*40)

    # 1. 创建 DataLoader (测试 collate_fn)
    loader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)
    batch = next(iter(loader))
    src_input, tgt_input = batch

    print(f"📦 Batch Shapes:")
    for key, val in src_input.items():
        if isinstance(val, torch.Tensor):
            print(f"  - {key}: {val.shape}")
    
    # 检查输入维度是否为 (Batch, Time, 21, 3)
    if src_input['left'].shape[-2:] != (21, 3):
        print(f"❌ 失败: 左手张量形状错误，期望 (..., 21, 3)，实际 {src_input['left'].shape}")
        return

    # 2. 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️  正在 {device} 上初始化模型...")
    
    try:
        model = Uni_Sign(args).to(device)
        print("✅ 模型初始化成功 (Body分支已移除)")
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return

    # 3. 数据上送设备
    for k, v in src_input.items():
        if isinstance(v, torch.Tensor): src_input[k] = v.to(device)
    for k, v in tgt_input.items():
        if isinstance(v, torch.Tensor): tgt_input[k] = v.to(device)

    # 4. 前向传播 (Forward)
    try:
        output = model(src_input, tgt_input)
        loss = output['loss']
        print(f"✅ 前向传播成功! Loss: {loss.item():.4f}")
    except RuntimeError as e:
        print(f"❌ 前向传播崩溃: {e}")
        print("💡 提示: 检查 models.py 中的 forward 函数，是否还有 body_feat 的残留代码？")
        return

    # 5. 反向传播测试 (Backward)
    try:
        loss.backward()
        print("✅ 反向传播成功! 梯度计算正常。")
    except Exception as e:
        print(f"❌ 反向传播失败: {e}")

    # 6. 生成测试 (Generate)
    print("\n🧪 测试 3: 推理生成 (Generate)")
    try:
        # 构造推理输入
        inference_input = {
            "inputs_embeds": output["inputs_embeds"], # 复用 forward 产生的 embedding
            "attention_mask": src_input["attention_mask"]
        }
        # 或者直接用原始输入测试 generate 函数的预处理逻辑
        generated = model.generate(src_input, max_new_tokens=10)
        print(f"✅ 生成成功! 输出 Shape: {generated.shape}")
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        print("💡 提示: 检查 models.py 的 generate 函数是否去除了 body 处理逻辑？")

if __name__ == "__main__":
    # 运行测试
    ds = test_data_pipeline()
    if ds is not None:
        test_model_forward(ds)
    else:
        print("⚠️ 跳过模型测试，因为数据加载失败。")