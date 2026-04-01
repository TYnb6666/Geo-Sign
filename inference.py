#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Geo-Sign Inference Script
使用多个模型检查点进行推理并输出预测语句

用法:
    python inference.py --checkpoint_paths <path1> <path2> <path3> \
                        --modes hand hand_body hand_body_face \
                        --output_dir ./inference_results

Mode 说明:
    - hand: 仅使用手部数据 (left, right)
    - hand_body: 使用手部+身体数据 (left, right, body)
    - hand_body_face: 使用全部数据 (left, right, body, face_all)
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import T5Tokenizer

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import Uni_Sign
from datasets import load_pickle_compatible, hybrid_fill_nan, load_part_kp_4d
from config import mt5_path, test_label_paths, pose_dirs


# Mode 到数据键的映射 (与 models.py 和 datasets.py 保持一致)
MODE_TO_KEYS = {
    "hand": ["left", "right"],
    "hand_body": ["body", "left", "right"],
    "hand_body_face": ["body", "left", "right", "face_all"]
}

# Mode 对应模型初始化时的 modes 列表
# 注意：模型内部顺序需要与 models.py 中保持一致
MODE_TO_MODEL_MODES = {
    "hand": ["left", "right"],
    "hand_body": ["body", "left", "right"],
    "hand_body_face": ["body", "left", "right", "face_all"]
}

VALID_MODES = list(MODE_TO_KEYS.keys())


def get_inference_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser('Geo-Sign Inference Script')

    # 模型检查点路径 (支持多个)
    parser.add_argument('--checkpoint_paths', nargs='+', type=str, required=True,
                        help='模型检查点路径列表')
    parser.add_argument('--checkpoint_names', nargs='+', type=str, default=None,
                        help='模型名称列表 (用于结果展示)')
    parser.add_argument('--modes', nargs='+', type=str, required=True,
                        help=f'每个模型对应的数据模式: {VALID_MODES}')

    # 数据参数
    parser.add_argument('--dataset', default='CSL_Daily', type=str,
                        help='数据集名称')
    parser.add_argument('--data_path', default=None, type=str,
                        help='自定义测试数据路径 (可选)')
    parser.add_argument('--max_length', default=256, type=int,
                        help='最大帧长度')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='批次大小')
    parser.add_argument('--num_samples', default=-1, type=int,
                        help='推理样本数量 (-1 表示全部)')

    # 输出参数
    parser.add_argument('--output_dir', default='./inference_results', type=str,
                        help='输出目录')
    parser.add_argument('--save_predictions', action='store_true',
                        help='保存预测结果到文件')

    # 生成参数
    parser.add_argument('--num_beams', default=4, type=int,
                        help='Beam search 数量')
    parser.add_argument('--max_new_tokens', default=100, type=int,
                        help='最大生成token数')

    # 设备参数
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='设备')
    parser.add_argument('--input_channels', default=4, type=int,
                        help='输入通道数')

    args = parser.parse_args()

    # 验证 modes 参数
    for mode in args.modes:
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {VALID_MODES}")

    # 验证 checkpoint_paths 和 modes 数量匹配
    if len(args.checkpoint_paths) != len(args.modes):
        raise ValueError(
            f"Number of checkpoint_paths ({len(args.checkpoint_paths)}) "
            f"must match number of modes ({len(args.modes)})"
        )

    return args


class InferenceDataset(torch.utils.data.Dataset):
    """推理用数据集 - 加载所有数据，根据mode筛选"""

    def __init__(self, args, data_path=None):
        self.args = args
        self.max_length = args.max_length

        # 加载数据
        if data_path:
            label_path = data_path
        else:
            label_path = test_label_paths[args.dataset]

        # 加载标签文件
        import pickle
        import gzip
        import json

        try:
            with gzip.open(label_path, 'rb') as f:
                self.raw_data = pickle.load(f)
        except:
            try:
                with open(label_path, 'rb') as f:
                    self.raw_data = pickle.load(f)
            except:
                with open(label_path, 'r', encoding='utf-8') as f:
                    self.raw_data = json.load(f)

        self.list = list(self.raw_data.keys())

        # 设置 pose_dir
        self.pose_dir = pose_dirs.get(args.dataset, "")

        # 限制样本数量
        if args.num_samples > 0:
            self.list = self.list[:args.num_samples]

        print(f"[Inference] Loaded {len(self.list)} samples.")

        # 初始化 tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(mt5_path, legacy=False)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        key = self.list[index]
        sample = self.raw_data[key]
        text = sample.get('text', '')
        name_sample = sample.get('name', key)
        video_rel_path = sample.get('video_path', f"{name_sample}.mp4")

        # 加载骨骼数据 (加载所有数据)
        pose_sample = self.load_pose(video_rel_path)

        return {
            "name": name_sample,
            "pose": pose_sample,
            "text": text
        }

    def load_pose(self, path):
        """加载骨骼数据 - 加载所有可用的数据"""
        video_name = Path(path).stem

        # 构建路径
        pkl_path = Path(self.pose_dir) / "processed" / video_name / "data" / "keypoints.pkl"

        if not pkl_path.exists():
            pkl_path = Path(self.pose_dir) / video_name / "data" / "keypoints.pkl"
            if not pkl_path.exists():
                raise FileNotFoundError(f"PKL not found: {pkl_path}")

        # 加载数据
        data = load_pickle_compatible(str(pkl_path))

        # 清洗数据
        if 'left_hand' in data and 'right_hand' in data:
            data['left_hand'] = hybrid_fill_nan(data['left_hand'])
            data['right_hand'] = hybrid_fill_nan(data['right_hand'])
        else:
            raise ValueError(f"Invalid data format in {pkl_path}")

        if 'pose' in data:
            data['pose'] = hybrid_fill_nan(data['pose'])

        if 'face' in data:
            data['face'] = np.nan_to_num(data['face'], nan=0.0).astype(np.float32)

        # 采样
        duration = len(data['left_hand'])
        if duration > self.max_length:
            tmp = list(range(duration))[:self.max_length]
        else:
            tmp = list(range(duration))

        # 提取 4D 特征 (提取所有数据)
        kps_with_scores = load_part_kp_4d(data, tmp)

        return kps_with_scores

    def collate_fn(self, batch):
        """批处理函数 - 返回所有数据，推理时根据mode筛选"""
        src_input = {}
        tgt_input = {}

        # 处理所有 Pose Data (body, left, right, face_all)
        for mode in ['body', 'left', 'right', 'face_all']:
            if mode in batch[0]['pose']:
                features = [torch.from_numpy(b['pose'][mode]).float() for b in batch]
                padded = pad_sequence(features, batch_first=True, padding_value=0.0)
                src_input[mode] = padded

        # 处理 Attention Mask
        lengths = [len(b['pose']['left']) for b in batch]
        max_len = max(lengths)
        bs = len(batch)

        attention_mask = torch.zeros((bs, max_len), dtype=torch.long)
        for i, l in enumerate(lengths):
            attention_mask[i, :l] = 1
        src_input['attention_mask'] = attention_mask

        # 处理文本 prefix
        prefix = "translate sign to text: "
        prefix_inputs = self.tokenizer([prefix] * bs, return_tensors="pt", padding=True)
        src_input['prefix_ids'] = prefix_inputs.input_ids
        src_input['prefix_mask'] = prefix_inputs.attention_mask

        # 保存元信息
        tgt_input['gt_sentence'] = [b['text'] for b in batch]
        tgt_input['names'] = [b['name'] for b in batch]

        return src_input, tgt_input


def create_model_with_mode(args, mode: str) -> Uni_Sign:
    """根据 mode 创建模型，动态设置 modes"""
    from stgcn_layers import Graph, get_stgcn_chain
    import torch.nn as nn
    import math
    import warnings

    # 获取该 mode 对应的模型 modes
    model_modes = MODE_TO_MODEL_MODES[mode]

    # 创建一个简单的 args 副本，用于模型初始化
    class ModelArgs:
        pass

    model_args = ModelArgs()
    for k, v in vars(args).items():
        setattr(model_args, k, v)

    # 动态创建模型
    class DynamicUni_Sign(nn.Module):
        def __init__(self, args, modes):
            super().__init__()
            self.args = args
            self.modes = modes

            initial_gcn_dim = 64
            initial_input_dim = getattr(args, 'input_channels', 4)

            print(f"✅ Dynamic Uni-Sign Model Initialized with Input Channels: {initial_input_dim}")
            print(f"✅ Modes: {self.modes}")

            # 1. Build Graph & Projection Layers
            self.graph, As, self.proj_linear = {}, [], nn.ModuleDict()
            for m in self.modes:
                g = Graph(layout=m, strategy="distance", max_hop=1)
                self.graph[m] = g
                As.append(torch.tensor(g.A, dtype=torch.float32, requires_grad=False))
                self.proj_linear[m] = nn.Linear(initial_input_dim, initial_gcn_dim)

            # 2. Build GCN Modules
            self.gcn_modules = nn.ModuleDict()
            self.fusion_gcn_modules = nn.ModuleDict()

            final_dim_gcn = -1
            for i, m in enumerate(self.modes):
                current_spatial_k = As[i].shape[0]
                gcn, d_mid = get_stgcn_chain(initial_gcn_dim, "spatial", (1, current_spatial_k), As[i].clone(), True)
                fus, d_out = get_stgcn_chain(d_mid, "temporal", (5, current_spatial_k), As[i].clone(), True)
                if i == 0:
                    final_dim_gcn = d_out
                self.gcn_modules[m] = gcn
                self.fusion_gcn_modules[m] = fus

            # Share weights for left/right hands
            if "right" in self.modes and "left" in self.modes:
                self.gcn_modules["left"] = self.gcn_modules["right"]
                self.fusion_gcn_modules["left"] = self.fusion_gcn_modules["right"]
                self.proj_linear["left"] = self.proj_linear["right"]

            # 3. Projection to mT5
            concat_dim = final_dim_gcn * len(self.modes)
            self.part_para = nn.Parameter(torch.zeros(concat_dim))

            from transformers import MT5ForConditionalGeneration, T5Tokenizer
            mt5_cfg = MT5ForConditionalGeneration.from_pretrained(mt5_path).config
            self.mt5_model = MT5ForConditionalGeneration.from_pretrained(mt5_path)
            self.mt5_tokenizer = T5Tokenizer.from_pretrained(mt5_path, legacy=False)
            self.mt5_dim = mt5_cfg.d_model
            self.pose_proj = nn.Linear(concat_dim, self.mt5_dim)

            self.apply(self._init_weights)

        @staticmethod
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        @torch.no_grad()
        def generate(self, pc, *, max_new_tokens=100, num_beams=4, **kwargs):
            compute_dtype = self.pose_proj.weight.dtype
            feats = []
            active_modes = [m for m in self.modes if m in pc]

            body_feat = None
            for part in active_modes:
                x = self.proj_linear[part](pc[part].to(dtype=compute_dtype)).permute(0, 3, 1, 2)
                gcn_out = self.gcn_modules[part](x)

                if part == 'body':
                    body_feat = gcn_out
                else:
                    if body_feat is not None:
                        if part == 'left':
                            gcn_out = gcn_out + body_feat[..., 7][..., None].detach()
                        elif part == 'right':
                            gcn_out = gcn_out + body_feat[..., 8][..., None].detach()
                        elif part == 'face_all':
                            gcn_out = gcn_out + body_feat[..., 0][..., None].detach()

                gcn_out = self.fusion_gcn_modules[part](gcn_out)
                pool_sp = gcn_out.mean(dim=-1).transpose(1, 2)
                feats.append(pool_sp)

            concatenated_feats = torch.cat(feats, dim=-1)
            pose_features_biased = concatenated_feats
            if len(active_modes) == len(self.modes):
                pose_features_biased += self.part_para

            pose_emb = self.pose_proj(pose_features_biased)
            prefix_ids = pc["prefix_ids"].long()
            prefix_mask = pc["prefix_mask"]

            inputs_embeds = torch.cat([self.mt5_model.encoder.embed_tokens(prefix_ids), pose_emb], dim=1)
            attention_mask = torch.cat([prefix_mask, pc["attention_mask"]], dim=1)

            return self.mt5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                **kwargs
            )

    return DynamicUni_Sign(model_args, model_modes)


def load_model(checkpoint_path: str, args, device: torch.device, mode: str):
    """加载模型"""
    print(f"\nLoading model from: {checkpoint_path}")
    print(f"  Mode: {mode}")

    # 根据模式创建模型
    model = create_model_with_mode(args, mode)

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model', checkpoint)

    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a valid state_dict.")

    # 加载权重
    ret = model.load_state_dict(state_dict, strict=False)
    if ret.missing_keys:
        # 过滤掉与当前mode无关的missing keys
        relevant_missing = [k for k in ret.missing_keys if any(k.startswith(p) for p in MODE_TO_KEYS[mode])]
        if relevant_missing:
            print(f"  Warning - Missing relevant keys: {relevant_missing[:5]}...")
    if ret.unexpected_keys:
        print(f"  Unexpected keys: {ret.unexpected_keys[:5]}...")

    model.to(device)
    model.eval()

    print(f"  Model loaded successfully!")
    return model


@torch.no_grad()
def inference_single_model(
    model: Uni_Sign,
    dataloader: DataLoader,
    device: torch.device,
    args,
    mode: str
) -> List[Dict[str, Any]]:
    """对单个模型进行推理"""
    results = []
    tokenizer = model.mt5_tokenizer

    # 获取当前mode需要的keys
    active_keys = MODE_TO_KEYS[mode]

    print(f"  Active keys for mode '{mode}': {active_keys}")

    for step, (src_input, tgt_input) in enumerate(dataloader):
        # 移动数据到设备
        for key in src_input:
            if isinstance(src_input[key], torch.Tensor):
                src_input[key] = src_input[key].to(device, non_blocking=True)

        # 根据 mode 准备生成输入
        generation_input = {
            "prefix_ids": src_input["prefix_ids"],
            "prefix_mask": src_input["prefix_mask"],
            "attention_mask": src_input["attention_mask"]
        }

        # 只添加当前mode需要的keys
        for key in active_keys:
            if key in src_input:
                generation_input[key] = src_input[key]

        # 生成
        output_ids = model.generate(
            pc=generation_input,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams
        )

        # 解码
        predictions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        # 保存结果
        for i, (pred, gt, name) in enumerate(zip(predictions, tgt_input['gt_sentence'], tgt_input['names'])):
            results.append({
                "name": name,
                "prediction": pred,
                "ground_truth": gt
            })

        if (step + 1) % 10 == 0:
            print(f"  Processed {step + 1}/{len(dataloader)} batches")

    return results


def print_comparison_results(
    all_results: Dict[str, List[Dict]],
    model_modes: Dict[str, str],
    num_display: int = 10
):
    """打印对比结果"""
    print("\n" + "=" * 80)
    print("推理结果对比")
    print("=" * 80)

    model_names = list(all_results.keys())
    num_samples = len(list(all_results.values())[0])

    for i in range(min(num_display, num_samples)):
        print(f"\n--- 样本 {i+1} ---")

        # 打印 ground truth
        gt = all_results[model_names[0]][i]['ground_truth']
        name = all_results[model_names[0]][i]['name']
        print(f"样本名称: {name}")
        print(f"真实标注: {gt}")
        print("-" * 40)

        # 打印各模型预测
        for model_name in model_names:
            pred = all_results[model_name][i]['prediction']
            mode = model_modes.get(model_name, "unknown")
            print(f"[{mode}] {model_name}: {pred}")


def save_results(
    all_results: Dict[str, List[Dict]],
    model_modes: Dict[str, str],
    output_dir: str
):
    """保存结果到文件"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_names = list(all_results.keys())

    # 保存每个模型的结果
    for model_name in model_names:
        results = all_results[model_name]
        safe_name = model_name.replace("/", "_").replace("\\", "_")
        mode = model_modes.get(model_name, "unknown")

        # 保存为文本文件
        txt_path = output_path / f"{safe_name}_{mode}_predictions.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Mode: {mode}\n")
            f.write("=" * 40 + "\n")
            for r in results:
                f.write(f"Name: {r['name']}\n")
                f.write(f"GT: {r['ground_truth']}\n")
                f.write(f"Pred: {r['prediction']}\n")
                f.write("-" * 40 + "\n")
        print(f"Saved: {txt_path}")

    # 保存对比结果
    comparison_path = output_path / "comparison.txt"
    with open(comparison_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("模型对比结果\n")
        f.write("=" * 80 + "\n\n")

        # 写入模型信息
        for model_name in model_names:
            mode = model_modes.get(model_name, "unknown")
            f.write(f"{model_name}: mode={mode}\n")
        f.write("\n")

        num_samples = len(list(all_results.values())[0])

        for i in range(num_samples):
            f.write(f"\n--- 样本 {i+1} ---\n")
            gt = all_results[model_names[0]][i]['ground_truth']
            name = all_results[model_names[0]][i]['name']
            f.write(f"样本名称: {name}\n")
            f.write(f"真实标注: {gt}\n")
            f.write("-" * 40 + "\n")

            for model_name in model_names:
                pred = all_results[model_name][i]['prediction']
                mode = model_modes.get(model_name, "unknown")
                f.write(f"[{mode}] {model_name}: {pred}\n")

    print(f"\nComparison saved to: {comparison_path}")


def main():
    args = get_inference_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 处理检查点名称
    checkpoint_paths = args.checkpoint_paths
    modes = args.modes

    if args.checkpoint_names:
        checkpoint_names = args.checkpoint_names
    else:
        # 使用文件名作为模型名称
        checkpoint_names = [Path(p).parent.name + "_" + Path(p).stem for p in checkpoint_paths]

    # 验证数量匹配
    if len(checkpoint_names) != len(checkpoint_paths):
        raise ValueError("checkpoint_names数量必须与checkpoint_paths相同")

    # 创建数据集和数据加载器
    print("\n创建数据集...")
    dataset = InferenceDataset(args, args.data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=dataset.collate_fn,
        sampler=SequentialSampler(dataset)
    )

    # 存储所有模型的结果和对应的mode
    all_results = {}
    model_modes = {}

    # 对每个模型进行推理
    for ckpt_path, model_name, mode in zip(checkpoint_paths, checkpoint_names, modes):
        print(f"\n{'='*60}")
        print(f"模型: {model_name}")
        print(f"检查点: {ckpt_path}")
        print(f"数据模式: {mode}")
        print(f"{'='*60}")

        # 加载模型
        model = load_model(ckpt_path, args, device, mode)

        # 推理
        results = inference_single_model(model, dataloader, device, args, mode)
        all_results[model_name] = results
        model_modes[model_name] = mode

        # 打印该模型的统计信息
        print(f"\n完成推理，共 {len(results)} 个样本")

    # 打印对比结果
    print_comparison_results(all_results, model_modes, num_display=20)

    # 保存结果
    if args.save_predictions:
        save_results(all_results, model_modes, args.output_dir)

    print("\n推理完成！")


if __name__ == '__main__':
    main()
