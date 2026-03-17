import os
import torch
import argparse
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import T5Tokenizer

# --- 引入项目模块 ---
# 确保 config.py, datasets.py, models.py, SLRT_metrics.py 在同一目录下
from config import mt5_path, test_label_paths
from datasets import S2T_Dataset
from models import Uni_Sign
from SLRT_metrics import translation_performance

# ==========================================
# 1. 辅助函数 (不依赖外部 utils 以防报错)
# ==========================================
def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    return batch

def get_args():
    parser = argparse.ArgumentParser(description="Standalone Inference Script")
    
    # 核心路径
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best_checkpoint.pth')
    parser.add_argument('--dataset', default='CSL_Daily', type=str)
    parser.add_argument('--output_file', default='test_predictions.txt', type=str)
    
    # 模型参数 (必须与训练一致)
    parser.add_argument('--input_channels', default=2, type=int, help='2 for xy, 3 for xyz')
    parser.add_argument('--hidden_dim', default=768, type=int)
    parser.add_argument('--gcn_out_dim', default=256, type=int)
    
    # 推理参数
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_beams', default=4, type=int)
    parser.add_argument('--max_new_tokens', default=100, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    
    # 占位参数 (Uni_Sign 初始化必须需要，但推理不用)
    parser.add_argument('--task', default='SLT', type=str)
    parser.add_argument('--rgb_support', action='store_true')
    parser.add_argument('--hyp_dim', default=256, type=int)
    parser.add_argument('--use_hyperbolic', action='store_true')
    parser.add_argument('--init_c', default=1.5, type=float)
    parser.add_argument('--hyp_text_emb_src', default='token', type=str)
    parser.add_argument('--hyp_text_cmp', default='token', type=str)
    parser.add_argument('--drop_last', action='store_true')
    parser.add_argument('--max_length', default=300, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🚀 Start testing on {device}...")

    # ------------------------------------------------------------------
    # 1. 加载 Tokenizer
    # ------------------------------------------------------------------
    print(f"📚 Loading Tokenizer from {mt5_path}...")
    tokenizer = T5Tokenizer.from_pretrained(mt5_path, legacy=False)

    # ------------------------------------------------------------------
    # 2. 初始化模型 & 加载权重
    # ------------------------------------------------------------------
    print(f"🏗️  Building Model (Input Channels: {args.input_channels})...")
    model = Uni_Sign(args)
    
    print(f"📥 Loading Checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # 自动处理 DeepSpeed 保存的 'module.' 前缀
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    # 加载参数 (strict=False 允许忽略一些无关参数)
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"⚠️  Missing keys (safe if unused): {len(missing)}")
    
    model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # 3. 准备数据
    # ------------------------------------------------------------------
    print(f"📂 Loading Test Data: {args.dataset}...")
    test_path = test_label_paths[args.dataset]
    # 实例化数据集
    test_set = S2T_Dataset(path=test_path, args=args, phase='test')
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=test_set.collate_fn
    )
    print(f"✅ Loaded {len(test_set)} samples.")

    # ------------------------------------------------------------------
    # 4. 推理循环
    # ------------------------------------------------------------------
    print("⚡ Running Inference...")
    hypotheses = [] # 预测
    references = [] # 答案

    with torch.no_grad():
        for batch in tqdm(test_loader):
            src_input, tgt_input = batch
            src_input = move_to_device(src_input, device)
            
            # 生成
            generated_out = model.generate(
                src_input, 
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams
            )
            
            # 解码预测结果
            batch_preds = tokenizer.batch_decode(generated_out, skip_special_tokens=True)
            
            # 解码标准答案 (处理 Label 中的 -100)
            gt_ids = tgt_input['labels_ids']
            gt_ids = torch.where(gt_ids != -100, gt_ids, tokenizer.pad_token_id)
            batch_gts = tokenizer.batch_decode(gt_ids, skip_special_tokens=True)
            
            hypotheses.extend(batch_preds)
            references.extend(batch_gts)

    # ------------------------------------------------------------------
    # 5. 计算指标 & 保存
    # ------------------------------------------------------------------
    print("\n📊 Calculating Metrics...")
    
    # 预处理文本 (去空格，因为中文计算BLEU通常按字)
    # CSL 数据集的处理习惯：预测要去空格，参考要去空格且统一标点
    hyp_clean = [' '.join(list(h.replace(" ", "").replace("\n", ""))) for h in hypotheses]
    ref_clean = [' '.join(list(r.replace(" ", "").replace("\n", "").replace("，", ",").replace("？", "?"))) for r in references]

    # 调用 SLRT_metrics 计算
    try:
        bleu_dict, rouge_score = translation_performance(ref_clean, hyp_clean)
        print("\n" + "="*40)
        print(f"🏆 Final Results on {args.dataset}")
        print("="*40)
        print(f"BLEU-1: {bleu_dict['bleu1']:.2f}")
        print(f"BLEU-2: {bleu_dict['bleu2']:.2f}")
        print(f"BLEU-3: {bleu_dict['bleu3']:.2f}")
        print(f"BLEU-4: {bleu_dict['bleu4']:.2f}")
        print(f"ROUGE : {rouge_score:.2f}")
        print("="*40 + "\n")
    except Exception as e:
        print(f"⚠️ Metrics calculation failed: {e}")
        print("Saving raw text anyway...")

    # 保存结果到文件
    with open(args.output_file, "w", encoding="utf-8") as f:
        for p, g in zip(hypotheses, references):
            f.write(f"PRED: {p}\n")
            f.write(f"GT:   {g}\n")
            f.write("-"*30 + "\n")
            
    print(f"💾 Translations saved to: {args.output_file}")

if __name__ == "__main__":
    main()