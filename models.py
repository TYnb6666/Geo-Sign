from __future__ import annotations
import contextlib, math, warnings
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MT5ForConditionalGeneration, T5Tokenizer

# ---------- project-specific --------------------------------------------------
from stgcn_layers import Graph, get_stgcn_chain
from config import mt5_path

# ============================================================================ #
#  Helper: truncated normal initialiser
# ============================================================================ #
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in trunc_normal_", stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1).erfinv_()
        tensor.mul_(std * math.sqrt(2.0)).add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

# ============================================================================ #
#  Uni-Sign Model (With Body Support)
# ============================================================================ #
class Uni_Sign(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.text_decoder = getattr(args, "text_decoder", "mt5")

        # 支持 body, left, right, face_all 四种模式
        self.modes = ["body", "left", "right", "face_all"]

        initial_gcn_dim = 64

        # [关键修改] 动态读取 input_channels
        initial_input_dim = getattr(args, 'input_channels', 4)  # 默认 4D: [x, y, z, conf]

        print(f"✅ Uni-Sign Model Initialized with Input Channels: {initial_input_dim}")
        print(f"✅ Modes: {self.modes}")

        # 1. Build Graph & Projection Layers
        self.graph, As, self.proj_linear = {}, [], nn.ModuleDict()
        for m in self.modes:
            g = Graph(layout=m, strategy="distance", max_hop=1)
            self.graph[m] = g
            As.append(torch.tensor(g.A, dtype=torch.float32, requires_grad=False))
            self.proj_linear[m] = nn.Linear(initial_input_dim, initial_gcn_dim)

        # 2. Build GCN Modules (Independent Streams)
        self.gcn_modules        = nn.ModuleDict()
        self.fusion_gcn_modules = nn.ModuleDict()

        final_dim_gcn = -1
        for i, m in enumerate(self.modes):
            current_spatial_k = As[i].shape[0]  # body: 9, left/right: 21
            gcn, d_mid = get_stgcn_chain(initial_gcn_dim, "spatial", (1, current_spatial_k), As[i].clone(), True)
            fus, d_out = get_stgcn_chain(d_mid, "temporal", (5, current_spatial_k), As[i].clone(), True)
            if i == 0: final_dim_gcn = d_out
            self.gcn_modules[m]        = gcn
            self.fusion_gcn_modules[m] = fus

        # Share weights for left/right hands (but NOT body)
        if "right" in self.modes and "left" in self.modes:
            self.gcn_modules["left"]        = self.gcn_modules["right"]
            self.fusion_gcn_modules["left"] = self.fusion_gcn_modules["right"]
            self.proj_linear["left"]        = self.proj_linear["right"]

        # 3. Projection to mT5
        concat_dim    = final_dim_gcn * len(self.modes)
        self.part_para = nn.Parameter(torch.zeros(concat_dim))
        self.mt5_tokenizer = T5Tokenizer.from_pretrained(mt5_path, legacy=False)
        self.vocab_size = self.mt5_tokenizer.vocab_size

        if self.text_decoder == "mt5":
            mt5_cfg = MT5ForConditionalGeneration.from_pretrained(mt5_path).config
            self.mt5_model = MT5ForConditionalGeneration.from_pretrained(mt5_path)
            self.mt5_dim = mt5_cfg.d_model
            self.pose_proj = nn.Linear(concat_dim, self.mt5_dim)
        elif self.text_decoder == "transformer":
            self.mt5_model = None
            self.mt5_dim = int(getattr(args, "hidden_dim", 768))
            self.pose_proj = nn.Linear(concat_dim, self.mt5_dim)
            self.prefix_embed = nn.Embedding(self.vocab_size, self.mt5_dim)
            self.tgt_embed = nn.Embedding(self.vocab_size, self.mt5_dim)
            self.pos_embed = nn.Embedding(2048, self.mt5_dim)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.mt5_dim,
                nhead=8,
                dim_feedforward=self.mt5_dim * 4,
                dropout=0.1,
                batch_first=True,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
            self.lm_head = nn.Linear(self.mt5_dim, self.vocab_size)
        else:
            raise ValueError(f"Unsupported text_decoder: {self.text_decoder}")

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, src_input: Dict, tgt_input: Dict) -> Dict[str, torch.Tensor]:
        if self.mt5_tokenizer is None:
            raise RuntimeError("Tokenizer not loaded.")

        out = {}
        compute_dtype = self.pose_proj.weight.dtype
        autocast_ctx = contextlib.nullcontext()

        # ========== 1. Pose Encoding (Visual Encoder) ===================
        with autocast_ctx:
            feats = []
            active_modes = [m for m in self.modes if m in src_input]
            if not active_modes: raise ValueError("src_input contains no data for any defined modes.")

            # 先处理 body，保存 body_feat 用于后续手部特征的相对位置编码
            body_feat = None
            for part in active_modes:
                # (Batch, T, Nodes, C) -> Linear -> Permute -> (Batch, Hidden, T, Nodes)
                x = self.proj_linear[part](src_input[part].to(dtype=compute_dtype)).permute(0,3,1,2)

                # Spatial GCN
                gcn_out = self.gcn_modules[part](x)

                # 保存 body 特征用于相对位置编码
                if part == 'body':
                    body_feat = gcn_out
                else:
                    # 手部/面部特征加上对应 body 关键点的相对位置
                    # body 关键点索引: 0:nose, 1:左耳, 2:右耳, 3:左肩, 4:右肩,
                    #                 5:左肘, 6:右肘, 7:左腕, 8:右腕
                    if body_feat is not None:
                        if part == 'left':
                            # 左手使用左腕 (index 7)
                            gcn_out = gcn_out + body_feat[..., 7][..., None].detach()
                        elif part == 'right':
                            # 右手使用右腕 (index 8)
                            gcn_out = gcn_out + body_feat[..., 8][..., None].detach()
                        elif part == 'face_all':
                            # 面部使用鼻子 (index 0)
                            gcn_out = gcn_out + body_feat[..., 0][..., None].detach()

                # Temporal GCN
                gcn_out = self.fusion_gcn_modules[part](gcn_out)

                # Global Average Pooling
                pool_sp = gcn_out.mean(dim=-1).transpose(1,2)
                feats.append(pool_sp)

            # Concatenate body, left, right features
            concatenated_feats = torch.cat(feats, dim=-1)
            pose_features_biased = concatenated_feats

            if len(active_modes) == len(self.modes):
                pose_features_biased = concatenated_feats + self.part_para

            # Project to mT5 dimension
            pose_emb = self.pose_proj(pose_features_biased)

            # ========== 2. Text Decoding (mT5) ==========================
            prefix_ids    = src_input["prefix_ids"].long()
            prefix_mask   = src_input["prefix_mask"]
            labels        = tgt_input["labels_ids"].long()
            labels_masked = labels.clone()
            labels_masked[labels_masked == self.mt5_tokenizer.pad_token_id] = -100

            if self.text_decoder == "mt5":
                inputs_embeds = torch.cat([self.mt5_model.encoder.embed_tokens(prefix_ids), pose_emb], dim=1)
                attention_mask= torch.cat([prefix_mask, src_input["attention_mask"]], dim=1)

                mt5_out = self.mt5_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                         labels=labels_masked, return_dict=True, output_hidden_states=True)
                logits  = mt5_out.logits
                
                ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                                          labels_masked.view(-1),
                                          label_smoothing=getattr(self.args, 'label_smoothing', 0.0),
                                          ignore_index=-100)
            else:
                prefix_emb = self.prefix_embed(prefix_ids)
                inputs_embeds = torch.cat([prefix_emb, pose_emb], dim=1)
                attention_mask = torch.cat([prefix_mask, src_input["attention_mask"]], dim=1)
                memory_key_padding_mask = ~attention_mask.bool()

                y_in = labels[:, :-1]
                y_out = labels[:, 1:]
                y_out_masked = y_out.clone()
                y_out_masked[y_out_masked == self.mt5_tokenizer.pad_token_id] = -100

                tgt_emb = self.tgt_embed(y_in) + self.pos_embed(
                    torch.arange(y_in.size(1), device=y_in.device).unsqueeze(0)
                )
                tgt_key_padding_mask = (y_in == self.mt5_tokenizer.pad_token_id)
                causal_mask = torch.triu(
                    torch.full((y_in.size(1), y_in.size(1)), float("-inf"), device=y_in.device),
                    diagonal=1
                )

                dec_out = self.decoder(
                    tgt=tgt_emb,
                    memory=inputs_embeds,
                    tgt_mask=causal_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask
                )
                logits = self.lm_head(dec_out)
                ce_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y_out_masked.reshape(-1),
                    label_smoothing=getattr(self.args, 'label_smoothing', 0.0),
                    ignore_index=-100
                )
            
            out["ce_loss"] = ce_loss.detach()
            out["loss"] = ce_loss

        out.update({
            "margin_loss": torch.tensor(0.0), 
            "alpha": torch.tensor(1.0),
            "inputs_embeds": inputs_embeds.detach(),
            "attention_mask": attention_mask.detach(),
            "eval_figure_data": {} 
        })
        
        return out

    @torch.no_grad()
    def generate(self, pc: Dict[str, torch.Tensor],
                 *, max_new_tokens: int = 100, num_beams: int = 4, **kwargs) -> torch.Tensor:

        if not {"inputs_embeds", "attention_mask"} <= pc.keys():
            if "body" in pc or "left" in pc or "right" in pc:
                compute_dtype = self.pose_proj.weight.dtype
                feats = []
                active_modes = [m for m in self.modes if m in pc]

                # 先处理 body
                body_feat = None
                for part in active_modes:
                    x = self.proj_linear[part](pc[part].to(dtype=compute_dtype)).permute(0,3,1,2)
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
                    pool_sp = gcn_out.mean(dim=-1).transpose(1,2)
                    feats.append(pool_sp)

                concatenated_feats = torch.cat(feats, dim=-1)
                pose_features_biased = concatenated_feats
                if len(active_modes) == len(self.modes):
                     pose_features_biased += self.part_para

                pose_emb = self.pose_proj(pose_features_biased)
                prefix_ids    = pc["prefix_ids"].long()
                prefix_mask   = pc["prefix_mask"]
                if self.text_decoder == "mt5":
                    prefix_emb = self.mt5_model.encoder.embed_tokens(prefix_ids)
                else:
                    prefix_emb = self.prefix_embed(prefix_ids)
                inputs_embeds  = torch.cat([prefix_emb, pose_emb], dim=1)
                attention_mask = torch.cat([prefix_mask, pc["attention_mask"]], dim=1)
                pc_out = {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
            else:
                raise ValueError("generate: need 'inputs_embeds' or 'body/left/right' inputs.")
        else:
            pc_out = pc

        if self.text_decoder == "mt5":
            return self.mt5_model.generate(
                inputs_embeds  = pc_out["inputs_embeds"],
                attention_mask = pc_out["attention_mask"],
                max_new_tokens = max_new_tokens,
                num_beams      = num_beams,
                **kwargs
            )

        memory = pc_out["inputs_embeds"]
        memory_key_padding_mask = ~pc_out["attention_mask"].bool()
        batch_size = memory.size(0)
        start_id = self.mt5_tokenizer.pad_token_id
        eos_id = self.mt5_tokenizer.eos_token_id
        ys = torch.full((batch_size, 1), start_id, dtype=torch.long, device=memory.device)

        for _ in range(max_new_tokens):
            tgt_emb = self.tgt_embed(ys) + self.pos_embed(
                torch.arange(ys.size(1), device=ys.device).unsqueeze(0)
            )
            causal_mask = torch.triu(
                torch.full((ys.size(1), ys.size(1)), float("-inf"), device=ys.device),
                diagonal=1
            )
            dec_out = self.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=(ys == self.mt5_tokenizer.pad_token_id),
                memory_key_padding_mask=memory_key_padding_mask,
            )
            next_token = self.lm_head(dec_out[:, -1]).argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            if eos_id is not None and torch.all(next_token.squeeze(1) == eos_id):
                break

        return ys[:, 1:]

# ============================================================================ #
#  Helper function for checkpoint saving (Required by fine_tuning.py)
# ============================================================================ #
def get_requires_grad_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Helper function to get parameters that require gradients.
    Used by fine_tuning.py to save model checkpoints efficiently.
    """
    param_req = {n: p.requires_grad for n, p in model.named_parameters()}
    dup_map = {k.replace("left","right"):v for k,v in param_req.items() if "left" in k}
    param_req.update(dup_map)
    return {k:v for k,v in model.state_dict().items() if param_req.get(k, False)}
