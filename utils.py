"""
This file is modified from:
https://github.com/facebookresearch/deit/blob/main/utils.py

Includes miscellaneous functions, distributed helpers, DeepSpeed config,
and argument parsing for Uni-Sign training.
"""

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import io
import os
import time, random
import numpy as np
from collections import defaultdict, deque
import datetime
import warnings

import torch
import torch.nn.functional as F
import argparse
import torch.backends.cudnn as cudnn
import pickle
import gzip

# --- DeepSpeed Integration ---
try:
    import deepspeed
    import deepspeed.comm as dist
    _deepspeed_available = True
except ImportError:
    warnings.warn("DeepSpeed not installed. Distributed training and ZeRO features will be unavailable. `pip install deepspeed`")
    class DummyDist:
        def is_available(self): return False
        def is_initialized(self): return False
        def get_world_size(self): return 1
        def get_rank(self): return 0
        def barrier(self): pass
        def all_reduce(self, tensor, op=None): pass 
        def all_gather_object(self, object_list, obj): object_list[0] = obj
        def broadcast(self, tensor, src): pass
    dist = DummyDist()
    _deepspeed_available = False

# =============================================================================
#  Distributed Helpers
# =============================================================================

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode_ds(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.gpu = 0 
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    
    if _deepspeed_available:
        deepspeed.init_distributed(dist_backend=args.dist_backend)
    else:
        print("DeepSpeed not available, skipping init_distributed")

    setup_for_distributed(args.rank == 0)

# =============================================================================
#  DeepSpeed Initialization Wrapper
# =============================================================================

def init_deepspeed(args, model, optimizer, lr_scheduler):
    if not _deepspeed_available or not args.deepspeed:
        return model, optimizer, lr_scheduler

    if not args.deepspeed_config:
        ds_config = {
            "train_batch_size": args.batch_size * args.world_size * args.gradient_accumulation_steps,
            "train_micro_batch_size_per_gpu": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "steps_per_print": 10,
            "gradient_clipping": args.gradient_clipping,
            "zero_optimization": {
                "stage": 0,
                "offload_param": {"device": "none", "pin_memory": True},
                "offload_optimizer": {"device": "none", "pin_memory": True}
            },
            "fp16": {"enabled": args.dtype == 'fp16'},
            "bf16": {"enabled": args.dtype == 'bf16'},
            "wall_clock_breakdown": False
        }
    else:
        ds_config = args.deepspeed_config

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=ds_config
    )
    
    return model, optimizer, lr_scheduler

# =============================================================================
#  Misc Helper Functions
# =============================================================================

def count_parameters_in_MB(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    return batch

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True

# =============================================================================
#  Metrics & Logging
# =============================================================================

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        # 【修复】防止空序列报错
        if len(self.deque) == 0:
            return 0.0
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        # 【修复】防止空序列报错
        if len(self.deque) == 0:
            return 0.0
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        # 【修复】防止除以零
        if self.count == 0:
            return 0.0
        return self.total / self.count

    @property
    def max(self):
        # 【修复】防止 max() 对空序列报错 (本次报错的核心原因)
        if len(self.deque) == 0:
            return 0.0
        return max(self.deque)

    @property
    def value(self):
        # 【修复】防止空序列报错
        if len(self.deque) == 0:
            return 0.0
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'MetricLogger' object has no attribute '{}'".format(attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}MB'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

# =============================================================================
#  Argument Parser
# =============================================================================

def get_args_parser():
    parser = argparse.ArgumentParser('Uni-Sign scripts', add_help=False)
    
    # --- Training Parameters ---
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw")')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=[0.9, 0.999], type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: 0.9 0.999)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay (default: 0.01)')
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate (default: 3e-4)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--dtype', type=str, default='fp32', choices=['fp16', 'bf16', 'fp32'],
                        help='Training precision')

    # --- Model Parameters ---
    parser.add_argument('--hidden_dim', default=768, type=int, help='Transformer hidden dimension')
    parser.add_argument('--gcn_out_dim', default=256, type=int, help='GCN output dimension')
    parser.add_argument('--text_decoder', default='mt5', type=str, choices=['mt5', 'transformer'],
                        help='Text decoder backend: pretrained mT5 or lightweight Transformer decoder')
    
    # 【新增】支持自定义输入通道数
    parser.add_argument('--input_channels', default=3, type=int, 
                        help='Input dimensionality (2 for xy, 3 for xyz, 7 for imu)')

    # --- Dataset Parameters ---
    parser.add_argument('--dataset', default='CSL_Daily', type=str, help='dataset name')
    parser.add_argument('--task', default='SLT', type=str, help='task name')
    parser.add_argument('--max_length', default=256, type=int, help='max frame length')
    parser.add_argument('--max_eval_samples', default=1000, type=int, help='max eval samples')

    # --- Checkpoints & Output ---
    parser.add_argument('--output_dir', default='', help='path to save checkpoints')
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--auto_resume', action='store_true',
                        help='Auto resume from latest checkpoint_*.pth in output_dir')
    parser.add_argument('--resume_checkpoint', default='',
                        help='Explicit checkpoint path for resume (overrides auto latest)')
    parser.add_argument('--load_checkpoint_dir', default='', help='load checkpoint dir')
    parser.add_argument('--save_batch_name', default='testing', help='name for saved batch')
    parser.add_argument('--save_one_batch', action='store_true', help='save one batch for debug')
    parser.add_argument('--save_eval_predictions', action='store_true',
                        help='Save eval predictions/references as JSONL files')
    parser.add_argument('--save_eval_raw_2d', action='store_true',
                        help='Save raw 2D keypoints (x,y) with predictions during evaluation')

    # --- Hyperbolic / Graph Parameters ---
    parser.add_argument('--use_hyperbolic', action='store_true', help='Use Hyperbolic Geometry')
    parser.add_argument('--hyp_dim', default=256, type=int, help='Hyperbolic dimension')
    parser.add_argument('--init_c', default=1.5, type=float, help='Initial curvature')
    parser.add_argument('--hyp_lr', default=1e-3, type=float, help='Hyperbolic learning rate')
    parser.add_argument('--hyp_stabilize', default=100, type=int, help='Stabilize steps')
    parser.add_argument('--alpha', default=1.0, type=float, help='Loss weight alpha')
    parser.add_argument('--alpha_reg', default=0.01, type=float, help='Regularization weight')
    parser.add_argument('--label_smoothing', default=0.2, type=float, help='Label smoothing')
    parser.add_argument('--label_smoothing_hyp', default=0.2, type=float, help='Hyp label smoothing')
    parser.add_argument('--hyp_text_emb_src', default='token', type=str, choices=['token', 'decoder'])
    parser.add_argument('--hyp_text_cmp', default='token', type=str, choices=['pooled', 'attn', 'token'])

    # --- Distributed & System ---
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
                        help='Number of steps to accumulate gradients')
    parser.add_argument('--gradient_clipping', default=1.0, type=float, help='Gradient clipping value')
    parser.add_argument('--manual_grad_clip', action='store_true', help='Use manual gradient clipping')
    parser.add_argument('--clip_grad_norm_euclid', default=1.0, type=float, help='Euclidean clip norm')
    parser.add_argument('--clip_grad_norm_hyp', default=0.1, type=float, help='Hyperbolic clip norm')
    
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--quick_break', default=-1, type=int, help='Break epoch after N steps (for debugging)')
    
    # --- Generation Parameters ---
    parser.add_argument('--num_beams', default=4, type=int, metavar='N',
                        help='Number of beams for generation during evaluation (default: 4)')
    parser.add_argument('--max_tgt_len', default=100, type=int, metavar='LEN',
                        help='Max new tokens for generation during evaluation (default: 100)')

    # --- Logging ---
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
    parser.add_argument('--rgb_support', action='store_true', help='Enable RGB features')
    parser.add_argument('--wandb_project', default='Uni-Sign-Hyperbolic-v4', type=str,
                        help='WandB project name')
    parser.add_argument('--run-name', dest='wandb_run_name', type=str, default=None,
                        help='Explicit WandB run name')

    # Standard Distributed args
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--local_rank', default=0, type=int)
    
    # DeepSpeed specific
    parser.add_argument('--deepspeed', action='store_true', help='Enable DeepSpeed')
    parser.add_argument('--deepspeed_config', default=None, type=str, help='DeepSpeed config file')

    return parser
