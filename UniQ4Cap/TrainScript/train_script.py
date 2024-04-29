import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from dist_utils import get_rank
from config import Config
def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--cfg-path", help="path to configuration file.",default='/mnt/ha/bd2/LAVIS/mm_llm/script/query_token_64/pretrain_stage1.yaml')
    parser.add_argument("--local-rank", help="local device id on current node", type=int,default=0)
    parser.add_argument("--world_size", help="local device id on current node", type=int, default=8)

    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def main():
    args = parse_args()    
    cfg = Config(args)
    setup_seeds(cfg)

    # dataset path config
    root_dir = cfg.run_cfg.get("root_dir", None)
    data_path = cfg.run_cfg.get("data_path", None)
    ann_path = cfg.run_cfg.get("ann_path", None)
    # load the dataset
    data_set = DataSet(
        data_path=root_dir + data_path,
        annotation_path=root_dir + ann_path,
        video_processor=vision_processor,
        audio_processor=audio_processor)

    # -------- load the model -------- #
    model = 

    # --------optimizer part--------- #
    # 1.get the parameters to optimize (Q-Former part)
    lr_scale = cfg.run_cfg.get("lr_layer_decay", 1)
    weight_decay = cfg.run_cfg.get("weight_decay", 0.05)
    optim_params = model.module.get_optimizer_params(weight_decay, lr_scale)
    beta2 = cfg.run_cfg.get("beta2", 0.999)
    
    # 3.Our Greatest optimizer
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=float(cfg.run_cfg.init_lr),
        betas=(0.9, beta2),
    )

        # --------learning rate scheduler-------- #
    # learning rate warm up tricks
    from lavis.common.optims import LinearWarmupCosineLRScheduler
    amp = cfg.run_cfg.get("amp", False)
    if amp:
        scaler = torch.cuda.amp.GradScaler()

    max_epoch = cfg.run_cfg.max_epoch
    min_lr = cfg.run_cfg.min_lr
    init_lr = cfg.run_cfg.init_lr
    decay_rate = cfg.run_cfg.get("lr_decay_rate", None)
    warmup_start_lr = cfg.run_cfg.get("warmup_lr", -1)
    warmup_steps = cfg.run_cfg.get("warmup_steps", 0)

    lr_scheduler = LinearWarmupCosineLRScheduler(
        optimizer=optimizer,
        max_epoch=max_epoch,
        min_lr=min_lr,
        init_lr=init_lr,
        decay_rate=decay_rate,
        warmup_start_lr=warmup_start_lr,
        warmup_steps=warmup_steps,
    )

    