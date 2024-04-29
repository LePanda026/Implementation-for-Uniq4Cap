import torch
import random
import logging
from mm_llm.trainer import trainer
import argparse
import numpy as np
    # Distributed Data Parallel Sets
import os
from datetime import datetime
import torch.backends.cudnn as cudnn

from mm_llm.utils.config import Config
from mm_llm.utils.dist_utils import get_rank, init_distributed_mode
from mm_llm.utils.logger import setup_logger

from mm_llm.datasets.caption_dataset import CaptionDataset
from mm_llm.model.encoder import AudioConfig as AudioProcConfig
from mm_llm.model.encoder import VideoConfig as VideoProcConfig
from mm_llm.datasets.processors import VideoProcess, AudioProcess
from mm_llm.model.mm_blip2_models.mm_blip2_qformer import Blip2Qformer
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from mm_llm.model.encoder.noalign.beats.audio_processor import BeatsAudioProcessor

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--cfg-path", help="path to configuration file.",default='./mm_llm/script/pretrain_stage1_noprealign_noitc.yaml')
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
    print(args.world_size)
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.local_rank)
    print(f'current rank: {args.local_rank}')
    torch.cuda.set_device(args.local_rank)
    cfg = Config(args)
    # torch.distributed.init_process_group(backend='gloo',
    #                                      init_method='file:///mnt/ha/bd2/LAVIS/switch.txt',
    #                                      world_size=args.world_size,
    #                                      rank=args.local_rank)
    torch.distributed.init_process_group(backend='nccl',
                                         world_size=args.world_size,
                                         rank=args.local_rank)

    # set up the random seeds
    setup_seeds(cfg)
    # set up the logger
    setup_logger()
    cfg.pretty_print()
    # get the current job_id
    job_id = datetime.now().strftime("%Y%m%d%H%M")[:-1]

    # 获取 processor参数和 processor
    pre_align = cfg.model_cfg.get("pre_align",True)
    vision_proc_cfg = VideoProcConfig.from_pretrained(cfg.model_cfg.get("vit_config",None))
    audio_proc_cfg = AudioProcConfig.from_pretrained(cfg.model_cfg.get("aud_config",None))
    vision_processor = VideoProcess(vision_proc_cfg)
    if pre_align:
        audio_processor = AudioProcess(audio_proc_cfg)
    else:
        audio_processor = BeatsAudioProcessor(model_name='iter3', sampling_rate=16000, n_frames=1, frame_length=512, is_eval=False)

    # dataset path config
    root_dir = cfg.run_cfg.get("root_dir", None)
    data_path = cfg.run_cfg.get("data_path", None)
    ann_path = cfg.run_cfg.get("ann_path", None)
    # load the dataset
    data_set = CaptionDataset(
        data_path=root_dir + data_path,
        annotation_path=root_dir + ann_path,
        video_processor=vision_processor,
        audio_processor=audio_processor)

    # Dataloader (DDP)
    batch_size = cfg.run_cfg.get("batch_size", 256)
    print(batch_size)
    # data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True,drop_last=True)
    sampler = DistributedSampler(data_set)
    data_loader = DataLoader(data_set, num_workers=12,batch_size=batch_size, sampler=sampler)

    # -------- load the model -------- #
    model = Blip2Qformer.from_config(cfg.model_cfg)
    print("loading the model done")
    model = DDP(model.cuda(args.local_rank), device_ids=[args.local_rank])
    #电脑

    # --------optimizer part--------- #
    # 1.get the parameters to optimize (Q-Former part)
    lr_scale = cfg.run_cfg.get("lr_layer_decay", 1)
    weight_decay = cfg.run_cfg.get("weight_decay", 0.05)
    optim_params = model.module.get_optimizer_params(weight_decay, lr_scale)

    # 2.calculate the number of trainable parameters
    num_parameters = 0
    for p_group in optim_params:
        for p in p_group["params"]:
            num_parameters += p.data.nelement()
    logging.info("number of trainable parameters: {}".format(num_parameters))
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

    # --------trainer!--------- #

    train_cls = trainer(
        run_config=cfg.run_cfg,
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
        job_id=job_id,
        local_rank=args.local_rank
    )

    train_cls.train()

if __name__ == "__main__":
    main()
