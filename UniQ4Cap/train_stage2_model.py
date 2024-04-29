import argparse
import os
import random

import numpy as np
import torch
# import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.registry import registry
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners import *
from lavis.tasks import *


from mm_llm.datasets.caption_dataset import CaptionDataset
from mm_llm.model.encoder import AudioConfig as AudioProcConfig
from mm_llm.model.encoder import VideoConfig as VideoProcConfig
from mm_llm.datasets.processors import VideoProcess, AudioProcess


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", help="path to configuration file.",default='lavis/projects/xinstruct_blip/train/vicuna7b/2stage.yaml')
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # cudnn.benchmark = False
    # cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    cfg = Config(parse_args())

    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    # set after init_distributed_mode() to only log on master.
    setup_logger()
    cfg.pretty_print()

    # task = tasks.setup_task(cfg)
    # datasets = task.build_datasets(cfg)
    # model = task.build_model(cfg)
    model_cfg = cfg.model_cfg

    # processor 参数路径
    proc_cfg_path = './configs/'
    # 获取 processor参数和 processor
    vision_proc_cfg = VideoProcConfig.from_pretrained(cfg.model_cfg.get("vit_config", None))
    audio_proc_cfg = AudioProcConfig.from_pretrained(cfg.model_cfg.get("aud_config", None))
    vision_processor = VideoProcess(vision_proc_cfg)
    audio_processor = AudioProcess(audio_proc_cfg)

    # 数据集路径参数
    root_dir = '/mnt/d/datasets/'
    data_path = '/coco_vat_test/'
    ann_path = '/processed_coco_vat_test.json'
    # 加载数据集
    datasets = {}
    data_set = CaptionDataset(
        data_path=root_dir + data_path,
        annotation_path=root_dir + ann_path,
        video_processor=vision_processor,
        audio_processor=audio_processor)
    # TODO: split the train and validate dataset
    datasets['train'] = data_set
    datasets['val'] = data_set

    # Dataloader
    # TODO: make the distributed data loader run #
    from torch.utils.data import DataLoader, DistributedSampler
    data_loader = DataLoader(data_set, batch_size=500, shuffle=True, drop_last=True)

    from mm_llm.datasets.data_utils import prepare_sample
    # from lavis.models.blip2_models.blip2_opt import Blip2OPT
    # from mm_llm.model.stage2.mm_blip2_llm import Blip2VicunaXInstruct
    # model = Blip2VicunaXInstruct.from_config(model_cfg).to('cuda')

    for samples in data_loader:
        samples = prepare_sample(samples,0)
        print()
        # output = model(samples)

if __name__ == "__main__":
    main()
