"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from tqdm import tqdm

import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

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

from lavis.datasets.dataset.load_dataset import build_mine_datasets
# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners import *
from lavis.tasks import *


from lavis.models.encoder import AudioConfig as AudioProcConfig
from lavis.models.encoder import VideoConfig as VideoProcConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", help="path to configuration file.",default='./lavis/projects/blip2/train/pretrain_stage1.yaml')
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

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    job_id = now()
    cfg = Config(parse_args())
    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    setup_logger()
    cfg.pretty_print()
    task = tasks.setup_task(cfg)

    root_dir = '/mnt/d/datasets/'
    data_path = '/coco_vat_test/'
    ann_path = '/processed_coco_vat_test.json'
    proc_cfg_path = '/home/knight/LAVIS/lavis/configs/'

    vision_proc_cfg = VideoProcConfig.from_pretrained(proc_cfg_path+'video_huge_config.json')
    audio_proc_cfg = AudioProcConfig.from_pretrained(proc_cfg_path+'audio_config.json')
    from lavis.datasets.dataset.caption_dataset import CaptionDataset
    from lavis.datasets.dataset.processors import VideoProcess,AudioProcess
    vision_processor = VideoProcess(vision_proc_cfg)
    audio_processor = AudioProcess(audio_proc_cfg)
    datasets = {}
    datasets['train'] = CaptionDataset(
                            data_path=root_dir+data_path,
                            annotation_path=root_dir+ann_path,
                            video_processor=vision_processor,
                            audio_processor=audio_processor)
    datasets['val'] = CaptionDataset(
        data_path=root_dir + data_path,
        annotation_path=root_dir + ann_path,
        video_processor=vision_processor,
        audio_processor=audio_processor)

    # from lavis.models.mm_blip2_models.mm_blip2_qformer import Blip2Qformer
    # model = Blip2Qformer.from_config(cfg.model_cfg)
    model = task.build_model(cfg)
    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets)

    runner.train()
if __name__ == "__main__":
    main()
