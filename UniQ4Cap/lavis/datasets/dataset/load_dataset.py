import os
import warnings
from lavis.processors.blip_processors import BlipImageTrainProcessor,BlipImageEvalProcessor,BlipCaptionProcessor
from lavis.datasets.datasets.caption_datasets import CaptionDataset
from lavis.datasets.datasets.coco_caption_datasets import COCOCapEvalDataset

def build_mine_datasets(cfg):

    vis_processors = dict()
    text_processors = dict()


    datasets_config = cfg.datasets_cfg
    name = 'coco_caption'
    datasets_config = datasets_config[name]

    # processor information
    vis_proc_cfg = datasets_config.vis_processor
    txt_proc_cfg = datasets_config.text_processor

    if vis_proc_cfg is not None:
        vis_train_cfg = vis_proc_cfg.get("train")
        vis_eval_cfg = vis_proc_cfg.get("eval")

        vis_processors["train"] = BlipImageTrainProcessor(vis_train_cfg)
        vis_processors["eval"] = BlipImageEvalProcessor(vis_eval_cfg)

    if txt_proc_cfg is not None:
        txt_train_cfg = txt_proc_cfg.get("train")
        txt_eval_cfg = txt_proc_cfg.get("eval")

        text_processors["train"] = BlipCaptionProcessor(txt_train_cfg)
        text_processors["eval"] = BlipCaptionProcessor(txt_eval_cfg)


    # annotation and image information
    build_info = datasets_config.build_info
    ann_info = build_info.annotations
    vis_info = build_info.images
    datasets = dict()

    for split in ann_info.keys():
        if split not in ["train", "val", "test"]:
            continue
        is_train = split == "train"
        # annotation path
        ann_paths = ann_info.get(split).storage
        if isinstance(ann_paths, str):
            ann_paths = [ann_paths]

        abs_ann_paths = []
        for ann_path in ann_paths:
            if not os.path.isabs(ann_path):
                ann_path = os.path.expanduser(
                    os.path.join('/mnt/c/Users/24600/Desktop/LAVIS/lavis/datasets', ann_path))
            abs_ann_paths.append(ann_path)
        ann_paths = abs_ann_paths
        vis_path = vis_info.storage
        if not os.path.isabs(vis_path):
            # vis_path = os.path.join(utils.get_cache_path(), vis_path)
            vis_path = os.path.expanduser(os.path.join('/mnt/c/Users/24600/Desktop/LAVIS/lavis/datasets', vis_path))
        if not os.path.exists(vis_path):
            warnings.warn("storage path {} does not exist.".format(vis_path))
        dataset_cls = CaptionDataset if is_train else COCOCapEvalDataset

        vis_processor = (
            vis_processors["train"]
            if is_train
            else vis_processors["eval"]
        )
        text_processor = (
            text_processors["train"]
            if is_train
            else text_processors["eval"]
        )

        datasets[split] = dataset_cls(
            vis_processor=vis_processor,
            text_processor=text_processor,
            ann_paths=ann_paths,
            vis_root=vis_path,
        )

    return datasets