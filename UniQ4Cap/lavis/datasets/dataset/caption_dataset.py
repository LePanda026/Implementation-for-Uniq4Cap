"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from lavis.datasets.dataset.dataset import BaseDataset


from PIL import Image


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )


class CaptionDataset(BaseDataset, __DisplMixin):
    def __init__(self,
                 data_path,annotation_path,
                 video_processor,audio_processor):

        # Here we have read the annotations and initialize the processor
        super().__init__(data_path,annotation_path,video_processor,audio_processor)

        # Add the Image Ids
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["instance_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):
        # get the video's name by index
        data = self.annotation[index]
        video_name = audio_name = data.get("id","")
        caption = data.get("sound_mplug","")

        video_path = os.path.join(self.data_path,video_name + self.end_video)
        audio_path = os.path.join(self.data_path,audio_name + self.end_audio)
        processed_video = self.video_processor(video_path)
        processed_audio = self.audio_processor(audio_path)

        return {
            "name":video_name,
            "video": processed_video,
            "audio": processed_audio,
            "text": caption,
            "image_id": data["instance_id"]
        }


        # ann = self.annotation[index]
        # image_path = os.path.join(self.vis_root, ann["image"])
        # try:
        #     image = Image.open(image_path).convert("RGB")
        # except:
        #     return None
        #
        # image = self.vis_processor(image)
        # caption = self.text_processor(ann["caption"])
        #
        # return {
        #     "image": image,
        #     "text_input": caption,
        #     # "image_id": ann["image_id"]
        # }

class CaptionEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        return {

            "image": image,
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
        }

class CaptionInstructDataset(CaptionDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            data['text_output'] = data["text_input"]
            data['text_input'] = self.text_processor("")
        return data
