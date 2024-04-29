import cv2
import torchaudio
import decord
import numpy as np
import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import NormalizeVideo, RandomHorizontalFlipVideo, CenterCropVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
import json
import os
from languagebind.video.configuration_video import CLIPVisionConfig as VideoConfig
from languagebind.audio.configuration_audio import CLIPVisionConfig as AudioConfig

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
DEFAULT_AUDIO_FRAME_SHIFT_MS = 10
vision_config_path = '/home/w/LanguageBind/encoder/video_config.json'
audio_config_path = '/home/w/LanguageBind/encoder/audio_config.json'
audio_path = '/home/w/LanguageBind/assets/audio/1.wav'
vision_encoder_config = VideoConfig.from_pretrained(vision_config_path)
model_vision_encoder_config = AudioConfig.from_pretrained(audio_config_path)


def make_list_of_images(x):
    if not isinstance(x, list):
        return [x]
    return x

def process_images(images, config, tokenizer=None, **kwargs):
    encoding = {}  # 初始化encoding
    images = make_list_of_images(images)
    image_features = [load_and_transform_video(image, get_video_transform(config),
                                               video_decode_backend=config.video_decode_backend,
                                               num_frames=config.num_frames) for image in images]
    image_features = torch.stack(image_features)
    encoding["pixel_values"] = image_features
    return encoding
def load_and_transform_video(
        video_path,
        transform,
        video_decode_backend='opencv',
        clip_start_sec=0.0,
        clip_end_sec=None,
        num_frames=8,
):
    if video_decode_backend == 'pytorchvideo':
        #  decord pyav
        video = EncodedVideo.from_path(video_path, decoder="decord", decode_audio=False)
        duration = video.duration
        start_sec = clip_start_sec  # secs
        end_sec = clip_end_sec if clip_end_sec is not None else duration  # secs
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
        video_outputs = transform(video_data)

    elif video_decode_backend == 'decord':
        decord.bridge.set_bridge('torch')
        decord_vr = VideoReader(video_path, ctx=cpu(0))
        duration = len(decord_vr)
        frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)
        video_data = decord_vr.get_batch(frame_id_list)
        video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        video_outputs = transform(video_data)

    elif video_decode_backend == 'opencv':
        cv2_vr = cv2.VideoCapture(video_path)
        duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)

        video_data = []
        for frame_idx in frame_id_list:
            cv2_vr.set(1, frame_idx)
            _, frame = cv2_vr.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_data.append(torch.from_numpy(frame).permute(2, 0, 1))
        cv2_vr.release()
        video_data = torch.stack(video_data, dim=1)
        video_outputs = transform(video_data)
    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv)')
    return video_outputs
def get_video_transform(config):
    if config.video_decode_backend == 'pytorchvideo':
        transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(config.num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                    ShortSideScale(size=224),
                    CenterCropVideo(224),
                    RandomHorizontalFlipVideo(p=0.5),
                ]
            ),
        )

    elif config.video_decode_backend == 'decord':

        transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=224),
                CenterCropVideo(224),
                RandomHorizontalFlipVideo(p=0.5),
            ]
        )

    elif config.video_decode_backend == 'opencv':
        transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=224),
                CenterCropVideo(224),
                RandomHorizontalFlipVideo(p=0.5),
            ]
        )
    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv)')
    return transform

# audio
def load_and_transform_audio(audio_path,transform):
    waveform_and_sr = torchaudio_loader(audio_path)
    audio_outputs = transform(waveform_and_sr)
    return audio_outputs
def torchaudio_loader(path):
    return torchaudio.load(path)
class AudioTransform:
    def __init__(self, args):
        self.sample_rate = args.audio_sample_rate
        self.num_mel_bins = args.num_mel_bins
        self.target_length = args.target_length
        self.audio_mean = args.audio_mean
        self.audio_std = args.audio_std
        self.mean = []
        self.std = []
        # mean=-4.2677393
        # std=4.5689974
        # self.norm = transforms.Normalize(mean=self.audio_mean, std=self.audio_std)


    def __call__(self, audio_data_and_origin_sr):
        audio_data, origin_sr = audio_data_and_origin_sr
        if self.sample_rate != origin_sr:
            # print(audio_data.shape, origin_sr)
            audio_data = torchaudio.functional.resample(audio_data, orig_freq=origin_sr, new_freq=self.sample_rate)
        waveform_melspec = self.waveform2melspec(audio_data)
        return waveform_melspec


    def waveform2melspec(self, audio_data):
        mel = self.get_mel(audio_data)
        if mel.shape[0] > self.target_length:
            # split to three parts
            chunk_frames = self.target_length
            total_frames = mel.shape[0]
            ranges = np.array_split(list(range(0, total_frames - chunk_frames + 1)), 3)
            # print('total_frames-chunk_frames:', total_frames-chunk_frames,
            #       'len(audio_data):', len(audio_data),
            #       'chunk_frames:', chunk_frames,
            #       'total_frames:', total_frames)
            if len(ranges[1]) == 0:  # if the audio is too short, we just use the first chunk
                ranges[1] = [0]
            if len(ranges[2]) == 0:  # if the audio is too short, we just use the first chunk
                ranges[2] = [0]
            # randomly choose index for each part
            idx_front = np.random.choice(ranges[0])
            idx_middle = np.random.choice(ranges[1])
            idx_back = np.random.choice(ranges[2])
            # idx_front = ranges[0][0]  # fixed
            # idx_middle = ranges[1][0]
            # idx_back = ranges[2][0]
            # select mel
            mel_chunk_front = mel[idx_front:idx_front + chunk_frames, :]
            mel_chunk_middle = mel[idx_middle:idx_middle + chunk_frames, :]
            mel_chunk_back = mel[idx_back:idx_back + chunk_frames, :]
            # print(total_frames, idx_front, idx_front + chunk_frames, idx_middle, idx_middle + chunk_frames, idx_back, idx_back + chunk_frames)
            # stack
            mel_fusion = torch.stack([mel_chunk_front, mel_chunk_middle, mel_chunk_back], dim=0)
        elif mel.shape[0] < self.target_length:  # padding if too short
            n_repeat = int(self.target_length / mel.shape[0]) + 1
            # print(self.target_length, mel.shape[0], n_repeat)
            mel = mel.repeat(n_repeat, 1)[:self.target_length, :]
            mel_fusion = torch.stack([mel, mel, mel], dim=0)
        else:  # if equal
            mel_fusion = torch.stack([mel, mel, mel], dim=0)
        mel_fusion = mel_fusion.transpose(1, 2)  # [3, target_length, mel_bins] -> [3, mel_bins, target_length]

        # self.mean.append(mel_fusion.mean())
        # self.std.append(mel_fusion.std())
        mel_fusion = (mel_fusion - self.audio_mean) / (self.audio_std * 2)
        return mel_fusion

    def get_mel(self, audio_data):
        # mel shape: (n_mels, T)
        audio_data -= audio_data.mean()
        mel = torchaudio.compliance.kaldi.fbank(
            audio_data,
            htk_compat=True,
            sample_frequency=self.sample_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=self.num_mel_bins,
            dither=0.0,
            frame_length=25,
            frame_shift=DEFAULT_AUDIO_FRAME_SHIFT_MS,
        )
        return mel  # (T, n_mels)
def get_audio_transform(config):
    return AudioTransform(config)

def process_audio(audios):
    encoding = {}  # 初始化encoding
    if audios is not None:
        audios = make_list_of_images(audios)
        audio_features = [load_and_transform_audio(audio, get_audio_transform(model_vision_encoder_config)) for audio in audios]
        audio_features = torch.stack(audio_features)
        encoding["pixel_values"] = audio_features
    return encoding

class fuck_dataset(Dataset):
    def __init__(self, data_path,annotation_path,end_video = '.mp4',end_audio = '.wav'):
        self.data_path = data_path
        self.annotation_path =annotation_path
        self.end_video = end_video
        self.end_audio = end_audio
        # Assuming annotation is stored in a single JSON file
        with open(self.annotation_path, 'r') as annotation_file:
            self.annotation_data = json.load(annotation_file)

        self.video_names = list(self.annotation_data.keys())
    def __getitem__(self,i):
        video_name = self.video_names[i]
        annotation_dict = self.annotation_data[video_name]
        sound_mplug = annotation_dict.get("sound_mplug", "")
        video_path = os.path.join(self.data_path, video_name + self.end_video)
        process_images_=process_images(video_path, vision_encoder_config)["pixel_values"]
        audio_path= os.path.join(self.data_path, video_name + self.end_audio)
        process_audio_=process_audio(audio_path)["pixel_values"]
        # Return sound_mplug and video_name as a tuple
        return sound_mplug,process_images_,process_audio_

    def __len__(self):
        return len(self.video_names)