import torch
from .video.fuckprocessing_video import fuckLanguageBindVideoProcessor
from .video.modeling_video import CLIPVisionTransformer
from .video.configuration_video import CLIPVisionConfig

class ClipVideo:
    def __init__(self, model_path, config_path,freeze=True):
        self.model_vision_encoder_config = CLIPVisionConfig.from_pretrained(config_path)
        self.model = CLIPVisionTransformer(self.model_vision_encoder_config)
        self.model.load_state_dict(torch.load(model_path,map_location='cpu'))
        self.audio_process = fuckLanguageBindVideoProcessor(self.model.config)
        if freeze:
            for name, param in self.model.named_parameters():
                param.requires_grad = False
            # self.model.eval()