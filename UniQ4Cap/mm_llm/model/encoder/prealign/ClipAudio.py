import math
import torch
from torch.nn import functional as F
from torch import nn
from .audio.fuckprocessing_audio import fuckLanguageBindAudioProcessor
from .audio.modeling_audio import CLIPVisionTransformer
from .audio.configuration_audio import CLIPVisionConfig

class ClipAudio:
    def __init__(self, model_path, config_path,freeze=True):
        self.model_path = model_path
        self.config_path = config_path
        self.freeze=freeze
        self.model = self.initialize_model()

    def initialize_model(self):
        # 加载配置文件
        model_vision_encoder_config = CLIPVisionConfig.from_pretrained(self.config_path)
        model_vision_encoder = CLIPVisionTransformer(model_vision_encoder_config)
        # 调整位置嵌入
        self.resize_position_embedding(model_vision_encoder.embeddings, model_vision_encoder_config)

        state_dict = torch.load(self.model_path,map_location='cpu')
        model_vision_encoder.load_state_dict(state_dict)
        if self.freeze:
            for name,param in model_vision_encoder.named_parameters():
                param.requires_grad = False
            # model_vision_encoder.eval()
        # TODO: Check the code
        return model_vision_encoder

    def resize_position_embedding(self, m, vision_config):
        if vision_config.num_mel_bins != 0 and vision_config.target_length != 0:
            m.image_size = [vision_config.num_mel_bins, vision_config.target_length]
        m.config.image_size = [m.image_size, m.image_size] if isinstance(m.image_size, int) else m.image_size
        old_pos_embed_state_dict = m.position_embedding.state_dict()
        old_pos_embed = old_pos_embed_state_dict['weight']
        dtype = old_pos_embed.dtype
        grid_size = [m.config.image_size[0] // m.patch_size, m.config.image_size[1] // m.patch_size]
        extra_tokens = 1
        new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
        if new_seq_len == old_pos_embed.shape[0]:
            return

        m.num_patches = grid_size[0] * grid_size[1]
        m.num_positions = m.num_patches + 1
        m.register_buffer("position_ids", torch.arange(m.num_positions).expand((1, -1)))
        new_position_embedding = nn.Embedding(m.num_positions, m.embed_dim)

        if extra_tokens:
            pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
        else:
            pos_emb_tok, pos_emb_img = None, old_pos_embed
        old_grid_size = [int(math.sqrt(len(pos_emb_img)))] * 2

        pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
        pos_emb_img = F.interpolate(
            pos_emb_img,
            size=grid_size,
            mode='bicubic',
            antialias=True,
            align_corners=False,
        )
        pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
        if pos_emb_tok is not None:
            new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
        else:
            new_pos_embed = pos_emb_img
        old_pos_embed_state_dict['weight'] = new_pos_embed.to(dtype)
        m.position_embedding = new_position_embedding
        m.position_embedding.load_state_dict(old_pos_embed_state_dict)

    def process_audio(self):
        audio_processor = fuckLanguageBindAudioProcessor(self.model.config)
        data = audio_processor([self.audio_path])

        with torch.no_grad():
            out = self.model(**data)

        return out.last_hidden_state, data['pixel_values']