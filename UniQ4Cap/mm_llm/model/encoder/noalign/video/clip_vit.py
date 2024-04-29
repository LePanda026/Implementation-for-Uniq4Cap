from collections import OrderedDict
from itertools import repeat
import collections.abc
import math

import torch
import torch.nn.functional as F
from torch import nn

from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper

from lavis.models.eva_vit import convert_weights_to_fp16
from lavis.common.dist_utils import download_cached_file
from einops import rearrange


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, use_grad_checkpointing=False):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        if use_grad_checkpointing:
            self.attn = checkpoint_wrapper(self.attn)
            self.mlp = checkpoint_wrapper(self.mlp)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None,
                 use_grad_checkpointing=False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask, use_grad_checkpointing and i > 12) for i in
              range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int,
                 use_grad_checkpointing: bool):
        super().__init__()
        self.input_resolution = input_resolution
        self.num_features = width
        self.num_heads = heads
        self.num_patches = (input_resolution // patch_size) ** 2
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.num_patches + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, use_grad_checkpointing=use_grad_checkpointing)

    #         self.ln_final = LayerNorm(width)

    def forward(self, x: torch.Tensor):

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        #         x = self.ln_final(x)
        return x

    def get_num_layer(self, var_name=""):
        if var_name in ("class_embedding", "positional_embedding", "conv1", "ln_pre"):
            return 0
        elif var_name.startswith("transformer.resblocks"):
            layer_id = int(var_name.split('.')[2])
            return layer_id + 1
        else:
            return len(self.transformer.resblocks)

        # From PyTorch internals



class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x, B, T):
        if not self.training or self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        if T == 1:
            rand = torch.randn(batch, num_tokens)
            patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices
        else:
            rand = torch.randn(B, num_tokens)
            patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices
            patch_indices_keep = patch_indices_keep.unsqueeze(1).repeat(1, T, 1)
            patch_indices_keep = rearrange(patch_indices_keep, 'b t n -> (b t) n')


        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class VisionTransformer3D(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, num_frames: int,
                 use_grad_checkpointing: bool):
        super().__init__()
        self.input_resolution = input_resolution
        self.num_features = width
        self.num_heads = heads
        self.num_patches = (input_resolution // patch_size) ** 2
        self.width = width
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.patch_dropout = PatchDropout(0.8)

        self.conv1 = nn.Conv3d(
            in_channels=3,
            out_channels=self.width,
            kernel_size=(1,self.patch_size,self.patch_size),
            stride=(1,self.patch_size,self.patch_size),
            bias=False
        )

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        class_embedding = nn.Parameter(self.class_embedding.data.repeat(self.num_frames, 1))
        self.class_embedding = class_embedding
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.num_patches + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, use_grad_checkpointing=use_grad_checkpointing)


    def forward(self, x: torch.Tensor):

        B,_,T,_,_ = x.shape
        # x's shape is b c t h w
        patch_embeds = self.conv1(x)
        patch_embeds = rearrange(patch_embeds,'b c t h w -> b t (h w) c')
        class_embeds = self.class_embedding.unsqueeze(1).unsqueeze(0).repeat(B,1,1,1)
        embeddings = torch.cat([class_embeds,patch_embeds],dim=2) # b t hw+1 c
        embeddings = embeddings + self.positional_embedding
        embeddings = rearrange(embeddings,'b t hw_1 c -> (b t) hw_1 c')
        hidden_states = self.patch_dropout(embeddings,B,T)
        hidden_states = self.ln_pre(hidden_states)
        output = self.transformer(hidden_states)
        return output.reshape(B,-1,1024)


        # x = self.conv1(x)  # shape = [*, width, grid, grid]
        # x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        # x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # x = torch.cat(
        #     [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
        #      x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        # x = x + self.positional_embedding.to(x.dtype)
        # x = self.ln_pre(x)
        #
        # x = x.permute(1, 0, 2)  # NLD -> LND
        # x = self.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        #
        # #         x = self.ln_final(x)


    def get_num_layer(self, var_name=""):
        if var_name in ("class_embedding", "positional_embedding", "conv1", "ln_pre"):
            return 0
        elif var_name.startswith("transformer.resblocks"):
            layer_id = int(var_name.split('.')[2])
            return layer_id + 1
        else:
            return len(self.transformer.resblocks)

        # From PyTorch internals


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def interpolate_pos_embed(model, state_dict, interpolation: str = 'bicubic', seq_dim=1):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('positional_embedding', None)

    grid_size = round((model.positional_embedding.shape[0] - 1) ** 0.5)
    if old_pos_embed is None:
        return
    grid_size = to_2tuple(grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed

    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    print('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        align_corners=True,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['positional_embedding'] = new_pos_embed


def create_clip_vit_L(img_size=224, use_checkpoint=False, precision="fp16"):
    model = VisionTransformer(
        input_resolution=img_size,
        patch_size=14,
        width=1024,
        layers=23,
        heads=16,
        use_grad_checkpointing=use_checkpoint,
    )
    url = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/clip_vit_L.pth"
    cached_file = download_cached_file(
        url, check_hash=False, progress=True
    )
    state_dict = torch.load(cached_file, map_location="cpu")
    interpolate_pos_embed(model, state_dict)

    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    # print(incompatible_keys)

    if precision == "fp16":
        convert_weights_to_fp16(model)
    return model
