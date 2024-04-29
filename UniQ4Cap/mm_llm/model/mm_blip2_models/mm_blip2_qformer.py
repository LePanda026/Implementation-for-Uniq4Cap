import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F


from mm_llm.model.encoder import ClipVideo,ClipAudio
from mm_llm.model.mm_blip2_models.blip_outputs import BlipOutput, BlipOutputFeatures
from mm_llm.model.mm_blip2_models.mm_blip2 import Blip2Base,compute_sim_matrix,disabled_train
from mm_llm.model.base_model import all_gather_with_grad, concat_all_gather
from mm_llm.model.encoder.noalign.video.clip_vit import VisionTransformer3D
from mm_llm.model.encoder.noalign.beats.beats_encoder import BeatsEncoder

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class Blip2Qformer(Blip2Base):
    def __init__(
        self,
        vit_config=None,
        aud_config=None,
        video_embed_dim=1024,
        audio_embed_dim=1024,
        vit_encoder_pretrained="",
        aud_encoder_pretrained="",
        freeze_encoder=True,

        bert_pretrained="bert-base-cased",
        qformer_pretrained="",
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        pre_align=True,

    ):
        super().__init__()

        # Encoder Part
        self.pre_align = pre_align
        self.video_embed_dim = video_embed_dim
        self.audio_embed_dim = audio_embed_dim
        # load the video encoder
        self.video_encoder,self.ln_video = self.init_video_encoder(
            vit_encoder_pretrained=vit_encoder_pretrained,
            vit_config=vit_config,
            freeze_encoder=freeze_encoder
        )
        # load the audio encoder
        self.audio_encoder,self.ln_audio = self.init_audio_encoder(
            aud_encoder_pretrained=aud_encoder_pretrained,
            aud_config=aud_config,
            freeze_encoder=freeze_encoder
        )
        qformer_embeds = 1408
        self.video_project = nn.Linear(video_embed_dim,qformer_embeds)
        self.audio_project = nn.Linear(audio_embed_dim,qformer_embeds)

        logging.info("loading the freeze Vision & Audio Encoder Successfully")

        self.tokenizer = self.init_tokenizer(bert_pretrained)
        self.num_query_token = num_query_token
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token,bert_pretrained,qformer_embeds, cross_attention_freq
        )

        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len

    def forward(self, samples):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # video = samples["video"].to(device)
        # audio = samples["audio"].to(device)
        # text = samples["text"]

        video = samples["video"]
        audio = samples["audio"]
        text = samples["text"]
        # encode
        video_output = self.video_encoder(video)
        audio_output = self.audio_encoder(audio)

        # layer norm
        video_embeds = self.ln_video(video_output)
        audio_embeds = self.ln_audio(audio_output)

        # project the features to fit the dimension of q-former
        video_embeds = self.video_project(video_embeds)
        audio_embeds = self.audio_project(audio_embeds)

        # Add the batch dimensions
        query_tokens = self.query_tokens.expand(video_embeds.shape[0], -1, -1)
        concat_embds = torch.concat((video_embeds,audio_embeds),dim=1)
        concat_atts = torch.ones(concat_embds.size()[:-1], dtype=torch.long).to(
            video.device)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=concat_embds,
            encoder_attention_mask=concat_atts,
            use_cache=True,
            return_dict=True,
        )

        # concat_features = F.normalize(
        #     self.vision_proj(query_output.last_hidden_state), dim=-1
        # )

        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(video.device)

        # text_output = self.Qformer.bert(
        #     text_tokens.input_ids,
        #     attention_mask=text_tokens.attention_mask,
        #     return_dict=True,
        # )
        # text_feat = F.normalize(
        #     self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        # )

        # ###============== Image-text Contrastive ===================###
        # image_feats_all = concat_all_gather(
        #     concat_features
        # )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        # text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]

        # sim_q2t = torch.matmul(
        #     concat_features.unsqueeze(1), text_feat_all.unsqueeze(-1)
        # ).squeeze()
        # # [batch_size, batch_size*num_gpu, num_query_tokens]
        
        # # image-text similarity: aggregate across all query tokens
        # sim_i2t, _ = sim_q2t.max(-1)
        # sim_i2t = sim_i2t / self.temp

        # # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        # sim_t2q = torch.matmul(
        #     text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
        # ).squeeze()

        # # text-image similarity: aggregate across all query tokens
        # sim_t2i, _ = sim_t2q.max(-1)
        # sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]
        # rank = dist.get_rank()
        # bs = video.size(0)
        # targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
        #     video.device
        # )

        # loss_itc = (
        #     F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
        #     + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        # ) / 2

        # ###============== Image-text Matching ===================###
        # text_input_ids_world = concat_all_gather(text_tokens.input_ids)
        # text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
        # concat_embds_world = all_gather_with_grad(concat_embds)
        # with torch.no_grad():
        #     sim_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)
        #     sim_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)

        #     weights_t2i = F.softmax(sim_t2i, dim=1)
        #     weights_i2t = F.softmax(sim_i2t, dim=1)

        # # select a negative image for each text
        # concat_embeds_neg = []
        # for b in range(bs):
        #     neg_idx = torch.multinomial(weights_t2i[b], 1).item()
        #     concat_embeds_neg.append(concat_embds_world[neg_idx])
        # concat_embeds_neg = torch.stack(concat_embeds_neg, dim=0)

        # # select a negative text for each image
        # text_ids_neg = []
        # text_atts_neg = []
        # for b in range(bs):
        #     neg_idx = torch.multinomial(weights_i2t[b], 1).item()
        #     text_ids_neg.append(text_input_ids_world[neg_idx])
        #     text_atts_neg.append(text_attention_mask_world[neg_idx])

        # text_ids_neg = torch.stack(text_ids_neg, dim=0)
        # text_atts_neg = torch.stack(text_atts_neg, dim=0)

        # text_ids_all = torch.cat(
        #     [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
        # )  # pos, pos, neg
        # text_atts_all = torch.cat(
        #     [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
        #     dim=0,
        # )

        # query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        # query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
        #     video.device
        # )
        # attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        # concat_embeds_all = torch.cat(
        #     [concat_embds, concat_embeds_neg, concat_embds], dim=0
        # )  # pos, neg, pos
        # concat_atts_all = torch.ones(concat_embeds_all.size()[:-1], dtype=torch.long).to(
        #     video.device
        # )

        # output_itm = self.Qformer.bert(
        #     text_ids_all,
        #     query_embeds=query_tokens_itm,
        #     attention_mask=attention_mask_all,
        #     encoder_hidden_states=concat_embeds_all,
        #     encoder_attention_mask=concat_atts_all,
        #     return_dict=True,
        # )

        # vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        # vl_output = self.itm_head(vl_embeddings)
        # logits = vl_output.mean(dim=1)

        # itm_labels = torch.cat(
        #     [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
        #     dim=0,
        # ).to(video.device)
        # loss_itm = F.cross_entropy(logits, itm_labels)

        ##================= Image Captioning ========================##
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            video.device
        )
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            output_attentions=True,
            return_dict=True,
            labels=labels,
        )
        crossattentions = lm_output.cross_attentions
        attentions = lm_output.attentions
        loss_lm = lm_output.loss

        # here are three losses


        # return BlipOutput(
        #     loss=loss_itc + loss_itm + loss_lm,
        #     loss_itc=loss_itc,
        #     loss_itm=loss_itm,
        #     loss_lm=loss_lm,
        
        return BlipOutput(
            loss= loss_lm,
            loss_lm=loss_lm,
        )

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=1,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        video = samples['video']
        audio = samples['audio']

        video_output = self.video_encoder(video)
        audio_output = self.audio_encoder(audio)

        video_embds = self.ln_video(video_output)
        audio_embds = self.ln_audio(audio_output)

        video_embds = self.video_project(video_embds)
        audio_embds = self.audio_project(audio_embds)

        concat_embds = torch.concat((video_embds,audio_embds),dim=1)

        # import nucleus sampling to improve the diversity of the generated captionings
        if not use_nucleus_sampling:
            concat_embds = concat_embds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1

        concat_attns = torch.ones(concat_embds.size()[:-1], dtype=torch.long).to(
            video.device
        )

        model_kwargs = {
            "encoder_hidden_states": concat_embds,
            "encoder_attention_mask": concat_attns,
        }

        input_ids = (
            torch.LongTensor(video.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(video.device)
        )

        query_tokens = self.query_tokens.expand(concat_embds.shape[0], -1, -1)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=True,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    def init_video_encoder(self,vit_encoder_pretrained='',vit_config=None,freeze_encoder=True):
        if self.pre_align:
            video_encoder = ClipVideo(vit_encoder_pretrained,vit_config,freeze_encoder).model
        else:
            video_encoder = VisionTransformer3D(
                                input_resolution=224,
                                patch_size=14,
                                width=1024,
                                layers=23,
                                heads=16,
                                use_grad_checkpointing=False,
                                num_frames=8
                            )
            video_encoder.load_state_dict(torch.load(vit_encoder_pretrained,map_location='cpu'))
            if freeze_encoder:
                for name,param in video_encoder.named_parameters():
                    param.requires_grad = False
        ln_video_encoder = LayerNorm(self.video_embed_dim)
        return video_encoder,ln_video_encoder

    def init_audio_encoder(self,aud_encoder_pretrained='',aud_config=None,freeze_encoder=True):
        if self.pre_align:
            audio_encoder = ClipAudio(aud_encoder_pretrained,aud_config,freeze_encoder).model
        else:
            audio_encoder = BeatsEncoder(aud_encoder_pretrained)
            if freeze_encoder:
                for name,param in audio_encoder.named_parameters():
                    param.requires_grad = False
        ln_audio_encoder = LayerNorm(self.audio_embed_dim)
        return audio_encoder,ln_audio_encoder



    def forward_image(self, image):
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, image_embeds

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, config):
        # Config and Checkpoint Paths
        pre_align = config.get("pre_align")
        vit_config = config.get("vit_config")
        aud_config = config.get("aud_config")
        video_embed_dim = config.get("video_encoder_embed")
        audio_embed_dim = config.get("audio_encoder_embed")
        vit_encoder_pretrained_ckpt = config.get("vit_pretrained")
        aud_encoder_pretrained_ckpt = config.get("aud_pretrained")
        bert_pretrained_ckpt = config.get("bert_pretrained")

        # Some other params
        num_query_token = config.get("num_query_token")
        cross_attention_freq = config.get("cross_attention_freq")
        freeze_encoder = config.get("freeze_encoder")
        embed_dim = config.get("embed_dim")
        max_txt_len = config.get("max_txt_len")

        model = cls(
            pre_align=pre_align,
            vit_config=vit_config,
            aud_config=aud_config,
            bert_pretrained=bert_pretrained_ckpt,
            video_embed_dim=video_embed_dim,
            audio_embed_dim=audio_embed_dim,
            vit_encoder_pretrained=vit_encoder_pretrained_ckpt,
            aud_encoder_pretrained=aud_encoder_pretrained_ckpt,
            freeze_encoder=freeze_encoder,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            embed_dim=embed_dim,
            max_txt_len=max_txt_len
        )
        # use this method to
        model.load_checkpoint_from_config(config)


        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)
