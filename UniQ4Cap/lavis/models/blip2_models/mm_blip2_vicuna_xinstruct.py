"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

"""
Requires Transformer 4.28 and above, implementation may change according the Llama implementation
"""
import logging
import os
import string

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)
from torch.nn.modules.module import _IncompatibleKeys

from lavis.common.dist_utils import download_cached_file
from lavis.common.utils import is_url
from lavis.models.base_model import BaseModel
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
from lavis.models.blip2_models.mm2_blip2 import Blip2Base, LayerNorm
from lavis.models.encoder import ClipVideo, ClipAudio
from lavis.processors.blip_processors import BlipCaptionProcessor
from lavis.tasks.multimodal_classification import MultimodalClassificationTask


class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)


class Blip2VicunaXInstruct(Blip2Base):
    """
    BLIP2 Vicuna model.
    Supported model types:
        - vicuna7b
        - vicuna13b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_vicuna_xinstruct", "vicuna7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "vicuna7b": "configs/models/blip2/blip2_xinstruct_vicuna7b.yaml",
        "vicuna13b": "configs/models/blip2/blip2_xinstruct_vicuna13b.yaml",
    }

    SEQUENCIAL_ENCODERS = [
        "eva_clip_g", 
        "beats"
    ]

    SEQUENCIAL_MODALITIES = [
        "video", 
        "audio"
    ]

    MODALITY_TO_CUE = {
        "image": " image: ",
        "pc": " 3d: ",
        "video": " video: ",
        "audio": " audio: ",
    }

    def __init__(
        self,
        # --------- Encoder And Q-Former part --------- #
        vit_config=None,
        aud_config=None,

        vit_encoder_pretrained="",
        aud_encoder_pretrained="",
        bert_pretrained="bert-base-cased",
        qformer_pretrained="",
        freeze_encoder=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        # --------- Encoder And Q-Former part --------- #

        modalities = ["image", "pc", "audio", "video"],
        use_cues=True,
        qformer_text_input=True,
        llm_text_input=False,
        apply_lemmatizer=False,
        
        ## llm model parameters
        llm_model="",
        lora_model="",
        lora=False,

        ## generation parameters
        prompt="",
        prefix="",
        postfix="",
        max_output_txt_len=256,
        special_qformer_input_prompt=False,
        enumerate_inputs=False,
        add_space=False,
        remove_start=False,
        clean_tokenization=False, # if set to true removes whitespace from cue, and start token from prompt. 

        ## shared Q-former setup
        shared_qformer=False,
        joint_video_audio=False,

        ## DisCRN
        use_caption=False,
        use_describe=False,

        ## classification setup
        predict_with_gen=False,
        format_candidates_prompt="{}",

        ## projection only parameters
        projection_only=False,
        proj_dim=1,
        ):
        super().__init__()

        # --------- Encoder And Q-Former part --------- #
        self.tokenizer = self.init_tokenizer(bert_pretrained)
        self.video_encoder = ClipVideo(vit_encoder_pretrained, vit_config, freeze=freeze_encoder).model
        self.audio_encoder = ClipAudio(aud_encoder_pretrained, aud_config, freeze=freeze_encoder).model
        self.ln_encoder = LayerNorm(1024)
        encoder_embeds = 1024
        qformer_embeds = 1408
        self.encoder_project = nn.Linear(encoder_embeds, qformer_embeds)
        logging.info("loading the freeze Vision & Audio Encoder Successfully")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token=num_query_token,
            feature_width=qformer_embeds,
            cross_attention_freq=cross_attention_freq,
            bert_pretrained=bert_pretrained,
            pretrained_qformer=qformer_pretrained
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
        # --------- Encoder And Q-Former part --------- #

        from transformers import LlamaTokenizer
        from lavis.models.blip2_models.modeling_llama import LlamaForCausalLM
        logging.info(f"Using modalities {modalities}")
        self.modalities = modalities

        logging.info(f"Shared Qformer is set to {shared_qformer}")
        self.shared_qformer = shared_qformer

        logging.info(f"Video-audio interleaving is set to {joint_video_audio}")
        self.joint_video_audio = joint_video_audio

        logging.info(f"Using Spacy en_core_wb_sm lemmatizer is set to {apply_lemmatizer}")
        self._lemmatizer = None
        self.apply_lemmatizer = apply_lemmatizer

        logging.info(f"Qformer text input {qformer_text_input} and LLM Text Input {llm_text_input}")
        self.qformer_text_input = qformer_text_input
        self.llm_text_input = llm_text_input

        self.projection_only = projection_only
        self.proj_dim = proj_dim
        logging.info(f"Projection only setup is set to {projection_only} with dimension {proj_dim}")

        ### Set up LLM ###
        logging.info(f"Setting up llm model {llm_model}")
        self.lora = lora
        print(f"Lora is set to {self.lora}")
        self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side="left")
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        if self.lora:
            # https://github.com/lxe/llama-peft-tuner/blob/main/finetune_peft.py
            self.llm_model = LlamaForCausalLM.from_pretrained(
            llm_model,
            load_in_8bit=True,
            torch_dtype=torch.float16)
            self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,
                lora_alpha=32, lora_dropout=0.1,
                target_modules=['q_proj', 'v_proj']
            )
            self.llm_model.gradient_checkpointing_enable()
            self.llm_model.enable_input_require_grads()
            self.llm_model.lm_head = CastOutputToFloat(self.llm_model.lm_head)
            self.llm_model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
            self.llm_hidden_size = self.llm_model.config.hidden_size
            self.llm_model = get_peft_model(self.llm_model, self.peft_config)
            self.lora_model = lora_model
        else:
            self.llm_model = LlamaForCausalLM.from_pretrained(
            llm_model, torch_dtype=torch.float16
            )
            self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
            self.llm_hidden_size = self.llm_model.config.hidden_size
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False
        self.llm_projection = nn.Linear(768, self.llm_hidden_size)
        self.clean_tokenization = clean_tokenization
        logging.info(f"Clean tokenization is set to {self.clean_tokenization}")

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt
        
        self.prefix = prefix
        if self.prefix:
            self.tokenized_prefix = self.llm_tokenizer(self.prefix, return_tensors="pt")

        self.postfix = postfix
        if type(self.postfix) != str or not self.postfix:
            self.postfix = ""
        logging.info(f"Using prefix set to {self.prefix} and postfix set to {self.postfix}.")

        ## generation parameters
        self.use_caption=use_caption
        self.use_describe=use_describe
        self.predict_with_gen=predict_with_gen
        self.format_candidates_prompt=format_candidates_prompt
        self.special_qformer_input_prompt=special_qformer_input_prompt
        self.enumerate_inputs=enumerate_inputs
        self.add_space=add_space
        self.remove_start=remove_start

        if self.projection_only:
            self.qformer_text_input=False



    def forward(self, samples):
        # -------- Encoder And Q-Former part --------- #

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        video_embeds = None
        audio_embeds = None
        if 'video' in samples.keys():
            video = samples["video"].to(device)
            video_output = self.video_encoder(video).last_hidden_state
            video_output = video_output.reshape(int(video_output.shape[0] / 8), -1, 1024)
            video_embeds = self.ln_encoder(video_output)
            video_embeds = self.encoder_project(video_embeds)

        if 'audio' in samples.keys():
            audio = samples["audio"].to(device)
            audio_output = self.audio_encoder(audio).last_hidden_state
            audio_embeds = self.ln_encoder(audio_output)
            audio_embeds = self.encoder_project(audio_embeds)
        text = samples["text"]

        # Add the Batch dimensions
        query_tokens = self.query_tokens.expand(len(text), -1, -1)

        # TODO: 到底要不要在这加个位置编码？
        # concat_embds = torch.concat((video_embeds, audio_embeds), dim=1)
        concat_embds = torch.cat((video_embeds, audio_embeds),
                                  dim=1) if video_embeds is not None and audio_embeds is not None else video_embeds if video_embeds is not None else audio_embeds
        concat_atts = torch.ones(concat_embds.size()[:-1], dtype=torch.long).to(
            self.device)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=concat_embds,
            encoder_attention_mask=concat_atts,
            use_cache=True,
            return_dict=True,
        )
        # -------- Encoder And Q-Former part --------- #


        # -------- Llm Projection ---------

        query_output = query_output.last_hidden_state
        inputs_llm = self.llm_projection(query_output)
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(self.device)
        # -------- Llm Projection --------- #


        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'

        if self.llm_text_input:
            input_sequence = [f"{t}{self.postfix}" for t in samples['text_input']] if self.postfix else samples['text']
            text_input_tokens = self.llm_tokenizer(
                input_sequence,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens= not self.clean_tokenization
            ).to(self.device)

        self.llm_tokenizer.truncation_side = 'right'
        output_sequence = [t + self.llm_tokenizer.eos_token for t in samples['text']]
        text_output_tokens = self.llm_tokenizer(
            output_sequence,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(self.device)

        if self.llm_text_input:
            llm_tokens, input_part_targets_len = self.concat_text_input_output(
                text_input_tokens.input_ids,
                text_input_tokens.attention_mask,
                text_output_tokens.input_ids,
                text_output_tokens.attention_mask,
            )
        else:
            llm_tokens = text_output_tokens
            input_part_targets_len = [0 for _ in range(llm_tokens['input_ids'].shape[0])] # input length is 0

        # do not apply loss to the padding
        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
        )

        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])

        bs = inputs_embeds.shape[0]

        att_list = []
        inp_list = []
        att_list.extend([atts_llm])
        inp_list.extend([inputs_llm])
       
        # do not apply loss to the query tokens
        empty_targets = (
            torch.ones(torch.cat(att_list, dim=1).size(), dtype=torch.long).to(self.device).fill_(-100)
        )

        # append llm prompt + output to queries
        att_list.append(llm_tokens['attention_mask'])
        inp_list.append(inputs_embeds)

        inputs_embeds = torch.cat(inp_list, dim=1)
        attention_mask = torch.cat(att_list, dim=1)
        targets = torch.cat([empty_targets, targets], dim=1)

        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        loss = outputs.loss

        return {"loss": loss}

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len

    @torch.no_grad()
    def get_query_outputs(
        self,
        samples
        ):
        if samples == None or samples == {}:
            return 

        curr_modalities = [modality for modality in self.modalities if modality in samples]
        if len(curr_modalities) == 0:
            print("Model modalities do not match sample modalities.")
            return
        
        # get batch size
        bs = None
        for modality in curr_modalities:
            data = samples[modality]
            bs = data.size(0)
            break
        
        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        elif "text_input" in samples.keys():
            prompt = samples["text_input"]
        else:
            prompt = self.prompt

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        embeds = {}
        query_tokens = {}
        data_atts = {}

        for modality in curr_modalities:
            data = samples[modality]
            ln = getattr(self, f"{modality}_ln")
            encoder = getattr(self, f"{modality}_encoder")
            if modality == "video" and self.video_enc_name in Blip2VicunaXInstruct.SEQUENCIAL_ENCODERS:
                embeds[modality] = []
                data_atts[modality] = []
                for j in range(data.size(2)):
                    this_frame = data[:,:,j,:,:]
                    with self.maybe_autocast():
                        embeds[modality].append(ln(encoder(this_frame)))
                        if self.shared_qformer:
                            embeds[modality][-1] = getattr(self, f"{modality}_encoder_projection")(embeds[modality][j])
                        data_atts[modality].append(torch.ones(embeds[modality][j].size()[:-1], dtype=torch.long).to(self.device))
                # B, Token Size, LM EMB
                query_tokens[modality] = getattr(self, f"{modality}_query_tokens").expand(data.size(0), -1, -1)

            elif modality == 'audio' and self.audio_enc_name in Blip2VicunaXInstruct.SEQUENCIAL_ENCODERS:
                embeds[modality] = []
                data_atts[modality] = []
                for j in range(data.size(1)):
                    this_frame = data[:,j,:,:]
                    with self.maybe_autocast():
                        embeds[modality].append(ln(encoder(this_frame)))
                        if self.shared_qformer:
                            embeds[modality][j] = getattr(self, f"{modality}_encoder_projection")(embeds[modality][j])
                    data_atts[modality].append(torch.ones(embeds[modality][j].size()[:-1], dtype=torch.long).to(self.device))
                # B, Token Size, LM EMB
                if not self.projection_only and not getattr(self, f"projection_only_{modality}"):
                    query_tokens[modality] = getattr(self, f"{modality}_query_tokens").expand(data.size(0), -1, -1)
            else:
                with self.maybe_autocast():
                    embeds[modality] = ln(encoder(data))
                if len(embeds[modality].size()) == 2:
                    # B, C, D
                    embeds[modality] = embeds[modality].unsqueeze(1)
                # B, C
                if self.shared_qformer:
                    embeds[modality] = getattr(self, f"{modality}_encoder_projection")(embeds[modality])
                
                data_atts[modality] = torch.ones(embeds[modality].size()[:-1], dtype=torch.long).to(self.device)
            
                # B, Token Size, LM EMB
                if not self.projection_only and not getattr(self, f"projection_only_{modality}"):
                    query_tokens[modality] = getattr(self, f"{modality}_query_tokens").expand(embeds[modality].shape[0], -1, -1)

        query_outputs = {}
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(self.device)

           
            Qformer_atts = {}
            query_atts = {}
            num = {}
            for modality in curr_modalities:
                # B, Token Size
                if not self.projection_only and not getattr(self, f"projection_only_{modality}"):
                    query_atts[modality] = torch.ones(query_tokens[modality].size()[:-1], dtype=torch.long).to(self.device)
                    # B, Token Size + Inp Size
                    Qformer_atts[modality] = torch.cat([query_atts[modality],text_Qformer.attention_mask],dim=1)
                if modality in Blip2VicunaXInstruct.SEQUENCIAL_MODALITIES and getattr(self, f'{modality}_enc_name') in Blip2VicunaXInstruct.SEQUENCIAL_ENCODERS:
                    num[modality] = len(embeds[modality])
                    bs = embeds[modality][0].shape[0]
                    indices = [j_+r for r,j in enumerate([[i*bs for i in range(num[modality])]]*bs) for j_ in j]
                    reordered_embeds = torch.cat(embeds[modality])[indices]
                    reordered_atts = torch.cat(data_atts[modality])[indices]
                    if self.projection_only or getattr(self, f"projection_only_{modality}"):
                        if self.proj_dim != 1:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.mean(1,keepdim=True)).view(bs*num[modality], self.num_query_token, -1)
                        else:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.view(reordered_embeds.shape[0],-1))
                        continue
                    query_output = getattr(self, f"{modality}_Qformer").bert(
                        text_Qformer.input_ids.repeat(num[modality], 1),
                        attention_mask=Qformer_atts[modality].repeat(num[modality], 1),
                        query_embeds=query_tokens[modality].repeat(num[modality], 1, 1),
                        encoder_hidden_states=reordered_embeds,
                        encoder_attention_mask=reordered_atts,
                        return_dict=True,
                    )
                    query_outputs[modality] = query_output
                else:
                    if self.projection_only or getattr(self, f"projection_only_{modality}"):
                        if self.proj_dim != 1:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality].mean(1, keepdim=True)).reshape(bs, self.num_query_token,-1)
                        else:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality]).reshape(bs, self.num_query_token,-1)
                        continue  
                    query_outputs[modality] = getattr(self, f"{modality}_Qformer").bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts[modality],
                        query_embeds=query_tokens[modality], 
                        encoder_hidden_states=embeds[modality].to(torch.float32), 
                        encoder_attention_mask=data_atts[modality], 
                        return_dict=True,
                    )
        else:
            num = {}
            for modality in curr_modalities:
                if modality in Blip2VicunaXInstruct.SEQUENCIAL_MODALITIES and getattr(self, f'{modality}_enc_name') in Blip2VicunaXInstruct.SEQUENCIAL_ENCODERS:
                    num[modality] = len(embeds[modality])
                    bs  = embeds[modality][0].shape[0]
                    indices = [j_+r for r,j in enumerate([[i*bs for i in range(num[modality])]]*bs) for j_ in j]
                    reordered_embeds = torch.cat(embeds[modality])[indices]
                    reordered_atts = torch.cat(data_atts[modality])[indices]
                    if self.projection_only or getattr(self, f"projection_only_{modality}"):
                        if self.proj_dim != 1:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.mean(1,keepdim=True)).view(bs*num, self.num_query_token, -1)
                        else:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.view(reordered_embeds.shape[0],-1))
                        continue
                    query_output = getattr(self, f"{modality}_Qformer").bert(
                        query_embeds=query_tokens[modality].repeat(num[modality], 1, 1),
                        encoder_hidden_states=reordered_embeds,
                        encoder_attention_mask=reordered_atts,
                        return_dict=True,
                    )
                    query_outputs[modality] = query_output
                else:   
                    if self.projection_only or getattr(self, f"projection_only_{modality}"):
                        if self.proj_dim != 1:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality].mean(1, keepdim=True)).reshape(bs, self.num_query_token,-1)
                        else:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality]).reshape(bs, self.num_query_token,-1)
                        continue  
                    query_outputs[modality] = getattr(self, f"{modality}_Qformer").bert(
                        query_embeds=query_tokens[modality],
                        encoder_hidden_states=embeds[modality].to(torch.float32), # pc data is floa16.
                        encoder_attention_mask=data_atts[modality],
                        return_dict=True,
                    )

        for modality in curr_modalities:
            if modality in Blip2VicunaXInstruct.SEQUENCIAL_MODALITIES and getattr(self, f'{modality}_enc_name') in Blip2VicunaXInstruct.SEQUENCIAL_ENCODERS:
                if self.projection_only or getattr(self, f"projection_only_{modality}"):  
                    if self.proj_dim != 1:
                        query_outputs[f'llm_proj_{modality}'] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].unsqueeze(1)).reshape(bs*num, self.num_query_token, -1)
                    else:
                        query_outputs[f'llm_proj_{modality}'] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality]).reshape(bs*num, self.num_query_token, -1)
                    query_outputs[f'llm_proj_{modality}'] = query_outputs[f'llm_proj_{modality}'].reshape(bs, num[modality], self.num_query_token, -1).contiguous().view(bs, num[modality]*self.num_query_token, -1)
                    query_outputs[modality] = query_outputs[modality].view(bs, num[modality]*self.num_query_token, -1)
                else:
                    query_outputs[f'llm_proj_{modality}']  = getattr(self, f"{modality}_llm_proj")(query_outputs[modality]['last_hidden_state'][:,:query_tokens[modality].size(1),:]).contiguous().view(bs, num[modality]*self.num_query_token, -1)
                    query_outputs[modality] = query_outputs[modality]['last_hidden_state'][:,:query_tokens[modality].size(1),:].contiguous().view(bs, num[modality]*self.num_query_token, -1)


            else:
                if self.projection_only or getattr(self, f"projection_only_{modality}"):
                    if self.proj_dim == 1:
                        query_outputs[f'llm_proj_{modality}'] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].mean(-1)).reshape(bs, self.num_query_token, -1)
                    else:
                        query_outputs[f'llm_proj_{modality}']= getattr(self, f"{modality}_llm_proj")(query_outputs[modality].reshape(bs, self.num_query_token, -1))
                else:
                    query_outputs[modality] = query_outputs[modality].last_hidden_state[:,:query_tokens[modality].size(1),:]
                    query_outputs[f'llm_proj_{modality}'] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality])

        for modality in curr_modalities:
            query_outputs[f'embeds_{modality}'] = embeds[modality]
        return query_outputs 

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
        special_qformer_input_prompt=False
        ):
        self.llm_tokenizer.padding_side = "left"

        if samples == None or samples == {}:
            return 

        if 'modalities' in samples:
            curr_modalities = samples['modalities'][0] if isinstance(samples['modalities'][0], list) else  samples['modalities']
        elif self.joint_video_audio:
            curr_modalities = ["video", "audio"]
        else:
            curr_modalities = [modality for modality in self.modalities if modality in samples]

        
        if len(curr_modalities) == 0:
            print("Model modalities do not match sample modalities.")
            return
            
        # get batch size
        bs = None
        for modality in curr_modalities:
            data = samples[modality]
            if isinstance(data, torch.Tensor):
                bs = data.size(0)
            else:
                bs = len(data)
            break
        
        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        elif self.prompt and 'text_input' in samples and '{}' in self.prompt:
            prompt = [self.prompt.format(t) for t in samples["text_input"]]
        elif "text_input" in samples.keys():
            prompt = samples["text_input"]
        else:
            prompt = self.prompt

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."            

        # For TextCaps
        if "ocr_tokens" in samples.keys() and "{}" in prompt[0]:
            prompt = [p.format(', '.join(samples['ocr_tokens'][i][:30])) for i, p in enumerate(prompt)]


        if 'discrn' in samples and self.use_caption: ## discriminatory reasoning
            if self.postfix:
                prompt = [f'{t}{self.postfix}' for t in prompt]
            if self.enumerate_inputs:
                prompt = [f'{self.prefix}(a){Blip2VicunaXInstruct.MODALITY_TO_CUE[samples["modalities"][i][0]] if self.use_cues else " "}{samples["baseline_captions"][i][0]} (b){Blip2VicunaXInstruct.MODALITY_TO_CUE[samples["modalities"][i][1]] if self.use_cues else " "}{samples["baseline_captions"][i][1]} {prompt[i]}' for i in range(bs)]
            else:
                prompt = [f'{self.prefix}{Blip2VicunaXInstruct.MODALITY_TO_CUE[samples["modalities"][i][0]]}{samples["baseline_captions"][i][0] if self.use_cues else " "}{Blip2VicunaXInstruct.MODALITY_TO_CUE[samples["modalities"][i][1]] if self.use_cues else " "}{samples["baseline_captions"][i][1]} {prompt[i]}' for i in range(bs)]
            llm_tokens = self.llm_tokenizer(
                prompt,
                padding="longest",
                return_tensors="pt"
            ).to(self.device)
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
        
            with self.maybe_autocast():
                outputs = self.llm_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=llm_tokens.attention_mask,
                    do_sample=use_nucleus_sampling,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_length=max_length,
                    min_length=min_length,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                )
        
            outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
            output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text = [o.strip() for o in output_text]
            # print(output)
            return output_text

        query_tokens = {}
        for modality in curr_modalities:
            if not self.projection_only and not getattr(self, f"projection_only_{modality}"):
                query_tokens[modality] = getattr(self, f"{modality}_query_tokens").expand(bs, -1, -1)
        if self.qformer_text_input:
            if self.special_qformer_input_prompt or special_qformer_input_prompt:  
                qformer_prompt = special_qformer_input_prompt if special_qformer_input_prompt else self.special_qformer_input_prompt
                qformer_prompt = [qformer_prompt] * len(prompt)
                if "text_input" in samples.keys():
                    if type(samples["text_input"][0]) == list:
                        qformer_prompt = [qformer_prompt[i].format(*samples["text_input"][i]) for i in range(len(qformer_prompt))]
                    else:
                        qformer_prompt = [qformer_prompt[i].format(samples["text_input"][i]) for i in range(len(qformer_prompt))]
                text_Qformer = self.tokenizer(
                    qformer_prompt,
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(self.device)

            elif self.use_describe:
                modality2prompt = {
                    "video": "a short description of the video",
                    "audio": "an audio that shows",
                    "image": "a short image caption",
                    "pc": "a 3d model of"
                }
                qformer_prompt = [modality2prompt[modality] for _ in samples['text_input']]

                text_Qformer = self.tokenizer(
                    qformer_prompt,
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(self.device)

            else:
                text_Qformer = self.tokenizer(
                    prompt,
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(self.device)

            Qformer_atts = {}
            query_atts = {}

            for modality in curr_modalities:
                if not  getattr(self, f"projection_only_{modality}"):
                    # B, Token Size
                    query_atts[modality] = torch.ones(query_tokens[modality].size()[:-1], dtype=torch.long).to(self.device)
                    # B, Token Size + Inp Size
                    Qformer_atts[modality] = torch.cat([query_atts[modality],text_Qformer.attention_mask],dim=1)

        embeds = {}
        data_atts = {}
        for modality in curr_modalities:
            data = samples[modality]
            ln = getattr(self, f"{modality}_ln")
            encoder = getattr(self, f"{modality}_encoder")
            if modality == "video" and "clip" in self.video_enc_name:
                embeds[modality] = []
                data_atts[modality] = []
                for j in range(data.size(2)):
                    this_frame = data[:,:,j,:,:]
                    with self.maybe_autocast():
                        embeds[modality].append(ln(encoder(this_frame)))
                        if self.shared_qformer:
                            embeds[modality][j] = getattr(self, f"{modality}_encoder_projection")(embeds[modality][j])
                        data_atts[modality].append(torch.ones(embeds[modality][j].size()[:-1], dtype=torch.long).to(self.device))
            elif modality == 'audio' and 'beats' in self.audio_enc_name:
                embeds[modality] = []
                data_atts[modality] = []
                for j in range(data.size(1)):
                    this_frame = data[:,j,:,:]
                    with self.maybe_autocast():
                        embeds[modality].append(ln(encoder(this_frame)))
                        if self.shared_qformer:
                            embeds[modality][j] = getattr(self, f"{modality}_encoder_projection")(embeds[modality][j])
                    data_atts[modality].append(torch.ones(embeds[modality][j].size()[:-1], dtype=torch.long).to(self.device))
            else:
                with self.maybe_autocast():
                    embeds[modality] = ln(encoder(data))
                if len(embeds[modality].size()) == 2:
                    embeds[modality] = embeds[modality].unsqueeze(1)
                if self.shared_qformer:
                    with self.maybe_autocast():
                        embeds[modality] = getattr(self, f"{modality}_encoder_projection")(embeds[modality])
                data_atts[modality] = torch.ones(embeds[modality].size()[:-1], dtype=torch.long).to(self.device)
            
        query_outputs = {}
        num = {}
        if self.qformer_text_input:
            for modality in curr_modalities:
                if modality in Blip2VicunaXInstruct.SEQUENCIAL_MODALITIES and getattr(self, f'{modality}_enc_name') in Blip2VicunaXInstruct.SEQUENCIAL_ENCODERS:
                    num[modality] = len(embeds[modality])
                    bs = embeds[modality][0].shape[0]
                    indices = [j_+r for r,j in enumerate([[i*bs for i in range(num[modality])]]*bs) for j_ in j]
                    reordered_embeds = torch.cat(embeds[modality])[indices]
                    reordered_atts = torch.cat(data_atts[modality])[indices]
                    if self.projection_only or getattr(self, f"projection_only_{modality}"):
                        if self.proj_dim != 1:
                                query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.mean(1,keepdim=True)).view(bs*num[modality], self.num_query_token, -1)
                        else:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.view(reordered_embeds.shape[0],-1))
                        continue
                    query_output = getattr(self, f"{modality}_Qformer").bert(
                        text_Qformer.input_ids.repeat(num[modality], 1),
                        attention_mask=Qformer_atts[modality].repeat(num[modality], 1),
                        query_embeds=query_tokens[modality].repeat(num[modality], 1, 1),
                        encoder_hidden_states=reordered_embeds,
                        encoder_attention_mask=reordered_atts,
                        return_dict=True,
                    )
                    query_outputs[modality] = query_output
                else:
                    bs = embeds[modality].shape[0]
                    if self.projection_only or getattr(self, f"projection_only_{modality}"):
                        if self.proj_dim != 1:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality].mean(1, keepdim=True)).reshape(bs, self.num_query_token,-1)
                        else:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality]).reshape(bs, self.num_query_token,-1)
                        continue  
                    query_outputs[modality] = getattr(self, f"{modality}_Qformer").bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts[modality],
                        query_embeds=query_tokens[modality],
                        encoder_hidden_states=embeds[modality].to(torch.float32),
                        encoder_attention_mask=data_atts[modality],
                        return_dict=True,
                    )
        else:
            for modality in curr_modalities:
                if modality in Blip2VicunaXInstruct.SEQUENCIAL_MODALITIES and getattr(self, f'{modality}_enc_name') in Blip2VicunaXInstruct.SEQUENCIAL_ENCODERS:
                    num[modality] = len(embeds[modality])
                    bs = embeds[modality][0].shape[0]
                    indices = [j_+r for r,j in enumerate([[i*bs for i in range(num[modality])]]*bs) for j_ in j]
                    reordered_embeds = torch.cat(embeds[modality])[indices]
                    reordered_atts = torch.cat(data_atts[modality])[indices]
                    if self.projection_only or getattr(self, f"projection_only_{modality}"):
                        if self.proj_dim != 1:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.mean(1,keepdim=True)).view(bs*num[modality], self.num_query_token, -1)
                        else:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.view(reordered_embeds.shape[0],-1))
                        continue
                    query_output = getattr(self, f"{modality}_Qformer").bert(
                        query_embeds=query_tokens[modality].repeat(num[modality], 1, 1),
                        encoder_hidden_states=reordered_embeds,
                        encoder_attention_mask=reordered_atts,
                        return_dict=True,
                    )
                    query_outputs[modality] = query_output
                else:
                    bs = embeds[modality].shape[0]
                    if self.projection_only or getattr(self, f"projection_only_{modality}"):
                        with self.maybe_autocast():
                            if self.proj_dim != 1:
                                query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality].mean(1, keepdim=True)).reshape(bs, self.num_query_token,-1)
                            else:
                                query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality]).reshape(bs, self.num_query_token,-1)
                            continue  
                    query_outputs[modality] = getattr(self, f"{modality}_Qformer").bert(
                        query_embeds=query_tokens[modality],
                        encoder_hidden_states=embeds[modality].to(torch.float32),
                        encoder_attention_mask=data_atts[modality],
                        return_dict=True,
                    )
    
        inputs_llm = {}
        atts_llm = {}
        enumeration = {}

        for i,modality in enumerate(curr_modalities):
            if modality in Blip2VicunaXInstruct.SEQUENCIAL_MODALITIES and getattr(self, f'{modality}_enc_name') in Blip2VicunaXInstruct.SEQUENCIAL_ENCODERS:
                if self.projection_only or getattr(self, f"projection_only_{modality}"):
                    if self.proj_dim != 1:
                        inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].unsqueeze(1)).reshape(bs*num[modality], self.num_query_token, -1)
                    else:
                        inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].reshape(bs*num, self.num_query_token, -1))
                    inputs_llm[modality] = inputs_llm[modality].reshape(bs, num[modality], self.num_query_token, -1).view(bs, num[modality]*self.num_query_token, -1)
                    atts_llm[modality] =  torch.ones(inputs_llm[modality].size()[:-1], dtype=torch.long).to(self.device)
                    continue
                # num*bs, num query tokens, llm emb size
                inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].last_hidden_state[:,:query_tokens[modality].size(1),:]) 
                # bs, num, num query tokens, llm emb size -> bs, num*num query tokens, llm emb size
                inputs_llm[modality] = inputs_llm[modality].reshape(bs, num[modality], self.num_query_token, -1).view(bs,  num[modality]*self.num_query_token, -1)
                atts_llm[modality] =  torch.ones(inputs_llm[modality].size()[:-1], dtype=torch.long).to(self.device)
            else:
                if self.projection_only or getattr(self, f"projection_only_{modality}"):
                    if self.proj_dim == 1:
                        inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].mean(-1)).reshape(bs, self.num_query_token, -1)
                    else:
                        inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].reshape(bs, self.num_query_token, -1))
                    atts_llm[modality] =  torch.ones(inputs_llm[modality].size()[:-1], dtype=torch.long).to(self.device)
                    continue
                inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality]['last_hidden_state'][:,:query_tokens[modality].size(1),:])
                atts_llm[modality] = torch.ones(inputs_llm[modality].size()[:-1], dtype=torch.long).to(self.device)
            if self.enumerate_inputs:
                enumeration[modality] = self.llm_tokenizer(
                [f"{'' if i == 0 else ' '}({chr(97+i)}) " for _ in prompt],
                return_tensors="pt",
                add_special_tokens=False if (i!= 0 or self.prefix) else True
                ).to(self.device)

        ## remove trailing whitespace 
        prompt = [p.strip() for p in prompt]

        if 'dialog' in samples:
            llm_tokens = self.llm_tokenizer(
                [f"{d} {p}" if d else p for d, p in zip(samples['dialog'], prompt)],
                padding="longest",
                return_tensors="pt",
                add_special_tokens= not self.clean_tokenization
            ).to(self.device)
        else:
            llm_tokens = self.llm_tokenizer(
                [f"{p}{self.postfix}" for p in prompt] if self.postfix else prompt,
                padding="longest",
                return_tensors="pt",
                add_special_tokens= not self.clean_tokenization
            ).to(self.device)
        bs = llm_tokens.input_ids.shape[0]

        att_list = []
        inp_list = []
        if self.prefix:
            att_list = [self.tokenized_prefix.attention_mask.repeat(bs, 1).to(self.device)]
            inp_list = [self.llm_model.get_input_embeddings()(self.tokenized_prefix.input_ids.to(self.device)).repeat(bs, 1, 1)]            

        if self.joint_video_audio:
            for pos in range(num['video']):
                if self.enumerate_inputs:
                    enumeration_pos = self.llm_tokenizer(
                        [f"{'' if pos == 0 else ' '}({chr(97+pos)}) " for _ in prompt],
                        return_tensors="pt",
                        add_special_tokens=False if (pos!= 0 or self.prefix) else True
                        ).to(self.device)
                    enumeration_inputs_llm = self.llm_model.get_input_embeddings()(enumeration_pos.input_ids)
                    enumeration_atts_llm = enumeration_pos.attention_mask.to(self.device)
                    inp_list.extend([enumeration_inputs_llm])
                    att_list.extend([enumeration_atts_llm])
                if self.use_cues:
                    for modality in ['video', 'audio']:
                        if self.clean_tokenization:
                            if self.prefix or pos > 1 or self.enumerate_inputs or modality == 'audio':
                                att_list.extend([torch.tensor(self.tokenized_cue[modality].attention_mask[:,1:]).to(self.device).repeat(atts_llm[modality].shape[0], 1), atts_llm[modality].view(bs,  num[modality], self.num_query_token)[:, pos, :]])
                                inp_list.extend([self.emb_cue[modality][:,1:].to(self.device).repeat(inputs_llm[modality].shape[0], 1, 1), inputs_llm[modality].view(bs,  num[modality], self.num_query_token, -1)[:, pos, :, :]])
                                continue
                        att_list.extend([torch.tensor(self.tokenized_cue[modality].attention_mask).to(self.device).repeat(atts_llm[modality].shape[0], 1), atts_llm[modality].view(bs,  num[modality], self.num_query_token)[:, pos, :]])
                        inp_list.extend([self.emb_cue[modality].to(self.device).repeat(inputs_llm[modality].shape[0], 1, 1), inputs_llm[modality].view(bs,  num[modality], self.num_query_token, -1)[:, pos, :, :]])
                else:
                    att_list.extend([atts_llm[modality].view(bs, num[modality], self.num_query_token)[:, pos, :]])
                    inp_list.extend([inputs_llm[modality].view(bs, num[modality], self.num_query_token, -1)[:, pos, :, :]])
        else:
            for modality in curr_modalities:
                if self.enumerate_inputs:
                    enumeration_inputs_llm = self.llm_model.get_input_embeddings()(enumeration[modality].input_ids.to(self.device))
                    enumeration_atts_llm = enumeration[modality].attention_mask.to(self.device)
                    inp_list.extend([enumeration_inputs_llm])
                    att_list.extend([enumeration_atts_llm])
                if self.use_cues:
                    if self.clean_tokenization or self.remove_start:
                        if (modality==curr_modalities[0] and not (self.prefix or self.enumerate_inputs)):
                            att_list.extend([torch.tensor(self.tokenized_cue[modality].attention_mask).to(self.device).repeat(atts_llm[modality].shape[0], 1), atts_llm[modality]])
                            inp_list.extend([self.emb_cue[modality].to(self.device).repeat(inputs_llm[modality].shape[0], 1, 1), inputs_llm[modality]])
                        else:
                            att_list.extend([torch.tensor(self.tokenized_cue[modality].attention_mask[:,1:]).to(self.device).repeat(atts_llm[modality].shape[0], 1), atts_llm[modality]])
                            inp_list.extend([self.emb_cue[modality][:,1:].to(self.device).repeat(inputs_llm[modality].shape[0], 1, 1), inputs_llm[modality]])
                    else:
                        att_list.extend([torch.tensor(self.tokenized_cue[modality].attention_mask).to(self.device).repeat(atts_llm[modality].shape[0], 1), atts_llm[modality]])
                        inp_list.extend([self.emb_cue[modality].to(self.device).repeat(inputs_llm[modality].shape[0], 1, 1), inputs_llm[modality]])

                else:
                    att_list.extend([atts_llm[modality]])
                    inp_list.extend([inputs_llm[modality]])

                if self.add_space:
                    space_tok =  self.llm_tokenizer(
                        [f" " for _ in prompt],
                        return_tensors="pt",
                        add_special_tokens=False
                        )
                    space_inputs_llm = self.llm_model.get_input_embeddings()(space_tok.input_ids.to(self.device))
                    space_atts_llm = space_tok.attention_mask.to(self.device)
                    inp_list.extend([space_inputs_llm])
                    att_list.extend([space_atts_llm])

        att_list.append(llm_tokens.attention_mask)
        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
        inp_list.append(inputs_embeds)
       
        attention_mask = torch.cat(att_list, dim=1)
        inputs_embeds = torch.cat(inp_list, dim=1)

       
        with self.maybe_autocast():
            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
        outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [o.strip() for o in output_text]
        return output_text
    
    @torch.no_grad()
    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        **kwargs
        ):
        if samples == None or samples == {}:
            return None

        # get batch size
        bs = None
        if 'modalities' in samples:
            curr_modalities = samples['modalities'][0] if isinstance(samples['modalities'][0], list) else  samples['modalities']
        else:
            curr_modalities = [modality for modality in self.modalities if modality in samples]
        for modality in curr_modalities:
            data = samples[modality]
            if isinstance(data, torch.Tensor):
                bs = data.size(0)   
            else:
                bs = len(data)     
            break

        if "text_input" not in samples:
            samples["text_input"] = self.prompt
        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]] * bs
        text_input = samples['text_input']

        if not prompt and self.prompt:
            prompt=self.prompt
        if prompt:
            if prompt.count("{}") == 2:
                if 'ocr_tokens' in samples:
                    text_input = [
                        prompt.format(', '.join(samples['ocr_tokens'][i][:30]), samples["text_input"][i])
                    for i in range(len(samples["text_input"]))]
                elif 'choices' in samples:
                    text_input = []
                    for i in range(len(samples["text_input"])):
                        this_choices = [f"({string.ascii_lowercase[j]}) {ch}" for j, ch in enumerate(samples["choices"][i])]
                        this_choices = " ".join(this_choices)
                        text_input.append(prompt.format(samples["text_input"][i], this_choices))
            else:
                text_input = [prompt.format(question) for question in samples["text_input"]]
            samples["prompt"] = text_input

        if 'discrn' in samples and self.use_caption: ## discriminatory reasoning
            self.llm_tokenizer.padding_side = "left"

            text_input = samples['text_input'] if 'prompt' not in samples else samples['prompt']
            if self.postfix:
                text_input = [f'{t}{self.postfix}' for t in text_input]
            if self.enumerate_inputs:
                prompt = [f'{self.prefix}(a){Blip2VicunaXInstruct.MODALITY_TO_CUE[samples["modalities"][i][0]] if self.use_cues else " "}{samples["baseline_captions"][i][0]} (b){Blip2VicunaXInstruct.MODALITY_TO_CUE[samples["modalities"][i][1]] if self.use_cues else " "}{samples["baseline_captions"][i][1]} {text_input[i]}' for i in range(bs)]
            else:
                prompt = [f'{self.prefix}{Blip2VicunaXInstruct.MODALITY_TO_CUE[samples["modalities"][i][0]]}{samples["baseline_captions"][i][0] if self.use_cues else " "}{Blip2VicunaXInstruct.MODALITY_TO_CUE[samples["modalities"][i][1]] if self.use_cues else " "}{samples["baseline_captions"][i][1]} {text_input[i]}' for i in range(bs)]
            llm_tokens = self.llm_tokenizer(
                prompt,
                padding="longest",
                return_tensors="pt"
            ).to(self.device)

            with self.maybe_autocast():
                outputs = self.llm_model.generate(
                    inputs_embeds=self.llm_model.get_input_embeddings()(llm_tokens.input_ids),
                    attention_mask=llm_tokens.attention_mask,
                    do_sample=False,
                    num_beams=num_beams,
                    max_length=max_len,
                    min_length=min_len,
                    repetition_penalty=1.5,
                    # eos_token_id=self.eos_token_id,
                    length_penalty=length_penalty,
                )
            outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
            output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return output_text

        output_text = self.generate(
            samples,
            num_beams=num_beams,
            max_length=max_len,
            min_length=min_len,
            length_penalty=length_penalty
        )

        if "apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]:
            output_text = self._lemmatize(output_text)
        
        #vizwiz
        output_text = [o if o != "" else "unanswerable" for o in output_text]

        return output_text

    def predict(
        self,
        samples,
        candidates=None,
        n_segments=1,
        max_length=10,
        min_length=1,
        length_penalty=-1.,
        special_qformer_input_prompt=False
        ):

        self.llm_tokenizer.padding_side = "left"

        if candidates == None:
            candidates = self.candidates
        else:
            self.candidates = candidates # for the output targets.
        
        if self.predict_with_gen:
            output = self.generate(samples,max_length=max_length,min_length=min_length,length_penalty=length_penalty)
            result = []
            for text in output:
                text = BlipCaptionProcessor().pre_caption(text)
                pred_label = ""  # default to an empty string
                for cand in candidates:
                    cand = BlipCaptionProcessor().pre_caption(cand)
                    if cand in text.split(" "):
                        pred_label = cand
                        break  # stop as soon as we find a match
                result.append(pred_label)
            return {"predictions":result, "target": samples["label"]}


        # If candidates is a list of lists, each sample has its candidates, then we need to iterate one by one
        if type(candidates[0]) == list:
            results = []

            for i in range(samples["image"].size(0)):
                this_sample = {
                    "image": samples["image"][i].unsqueeze(0),
                    "prompt": samples["prompt"],
                }

                if "text_input" in samples.keys():
                    this_sample["text_input"] = [samples["text_input"][i]]

                if 'context' in samples.keys():
                    this_sample['context'] = [samples["context"][i]]

                if 'history' in samples.keys():
                    this_sample['history'] = [samples["history"][i]]

                if 'caption' in samples.keys():
                    this_sample['caption'] = [samples["caption"][i]]

                this_result = self._predict_class(this_sample, candidates[i], n_segments, special_qformer_input_prompt)
                results.append(this_result)

            try:
                results = torch.cat(results, dim=0)
            except:
                results = [res.tolist()[0] for res in results]

            return results

        return self._predict_class(samples, candidates, n_segments, special_qformer_input_prompt)

    def _predict_class(
        self,
        samples,
        candidates,
        n_segments=1,
        special_qformer_input_prompt=False,
        ):
        if list(samples.keys()) == []:
            return None
    
        if "prompt" in samples:
            prompt = samples["prompt"]
        else:
            prompt = self.prompt
        
        candidates = [self.format_candidates_prompt.format(c) for c in candidates]

        if 'modalities' in samples:
            curr_modalities = samples['modalities'][0] if isinstance(samples['modalities'][0], list) else  samples['modalities']
        else:
            curr_modalities = [modality for modality in self.modalities if modality in samples]
        
        # get batch size
        for modality in curr_modalities:
            data = samples[modality]
            if isinstance(data, torch.Tensor):
                bs = data.size(0)
            else:
                bs = len(data)
            break

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        if "text_input" in samples.keys():
            if type(samples["text_input"][0]) == list:
                prompt = [prompt[i].format(*samples["text_input"][i]) for i in range(len(prompt))]
            else:
                prompt = [prompt[i].format(samples["text_input"][i]) for i in range(len(prompt))]

        # scienceqa
        if 'context' in samples.keys() and samples['context'] != '':
            prompt = [f'context: {samples["context"][i]}. {prompt[i]}' for i in range(len(prompt))]

        # visual dialog
        if 'history' in samples.keys() and samples['history'][0] != '':
            prompt = [f'dialog history: {samples["history"][i]}\n{prompt[i]}' for i in range(len(prompt))]

        if 'caption' in samples.keys() and samples['caption'][0] != '':
            prompt = [f'This image has the caption "{samples["caption"][i]}". {prompt[i]}' for i in range(len(prompt))]


                
        if 'discrn' in samples and self.use_caption: ## discriminatory reasoning
            if self.postfix:
                prompt = [f'{p}{self.postfix}' for p in prompt]
            if self.enumerate_inputs:
                prompt = [f'{self.prefix}(a){Blip2VicunaXInstruct.MODALITY_TO_CUE[samples["modalities"][i][0]] if self.use_cues else " "}{samples["baseline_captions"][i][0]} (b){Blip2VicunaXInstruct.MODALITY_TO_CUE[samples["modalities"][i][1]] if self.use_cues else " "}{samples["baseline_captions"][i][1]} {prompt[i]}' for i in range(bs)]
            else:
                prompt = [f'{self.prefix}{Blip2VicunaXInstruct.MODALITY_TO_CUE[samples["modalities"][i][0]]}{samples["baseline_captions"][i][0] if self.use_cues else " "}{Blip2VicunaXInstruct.MODALITY_TO_CUE[samples["modalities"][i][1]] if self.use_cues else " "}{samples["baseline_captions"][i][1]} {prompt[i]}' for i in range(bs)]
            text_input_tokens = self.llm_tokenizer(
                prompt,
                padding="longest",
                return_tensors="pt"
            ).to(self.device)
        else:
            if not self.projection_only and not getattr(self, f"projection_only_{modality}"):
                query_tokens = {}
                for modality in self.modalities:
                    if modality not in samples:
                            continue
                    query_tokens[modality] = getattr(self, f"{modality}_query_tokens").expand(bs, -1, -1)
            
            if self.qformer_text_input:
                if self.special_qformer_input_prompt or special_qformer_input_prompt:
                    
                    qformer_prompt = special_qformer_input_prompt if special_qformer_input_prompt else self.special_qformer_input_prompt
                    qformer_prompt = [qformer_prompt] * len(prompt)
                    if "text_input" in samples.keys():
                        if type(samples["text_input"][0]) == list:
                            qformer_prompt = [qformer_prompt[i].format(*samples["text_input"][i]) for i in range(len(qformer_prompt))]
                        else:
                            qformer_prompt = [qformer_prompt[i].format(samples["text_input"][i]) for i in range(len(qformer_prompt))]

                    text_Qformer = self.tokenizer(
                        qformer_prompt,
                        padding='longest',
                        truncation=True,
                        max_length=self.max_txt_len,
                        return_tensors="pt",
                    ).to(self.device)
                elif self.use_describe:
                    modality2prompt = {
                    "video": "a short description of the video",
                    "audio": "an audio that shows",
                    "image": "a short image caption",
                    "pc": "a 3d model of"
                    }
                    qformer_prompt = [modality2prompt[modality] for _ in samples['text_input']]

                    # qformer_prompt = [f'Describe the {Blip2VicunaXInstruct.MODALITY_TO_CUE[modality].replace(":", "").strip() if modality != "pc" else "3d model"}.' for _ in samples["text_input"]]
                    text_Qformer = self.tokenizer(
                        qformer_prompt,
                        padding='longest',
                        truncation=True,
                        max_length=self.max_txt_len,
                        return_tensors="pt",
                    ).to(self.device)
                else:
                    text_Qformer = self.tokenizer(
                        prompt,
                        padding='longest',
                        truncation=True,
                        max_length=self.max_txt_len,
                        return_tensors="pt",
                    ).to(self.device)
                
                Qformer_atts = {}
                query_atts = {}
                
                for modality in curr_modalities:
                    # B, Token Size
                    query_atts[modality] = torch.ones(query_tokens[modality].size()[:-1], dtype=torch.long).to(self.device)
                    # B, Token Size + Inp Size
                    Qformer_atts[modality] = torch.cat([query_atts[modality],text_Qformer.attention_mask],dim=1)
                
            embeds = {}
            data_atts = {}
            for modality in curr_modalities:
                data = samples[modality]
                ln = getattr(self, f"{modality}_ln")
                encoder = getattr(self, f"{modality}_encoder")
                if modality == "video" and "clip" in self.video_enc_name:
                    embeds[modality] = []
                    data_atts[modality] = []
                    for j in range(data.size(2)):
                        this_frame = data[:,:,j,:,:]
                        with self.maybe_autocast():
                            embeds[modality].append(ln(encoder(this_frame)))
                            if self.shared_qformer:
                                embeds[modality][j] = getattr(self, f"{modality}_encoder_projection")(embeds[modality][j])
                            data_atts[modality].append(torch.ones(embeds[modality][j].size()[:-1], dtype=torch.long).to(self.device))

                elif modality == 'audio' and 'beats' in self.audio_enc_name:
                    embeds[modality] = []
                    data_atts[modality] = []
                    for j in range(data.size(1)):
                        this_frame = data[:,j,:,:]
                        with self.maybe_autocast():
                            embeds[modality].append(ln(encoder(this_frame)))
                            if self.shared_qformer:
                                embeds[modality][j] = getattr(self, f"{modality}_encoder_projection")(embeds[modality][j])
                        data_atts[modality].append(torch.ones(embeds[modality][j].size()[:-1], dtype=torch.long).to(self.device))
                else:
                    with self.maybe_autocast():
                        embeds[modality] = ln(encoder(data))
                    if len(embeds[modality].size()) == 2:
                        # B, C, D
                        embeds[modality] = embeds[modality].unsqueeze(1)
                    # B, C
                    if self.shared_qformer:
                        embeds[modality] = getattr(self, f"{modality}_encoder_projection")(embeds[modality])
                    data_atts[modality] = torch.ones(embeds[modality].size()[:-1], dtype=torch.long).to(self.device)
                
            query_outputs = {}
            num = {}
            if self.qformer_text_input:
                for modality in curr_modalities:
                    if modality in Blip2VicunaXInstruct.SEQUENCIAL_MODALITIES and getattr(self, f'{modality}_enc_name') in Blip2VicunaXInstruct.SEQUENCIAL_ENCODERS:
                        num[modality] = len(embeds[modality])
                        bs = embeds[modality][0].shape[0]
                        indices = [j_+r for r,j in enumerate([[i*bs for i in range(num[modality])]]*bs) for j_ in j]
                        reordered_embeds = torch.cat(embeds[modality])[indices]
                        reordered_atts = torch.cat(data_atts[modality])[indices]
                        if self.projection_only or getattr(self, f"projection_only_{modality}"):
                            if self.proj_dim != 1:
                                    query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.mean(1,keepdim=True)).view(bs*num[modality], self.num_query_token, -1)
                            else:
                                query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.view(reordered_embeds.shape[0],-1))
                            continue
                        query_output = getattr(self, f"{modality}_Qformer").bert(
                            text_Qformer.input_ids.repeat(num[modality], 1),
                            attention_mask=Qformer_atts[modality].repeat(num[modality], 1),
                            query_embeds=query_tokens[modality].repeat(num[modality], 1, 1),
                            encoder_hidden_states=reordered_embeds,
                            encoder_attention_mask=reordered_atts,
                            return_dict=True,
                        )
                        query_outputs[modality] = query_output
                    else:
                        bs = embeds[modality].shape[0]
                        if self.projection_only or getattr(self, f"projection_only_{modality}"):
                            if self.proj_dim != 1:
                                query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality].mean(1, keepdim=True)).reshape(bs, self.num_query_token,-1)
                            else:
                                query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality]).reshape(bs, self.num_query_token,-1)
                            continue  
                        query_outputs[modality] = getattr(self, f"{modality}_Qformer").bert(
                            text_Qformer.input_ids,
                            attention_mask=Qformer_atts[modality],
                            query_embeds=query_tokens[modality],
                            encoder_hidden_states=embeds[modality].to(torch.float32),
                            encoder_attention_mask=data_atts[modality],
                            return_dict=True,
                        )
            else:
                for modality in curr_modalities:
                    if modality in Blip2VicunaXInstruct.SEQUENCIAL_MODALITIES and getattr(self, f'{modality}_enc_name') in Blip2VicunaXInstruct.SEQUENCIAL_ENCODERS:
                        num[modality] = len(embeds[modality])
                        bs = embeds[modality][0].shape[0]
                        indices = [j_+r for r,j in enumerate([[i*bs for i in range(num[modality])]]*bs) for j_ in j]
                        reordered_embeds = torch.cat(embeds[modality])[indices]
                        reordered_atts = torch.cat(data_atts[modality])[indices]
                        if self.projection_only or getattr(self, f"projection_only_{modality}"):
                            if self.proj_dim != 1:
                                query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.mean(1,keepdim=True)).view(bs*num[modality], self.num_query_token, -1)
                            else:
                                query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.view(reordered_embeds.shape[0],-1))
                            continue
                        query_output = getattr(self, f"{modality}_Qformer").bert(
                            query_embeds=query_tokens[modality].repeat(num[modality], 1, 1),
                            encoder_hidden_states=reordered_embeds,
                            encoder_attention_mask=reordered_atts,
                            return_dict=True,
                        )
                        query_outputs[modality] = query_output
                    else:
                        bs = embeds[modality].shape[0]
                        if self.projection_only or getattr(self, f"projection_only_{modality}"):
                            if self.proj_dim != 1:
                                query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality].mean(1, keepdim=True)).reshape(bs, self.num_query_token,-1)
                            else:
                                query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality]).reshape(bs, self.num_query_token,-1)
                            continue  
                        query_outputs[modality] = getattr(self, f"{modality}_Qformer").bert(
                            query_embeds=query_tokens[modality],
                            encoder_hidden_states=embeds[modality].to(torch.float32),
                            encoder_attention_mask=data_atts[modality],
                            return_dict=True,
                        )
            
            inputs_llm = {}
            atts_llm = {}
            enumeration = {}
            # from pdb import set_trace; set_trace()
            for i,modality in enumerate(curr_modalities):
                if modality in Blip2VicunaXInstruct.SEQUENCIAL_MODALITIES and getattr(self, f'{modality}_enc_name') in Blip2VicunaXInstruct.SEQUENCIAL_ENCODERS:
                    if self.projection_only or getattr(self, f"projection_only_{modality}"):
                        if self.proj_dim != 1:
                            inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].unsqueeze(1)).reshape(bs*num[modality], self.num_query_token, -1)
                        else:
                            inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].reshape(bs*num, self.num_query_token, -1))
                        inputs_llm[modality] = inputs_llm[modality].reshape(bs, num[modality], self.num_query_token, -1).view(bs, num[modality]*self.num_query_token, -1)
                        atts_llm[modality] =  torch.ones(inputs_llm[modality].size()[:-1], dtype=torch.long).to(self.device)
                        continue
                    # num*bs, num query tokens, llm emb size
                    inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].last_hidden_state[:,:query_tokens[modality].size(1),:]) 
                    # bs, num, num query tokens, llm emb size -> bs, num*num query tokens, llm emb size
                    inputs_llm[modality] = inputs_llm[modality].reshape(bs, num[modality], self.num_query_token, -1).view(bs, num[modality]*self.num_query_token, -1)
                    atts_llm[modality] =  torch.ones(inputs_llm[modality].size()[:-1], dtype=torch.long).to(self.device)
                    
                else:
                    if self.projection_only or getattr(self, f"projection_only_{modality}"):
                        if self.proj_dim == 1:
                            inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].mean(-1)).reshape(bs, self.num_query_token, -1)
                        else:
                            inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].reshape(bs, self.num_query_token, -1))
                        atts_llm[modality] =  torch.ones(inputs_llm[modality].size()[:-1], dtype=torch.long).to(self.device)
                        continue
                    inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality]['last_hidden_state'][:,:query_tokens[modality].size(1),:])
                    atts_llm[modality] = torch.ones(inputs_llm[modality].size()[:-1], dtype=torch.long).to(self.device)
                if self.enumerate_inputs:
                    enumeration[modality] = self.llm_tokenizer(
                        [f"{'' if i == 0 else ' '}({chr(97+i)}) " for _ in prompt],
                        return_tensors="pt",
                        add_special_tokens=False if (i!= 0 or self.prefix) else True
                        ).to(self.device)
                    
            att_list = []
            inp_list = []
            if self.prefix:
                att_list = [self.tokenized_prefix.attention_mask.repeat(bs, 1).to(self.device)]
                inp_list = [self.llm_model.get_input_embeddings()(self.tokenized_prefix.input_ids.to(self.device)).repeat(bs, 1, 1)]            
        
            for modality in curr_modalities:
                if self.enumerate_inputs:
                    enumeration_inputs_llm = self.llm_model.get_input_embeddings()(enumeration[modality].input_ids.to(self.device))
                    enumeration_atts_llm = enumeration[modality].attention_mask.to(self.device)
                    inp_list.extend([enumeration_inputs_llm])
                    att_list.extend([enumeration_atts_llm])
                if self.use_cues:
                    if self.clean_tokenization or self.remove_start:
                        if (modality==curr_modalities[0] and not (self.prefix or self.enumerate_inputs)):
                            att_list.extend([torch.tensor(self.tokenized_cue[modality].attention_mask).to(self.device).repeat(atts_llm[modality].shape[0], 1), atts_llm[modality]])
                            inp_list.extend([self.emb_cue[modality].to(self.device).repeat(inputs_llm[modality].shape[0], 1, 1), inputs_llm[modality]])
                        else:
                            att_list.extend([torch.tensor(self.tokenized_cue[modality].attention_mask[:,1:]).to(self.device).repeat(atts_llm[modality].shape[0], 1), atts_llm[modality]])
                            inp_list.extend([self.emb_cue[modality][:,1:].to(self.device).repeat(inputs_llm[modality].shape[0], 1, 1), inputs_llm[modality]])
                    else:
                        att_list.extend([torch.tensor(self.tokenized_cue[modality].attention_mask).to(self.device).repeat(atts_llm[modality].shape[0], 1), atts_llm[modality]])
                        inp_list.extend([self.emb_cue[modality].to(self.device).repeat(inputs_llm[modality].shape[0], 1, 1), inputs_llm[modality]])

                else:
                    att_list.extend([atts_llm[modality]])
                    inp_list.extend([inputs_llm[modality]])

                if self.add_space:
                    space_tok =  self.llm_tokenizer(
                        [f" " for _ in prompt],
                        return_tensors="pt",
                        add_special_tokens=False
                        )
                    space_inputs_llm = self.llm_model.get_input_embeddings()(space_tok.input_ids.to(self.device))
                    space_atts_llm = space_tok.attention_mask.to(self.device)
                    inp_list.extend([space_inputs_llm])
                    att_list.extend([space_atts_llm])



            atts_llm = torch.cat(att_list, dim=1)
            empty_targets = torch.ones(atts_llm.size(), dtype=torch.long).to(self.device).fill_(-100)
            inputs_llm = torch.cat(inp_list, dim=1)


            self.llm_tokenizer.padding_side = "right"
            self.llm_tokenizer.truncation_side = 'left'


            text_input_tokens = self.llm_tokenizer(
                [f"{p}{self.postfix}" for p in prompt] if self.postfix else prompt,
                padding="longest",
                return_tensors="pt",
                add_special_tokens= not self.clean_tokenization
            ).to(self.device)

        self.llm_tokenizer.truncation_side = 'right'
        n_cands = len(candidates)
        with self.maybe_autocast():
            all_losses = []
            for n in range(n_segments):
                seg_len = n_cands // n_segments
                if n == (n_segments - 1):
                    seg_len = n_cands - seg_len * (n_segments - 1)

                start_i = n * (n_cands // n_segments)
                end_i = start_i + seg_len
                this_output_tokens = self.llm_tokenizer(
                    candidates[start_i:end_i],
                    return_tensors="pt",
                    padding="longest",
                    # truncation=True,
                    # max_length=self.max_output_txt_len,
                ).to(self.device)

                this_input_tokens_ids = text_input_tokens.input_ids.repeat_interleave(seg_len, dim=0)
                this_input_tokens_atts = text_input_tokens.attention_mask.repeat_interleave(seg_len, dim=0)

                this_output_tokens_ids = this_output_tokens.input_ids.repeat(bs, 1)
                this_output_tokens_atts = this_output_tokens.attention_mask.repeat(bs, 1)

                this_llm_tokens, this_input_targets_len = self.concat_text_input_output(
                    this_input_tokens_ids,
                    this_input_tokens_atts,
                    this_output_tokens_ids,
                    this_output_tokens_atts
                )

                this_llm_input_ids = this_llm_tokens['input_ids']
                this_llm_atts = this_llm_tokens['attention_mask']

                inputs_embeds = self.llm_model.get_input_embeddings()(this_llm_input_ids)

                if self.use_caption:
                    inputs_embeds = torch.cat([inputs_embeds], dim=1)
                    attention_mask = torch.cat([this_llm_atts], dim=1)
                else:
                    inputs_embeds = torch.cat([inputs_llm.repeat_interleave(seg_len, dim=0), inputs_embeds], dim=1)
                    attention_mask = torch.cat([atts_llm.repeat_interleave(seg_len, dim=0), this_llm_atts], dim=1)


                this_targets = this_llm_input_ids.masked_fill(this_llm_input_ids == self.llm_tokenizer.pad_token_id, -100)
        
                for i, l in enumerate(this_input_targets_len):
                    this_targets[i][:l] = -100
                
                if self.use_caption:
                    torch.cat([this_targets], dim=1)
                else:
                    this_targets = torch.cat([empty_targets.repeat_interleave(seg_len, dim=0), this_targets], dim=1)


                outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=this_targets,
                    reduction="none",
                )

                loss = outputs.loss

                loss = loss.reshape(bs, seg_len)
                all_losses.append(loss)

        all_losses = torch.cat(all_losses, dim=-1)
        all_losses = -all_losses
        output_class_ranks = torch.argsort(all_losses, dim=-1)
        return {"predictions": all_losses, "targets": torch.tensor([self.candidates.index(l) for l in samples["label"]])}

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer
    
    def get_optimizer_params(self, weight_decay, lr_scale=1):
        return BaseModel.get_optimizer_params(self, weight_decay, lr_scale=lr_scale)

    @classmethod
    def from_config(cls, cfg):
        vit_config = cfg.get("vit_config")
        aud_config = cfg.get("aud_config")
        vit_encoder_pretrained_ckpt = cfg.get("vit_pretrained")
        aud_encoder_pretrained_ckpt = cfg.get("aud_pretrained")
        bert_pretrained = cfg.get("bert_pretrained")
        qformer_pretrained = cfg.get("qformer_pretrained")

        freeze_encoder = cfg.get("freeze_encoder")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq")
        embed_dim = cfg.get("embed_dim")
        max_txt_len = cfg.get("mat_txt_len")
        # llm part args
        llm_model = cfg.get("llm_model")
        # TODO: change the prompt
        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 256)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        qformer_text_input = cfg.get("qformer_text_input", True)
        modalities = cfg.get("modalities", ["image"])
        shared_qformer = cfg.get("shared_qformer",False)

        llm_text_input = cfg.get("llm_text_input", True)
        lora = cfg.get("lora", False)
        prefix = cfg.get("prefix", "")
        postfix = cfg.get("postfix", "")

        joint_video_audio=cfg.get('joint_video_audio', False)
        use_caption=cfg.get('use_caption', False)
        use_describe=cfg.get('use_describe', False)
        predict_with_gen = cfg.get('predict_with_gen', False)
        format_candidates_prompt = cfg.get('format_candidates_prompt', "{}")
        special_qformer_input_prompt = cfg.get('special_qformer_input_prompt', False)
        enumerate_inputs = cfg.get('enumerate_inputs', False)
        add_space = cfg.get('add_space', True)
        projection_only = cfg.get('projection_only', False)

        lora_model = cfg.get('lora_model', '')

        remove_start=cfg.get('remove_start', False)
        proj_dim=cfg.get('proj_dim', 1)
        clean_tokenization=cfg.get('clean_tokenization', False)

        logging.info("Model Config Arguments:")
        logging.info(OmegaConf.to_yaml(cfg))

        model = cls(
            # Q-former Arguments
            vit_config=vit_config,
            aud_config=aud_config,
            vit_encoder_pretrained=vit_encoder_pretrained_ckpt,
            aud_encoder_pretrained=aud_encoder_pretrained_ckpt,
            bert_pretrained=bert_pretrained,
            qformer_pretrained=qformer_pretrained,

            freeze_encoder=freeze_encoder,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            embed_dim=embed_dim,
            max_txt_len=max_txt_len,

            modalities=modalities,
            qformer_text_input=qformer_text_input,
            llm_text_input=llm_text_input,
            apply_lemmatizer=apply_lemmatizer,

            llm_model=llm_model,
            lora_model=lora_model,
            lora = lora,
            shared_qformer=shared_qformer,

            prompt=prompt,
            prefix=prefix,
            postfix=postfix,
            max_output_txt_len=max_output_txt_len,

            joint_video_audio=joint_video_audio,
            use_caption=use_caption,
            use_describe=use_describe,
            predict_with_gen=predict_with_gen,
            format_candidates_prompt=format_candidates_prompt,
            special_qformer_input_prompt=special_qformer_input_prompt,
            enumerate_inputs=enumerate_inputs,
            add_space=add_space,
            projection_only=projection_only,

            remove_start= remove_start,
            proj_dim=proj_dim,
            clean_tokenization=clean_tokenization
        )

        stage1_url_or_filename = cfg.get("stage1_url_or_filename","")

        # if stage1_url_or_filename:
        #     model.load_from_pretrained(stage1_url_or_filename)

        # model.load_checkpoint_from_config(cfg)
        return model


    @classmethod
    def init_Qformer(cls, num_query_token, feature_width,cross_attention_freq=2, bert_pretrained=None,pretrained_qformer=None, load_attention=False):
        encoder_config = BertConfig.from_pretrained(bert_pretrained)
        encoder_config.encoder_width = feature_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        encoder_config.vocab_size += 1 # for special token [DEC]
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        if pretrained_qformer:
            url_or_filename=pretrained_qformer
            logging.info(f"Loading pretrained qformer weights and query tokens from {url_or_filename}")

            # load the pretrained checkpoint into cache
            if os.path.isfile(url_or_filename):
                checkpoint = torch.load(url_or_filename, map_location="cpu")
            else:
                raise RuntimeError("checkpoint url or path is invalid")
            # process the checkpoint
            state_dict = checkpoint["model"]
            query_tokens.data = state_dict['query_tokens']
            state_dict = {k: v for k, v in state_dict.items() if 'embedding' not in k and 'predictions' not in k}

            # load the processed checkpoint into model and get the query_tokens
            Qformer.load_state_dict(state_dict, strict=False)

        return Qformer, query_tokens

    def get_state_dict(self, url_or_filename, **kwargs):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        if "model" in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        return state_dict
    
    def load_from_pretrained(self, url_or_filename, **kwargs):
        state_dict = self.get_state_dict(url_or_filename)
        self.load_state_dict(state_dict, strict=False)
        logging.info("load checkpoint from %s" % url_or_filename)

    def load_checkpoint(self, url_or_filename, **kwargs):
        """
        Load from a finetuned checkpoint.

        This should expect no mismatch in the model keys and the checkpoint keys.
        """
        state_dict = self.get_state_dict(url_or_filename)
        self.load_state_dict(state_dict, strict=True)
        logging.info("load checkpoint from %s" % url_or_filename)
    
    def load_state_dict(self, state_dict, strict=True):
        # from pdb import set_trace; set_trace()
        unexpected_keys = []
        missing_keys = []
        if self.shared_qformer and not self.projection_only:
            ## Load Q-Former if it is not loaded from config
            if not getattr(self, "pretrained_shared_qformer"):
                shared_qformer_state_dict = {'.'.join(k.split('.')[1:]):v for k,v in state_dict.items() if "shared_Qformer" == k.split('.')[0]}
                msg = self.shared_Qformer.load_state_dict(shared_qformer_state_dict, strict=strict)
                missing_keys.extend(msg.missing_keys)
                ## Load query tokens
                if "shared_query_tokens" not in state_dict:
                    missing_keys.append("shared_query_tokens")
                else:
                    self.shared_query_tokens = state_dict["shared_query_tokens"]
                missing_keys.extend(msg.missing_keys)
                unexpected_keys.extend(msg.unexpected_keys)

                for modality in self.modalities:
                    # Map shared Qformer by reference to all modalities. 
                    setattr(self, f"{modality}_Qformer", self.shared_Qformer) 
                    getattr(self, f"{modality}_query_tokens").data =  state_dict[f"shared_query_tokens"]
                    # load encoder projections
                    modality_encoder_projection_dict = {'.'.join(k.split('.')[1:]):v for k,v in state_dict.items() if f"{modality}_encoder_projection" in k.split('.')[0]}
                    msg = getattr(self, f"{modality}_encoder_projection").load_state_dict(modality_encoder_projection_dict, strict=strict)
                    missing_keys.extend(msg.missing_keys)
                    unexpected_keys.extend(msg.unexpected_keys)
                    # load modality layer norm
                    if getattr(self,f"load_ln_type_{modality}") == "vision":
                        modality_ln_dict = {'.'.join(k.split('.')[1:]):v for k,v in state_dict.items() if f"ln_vision" in k.split('.')[0]}
                    else:
                        modality_ln_dict = {'.'.join(k.split('.')[1:]):v for k,v in state_dict.items() if f"{modality}_ln" in k.split('.')[0]}
                    msg = getattr(self, f"{modality}_ln").load_state_dict(modality_ln_dict, strict=strict)
                    missing_keys.extend(msg.missing_keys)
                    unexpected_keys.extend(msg.unexpected_keys)
            
            ## Load Shared LLM projection if not loaded by config
            if not getattr(self, "load_projection_shared"):  
                shared_llm_projection_dict = {'.'.join(k.split('.')[1:]):v for k,v in state_dict.items() if f"shared_llm_proj" in k.split('.')[0]}
                msg = self.shared_llm_proj.load_state_dict(shared_llm_projection_dict, strict=strict)    
                missing_keys.extend(msg.missing_keys)
                unexpected_keys.extend(msg.unexpected_keys)
                for modality in self.modalities:   
                    ## Map to modality projections by reference
                    msg = setattr(self, f"{modality}_llm_proj", self.shared_llm_proj)
        else:
            for modality in self.modalities:
                ## Load Q-Former if not loaded from config
                if not getattr(self, f"pretrained_{modality}_qformer") or ((self.projection_only or getattr(self, f"projection_only_{modality}")) and not getattr(self, f"projection_path_{modality}")):

                    if self.projection_only or getattr(self, f"projection_only_{modality}") :
                        if not getattr(self, f"projection_path_{modality}"):
                            logging.info(f"Loaded {modality} projection")
                            modality_qformer_state_dict = {'.'.join(k.split('.')[1:]):v for k,v in state_dict.items() if f"{modality}_projection" == k.split('.')[0]}
                            msg = getattr(self, f"{modality}_projection").load_state_dict(modality_qformer_state_dict, strict=strict)
                            missing_keys.extend(msg.missing_keys)
                            unexpected_keys.extend(msg.unexpected_keys)
                    else:
                        modality_qformer_state_dict = {'.'.join(k.split('.')[1:]):v for k,v in state_dict.items() if f"{modality}_Qformer" == k.split('.')[0]}
                        msg = getattr(self, f"{modality}_Qformer").load_state_dict(modality_qformer_state_dict, strict=strict)
                        missing_keys.extend(msg.missing_keys)
                        unexpected_keys.extend(msg.unexpected_keys)
                    ## Load query tokens
                    if not self.projection_only and not getattr(self, f"projection_only_{modality}"):
                        if f"{modality}_query_tokens" not in state_dict:
                            missing_keys.append(f"{modality}_query_tokens")
                        else:
                            logging.info(f"Loaded {modality} query tokens")
                            getattr(self, f"{modality}_query_tokens").data =  state_dict[f"{modality}_query_tokens"]
                    # load modality layer norm if not loaded from config
                    if getattr(self,f"load_ln_type_{modality}") == "vision":
                        logging.info(f"Loaded {modality} vision ln")
                        modality_ln_dict = {'.'.join(k.split('.')[1:]):v for k,v in state_dict.items() if f"ln_vision" in k.split('.')[0]}
                    else:
                        modality_ln_dict = {'.'.join(k.split('.')[1:]):v for k,v in state_dict.items() if f"{modality}_ln" in k.split('.')[0]}
                    msg = getattr(self, f"{modality}_ln").load_state_dict(modality_ln_dict, strict=strict)
                    missing_keys.extend(msg.missing_keys)
                    unexpected_keys.extend(msg.unexpected_keys)
                ## Load LLM projections if not loaded from config
                if not getattr(self, f"load_projection_{modality}") or  (getattr(self, f"projection_only_{modality}") or self.projection_only):
                    if not getattr(self, f"projection_path_{modality}"):
                        logging.info(f"Loaded {modality} llm  projection")
                        modality_llm_projection_dict = {'.'.join(k.split('.')[1:]):v for k,v in state_dict.items() if f"{modality}_llm_proj" in k.split('.')[0]}
                        msg = getattr(self, f"{modality}_llm_proj").load_state_dict(modality_llm_projection_dict, strict=strict)
                        missing_keys.extend(msg.missing_keys)
                        unexpected_keys.extend(msg.unexpected_keys)
        
        ## llm model is loaded from pretrained
        lora_state_dict = {'.'.join(k.split('.')[1:]):v for k,v in state_dict.items() if f"llm_model" in k.split('.')[0]}

        if not self.lora or len(lora_state_dict) == 0:
            unexpected_keys = [k for k in unexpected_keys if k.split('.')[0] != 'llm_model']
        else:
            msg = self.llm_model.load_state_dict({'.'.join(k.split('.')[1:]):v for k,v in state_dict.items() if f"llm_model" in k.split('.')[0]}, strict=False)
            missing_keys.extend(["llm_model."+k for k in msg.missing_keys])
        missing_keys = [k for k in missing_keys if 'encoder' not in k.split('.')[0]]
        missing_keys = [k for k in missing_keys if k.split('.')[0] != 'llm_model']
        return _IncompatibleKeys(missing_keys, unexpected_keys)
    

    def before_evaluation(self, dataset, task_type, **kwargs):
        if task_type == MultimodalClassificationTask:
            self.candidates = dataset.classnames
            print(self.candidates)