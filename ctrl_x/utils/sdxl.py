from types import MethodType
from typing import Any, Dict, List, Optional, Tuple

from diffusers.models.attention_processor import Attention
import torch
import torch.nn.functional as F
from torch import nn

from .feature import *
from .utils import *

COUNT = 0


def get_control_config(structure_schedule, appearance_schedule):
    s = structure_schedule
    a = appearance_schedule

    control_config =\
f"""control_schedule:
    #       structure_conv   structure_attn   appearance_attn  conv/attn
    encoder:                                                # (num layers)
        0: [[             ], [             ], [             ]]  # 2/0
        1: [[             ], [             ], [{a}, {a}     ]]  # 2/2
        2: [[             ], [             ], [{a}, {a}     ]]  # 2/2
    middle: [[            ], [             ], [             ]]  # 2/1
    decoder:
        0: [[{s}          ], [{s}, {s}, {s}], [0.0, {a}, {a}]]  # 3/3
        1: [[             ], [             ], [{a}, {a}     ]]  # 3/3
        2: [[             ], [             ], [             ]]  # 3/0

control_target:
    - [output_tensor]  # structure_conv   choices: {{hidden_states, output_tensor}}
    - [query, key]     # structure_attn   choices: {{query, key, value}}
    - [before]         # appearance_attn  choices: {{before, value, after}}

self_recurrence_schedule:
    - [0.1, 0.5, 2]  # format: [start, end, num_recurrence]"""

    return control_config


def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:

    # Notice that normalization is always applied before the real computation in the following blocks.
    # 0. Self-Attention
    batch_size = hidden_states.shape[0]

    if self.norm_type == "ada_norm":
        norm_hidden_states = self.norm1(hidden_states, timestep)
    elif self.norm_type == "ada_norm_zero":
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
        )
    elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
        norm_hidden_states = self.norm1(hidden_states)
    elif self.norm_type == "ada_norm_continuous":
        norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
    elif self.norm_type == "ada_norm_single":
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
    else:
        raise ValueError("Incorrect norm used")

    if self.pos_embed is not None:
        norm_hidden_states = self.pos_embed(norm_hidden_states)

    # 1. Prepare GLIGEN inputs
    cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
    gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

    attn_output = self.attn1(
        norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states,  # Here, pass the "encoder_hidden_states" i.e. text embeddings
        # to attn1 (self-attn) for text-guided cross-image attention. Only valid when using the AttentionProcessor2_0
        # below as the processor to maintain self-attn.
        attention_mask=attention_mask,
        **cross_attention_kwargs,
    )

    if self.norm_type == "ada_norm_zero":
        attn_output = gate_msa.unsqueeze(1) * attn_output
    elif self.norm_type == "ada_norm_single":
        attn_output = gate_msa * attn_output

    hidden_states = attn_output + hidden_states
    if hidden_states.ndim == 4:
        hidden_states = hidden_states.squeeze(1)

    # 1.2 GLIGEN Control
    if gligen_kwargs is not None:
        hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

    # 3. Cross-Attention
    if self.attn2 is not None:
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm2(hidden_states, timestep)
        elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm2(hidden_states)
        elif self.norm_type == "ada_norm_single":
            # For PixArt norm2 isn't applied here:
            # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
            norm_hidden_states = hidden_states
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
        else:
            raise ValueError("Incorrect norm")

        if self.pos_embed is not None and self.norm_type != "ada_norm_single":
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            **cross_attention_kwargs,
        )
        hidden_states = attn_output + hidden_states

    # 4. Feed-forward
    # i2vgen doesn't have this norm 🤷‍♂️
    if self.norm_type == "ada_norm_continuous":
        norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
    elif not self.norm_type == "ada_norm_single":
        norm_hidden_states = self.norm3(hidden_states)

    if self.norm_type == "ada_norm_zero":
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

    if self.norm_type == "ada_norm_single":
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

    if self._chunk_size is not None:
        # "feed_forward_chunk_size" can be used to save memory
        ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
    else:
        ff_output = self.ff(norm_hidden_states)

    if self.norm_type == "ada_norm_zero":
        ff_output = gate_mlp.unsqueeze(1) * ff_output
    elif self.norm_type == "ada_norm_single":
        ff_output = gate_mlp * ff_output

    hidden_states = ff_output + hidden_states
    if hidden_states.ndim == 4:
        hidden_states = hidden_states.squeeze(1)

    return hidden_states


def _chunked_feed_forward(ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int):
    # "feed_forward_chunk_size" can be used to save memory
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
        )

    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    ff_output = torch.cat(
        [ff(hid_slice) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
        dim=chunk_dim,
    )
    return ff_output


def convolution_forward(  # From <class 'diffusers.models.resnet.ResnetBlock2D'>, forward (diffusers==0.28.0)
    self,
    input_tensor: torch.Tensor,
    temb: torch.Tensor,
    *args,
    **kwargs,
) -> torch.Tensor:
    do_structure_control = self.do_control and self.t in self.structure_schedule

    hidden_states = input_tensor

    hidden_states = self.norm1(hidden_states)
    hidden_states = self.nonlinearity(hidden_states)

    if self.upsample is not None:
        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            input_tensor = input_tensor.contiguous()
            hidden_states = hidden_states.contiguous()
        input_tensor = self.upsample(input_tensor)
        hidden_states = self.upsample(hidden_states)
    elif self.downsample is not None:
        input_tensor = self.downsample(input_tensor)
        hidden_states = self.downsample(hidden_states)

    hidden_states = self.conv1(hidden_states)

    if self.time_emb_proj is not None:
        if not self.skip_time_act:
            temb = self.nonlinearity(temb)
        temb = self.time_emb_proj(temb)[:, :, None, None]

    if self.time_embedding_norm == "default":
        if temb is not None:
            hidden_states = hidden_states + temb
        hidden_states = self.norm2(hidden_states)
    elif self.time_embedding_norm == "scale_shift":
        if temb is None:
            raise ValueError(
                f" `temb` should not be None when `time_embedding_norm` is {self.time_embedding_norm}"
            )
        time_scale, time_shift = torch.chunk(temb, 2, dim=1)
        hidden_states = self.norm2(hidden_states)
        hidden_states = hidden_states * (1 + time_scale) + time_shift
    else:
        hidden_states = self.norm2(hidden_states)

    hidden_states = self.nonlinearity(hidden_states)

    hidden_states = self.dropout(hidden_states)
    hidden_states = self.conv2(hidden_states)

    # Feature injection and AdaIN (hidden_states)
    if do_structure_control and "hidden_states" in self.structure_target:
        hidden_states = feature_injection(hidden_states, batch_order=self.batch_order)

    if self.conv_shortcut is not None:
        input_tensor = self.conv_shortcut(input_tensor)

    output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

    # Feature injection and AdaIN (output_tensor)
    if do_structure_control and "output_tensor" in self.structure_target:
        output_tensor = feature_injection(output_tensor, batch_order=self.batch_order)

    return output_tensor


class AttnProcessor2_0:  # From <class 'diffusers.models.attention_processor.AttnProcessor2_0'> (diffusers==0.28.0)

    """
    Attn processor for attn1 in BasicAttentionBlock, this processor only performs self-attention which is always the
    case for attn1. This processor receives 'encoder_hidden_states' but do not perform cross-attention. Instead, those
    text embeddings are used to assist cross-image attention.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        do_structure_control = attn.do_control and attn.t in attn.structure_schedule
        do_appearance_control = attn.do_control and attn.t in attn.appearance_schedule

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        _encoder_hidden_states = encoder_hidden_states  # text embeddings
        no_encoder_hidden_states = True  # perform self-attn
        if no_encoder_hidden_states:
            encoder_hidden_states = hidden_states

        # _encoder_hidden_states = attn.norm_encoder_hidden_states(_encoder_hidden_states)

        if do_appearance_control:  # Assume we only have this for self attention
            hidden_states_normed = normalize(hidden_states, dim=-2)  # B H D C
            encoder_hidden_states_normed = normalize(encoder_hidden_states, dim=-2)

            query_normed = attn.to_q(hidden_states_normed)
            key_normed = attn.to_k(encoder_hidden_states_normed)

            inner_dim = key_normed.shape[-1]
            head_dim = inner_dim // attn.heads
            query_normed = query_normed.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key_normed = key_normed.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # Match query and key injection with structure injection (if injection is happening this layer)
            if do_structure_control:
                if "query" in attn.structure_target:
                    query_normed = feature_injection(query_normed, batch_order=attn.batch_order)
                if "key" in attn.structure_target:
                    key_normed = feature_injection(key_normed, batch_order=attn.batch_order)

        # text_key = attn.cross_to_k(_encoder_hidden_states)
        # query = attn.cross_to_q(hidden_states)
        # # print(text_key.shape, query.shape)  # why is text_key of the shape [4, 77, 640]? Why 77 when only using "A suit" as prompt?
        # # ['uncond', 'structure_cond', 'appearance_cond', 'cond']
        # object_indexes = attn.object_indexes
        # if text_key.shape[0] == 4 and do_appearance_control:
        #
        #     text_key = text_key[3]  # [tokens, dim]
        #     appearance_query = query[2]  # [patches, dim]
        #     output_query = query[3]  # [patches, dim]
        #
        #     for object_index in object_indexes:
        #         appearance_text = torch.matmul(appearance_query, text_key[object_index])
        #         output_text = torch.matmul(output_query, text_key[object_index])
        #         # print(appearance_text.shape, output_text.shape)
        #         visualize_grid_to_grid(appearance_text, 'attn_app_text', alpha=0.6)

            # save attention maps


        # Appearance transfer (before)
        if do_appearance_control and "before" in attn.appearance_target:
            hidden_states = hidden_states.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            hidden_states = appearance_transfer(hidden_states, query_normed, key_normed, batch_order=attn.batch_order, cross_maps=attn.cross_maps)
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

            if no_encoder_hidden_states:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        query = attn.to_q(hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Feature injection (query, key, and/or value)
        if do_structure_control:
            if "query" in attn.structure_target:
                query = feature_injection(query, batch_order=attn.batch_order)
            if "key" in attn.structure_target:
                key = feature_injection(key, batch_order=attn.batch_order)
            if "value" in attn.structure_target:
                value = feature_injection(value, batch_order=attn.batch_order)

        # Appearance transfer (value)
        if do_appearance_control and "value" in attn.appearance_target:
            value = appearance_transfer(value, query_normed, key_normed, batch_order=attn.batch_order)

        # The output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # Appearance transfer (after)
        if do_appearance_control and "after" in attn.appearance_target:
            hidden_states = appearance_transfer(hidden_states, query_normed, key_normed, batch_order=attn.batch_order)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Linear projection
        hidden_states = attn.to_out[0](hidden_states, *args)
        # Dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False, attention_map=False):
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    if attention_map:
        return attn_weight @ value, attn_weight
    return attn_weight @ value


class AttnProcessor2_0_multiple:  # From <class 'diffusers.models.attention_processor.AttnProcessor2_0'> (diffusers==0.28.0)

    """
    Attn processor for attn1 in BasicAttentionBlock, this processor only performs self-attention which is always the
    case for attn1. This processor receives 'encoder_hidden_states' but do not perform cross-attention. Instead, those
    text embeddings are used to assist cross-image attention.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        do_structure_control = attn.do_control and attn.t in attn.structure_schedule
        do_appearance_control = attn.do_control and attn.t in attn.appearance_schedule

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        _encoder_hidden_states = encoder_hidden_states  # text embeddings
        no_encoder_hidden_states = True  # perform self-attn
        if no_encoder_hidden_states:
            encoder_hidden_states = hidden_states

        # _encoder_hidden_states = attn.norm_encoder_hidden_states(_encoder_hidden_states)

        if do_appearance_control:  # Assume we only have this for self attention
            hidden_states_normed = normalize(hidden_states, dim=-2)  # B H D C
            encoder_hidden_states_normed = normalize(encoder_hidden_states, dim=-2)

            query_normed = attn.to_q(hidden_states_normed)
            key_normed = attn.to_k(encoder_hidden_states_normed)

            inner_dim = key_normed.shape[-1]
            head_dim = inner_dim // attn.heads
            query_normed = query_normed.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key_normed = key_normed.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # Match query and key injection with structure injection (if injection is happening this layer)
            if do_structure_control:
                if "query" in attn.structure_target:
                    query_normed = feature_injection(query_normed, batch_order=attn.batch_order)
                if "key" in attn.structure_target:
                    key_normed = feature_injection(key_normed, batch_order=attn.batch_order)

        # text_key = attn.cross_to_k(_encoder_hidden_states)
        # query = attn.cross_to_q(hidden_states)
        # # print(text_key.shape, query.shape)  # why is text_key of the shape [4, 77, 640]? Why 77 when only using "A suit" as prompt?
        # # ['uncond', 'structure_cond', 'appearance_cond', 'cond']
        # object_indexes = attn.object_indexes
        # if text_key.shape[0] == 4 and do_appearance_control:
        #
        #     text_key = text_key[3]  # [tokens, dim]
        #     appearance_query = query[2]  # [patches, dim]
        #     output_query = query[3]  # [patches, dim]
        #
        #     for object_index in object_indexes:
        #         appearance_text = torch.matmul(appearance_query, text_key[object_index])
        #         output_text = torch.matmul(output_query, text_key[object_index])
        #         # print(appearance_text.shape, output_text.shape)
        #         visualize_grid_to_grid(appearance_text, 'attn_app_text', alpha=0.6)

            # save attention maps


        # Appearance transfer (before)
        if do_appearance_control and "before" in attn.appearance_target:
            hidden_states = hidden_states.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            # print(hidden_states.shape)
            hidden_states = appearance_transfer_multiple(hidden_states, query_normed, key_normed, batch_order=attn.batch_order, cross_maps=attn.cross_maps)
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

            if no_encoder_hidden_states:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        query = attn.to_q(hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Feature injection (query, key, and/or value)
        if do_structure_control:
            if "query" in attn.structure_target:
                query = feature_injection(query, batch_order=attn.batch_order)
            if "key" in attn.structure_target:
                key = feature_injection(key, batch_order=attn.batch_order)
            if "value" in attn.structure_target:
                value = feature_injection(value, batch_order=attn.batch_order)

        # Appearance transfer (value)
        if do_appearance_control and "value" in attn.appearance_target:
            value = appearance_transfer(value, query_normed, key_normed, batch_order=attn.batch_order)

        # The output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # Appearance transfer (after)
        if do_appearance_control and "after" in attn.appearance_target:
            hidden_states = appearance_transfer(hidden_states, query_normed, key_normed, batch_order=attn.batch_order)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Linear projection
        hidden_states = attn.to_out[0](hidden_states, *args)
        # Dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def register_control(
    model,
    timesteps,
    control_schedule,  # structure_conv, structure_attn, appearance_attn
    control_target=[["output_tensor"], ["query", "key"], ["before"]],
    device="cuda",
    cross_maps=None,
    multiple=False
):
    # Assume timesteps in reverse order (T -> 0)
    for block_type in ["encoder", "decoder", "middle"]:
        blocks = {
            "encoder": model.unet.down_blocks,
            "decoder": model.unet.up_blocks,
            "middle": [model.unet.mid_block],
        }[block_type]

        control_schedule_block = control_schedule[block_type]
        if block_type == "middle":
            control_schedule_block = [control_schedule_block]

        for layer in range(len(control_schedule_block)):
            # Convolution
            num_blocks = len(blocks[layer].resnets) if hasattr(blocks[layer], "resnets") else 0
            for block in range(num_blocks):
                convolution = blocks[layer].resnets[block]
                convolution.structure_target = control_target[0]
                convolution.structure_schedule = get_schedule(
                    timesteps, get_elem(control_schedule_block[layer][0], block)
                )
                convolution.forward = MethodType(convolution_forward, convolution)

            # Self-attention
            num_blocks = len(blocks[layer].attentions) if hasattr(blocks[layer], "attentions") else 0
            for block in range(num_blocks):
                for transformer_block in blocks[layer].attentions[block].transformer_blocks:
                    transformer_block.forward = MethodType(forward, transformer_block)
                    attention = transformer_block.attn1
                    attention2 = transformer_block.attn2
                    attention.structure_target = control_target[1]
                    attention.structure_schedule = get_schedule(
                        timesteps, get_elem(control_schedule_block[layer][1], block)
                    )
                    attention.appearance_target = control_target[2]
                    attention.appearance_schedule = get_schedule(
                        timesteps, get_elem(control_schedule_block[layer][2], block)
                    )
                    attention.cross_to_k = attention2.to_k
                    attention.cross_to_q = attention2.to_q
                    attention.cross_maps = cross_maps
                    if not multiple:
                        attention.processor = AttnProcessor2_0()
                    else:
                        attention.processor = AttnProcessor2_0_multiple()
                    # attention2.processor = AttnProcessor2_1()


def register_attr(model, t, do_control, batch_order):
    for layer_type in ["encoder", "decoder", "middle"]:
        blocks = {"encoder": model.unet.down_blocks, "decoder": model.unet.up_blocks,
                  "middle": [model.unet.mid_block]}[layer_type]
        for layer in blocks:
            # Convolution
            for module in layer.resnets:
                module.t = t
                module.do_control = do_control
                module.batch_order = batch_order
            # Self-attention
            if hasattr(layer, "attentions"):
                for block in layer.attentions:
                    for module in block.transformer_blocks:
                        module.attn1.t = t
                        module.attn2.t = t
                        module.attn1.do_control = do_control
                        module.attn1.batch_order = batch_order
                        module.attn1.object_indexes = model.object_indexes
                        module.attn2.object_indexes = model.object_indexes
                        
                        
def create_cross_mask(
        str_masks: List[Tuple[torch.Tensor, str, float]],
        app_masks: List[Tuple[torch.Tensor, str, float]],
        resolution: int,
        classnames: List[str],
        device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a cross_mask matrix correlating structure and appearance images.
    (Modified for demonstration with flexible resolution)

    Args:
        str_masks (List[Tuple[torch.Tensor, str, float]]):
            List of tuples containing (segmentation mask, class name, confidence) for the structure image.
        app_masks (List[Tuple[torch.Tensor, str, float]]):
            List of tuples containing (segmentation mask, class name, confidence) for the appearance image.
        resolution (int):
            The resolution to downsample masks to.
        classnames (List[str]):
            List of all possible class names.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            A tuple containing:
                - A [resolution^2, resolution^2] tensor representing the cross_mask.
                - A [resolution, resolution] tensor representing the assigned labels.
    """

    # Validate resolution
    if resolution <= 0:
        raise ValueError("Resolution must be a positive integer.")

    def downsample_mask(mask: torch.Tensor) -> torch.Tensor:
        """
        Downsamples a single mask to the desired resolution using nearest neighbor interpolation.

        Args:
            mask (torch.Tensor):
                A binary mask tensor of shape [H, W].

        Returns:
            torch.Tensor:
                Downsampled binary mask of shape [resolution, resolution].
        """
        if mask.dtype != torch.float32:
            mask = mask.float()
        mask = mask.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
        down_mask = F.interpolate(mask, size=(resolution, resolution), mode='nearest').squeeze(0).squeeze(0)
        return down_mask.bool()

    # Step 1: Downsample all masks (both structure and appearance)
    processed_str_masks = [
        (downsample_mask(mask), cls, conf) for mask, cls, conf in str_masks
    ]

    processed_app_masks = [
        (downsample_mask(mask), cls, conf) for mask, cls, conf in app_masks
    ]

    # Step 2: Process appearance masks - keep only the highest confidence per class
    app_masks_sorted = sorted(processed_app_masks, key=lambda x: x[2], reverse=True)  # Sort by confidence descending
    app_masks_unique = {}
    for mask, cls, conf in app_masks_sorted:
        if cls not in app_masks_unique:
            app_masks_unique[cls] = mask  # Keep only the highest confidence mask per class

    # Step 3: Assign labels to structure image points
    labels = torch.full((resolution, resolution), -1, dtype=torch.long, device=device)  # Initialize all as background (-1)

    # Sort structure masks by descending confidence
    processed_str_masks_sorted = sorted(processed_str_masks, key=lambda x: x[2], reverse=True)

    # Initialize list of remaining masks
    remaining_masks = processed_str_masks_sorted.copy()

    used_classnames = set()  # Track used classnames in the current iteration

    while remaining_masks:
        progress_made = False  # To detect if any assignment happens in this iteration
        indexes = []
        for i, (mask, cls, conf) in enumerate(remaining_masks.copy()):
            if cls in used_classnames:
                continue  # Skip to prioritize other classnames first
            # Identify points in the mask that are not yet assigned
            assignable = mask & (labels == -1)
            if assignable.any():
                class_idx = classnames.index(cls)
                labels[assignable] = class_idx
                used_classnames.add(cls)
                progress_made = True
            # Remove the mask from remaining_masks regardless of assignment
            indexes.append(i)
        indexes.sort(reverse=True)
        for index in indexes:
            del remaining_masks[index]

        if not progress_made:
            # All classnames have been used; reset the used_classnames to allow reuse
            used_classnames.clear()

    # Step 4: Prepare appearance masks for correlation
    # Create a dictionary mapping class index to its corresponding appearance mask
    app_class_masks = {}
    for cls_idx, cls in enumerate(classnames):
        if cls in app_masks_unique:
            app_class_masks[cls_idx] = app_masks_unique[cls]
        else:
            app_class_masks[cls_idx] = None  # Indicate no appearance mask for this class

    # Step 5: Determine appearance background points
    if app_masks_unique:
        # Combine all appearance masks to find background
        app_union_mask = torch.zeros((resolution, resolution), dtype=torch.bool, device=device)
        for mask in app_masks_unique.values():
            app_union_mask |= mask
        appearance_background_mask = ~app_union_mask  # Background in appearance image
    else:
        # If no appearance masks, entire appearance image is background
        appearance_background_mask = torch.ones((resolution, resolution), dtype=torch.bool, device=device)

    appearance_background_flat = appearance_background_mask.view(-1)  # Shape: [resolution^2]

    # Step 6: Flatten labels for easier indexing
    labels_flat = labels.view(-1)  # Shape: [resolution^2]

    # Step 7: Initialize cross_mask with -inf
    cross_mask = torch.full((resolution * resolution, resolution * resolution), float('-inf'), device=device)

    # Step 8: Handle background points in structure image
    background_indices = (labels_flat == -1).nonzero(as_tuple=True)[0]
    if background_indices.numel() > 0:
        # Set -inf for all columns first
        cross_mask[background_indices, :] = float('-inf')
        # Identify appearance background column indices
        appearance_bg_indices = appearance_background_flat.nonzero(as_tuple=True)[0]
        # Set 0 for [background_struct, background_app]
        cross_mask[background_indices.unsqueeze(1), appearance_bg_indices] = 0

    # Step 9: Handle non-background points in structure image
    for cls_idx, cls in enumerate(classnames):
        # Indices in the structure image belonging to the current class
        str_class_indices = (labels_flat == cls_idx).nonzero(as_tuple=True)[0]
        if str_class_indices.numel() == 0:
            continue  # Skip if no points belong to this class

        app_mask = app_class_masks[cls_idx]
        if app_mask is not None:
            # Flattened appearance mask for the current class
            app_mask_flat = app_mask.view(-1)  # Shape: [resolution^2]
            if app_mask_flat.sum() == 0:
                # If the appearance mask is empty, treat as no mask
                cross_mask[str_class_indices, :] = 0
            else:
                # Set -inf for all columns first
                cross_mask[str_class_indices, :] = float('-inf')
                # Identify appearance class column indices
                appearance_cls_indices = app_mask_flat.nonzero(as_tuple=True)[0]
                # Set 0 for [struct_cls, appearance_cls]
                cross_mask[str_class_indices.unsqueeze(1), appearance_cls_indices] = 0
        else:
            # If no appearance mask for this class, set entire row to 0
            cross_mask[str_class_indices, :] = 0

    return cross_mask, labels


def create_cross_mask_multiple(
        str_masks: List[Tuple[torch.Tensor, str, float]],
        app_masks_list: List[List[Tuple[torch.Tensor, str, float]]],
        app_classnames: List[List[str]],
        resolution: int,
        classnames: List[str],
        device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a cross_mask matrix correlating structure and appearance images.
    (Modified for demonstration with flexible resolution)

    Return Shape:
    [resolution^2, app_list_len, resolution^2] tensor representing the cross_mask.
    """

    # Validate resolution
    if resolution <= 0:
        raise ValueError("Resolution must be a positive integer.")

    def downsample_mask(mask: torch.Tensor) -> torch.Tensor:
        if mask.dtype != torch.float32:
            mask = mask.float()
        mask = mask.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
        down_mask = F.interpolate(mask, size=(resolution, resolution), mode='nearest').squeeze(0).squeeze(0)
        return down_mask.bool()

    # Step 1: Downsample all masks (both structure and appearance)
    processed_str_masks = [
        (downsample_mask(mask), cls, conf) for mask, cls, conf in str_masks
    ]

    app_masks_unique_list = []
    for app_masks in app_masks_list:
        processed_app_masks = [
            (downsample_mask(mask), cls, conf) for mask, cls, conf in app_masks
        ]
        # Step 2: Process appearance masks - keep only the highest confidence per class
        app_masks_sorted = sorted(processed_app_masks, key=lambda x: x[2], reverse=True)  # Sort by confidence descending
        app_masks_unique = {}
        for mask, cls, conf in app_masks_sorted:
            if cls not in app_masks_unique:
                app_masks_unique[cls] = mask  # Keep only the highest confidence mask per class
        app_masks_unique_list.append(app_masks_unique)
        

    # Step 3: Assign labels to structure image points
    labels = torch.full((resolution, resolution), -1, dtype=torch.long, device=device)  # Initialize all as background (-1)

    # Sort structure masks by descending confidence
    processed_str_masks_sorted = sorted(processed_str_masks, key=lambda x: x[2], reverse=True)

    # Initialize list of remaining masks
    remaining_masks = processed_str_masks_sorted.copy()

    used_classnames = set()  # Track used classnames in the current iteration

    while remaining_masks:
        progress_made = False  # To detect if any assignment happens in this iteration
        indexes = []
        for i, (mask, cls, conf) in enumerate(remaining_masks.copy()):
            if cls in used_classnames:
                continue  # Skip to prioritize other classnames first
            # Identify points in the mask that are not yet assigned
            assignable = mask & (labels == -1)
            if assignable.any():
                class_idx = classnames.index(cls)
                labels[assignable] = class_idx
                used_classnames.add(cls)
                progress_made = True
            # Remove the mask from remaining_masks regardless of assignment
            indexes.append(i)
        indexes.sort(reverse=True)
        for index in indexes:
            del remaining_masks[index]

        if not progress_made:
            # All classnames have been used; reset the used_classnames to allow reuse
            used_classnames.clear()

    # Step 4: Prepare appearance masks for correlation
    # Create a dictionary mapping class index to its corresponding appearance mask
    app_class_masks = {}
    app_class_index = {}
    L = len(app_masks_unique_list)
    for cls_idx, cls in enumerate(classnames):
        for j in range(L):
            if cls in app_masks_unique_list[j] or j == len(app_masks_unique_list) - 1:
                app_class_index[cls_idx] = j
                if cls in app_masks_unique_list[j]:
                    app_class_masks[cls_idx] = app_masks_unique_list[j][cls]
                else:
                    app_class_masks[cls_idx] = None  # Indicate no appearance mask for this class
                break
    

    # Step 5: Determine appearance background points for the LAST IMAGE
    if app_masks_unique:
        # Combine all appearance masks to find background
        app_union_mask = torch.zeros((resolution, resolution), dtype=torch.bool, device=device)
        for mask in app_masks_unique_list[-1].values():
            app_union_mask |= mask
        appearance_background_mask = ~app_union_mask  # Background in appearance image
    else:
        # If no appearance masks, entire appearance image is background
        appearance_background_mask = torch.ones((resolution, resolution), dtype=torch.bool, device=device)

    appearance_background_flat = appearance_background_mask.view(-1)  # Shape: [resolution^2]
    # background_point_row = appearance_background_flat.unsqueeze(0).repeat(L, 1)
    # background_point_row[:L-1, :] = 0

    # Step 6: Flatten labels for easier indexing
    labels_flat = labels.view(-1)  # Shape: [resolution^2]

    # Step 7: Initialize cross_mask with -inf
    cross_mask = torch.full((resolution * resolution, L, resolution * resolution), float('-inf'), device=device)

    # Step 8: Handle background points in structure image
    background_indices = (labels_flat == -1).nonzero(as_tuple=True)[0]
    if background_indices.numel() > 0:
        # Set -inf for all columns first
        cross_mask[background_indices, :, :] = float('-inf')
        # Identify appearance background column indices
        appearance_bg_indices = appearance_background_flat.nonzero(as_tuple=True)[0]
        # Set 0 for [background_struct, background_app]
        # print(cross_mask.shape, background_indices.shape, appearance_bg_indices.shape)
        cross_mask[background_indices.unsqueeze(1), L-1, appearance_bg_indices.unsqueeze(0)] = 0
        # cross_mask[background_indices, L-1, :] = 0

    # Step 9: Handle non-background points in structure image
    for cls_idx, cls in enumerate(classnames):
        # Indices in the structure image belonging to the current class
        str_class_indices = (labels_flat == cls_idx).nonzero(as_tuple=True)[0]
        if str_class_indices.numel() == 0:
            continue  # Skip if no points belong to this class

        app_mask = app_class_masks[cls_idx]
        app_index = app_class_index[cls_idx]
        if app_mask is not None:
            # Flattened appearance mask for the current class
            app_mask_flat = app_mask.view(-1)  # Shape: [resolution^2]
            if app_mask_flat.sum() == 0:
                # If the appearance mask is empty, treat as no mask
                cross_mask[str_class_indices, L-1, :] = 0
            else:
                # Set -inf for all columns first
                cross_mask[str_class_indices, :, :] = float('-inf')
                # Identify appearance class column indices
                appearance_cls_indices = app_mask_flat.nonzero(as_tuple=True)[0]
                # Set 0 for [struct_cls, appearance_cls]
                # print(appearance_cls_indices)
                # print(str_class_indices)
                cross_mask[str_class_indices.unsqueeze(1), app_index, appearance_cls_indices.unsqueeze(0)] = 0
        else:
            # If no appearance mask for this class, set entire row to 0
            cross_mask[str_class_indices, L-1, :] = 0
    
    # print(cross_mask[205, 1, -1000:])
    # print(cross_mask[0, 1, -1000:])
    cross_mask = cross_mask.view(resolution**2, L*resolution**2)
    # print(cross_mask[205, :1000])
    # print(cross_mask[0, -1000:])

    return cross_mask, labels
