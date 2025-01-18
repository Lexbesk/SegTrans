import math

import torch.nn.functional as F

from .utils import *

RESOLUTIONS = {}


def get_schedule(timesteps, schedule):
    end = round(len(timesteps) * schedule)
    timesteps = timesteps[:end]
    return timesteps


def get_elem(l, i, default=0.0):
    if i >= len(l):
        return default
    return l[i]


def pad_list(l_1, l_2, pad=0.0):
    max_len = max(len(l_1), len(l_2))
    l_1 = l_1 + [pad] * (max_len - len(l_1))
    l_2 = l_2 + [pad] * (max_len - len(l_2))
    return l_1, l_2
    
    
def normalize(x, dim):
    x_mean = x.mean(dim=dim, keepdim=True)
    x_std = x.std(dim=dim, keepdim=True)
    x_normalized = (x - x_mean) / x_std
    return x_normalized


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False, attention_map=False, cross_maps=None):
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
    # print(attn_weight.shape, scale_factor)
    # print(attn_weight)
    resolution = attn_weight.shape[-1]
    cross_map = cross_maps[resolution]
    attn_weight += cross_map
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    if attention_map:
        return attn_weight @ value, attn_weight
    return attn_weight @ value


# https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
def appearance_mean_std(q_c_normed, k_s_normed, v_s, cross_maps=None):  # c: content, s: style, cross_maps: {"4096": cross image attention [4096, 4096], "1024": cross image attention [1024, 1024]}
    q_c = q_c_normed  # q_c and k_s must be projected from normalized features
    k_s = k_s_normed
    # print(q_c.shape, k_s.shape, v_s.shape)
    # if q_c.shape[2] not in RESOLUTIONS:
    #     RESOLUTIONS[q_c.shape[2]] = 0
    # print(RESOLUTIONS)
    assert cross_maps is not None and q_c.shape[2] in cross_maps
    # cross_map = cross_maps[q_c.shape[2]]
    # mean = F.scaled_dot_product_attention(q_c, k_s, v_s)  # Use scaled_dot_product_attention for efficiency
    # std = (F.scaled_dot_product_attention(q_c, k_s, v_s.square()) - mean.square()).relu().sqrt()
    mean = scaled_dot_product_attention(q_c, k_s, v_s, cross_maps=cross_maps)  # Use scaled_dot_product_attention for efficiency
    std = (scaled_dot_product_attention(q_c, k_s, v_s.square(), cross_maps=cross_maps) - mean.square()).relu().sqrt()
    
    return mean, std
    

def feature_injection(features, batch_order):
    assert features.shape[0] % len(batch_order) == 0
    features_dict = batch_tensor_to_dict(features, batch_order)
    features_dict["cond"] = features_dict["structure_cond"]
    features = batch_dict_to_tensor(features_dict, batch_order)
    return features


def appearance_transfer(features, q_normed, k_normed, batch_order, v=None, reshape_fn=None, cross_maps=None):
    assert features.shape[0] % len(batch_order) == 0

    features_dict = batch_tensor_to_dict(features, batch_order)
    q_normed_dict = batch_tensor_to_dict(q_normed, batch_order)
    k_normed_dict = batch_tensor_to_dict(k_normed, batch_order)
    v_dict = features_dict
    if v is not None:
        v_dict = batch_tensor_to_dict(v, batch_order)
    
    mean_cond, std_cond = appearance_mean_std(
        q_normed_dict["cond"], k_normed_dict["appearance_cond"], v_dict["appearance_cond"], cross_maps=cross_maps
    )

    if reshape_fn is not None:
        mean_cond = reshape_fn(mean_cond)
        std_cond = reshape_fn(std_cond)

    features_dict["cond"] = std_cond * normalize(features_dict["cond"], dim=-2) + mean_cond
    
    features = batch_dict_to_tensor(features_dict, batch_order)
    return features
