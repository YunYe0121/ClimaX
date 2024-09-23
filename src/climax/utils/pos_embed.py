# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------


import numpy as np
import torch
from climax.utils.data_utils import get_region_info

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model, new_size=(64, 128)):
    if "net.pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["net.pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        orig_num_patches = pos_embed_checkpoint.shape[-2]
        patch_size = model.patch_size
        w_h_ratio = 2
        orig_h = int((orig_num_patches // w_h_ratio) ** 0.5)
        orig_w = w_h_ratio * orig_h
        orig_size = (orig_h, orig_w)
        new_size = (new_size[0] // patch_size, new_size[1] // patch_size)
        # print (orig_size)
        # print (new_size)
        if orig_size[0] != new_size[0]:
            print("Interpolate PEs from %dx%d to %dx%d" % (orig_size[0], orig_size[1], new_size[0], new_size[1]))
            pos_tokens = pos_embed_checkpoint.reshape(-1, orig_size[0], orig_size[1], embedding_size).permute(
                0, 3, 1, 2
            )
            new_pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size[0], new_size[1]), mode="bicubic", align_corners=False
            )
            new_pos_tokens = new_pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            checkpoint_model["net.pos_embed"] = new_pos_tokens


def interpolate_channel_embed(checkpoint_model, new_len):
    if "net.channel_embed" in checkpoint_model:
        channel_embed_checkpoint = checkpoint_model["net.channel_embed"]
        old_len = channel_embed_checkpoint.shape[1]
        if new_len <= old_len:
            checkpoint_model["net.channel_embed"] = channel_embed_checkpoint[:, :new_len]

'''
This is written by YY while input data is only in the region of specific area.
'''
def interpolate_pos_embed_localization(
        model, checkpoint_model, new_size = (140, 180),
        region_name = 'Taiwan',
        lat_mat = np.arange(-90 + 0.25 / 2, 90, 0.25),
        lon_mat = np.arange(0, 360, 0.25),
    ):
    '''
    Argument:
        model:
            Climax model used.
        checkpoint:
            The pretrained model checkpoint used.
        new_size:
            The new 'patch' size after interpolation.
        region_name:
            The corresponding name for get_region_info function to get the information for specific area.
        lat_mat:
            The lat_matrix for get_region_info function to find the min_h & max_h.
        lon_mat:
            The lat_matrix for get_region_info function to find the min_w & max_w.
    '''
    if 'net.pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['net.pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        orig_num_patches = pos_embed_checkpoint.shape[-2]
        patch_size = model.patch_size
        '''
        This ratio is to calculate 'original size' patch num.
        In climax model, this will be 2.
        '''
        w_h_ratio = 2
        orig_h = int((orig_num_patches // w_h_ratio) ** 0.5)
        orig_w = w_h_ratio * orig_h
        orig_size = (orig_h, orig_w)
        new_size = (new_size[0] // patch_size, new_size[1] // patch_size)
        global_grid_size = (len(lat_mat), len(lon_mat))
        global_size = (global_grid_size[0] // patch_size, global_grid_size[1] // patch_size)
        '''
        All of these size are 'patch_size (the size after divided by patch)'!
        '''
        # print(f'orig_size: {orig_size}')
        # print(f'global_size: {global_size}')
        # print(f'new_size: {new_size}')
        '''
        Use a function to get the cropped localization embedding.
        Here I reuse the get_region_info function in climax.utils.data_utils.
        The lat and lon are gridded in 0.25 degree,
        which is the "really" maxium resolution for era5 dataset.
        '''
        region_dict = get_region_info(region_name, lat_mat, lon_mat, patch_size)
        min_h, max_h = region_dict['min_h'], region_dict['max_h']
        min_w, max_w = region_dict['min_w'], region_dict['max_w']
        '''
        Here I add some checks to make sure the value from get_region_info conform to my thought.
        '''
        assert min_h % patch_size == 0 and min_w % patch_size == 0
        assert (max_h + 1) % patch_size == 0 and (max_w + 1) % patch_size == 0
        # print(f'min_h: {min_h}')
        # print(f'max_h: {max_h}')
        # print(f'min_w: {min_w}')
        # print(f'max_w: {max_w}')
        start_h = min_h // patch_size
        end_h = (max_h + 1) // patch_size
        start_w = min_w // patch_size
        end_w = (max_w + 1) // patch_size
        # print( ((end_h - start_h), (end_w - start_w)) )
        assert ((end_h - start_h), (end_w - start_w)) == new_size
        
        '''
        Get the corresponding patch area with regional information.
        '''
        if orig_size[0] != global_size[0]:
            print('Interpolate positional embeddings from %dx%d to %dx%d.' % (orig_size[0], orig_size[1], global_size[0], global_size[1]))
            pos_tokens = pos_embed_checkpoint.reshape(-1, orig_size[0], orig_size[1], embedding_size).permute(
                0, 3, 1, 2
            )
            new_pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size = (global_size[0], global_size[1]), mode = 'bicubic', align_corners = False
            )
            '''
            Do the slicing operation according to geographic information.
            '''
            new_pos_tokens = new_pos_tokens[:, :, start_h : end_h, start_w : end_w]
            print(f'Then do the slicing operation from ({global_size[0]} x {global_size[1]}) to ({new_size[0]} x {new_size[1]}).')
            new_pos_tokens = new_pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            checkpoint_model["net.pos_embed"] = new_pos_tokens
