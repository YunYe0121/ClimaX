# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed, trunc_normal_

from climax.utils.pos_embed import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
)

from .parallelpatchembed import ParallelVarPatchEmbed


class ClimaX(nn.Module):
    '''Implements the ClimaX model as described in the paper,
    https://arxiv.org/abs/2301.10343

    Args:
        default_vars (list): list of default variables to be used for training
        img_size (list): image size of the input data
        patch_size (int): patch size of the input data
        embed_dim (int): embedding dimension
        depth (int): number of transformer layers
        decoder_depth (int): number of decoder layers
        num_heads (int): number of attention heads
        mlp_ratio (float): ratio of mlp hidden dimension to embedding dimension
        drop_path (float): stochastic depth rate
        drop_rate (float): dropout rate
        parallel_patch_embed (bool): whether to use parallel patch embedding
    '''

    def __init__(
        self,
        default_vars,
        img_size=[32, 64],
        patch_size=2,
        embed_dim=1024,
        depth=8,
        decoder_depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1,
        parallel_patch_embed=False,
    ):
        super().__init__()

        # TODO: remove time_history parameter
        self.img_size = img_size
        self.patch_size = patch_size
        self.default_vars = default_vars
        self.parallel_patch_embed = parallel_patch_embed
        # variable tokenization: separate embedding layer for each input variable
        if self.parallel_patch_embed:
            self.token_embeds = ParallelVarPatchEmbed(len(default_vars), img_size, patch_size, embed_dim)
            self.num_patches = self.token_embeds.num_patches
        else:
            self.token_embeds = nn.ModuleList(
                [PatchEmbed(img_size, patch_size, 1, embed_dim) for i in range(len(default_vars))]
            )
            self.num_patches = self.token_embeds[0].num_patches

        # variable embedding to denote which variable each token belongs to
        # helps in aggregating variables
        self.var_embed, self.var_map = self.create_var_embedding(embed_dim)

        # variable aggregation: a learnable query and a single-layer cross attention
        self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # positional embedding and lead time embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=True)
        self.lead_time_embed = nn.Linear(1, embed_dim)

        # --------------------------------------------------------------------------

        # ViT backbone
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                    drop=drop_rate,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # --------------------------------------------------------------------------

        # prediction head
        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(embed_dim, len(self.default_vars) * patch_size**2))
        self.head = nn.Sequential(*self.head)

        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialize pos_emb and var_emb
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        var_embed = get_1d_sincos_pos_embed_from_grid(self.var_embed.shape[-1], np.arange(len(self.default_vars)))
        self.var_embed.data.copy_(torch.from_numpy(var_embed).float().unsqueeze(0))

        # token embedding layer
        if self.parallel_patch_embed:
            for i in range(len(self.token_embeds.proj_weights)):
                w = self.token_embeds.proj_weights[i].data
                trunc_normal_(w.view([w.shape[0], -1]), std=0.02)
        else:
            for i in range(len(self.token_embeds)):
                w = self.token_embeds[i].proj.weight.data
                trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_var_embedding(self, dim):
        var_embed = nn.Parameter(torch.zeros(1, len(self.default_vars), dim), requires_grad=True)
        # TODO: create a mapping from var --> idx
        var_map = {}
        idx = 0
        for var in self.default_vars:
            var_map[var] = idx
            idx += 1
        return var_embed, var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        ids = np.array([self.var_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def get_var_emb(self, var_emb, vars):
        ids = self.get_var_ids(vars, var_emb.device)
        return var_emb[:, ids, :]

    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        '''
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        '''
        p = self.patch_size
        c = len(self.default_vars)
        h = self.img_size[0] // p if h is None else h // p
        w = self.img_size[1] // p if w is None else w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def aggregate_variables(self, x: torch.Tensor):
        '''
        x: B, V, L, D
        '''
        b, _, l, _ = x.shape
        x = torch.einsum('bvld->blvd', x)
        x = x.flatten(0, 1)  # BxL, V, D

        var_query = self.var_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.var_agg(var_query, x, x)  # BxL, D
        x = x.squeeze()

        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, D
        return x

    def forward_encoder(self, x: torch.Tensor, lead_times: torch.Tensor, variables):
        # x: `[B, V, H, W]` shape.

        if isinstance(variables, list):
            variables = tuple(variables)

        # tokenize each variable separately
        embeds = []
        var_ids = self.get_var_ids(variables, x.device)

        if self.parallel_patch_embed:
            x = self.token_embeds(x, var_ids)  # B, V, L, D
        else:
            for i in range(len(var_ids)):
                id = var_ids[i]
                embeds.append(self.token_embeds[id](x[:, i : i + 1]))
            x = torch.stack(embeds, dim=1)  # B, V, L, D

        # add variable embedding
        var_embed = self.get_var_emb(self.var_embed, variables)
        x = x + var_embed.unsqueeze(2)  # B, V, L, D

        # variable aggregation
        x = self.aggregate_variables(x)  # B, L, D

        # add pos embedding
        x = x + self.pos_embed

        # add lead time embedding
        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))  # B, D
        lead_time_emb = lead_time_emb.unsqueeze(1)
        x = x + lead_time_emb  # B, L, D

        x = self.pos_drop(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward(self, x, y, lead_times, variables, out_variables, metric, lat):
        '''Forward pass through the model.

        Args:
            x: `[B, Vi, H, W]` shape. Input weather/climate variables
            y: `[B, Vo, H, W]` shape. Target weather/climate variables
            lead_times: `[B]` shape. Forecasting lead times of each element of the batch.

        Returns:
            loss (list): Different metrics.
            preds (torch.Tensor): `[B, Vo, H, W]` shape. Predicted weather/climate variables.
        '''
        out_transformers = self.forward_encoder(x, lead_times, variables)  # B, L, D
        preds = self.head(out_transformers)  # B, L, V*p*p

        preds = self.unpatchify(preds)
        out_var_ids = self.get_var_ids(tuple(out_variables), preds.device)
        preds = preds[:, out_var_ids]

        if metric is None:
            loss = None
        else:
            loss = [m(preds, y, out_variables, lat) for m in metric]

        return loss, preds

    def evaluate(self, x, y, lead_times, variables, out_variables, transform, metrics, lat, clim, log_postfix):
        _, preds = self.forward(x, y, lead_times, variables, out_variables, metric=None, lat=lat)
        return [m(preds, y, transform, out_variables, lat, clim, log_postfix) for m in metrics]

'''
This variant ClimaX model is written by YYC, the man in CLLab.
'''
class ClimaXChannelMask(ClimaX):
    def __init__(
        self,
        default_vars,
        img_size = [32, 64],
        patch_size = 2,
        embed_dim = 1024,
        depth = 8,
        decoder_depth = 2,
        num_heads = 16,
        mlp_ratio = 4.0,
        drop_path = 0.1,
        drop_rate = 0.1,
        parallel_patch_embed = False,
        # 
        variables_to_mask = None,
        random_mask_variables = False,
        # 
    ):
        super().__init__(
            default_vars = default_vars,
            img_size = img_size,
            patch_size = patch_size,
            embed_dim = embed_dim,
            depth = depth,
            decoder_depth = decoder_depth,
            num_heads = num_heads,
            mlp_ratio = mlp_ratio,
            drop_path = drop_path,
            drop_rate = drop_rate,
            parallel_patch_embed = parallel_patch_embed,
        )

        '''
        Initialize variables to mask.
        '''
        self.variables_to_mask = variables_to_mask
        self.random_mask_variables = random_mask_variables

        '''
        Add a learnable mask token.
        '''
        # self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        L = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        V = len(variables_to_mask)
        self.mask_token = nn.Parameter(torch.zeros(V, L, embed_dim))

    '''
    Modify the forward_encoder fucntion do the 'variable masking'.
    '''
    def forward_encoder(
            self,
            x: torch.Tensor,
            lead_times: torch.Tensor,
            variables,
            is_pretraining: bool,
        ):
        
        # x: `[B, V, H, W]` shape
        if isinstance(variables, list):
            variables = tuple(variables)

        # tokenize each variable separately
        embeds = []
        var_ids = self.get_var_ids(variables, x.device)

        if self.parallel_patch_embed:
            x = self.token_embeds(x, var_ids)  # B, V, L, D
        else:
            for i in range(len(var_ids)):
                id = var_ids[i]
                embeds.append(self.token_embeds[id](x[:, i : i + 1]))
            x = torch.stack(embeds, dim=1)  # B, V, L, D

        # add variable embedding
        var_embed = self.get_var_emb(self.var_embed, variables)
        x = x + var_embed.unsqueeze(2)  # B, V, L, D

        '''
        Mask specific channels (variables).
        '''
        if is_pretraining:
            # B, V, L, D.
            x = self.mask_channels(x)

        # variable aggregation
        x = self.aggregate_variables(x)  # B, L, D

        # add pos embedding
        x = x + self.pos_embed

        # add lead time embedding
        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))  # B, D
        lead_time_emb = lead_time_emb.unsqueeze(1)
        x = x + lead_time_emb  # B, L, D

        x = self.pos_drop(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    '''
    Modify the fucntion forward to match modified 'forward_encoder function'.
    '''
    def forward(
            self,
            x,
            y,
            lead_times,
            variables,
            out_variables,
            metric,
            lat,
            # 
            is_pretraining: bool,
            # 
        ):
        '''
        Forward pass through the model.

        Args:
            x: `[B, Vi, H, W]` shape. Input weather/climate variables
            y: `[B, Vo, H, W]` shape. Target weather/climate variables
            lead_times: `[B]` shape. Forecasting lead times of each element of the batch.

        Returns:
            loss (list): Different metrics.
            preds (torch.Tensor): `[B, Vo, H, W]` shape. Predicted weather/climate variables.
        '''
        
        # out_transformers = self.forward_encoder(x, lead_times, variables)  # B, L, D
        # B, L, D.
        out_transformers = self.forward_encoder(
            x, lead_times, variables, is_pretraining,
        )  
        preds = self.head(out_transformers)  # B, L, V*p*p

        preds = self.unpatchify(preds)
        out_var_ids = self.get_var_ids(tuple(out_variables), preds.device)
        preds = preds[:, out_var_ids]

        if metric is None:
            loss = None
        else:
            loss = [m(preds, y, out_variables, lat) for m in metric]

        return loss, preds

    def evaluate(self,
                x, y, lead_times,
                variables, out_variables,
                transform, metrics, lat, clim, log_postfix
        ):
        _, preds = self.forward(
            x, y, lead_times,
            variables, out_variables,
            metric = None, lat = lat,
            # 
            is_pretraining = False,
            # 
            )
        return [m(preds, y, transform, out_variables, lat, clim, log_postfix) for m in metrics]

    '''
    Do the channel masking.
    '''
    def mask_channels(self, x):
        '''
        Masks specific channels (variables) in the input.

        Args:
            x: Input tensor of shape (B, V, L, D), where V is the number of variables.

        Returns:
            masked_x: The input with specified channels masked (replaced by the mask token).
        '''
        # Get the corresponding variable ids to do variable masking.
        if not self.random_mask_variables:
            mask_ids = self.get_var_ids(tuple(self.variables_to_mask), x.device)
        else:
            input_variable_len = x.shape[1]
            mask_ids = torch.randint(low = 0, high = input_variable_len, size = (len(self.variables_to_mask), ))
        
        # Mask the specified variables.
        masked_x = x.clone()
        
        masked_x[:, mask_ids] = self.mask_token.expand( x.shape[0], len(mask_ids), x.shape[2], x.shape[3] ).to(masked_x.dtype)

        return masked_x

'''
This variant ClimaX model is written by YYC, the man in CLLab.
'''
class ClimaXPatchMask(ClimaX):
    def __init__(
        self,
        default_vars,
        img_size = [32, 64],
        patch_size = 2,
        embed_dim = 1024,
        depth = 8,
        decoder_depth = 2,
        num_heads = 16,
        mlp_ratio = 4.0,
        drop_path = 0.1,
        drop_rate = 0.1,
        parallel_patch_embed = False,
        # 
        mask_ratio = 0.5,
        # 
    ):
        super().__init__(
            default_vars = default_vars,
            img_size = img_size,
            patch_size = patch_size,
            embed_dim = embed_dim,
            depth = depth,
            decoder_depth = decoder_depth,
            num_heads = num_heads,
            mlp_ratio = mlp_ratio,
            drop_path = drop_path,
            drop_rate = drop_rate,
            parallel_patch_embed = parallel_patch_embed,
        )

        self.mask_ratio = mask_ratio
        L = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.num_masked = int(L * self.mask_ratio)
    
        '''
        Add a learnable mask token.
        '''
        # self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(self.num_masked, embed_dim))

    '''
    Modify the forward_encoder fucntion do the 'patch masking'.
    '''
    def forward_encoder(
        self, x: torch.Tensor, lead_times: torch.Tensor, variables, is_pretraining):
        # x: `[B, V, H, W]` shape.

        if isinstance(variables, list):
            variables = tuple(variables)

        # tokenize each variable separately
        embeds = []
        var_ids = self.get_var_ids(variables, x.device)

        if self.parallel_patch_embed:
            x = self.token_embeds(x, var_ids)  # B, V, L, D
        else:
            for i in range(len(var_ids)):
                id = var_ids[i]
                embeds.append(self.token_embeds[id](x[:, i : i + 1]))
            x = torch.stack(embeds, dim=1)  # B, V, L, D

        # add variable embedding
        var_embed = self.get_var_emb(self.var_embed, variables)
        x = x + var_embed.unsqueeze(2)  # B, V, L, D

        # variable aggregation
        x = self.aggregate_variables(x)  # B, L, D

        # Apply masking.
        if is_pretraining:
            x, mask = self.mask_patches(x, self.mask_ratio)

        # add pos embedding
        x = x + self.pos_embed

        # add lead time embedding
        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))  # B, D
        lead_time_emb = lead_time_emb.unsqueeze(1)
        x = x + lead_time_emb  # B, L, D

        x = self.pos_drop(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # return x
        return x

    '''
    Modify the fucntion forward to match modified 'forward_encoder function'.
    This is modified by YYC.
    '''
    def forward(self, x, y, lead_times, variables, out_variables, metric, lat, is_pretraining):
        '''Forward pass through the model.

        Args:
            x: `[B, Vi, H, W]` shape. Input weather/climate variables
            y: `[B, Vo, H, W]` shape. Target weather/climate variables
            lead_times: `[B]` shape. Forecasting lead times of each element of the batch.

        Returns:
            loss (list): Different metrics.
            preds (torch.Tensor): `[B, Vo, H, W]` shape. Predicted weather/climate variables.
        '''
        # out_transformers = self.forward_encoder(x, lead_times, variables)  # B, L, D
        out_transformers = self.forward_encoder(x, lead_times, variables, is_pretraining)  # B, L, D
        preds = self.head(out_transformers)  # B, L, V*p*p

        preds = self.unpatchify(preds)
        out_var_ids = self.get_var_ids(tuple(out_variables), preds.device)
        preds = preds[:, out_var_ids]

        if metric is None:
            loss = None
        else:
            loss = [m(preds, y, out_variables, lat) for m in metric]

        return loss, preds

    def evaluate(self,
                x, y, lead_times,
                variables, out_variables,
                transform, metrics, lat, clim, log_postfix
        ):
        _, preds = self.forward(
            x, y, lead_times,
            variables, out_variables,
            metric = None, lat = lat,
            # 
            is_pretraining = False,
            # 
            )
        return [m(preds, y, transform, out_variables, lat, clim, log_postfix) for m in metrics]


    '''
    The patch mask function written by YYC.
    '''
    def mask_patches(self, x, mask_ratio = 0.5):
        '''
        Randomly masks a portion of the input patches.

        Args:
            x: Input patch embeddings of shape (B, L, D) where L is the number of patches.
            mask_ratio: The proportion of patches to mask.

        Returns:
            masked_x: The input with masked patches replaced by the mask token.
            mask: Boolean tensor indicating which patches were masked.
        '''
        B, L, D = x.shape
        num_masked = int(L * mask_ratio)

        assert num_masked == self.num_masked

        # Randomly select patches to mask.
        mask = torch.ones(L, dtype = torch.bool)
        mask_indices = torch.randperm(L)[: num_masked]
        mask[mask_indices] = False
        
        # Replace masked patches with the mask token.
        masked_x = x.clone()
        masked_x[:, mask_indices] = self.mask_token.expand(B, num_masked, D).to(masked_x.dtype)
        
        return masked_x, mask