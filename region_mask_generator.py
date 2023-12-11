# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed

from ptflops import get_model_complexity_info
import numpy as np


class GetSubnet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten(start_dim=1).sort()
        j = int((1 - k) * scores[0].numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten(start_dim=1)
        for i in range(len(idx)):
            flat_out[i,idx[i,:j]] = 0
            flat_out[i,idx[i,j:]] = 1
        # flat_out[idx[:,:j]] = 0
        # flat_out[idx[:,j:]] = 1
        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None
    
class CNN(nn.Module):
    def __init__(self, region_size=16, in_channels=4):
        super(CNN,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=16,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ELU(),
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=8,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ELU()
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=8,out_channels=1,kernel_size=region_size,stride=region_size),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.mask = None
        self.new_mask = None
        # self.f1 = nn.Flatten()
        # self.mlp = MLP_1L(in_size=7, cluster_size=cluster_size)
        # self.out=nn.Linear(968+cluster_size*cluster_size, cluster_size*cluster_size)
        # self.cluster_size = cluster_size
        # self.input_size = in_size    
    def update_last_mask(self, mask):
        self.mask.data.copy_(mask)

    def update_new_mask(self, mask):
        self.new_mask.data.copy_(mask)

    def init_masks(self, x):
        # with open('datasets/hm_train.npy', 'rb') as f:
        #     hm_train = np.load(f)
        # init_value = torch.tensor(hm_train/hm_train.max(), device=input.device, dtype=torch.float)
        # self.init_value = GetSubnet.apply(init_value.abs(), 0.5)
        self.mask = torch.zeros(1, 1, x.shape[2], x.shape[3]).to(x.device)
        self.new_mask = torch.zeros(1, 1, x.shape[2], x.shape[3]).to(x.device)

    def forward(self, x, frame_ids):
        if self.mask is None:
            self.init_masks(x)
        mask = self.mask.repeat(x.shape[0],1,1,1)
        if 1 in frame_ids:
            first_frame_id = torch.where(frame_ids==1)[0][0]
            mask[first_frame_id:] = self.new_mask.repeat(x.shape[0]-first_frame_id,1,1,1)
        x = torch.cat((x, mask), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.mlp(x)
        # x = self.out(x)
        return x

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        # self.obj_token = nn.Parameter(torch.zeros(1, num_patches, 1), requires_grad=False)  # object token

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        # self.decoder_embed = nn.Linear(embed_dim+1, 1, bias=True)

        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        # self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size[0]*patch_size[1] * in_chans, bias=True) # decoder to patch
        self.decoder_pred = nn.Linear(decoder_embed_dim, 1, bias=True)
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        p0, p1 = self.patch_embed.patch_size
        grid_size = self.patch_embed.img_size[0] // p0, self.patch_embed.img_size[1] // p1
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size, cls_token=False)
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], grid_size, cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=.02)
        # torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def update_obj_token(self, obj_token):
        self.obj_token.data.copy_(obj_token)
   
    def forward_encoder(self, x):
        # Shape of object token should be (1,num_patches,1) indicating if an object is present or not in the patch from last batch predictions
        # embed patches
        x = self.patch_embed(x)

        # append obj_token with pos_embed
        # pos_embed = torch.cat((self.pos_embed, self.obj_token), dim=-1)

        # add pos embed w/o cls token
        # obj_token = self.obj_token.repeat(x.shape[0], 1, 1)
        # x = torch.cat((x, torch.zeros_like(obj_token)), dim=-1) + pos_embed
        x = x + self.pos_embed

        # masking: length -> length * mask_ratio
        # x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        # cls_token = self.cls_token + self.pos_embed[:, :1, :]
        # cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x
        # return x

    def forward_decoder(self, x, ids_restore=None):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        # mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # # x_ = torch.cat([x[:, 1:1+ids_restore, :], mask_tokens, x[:, ids_restore+2:, :]], dim=1)  # no cls token
        # x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed
        # x = x + self.decoder_pos_embed[:, 1:, :]

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # # remove cls token
        # x = x[:, 1:, :]

        return x

    
    def forward(self, imgs, mask=None, org_imgs=None, mask_ratio=0.75, dataset='S7-ISP'):
        # obj_token = torch.zeros(1, self.patch_embed.num_patches, 1).to(imgs.device)
        latent = self.forward_encoder(imgs)
        # latent = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent)  # [N, L, p*p*3]
        # loss = self.forward_loss(imgs, pred, mask, org_imgs, dataset)
        # pred = pred[:,self.patch_embed.num_patches//2,:].reshape(-1,self.patch_embed.patch_size[0],self.patch_embed.patch_size[1])
        return pred


def mae_vit_base_patch16_dec512d8b(region_size=16, **kwargs):
    model = MaskedAutoencoderViT(
        patch_size=(region_size,region_size), 
        embed_dim=256, depth=2, num_heads=1,
        decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=1,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks

if __name__ == "__main__":
    model = mae_vit_base_patch16(img_size=(384,1280), in_chans=3)
    # torch.save(model, 'vit.pt')
    x = torch.rand((1,3,384,1280))
    # x = torch.rand((1,1,256,256))
    y = model(x)
    x = x.numpy()
    
    #np.save('MHA_input.npy', x)
    
    # print(x.shape)
    # print(y.shape)
    flops, params = get_model_complexity_info(model, (3, 384, 1280), as_strings=True, print_per_layer_stat=True)
    print('Flops:  ' + flops)
    print('Params: ' + params)