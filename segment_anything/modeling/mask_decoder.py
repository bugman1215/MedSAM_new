# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d




class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        num_feature_scales: int = 4,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )
        # 修改
        self.fusion_conv = nn.Conv2d(
            in_channels=transformer_dim * num_feature_scales,  # 假设每个特征图的通道数一致
            out_channels=transformer_dim,
            kernel_size=1
        )

    def fuse_multiscale_features(self, features: torch.Tensor) -> torch.Tensor:
        # 使用最后一个特征图的尺寸作为 target_size
        target_size = features[-1].size()[2:4]  # target_size 应该是 [height, width] 的形式

        # 打印 target_size 的值
        print(f"Target size: {target_size}")

        # 调整前12个特征图的通道数到256
        reduced_features = [
            self.channel_reduction(f.permute(0, 3, 1, 2)) if f.shape[1] == 768 else f
            for f in features
        ]

        resized_features = []
        for f in reduced_features:
            if f.dim() == 3:  # 如果是3D张量，增加一个 batch 维度
                f = f.unsqueeze(0)
            resized_features.append(F.interpolate(f, size=target_size, mode='bilinear', align_corners=False))

        concatenated_features = torch.cat(resized_features, dim=1)  # 沿着通道维度拼接
        fused_features = self.fusion_conv(concatenated_features)  # 融合

        return fused_features


    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        # 在这里加入clip prompt
        clip_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          clip_prompt_embeddings (torch.Tensor) : the embeddings of the clip model
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """



        # 将多尺度特征上/下采样到相同尺寸，然后融合
        merged_features = self.fuse_multiscale_features(image_embeddings)

        masks, iou_pred = self.predict_masks(
            image_embeddings=merged_features,  # 使用融合后的特征
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            clip_prompt_embeddings=clip_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # 选择单掩码或多掩码输出
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        return masks, iou_pred


    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        #添加clip参数
        clip_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        batch_size = 1

        if sparse_prompt_embeddings.shape[0] != image_embeddings.shape[0]:
            sparse_prompt_embeddings = sparse_prompt_embeddings.expand(image_embeddings.shape[0], -1, -1)
        if clip_prompt_embeddings.shape[0] != image_embeddings.shape[0]:
            clip_prompt_embeddings = clip_prompt_embeddings.expand(image_embeddings.shape[0], -1, -1)

        # 打印张量形状
        print(f"image_embeddings shape: {image_embeddings.shape}")
        print(f"sparse_prompt_embeddings shape: {sparse_prompt_embeddings.shape}")
        print(f"dense_prompt_embeddings shape: {dense_prompt_embeddings.shape}")
        print(f"clip_prompt_embeddings shape: {clip_prompt_embeddings.shape}")

        # Concatenate output tokens
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        clip_prompt_embeddings = clip_prompt_embeddings.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1, -1
        ).squeeze(2)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings, clip_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
