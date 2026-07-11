# -*- coding: utf-8 -*-
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
# from ultralytics.nn.modules import Proto, Conv, Detect
from .block import Proto      # 原本就在 block.py 裡
from .conv import Conv        # 原本在 conv.py 裡
from .head import Detect      # 原本在 head.py 裡



multiplier = 1
rfb = False

class ASFFV5(nn.Module):
    def __init__(self, level, ch, multiplier=1, rfb=False,num_heads=3, vis=False):
        super(ASFFV5, self).__init__()
        self.level = level
        if num_heads == 3:
            self.dim = [int(ch[2] * multiplier), int(ch[1] * multiplier),int(ch[0] * multiplier)]
        else:
            self.dim = [int(ch[3] * multiplier), int(ch[2] * multiplier), int(ch[1] * multiplier),int(ch[0] * multiplier)]



        self.inter_dim = self.dim[self.level]
        if level == 0:
            if num_heads == 4:
                self.stride_level_1 = Conv(int(ch[2] * multiplier), self.inter_dim, 3, 2)
                self.stride_level_2 = Conv(int(ch[1] * multiplier), self.inter_dim, 3, 2)
                self.expand = Conv(self.inter_dim, int(ch[3] * multiplier), 3, 1)
            else:
                self.stride_level_1 = Conv(int(ch[1] * multiplier), self.inter_dim, 3, 2)
                self.stride_level_2 = Conv(int(ch[0] * multiplier), self.inter_dim, 3, 2)
                self.expand = Conv(self.inter_dim, int(ch[2] * multiplier), 3, 1)
        elif level == 1:
            if num_heads == 4:
                self.compress_level_0 = Conv(int(ch[3] * multiplier), self.inter_dim, 1, 1)
                self.stride_level_2 = Conv(int(ch[1] * multiplier), self.inter_dim, 3, 2)
                self.expand = Conv(self.inter_dim, int(ch[2] * multiplier), 3, 1)
            else:
                self.compress_level_0 = Conv(int(ch[2] * multiplier), self.inter_dim, 1, 1)
                self.stride_level_2 = Conv(int(ch[0] * multiplier), self.inter_dim, 3, 2)
                self.expand = Conv(self.inter_dim, int(ch[1] * multiplier), 3, 1)
        elif level == 2:
            if num_heads == 4:
                self.compress_level_0 = Conv(int(ch[2] * multiplier), self.inter_dim, 1, 1)
                self.stride_level_2 = Conv(int(ch[0] * multiplier), self.inter_dim, 3, 2)
                self.expand = Conv(self.inter_dim, int(ch[1] * multiplier), 3, 1)
            else:
                self.compress_level_0 = Conv(int(ch[2] * multiplier), self.inter_dim, 1, 1)
                self.compress_level_1 = Conv(int(ch[1] * multiplier), self.inter_dim, 1, 1)
                self.expand = Conv(self.inter_dim, int(ch[0] * multiplier), 3, 1)

        elif level == 3 and num_heads == 4:
            self.compress_level_0 = Conv(int(ch[2] * multiplier), self.inter_dim, 1, 1)
            self.compress_level_1 = Conv(int(ch[1] * multiplier), self.inter_dim, 1, 1)
            self.expand = Conv(self.inter_dim, int(ch[0] * multiplier), 3, 1)


        # when adding rfb, we use half number of channels to save memory
        compress_c = 8 if rfb else 16
        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_levels = Conv(compress_c * 3, 3, 1, 1)
        self.vis = vis

    def forward(self, x):

        if len(x) == 3:
            x_level_0 = x[2]  # l
            x_level_1 = x[1]  # m
            x_level_2 = x[0]  # s
        elif len(x) == 4:
            x_level_0 = x[3]
            x_level_1 = x[2]
            x_level_2 = x[1]
            x_level_3 = x[0]

        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            if len(x) == 4:
                level_0_compressed = self.compress_level_0(x_level_1)
                level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
                level_1_resized = x_level_2
                level_2_resized = self.stride_level_2(x_level_3)
            else:
                level_0_compressed = self.compress_level_0(x_level_0)
                level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
                x_level_1_compressed = self.compress_level_1(x_level_1)
                level_1_resized = F.interpolate(x_level_1_compressed, scale_factor=2, mode='nearest')
                level_2_resized = x_level_2

        elif self.level == 3 and len(x) == 4:
            level_0_compressed = self.compress_level_0(x_level_1)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            x_level_1_compressed = self.compress_level_1(x_level_2)
            level_1_resized = F.interpolate(x_level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_3

        # 加权计算
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)

        # 拼接权重并进行归一化
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


class DetectASFF(Detect):
    def __init__(self, nc=80, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__(nc, ch)

        # 根据 num_heads 动态选择 fusion 数量
        self.l0_fusion = ASFFV5(level=0, ch=ch, multiplier=multiplier, rfb=rfb, num_heads=self.nl)
        self.l1_fusion = ASFFV5(level=1, ch=ch, multiplier=multiplier, rfb=rfb, num_heads=self.nl)
        self.l2_fusion = ASFFV5(level=2, ch=ch, multiplier=multiplier, rfb=rfb, num_heads=self.nl)

        if self.nl == 4:
            self.l3_fusion = ASFFV5(level=3, ch=ch, multiplier=multiplier, rfb=rfb, num_heads=self.nl)



    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        if self.nl == 3:
            x1 = self.l0_fusion(x)
            x2 = self.l1_fusion(x)
            x3 = self.l2_fusion(x)
            x = [x3, x2, x1]
        elif self.nl == 4:
            x1 = self.l0_fusion(x)
            x2 = self.l1_fusion(x)
            x3 = self.l2_fusion(x)
            x4 = self.l3_fusion(x)
            x = [x4, x3, x2, x1]

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)



class SegmentASFF(DetectASFF):
    """YOLOv8 Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = DetectASFF.forward(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))

