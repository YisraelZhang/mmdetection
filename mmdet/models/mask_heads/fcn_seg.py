import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import auto_fp16, force_fp32
from ..builder import build_loss
from ..utils import ConvModule

class SEG(nn.Module):
    def __init__(self,
                 num_convs=4,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 num_classes=2,
                 class_agnostic=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_mask=dict(
                     type="CrossEntropyLoss", use_mask=True, loss_weight=1.0
                 )):
        super(SEG, self).__init__()

        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.loss_mask = build_loss(loss_mask)

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i==0 else self.conv_out_channels
            )
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg
                )
            )
        out_channels = 1 if self.class_agnostic else self.num_classes
        self.conv_logits = nn.Conv2d(self.conv_out_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None

    def init_weights(self):
        for m in self.conv_logits:
            if m is None:
                continue
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    @auto_fp16()
    def forward(self, x, scales=None):
        for conv in self.convs:
            x = conv(x)
        mask_pred = self.conv_logits(x)
        return mask_pred

    @force_fp32(apply_to=('mask_pred',))
    def loss(self, mask_pred, mask_targets):
        loss = dict()
        if self.class_agnostic:
            loss_mask = self.loss_mask(mask_pred, mask_targets)
        else:
            loss_mask = self.loss_mask(mask_pred, mask_targets)
        loss['loss_mask'] = loss_mask
        return loss_mask