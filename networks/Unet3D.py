# -*- coding: utf-8 -*-
"""
An implementation of the 3D U-Net paper:
     Özgün Çiçek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, Olaf Ronneberger:
     3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation.
     MICCAI (2) 2016: 424-432
Note that there are some modifications from the original paper, such as
the use of batch normalization, dropout, and leaky relu here.
The implementation is borrowed from: https://github.com/ozan-oktay/Attention-Gated-Networks
"""
import math

import torch.nn as nn
import torch.nn.functional as F

from networks.networks_other import init_weights
from networks.utils import UnetConv3, UnetUp3, UnetUp3_CT


class ProjectionHead(nn.Module):
    """
    Projection head for 3D CNNs.
    """
    def __init__(self, proj_head_mode, n_filters_in, n_filters_hidden=256, n_filters_out=256):
        super(ProjectionHead, self).__init__()

        if proj_head_mode == 'none':
            self.proj_head = nn.Identity()
        elif proj_head_mode == 'convmlp':
            # need batchnorm or not?
            self.proj_head = nn.Sequential(
                nn.Conv3d(n_filters_in, n_filters_hidden, kernel_size=1),
                nn.BatchNorm3d(n_filters_hidden), # BNReLU
                nn.ReLU(), # BNReLU
                nn.Conv3d(n_filters_hidden, n_filters_out, kernel_size=1),
            )
        elif proj_head_mode == 'linear':
            self.proj_head = nn.Conv3d(n_filters_in, n_filters_out, kernel_size=1)

    def forward(self, x):
        return self.proj_head(x)


class unet_3D(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True,
                 proj_head_mode='linear'):
        super(unet_3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))

        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

        # projection (segmentation) head following the encoder
        self.projection_head = ProjectionHead(proj_head_mode=proj_head_mode, n_filters_in=256, n_filters_out=256)

        # classification head for last third stage of the decoder
        self.classification_head = nn.Conv3d(16 * 4, n_classes, kernel_size=1)

        # decide whether interpolate (4X upsample) the feature from the encoder
        self.is_interpolation = True
        self.scale_factor = 4.0
        self.align_corners = True

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        center = self.dropout1(center)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        up1 = self.dropout2(up1)

        # output of projection head
        center = self.projection_head(center)

        if self.is_interpolation:
            center = F.interpolate(input=center, scale_factor=self.scale_factor,
                                      mode='trilinear', align_corners=self.align_corners)

        # feed mid_feature into the classification head for contrastive loss
        mid = self.classification_head(up3)

        final = self.final(up1)

        return [final, mid], center

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p