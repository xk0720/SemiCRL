from torch.nn import Module, Sequential
from torch.nn import Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, AvgPool1d, Dropout3d
from torch.nn import ReLU, Sigmoid
import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F


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


class UNet(Module):
    # __                            __
    #  1|__   ________________   __|1
    #     2|__  ____________  __|2
    #        3|__  ______  __|3
    #           4|__ __ __|4
    # The convolution operations on either side are residual subject to 1*1 Convolution for channel homogeneity

    def __init__(self, in_dim=1, out_dim=2, feat_channels=[64, 256, 256, 512, 1024], residual='conv', n_classes=21,
                 proj_head_mode='linear'):
        # residual: conv for residual input x through 1*1 conv across every layer for downsampling, None for removal of residuals

        super(UNet, self).__init__()

        # Encoder downsamplers
        self.pool1 = MaxPool3d((2, 2, 2))
        self.pool2 = MaxPool3d((2, 2, 2))
        self.pool3 = MaxPool3d((2, 2, 2))
        self.pool4 = MaxPool3d((2, 2, 2))

        # Encoder convolutions
        self.conv_blk1 = Conv3D_Block(in_dim, feat_channels[0], residual=residual)
        self.conv_blk2 = Conv3D_Block(feat_channels[0], feat_channels[1], residual=residual)
        self.conv_blk3 = Conv3D_Block(feat_channels[1], feat_channels[2], residual=residual)
        self.conv_blk4 = Conv3D_Block(feat_channels[2], feat_channels[3], residual=residual)
        self.conv_blk5 = Conv3D_Block(feat_channels[3], feat_channels[4], residual=residual)

        # Decoder convolutions
        self.dec_conv_blk4 = Conv3D_Block(2 * feat_channels[3], feat_channels[3], residual=residual)
        self.dec_conv_blk3 = Conv3D_Block(2 * feat_channels[2], feat_channels[2], residual=residual)
        self.dec_conv_blk2 = Conv3D_Block(2 * feat_channels[1], feat_channels[1], residual=residual)
        self.dec_conv_blk1 = Conv3D_Block(2 * feat_channels[0], feat_channels[0], residual=residual)

        # Decoder upsamplers
        self.deconv_blk4 = Deconv3D_Block(feat_channels[4], feat_channels[3])
        self.deconv_blk3 = Deconv3D_Block(feat_channels[3], feat_channels[2])
        self.deconv_blk2 = Deconv3D_Block(feat_channels[2], feat_channels[1])
        self.deconv_blk1 = Deconv3D_Block(feat_channels[1], feat_channels[0])

        # Final 1*1 Conv Segmentation map
        self.one_conv = Conv3d(feat_channels[0], out_dim, kernel_size=1, stride=1, padding=0, bias=True)

        # Activation function
        self.sigmoid = Sigmoid()

        # projection (segmentation) head following the encoder
        self.projection_head = ProjectionHead(proj_head_mode=proj_head_mode, n_filters_in=256, n_filters_out=256)

        # classification head for last third stage of the decoder
        self.classification_head = nn.Conv3d(16 * 4, n_classes, kernel_size=1)

        # decide whether interpolate (4X upsample) the feature from the encoder
        self.is_interpolation = True
        self.scale_factor = 4.0
        self.align_corners = True

    def forward(self, x):
        # Encoder part

        x1 = self.conv_blk1(x)

        x_low1 = self.pool1(x1)
        x2 = self.conv_blk2(x_low1)

        x_low2 = self.pool2(x2)
        x3 = self.conv_blk3(x_low2)

        x_low3 = self.pool3(x3)
        x4 = self.conv_blk4(x_low3)

        x_low4 = self.pool4(x4)
        base = self.conv_blk5(x_low4)

        # Decoder part

        d4 = torch.cat([self.deconv_blk4(base), x4], dim=1)
        d_high4 = self.dec_conv_blk4(d4)

        d3 = torch.cat([self.deconv_blk3(d_high4), x3], dim=1)
        d_high3 = self.dec_conv_blk3(d3)
        d_high3 = Dropout3d(p=0.5)(d_high3)

        d2 = torch.cat([self.deconv_blk2(d_high3), x2], dim=1)
        d_high2 = self.dec_conv_blk2(d2)
        d_high2 = Dropout3d(p=0.5)(d_high2)

        d1 = torch.cat([self.deconv_blk1(d_high2), x1], dim=1)
        d_high1 = self.dec_conv_blk1(d1)

        seg = self.one_conv(d_high1)

        # output of projection head
        base = self.projection_head(base)

        if self.is_interpolation:
            base = F.interpolate(input=base, scale_factor=self.scale_factor,
                                      mode='trilinear', align_corners=self.align_corners)

        # feed mid_feature into the classification head for contrastive loss
        mid = self.classification_head(d_high3)

        #seg = self.sigmoid(self.one_conv(d_high1))

        return [seg, mid], base

class UNet_DTC(Module):
    # __                            __
    #  1|__   ________________   __|1
    #     2|__  ____________  __|2
    #        3|__  ______  __|3
    #           4|__ __ __|4
    # The convolution operations on either side are residual subject to 1*1 Convolution for channel homogeneity

    def __init__(self, in_dim=1, out_dim=2, feat_channels=[64, 256, 256, 512, 1024], residual='conv'):
        # residual: conv for residual input x through 1*1 conv across every layer for downsampling, None for removal of residuals

        super(UNet_DTC, self).__init__()

        # Encoder downsamplers
        self.pool1 = MaxPool3d((2, 2, 2))
        self.pool2 = MaxPool3d((2, 2, 2))
        self.pool3 = MaxPool3d((2, 2, 2))
        self.pool4 = MaxPool3d((2, 2, 2))

        # Encoder convolutions
        self.conv_blk1 = Conv3D_Block(in_dim, feat_channels[0], residual=residual)
        self.conv_blk2 = Conv3D_Block(feat_channels[0], feat_channels[1], residual=residual)
        self.conv_blk3 = Conv3D_Block(feat_channels[1], feat_channels[2], residual=residual)
        self.conv_blk4 = Conv3D_Block(feat_channels[2], feat_channels[3], residual=residual)
        self.conv_blk5 = Conv3D_Block(feat_channels[3], feat_channels[4], residual=residual)

        # Decoder convolutions
        self.dec_conv_blk4 = Conv3D_Block(2 * feat_channels[3], feat_channels[3], residual=residual)
        self.dec_conv_blk3 = Conv3D_Block(2 * feat_channels[2], feat_channels[2], residual=residual)
        self.dec_conv_blk2 = Conv3D_Block(2 * feat_channels[1], feat_channels[1], residual=residual)
        self.dec_conv_blk1 = Conv3D_Block(2 * feat_channels[0], feat_channels[0], residual=residual)

        # Decoder upsamplers
        self.deconv_blk4 = Deconv3D_Block(feat_channels[4], feat_channels[3])
        self.deconv_blk3 = Deconv3D_Block(feat_channels[3], feat_channels[2])
        self.deconv_blk2 = Deconv3D_Block(feat_channels[2], feat_channels[1])
        self.deconv_blk1 = Deconv3D_Block(feat_channels[1], feat_channels[0])

        # Final 1*1 Conv Segmentation map
        self.one_conv_1 = Conv3d(feat_channels[0], out_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.one_conv_2 = Conv3d(feat_channels[0], out_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.tanh = torch.nn.Tanh()

        # Activation function
        self.sigmoid = Sigmoid()

    def forward(self, x):
        # Encoder part

        x1 = self.conv_blk1(x)

        x_low1 = self.pool1(x1)
        x2 = self.conv_blk2(x_low1)

        x_low2 = self.pool2(x2)
        x3 = self.conv_blk3(x_low2)

        x_low3 = self.pool3(x3)
        x4 = self.conv_blk4(x_low3)

        x_low4 = self.pool4(x4)
        base = self.conv_blk5(x_low4)

        # Decoder part

        d4 = torch.cat([self.deconv_blk4(base), x4], dim=1)
        d_high4 = self.dec_conv_blk4(d4)

        d3 = torch.cat([self.deconv_blk3(d_high4), x3], dim=1)
        d_high3 = self.dec_conv_blk3(d3)
        d_high3 = Dropout3d(p=0.5)(d_high3)

        d2 = torch.cat([self.deconv_blk2(d_high3), x2], dim=1)
        d_high2 = self.dec_conv_blk2(d2)
        d_high2 = Dropout3d(p=0.5)(d_high2)

        d1 = torch.cat([self.deconv_blk1(d_high2), x1], dim=1)
        d_high1 = self.dec_conv_blk1(d1)

        seg = self.one_conv_1(d_high1)
        out_tanh = self.tanh(seg)
        seg = self.one_conv_2(d_high1)

        # seg = self.one_conv(d_high1)

        #seg = self.sigmoid(self.one_conv(d_high1))

        return out_tanh, seg



class Conv3D_Block(Module):

    def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1, residual=None):

        super(Conv3D_Block, self).__init__()

        self.conv1 = Sequential(
            Conv3d(inp_feat, out_feat, kernel_size=kernel,
                   stride=stride, padding=padding, bias=True),
            BatchNorm3d(out_feat),
            ReLU())

        self.conv2 = Sequential(
            Conv3d(out_feat, out_feat, kernel_size=kernel,
                   stride=stride, padding=padding, bias=True),
            BatchNorm3d(out_feat),
            ReLU())

        self.residual = residual

        if self.residual is not None:
            self.residual_upsampler = Conv3d(inp_feat, out_feat, kernel_size=1, bias=False)

    def forward(self, x):

        res = x

        if not self.residual:
            return self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x)) + self.residual_upsampler(res)


class Deconv3D_Block(Module):

    def __init__(self, inp_feat, out_feat, kernel=3, stride=2, padding=1):
        super(Deconv3D_Block, self).__init__()

        self.deconv = Sequential(
            ConvTranspose3d(inp_feat, out_feat, kernel_size=(kernel, kernel, kernel),
                            stride=(stride, stride, stride), padding=(padding, padding, padding), output_padding=1, bias=True),
            ReLU())

    def forward(self, x):
        return self.deconv(x)


class ChannelPool3d(AvgPool1d):

    def __init__(self, kernel_size, stride, padding):
        super(ChannelPool3d, self).__init__(kernel_size, stride, padding)
        self.pool_1d = AvgPool1d(self.kernel_size, self.stride, self.padding, self.ceil_mode)

    def forward(self, inp):
        n, c, d, w, h = inp.size()
        inp = inp.view(n, c, d * w * h).permute(0, 2, 1)
        pooled = self.pool_1d(inp)
        c = int(c / self.kernel_size[0])
        return inp.view(n, c, d, w, h)
