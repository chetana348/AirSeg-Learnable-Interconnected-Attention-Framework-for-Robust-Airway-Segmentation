import torch
import torch.nn as nn
from scripts.models.SCA import *


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        return self.conv(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = Conv(in_channels, out_channels)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Sequential(Conv(out_channels, out_channels), Embedding(out_channels))
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.down = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.InstanceNorm2d(out_channels))

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)

        if self.down is not None:
            identity = self.down(x)

        out += identity
        out = self.act(out)

        return out
    
class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, with_resblock=True, apply_norm=False, activation=False):
        super(DownConv, self).__init__()
        
        self.with_resblock = with_resblock
        self.apply_norm = apply_norm
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.InstanceNorm2d(out_channels) if self.apply_norm else None
        self.act = nn.ReLU(inplace=True) if self.activation else None
        self.resblock = ResBlock(out_channels, out_channels) if self.with_resblock else None

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out) if self.apply_norm else out
        out = self.act(out) if self.activation else out
        out = self.resblock(out) if self.with_resblock else out
        return out
    
class ResUNet(nn.Module):
    def __init__(self, num_classes=2, self_attn=False, pos_att=True, sem_att=True, ch_attn = True, sp_attn = True):
        super(ResUNet, self).__init__()
        self.self_attn = self_attn
        self.pos_att = pos_att
        self.sem_att = sem_att
        self.ch_attn = ch_attn
        self.sp_attn = sp_attn
        
        self.conv0 = ResBlock(1, 64)
        self.down1 = DownConv(64, 128)                        
        self.down2 = DownConv(128, 128)
        self.down3 = DownConv(128, 256)
        self.down4 = DownConv(256, 256)
        self.down5_1 = DownConv(256, 512, apply_norm=True, activation=True, with_resblock=False)
        self.down5_2 = DownConv(512, 256, stride=1, padding=1, apply_norm=True, activation=True, with_resblock=False)

        if self_attn:
            self.att = nn.Sequential(
                nn.LayerNorm(512),
                Attention(512)
            )
        if self.pos_att:
            self.pos_att1 = LAAM() 
            self.pos_att2 = LAAM()
        if self.sem_att:
            self.sem_att1 = CAM(256) 
            self.sem_att2 = CAM(256) 

        self.up_concat4 = Base(512, 256, 256, upshape=256, ch_attn = self.ch_attn, sp_attn = False)
        self.up_concat3 = Base(512, 256, 128, upshape=256, ch_attn = self.ch_attn, sp_attn = False)
        self.up_concat2 = Base(256, 128, 128, upshape=128, ch_attn = False, sp_attn = self.sp_attn)
        self.up_concat1 = Base(256, 128, 64, upshape=128, ch_attn = False, sp_attn = self.sp_attn)
        self.up_concat0 = Base(128, 64, 64, upshape=64, ch_attn = False, sp_attn = self.sp_attn)
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, inputs):
        h, w = inputs.shape[2], inputs.shape[3]
        feat0 = self.conv0(inputs)  # (64, 512, 512)
        feat1 = self.down1(feat0)  # (128, 256, 256)
        feat2 = self.down2(feat1)  # (128, 128, 128)
        feat3 = self.down3(feat2)  # (256, 64, 64)
        feat4 = self.down4(feat3)  # (256, 32, 32)
        feat5 = self.down5_1(feat4)  # (512, 16, 16)
        
        if self.self_attn:
            feat5 = feat5.flatten(2).transpose(1, 2)
            feat5 = feat5 + self.att(feat5)
            feat5 = feat5.transpose(1, 2).view(-1, 512, int(h / 32), int(w / 32))
        
        feat5 = self.down5_2(feat5)  # (256, 16, 16)

        if self.pos_att:
            feat5 = self.pos_att1(feat5)  # Positional Attention Module
            feat5 = self.pos_att2(feat5)  # Additional Positional Attention Module

        if self.sem_att:
            feat4 = self.sem_att1(feat4)  # Semantic Attention Module
            feat4 = self.sem_att2(feat4)  # Additional Semantic Attention Module

        up4 = self.up_concat4(feat4, feat5)  # (256, 32, 32)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)
        up0 = self.up_concat0(feat0, up1)
        final = self.final(up0)
        return final
