import torch
from torch import nn
import torch.nn.functional as F
from mscv import padding_forward
from torchvision import models

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        self.vgg_pretrained_features = models.vgg19(pretrained=True).features
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X, indices=None):
        if indices is None:
            indices = [2, 7, 12, 21, 30]
        out = []
        for i in range(indices[-1]):
            X = self.vgg_pretrained_features[i](X)
            if (i + 1) in indices:
                out.append(X)

        return out




class ConvLayer(torch.nn.Sequential):
    def __init__(self, conv, in_channels, out_channels, kernel_size, stride, padding=None, dilation=1, norm=None,
                 act=None):
        super(ConvLayer, self).__init__()
        padding = padding or dilation * (kernel_size - 1) // 2
        self.add_module('conv2d', conv(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation))
        if norm is not None:
            self.add_module('norm', norm(out_channels))
            # self.add_module('norm', norm(out_channels, track_running_stats=True))
        if act is not None:
            self.add_module('act', act)


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


#


class Grid(nn.Module): ##convolutional-grid 
    def __init__(self, n_feats):
        super(Grid, self).__init__()
        conv = nn.Conv2d
        act = nn.ReLU(True)
        norm = None
        self.n_feats = n_feats
        self.conv1 = ConvLayer(conv=conv, in_channels=n_feats, out_channels=n_feats,
                               kernel_size=3, stride=1, norm=norm, act=act)
        self.conv2 = ConvLayer(conv=conv, in_channels=n_feats, out_channels=n_feats,
                               kernel_size=5, stride=1, norm=norm, act=act)
        self.conv3 = ConvLayer(conv=conv, in_channels=n_feats, out_channels=n_feats,
                               kernel_size=7, stride=1, norm=norm, act=act)

        self.ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feats * 3, n_feats // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats // 16, n_feats * 3, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])

    def forward(self, x):
        input1 = x
        input2 = x
        input3 = x
        g1 = self.conv1(input1)
        g2 = self.conv1(g1) + self.conv2(input2)
        g3 = self.conv1(g2) + self.conv3(input3)

        w = self.ca(torch.cat([g1, g2, g3], dim=1))
        w = w.view(-1, 3, self.n_feats)[:, :, :, None, None]
        out = w[:, 0, ::] * g1 + w[:, 1, ::] * g2 + w[:, 2, ::] * g3
        return out + x



def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias).cuda()


class MultiScalePyramid(torch.nn.Module):
    def __init__(self, n_feats=256):
        super(MultiScalePyramid, self).__init__()
        self.sptial = SpatialGate()
        self.conv = Grid(n_feats)
        self.down = nn.MaxPool2d((2, 2))
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    @padding_forward(8)
    def forward(self, x):
        X_0A = x
        X_1A = self.conv(self.down(X_0A))
        X_2A = self.conv(self.down(X_1A))
        X_3A = self.conv(self.down(X_2A))

        X_3D = X_3A
        #
        X_2D = self.conv(X_2A + self.up(X_3D))
        X_1D = self.conv(X_1A + self.up(X_2D))
        X_0D = self.conv(X_0A + self.up(X_1D))

        output = self.sptial(X_0D)
        return output


#


class Net(torch.nn.Module):
    def __init__(self, in_channels=3, n_feats=256, norm=None,
                 bottom_kernel_size=3):
        super(Net, self).__init__()
        # Initial convolution layers
        conv = nn.Conv2d
        deconv = nn.ConvTranspose2d
        act = nn.ReLU(True)
        self.VGG = Vgg19(requires_grad=False)

        self.conv1 = ConvLayer(conv, in_channels, n_feats, kernel_size=bottom_kernel_size, stride=1, norm=None, act=act)
        self.VGGconv1 = ConvLayer(conv, 1472, n_feats, kernel_size=bottom_kernel_size, stride=1, norm=None, act=act)
        self.conv2 = ConvLayer(conv, n_feats * 2, n_feats * 2, kernel_size=3, stride=2, norm=norm, act=act)
        self.VGGconv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=2, norm=norm, act=act)
        self.conv3 = ConvLayer(conv, n_feats * 3, n_feats * 3, kernel_size=3, stride=2, norm=norm, act=act)
        self.VGGconv3 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=2, norm=norm, act=act)
        self.conv_last = ConvLayer(conv, n_feats * 4, n_feats, kernel_size=1, stride=1, norm=norm, act=act)
        self.MSP = nn.Sequential(*[MultiScalePyramid() for _ in range(1)])
        self.deconv1 = ConvLayer(deconv, n_feats, n_feats, kernel_size=4, stride=2, norm=None, act=act, padding=1)
        self.deconv2 = ConvLayer(deconv, n_feats, n_feats, kernel_size=4, stride=2, norm=None, act=act, padding=1)
        self.deconv3 = ConvLayer(deconv, n_feats, 3, kernel_size=3, stride=1, norm=None, act=act, padding=1)

    def forward(self, x):
        _, C, H, W = x.shape
        VGG_features_raw = self.VGG(x)

        VGG_features_resize = [F.interpolate(feature.detach(), size=(H, W), mode='bilinear', align_corners=False) for
                               feature in
                               VGG_features_raw]
        VGG_features = torch.cat(VGG_features_resize, 1)

        VGG_1 = self.VGGconv1(VGG_features)
        x1 = self.conv1(x)
        X_VGG_1 = torch.cat((VGG_1, x1), dim=1)  # channels = 512

        VGG_2 = self.VGGconv2(VGG_1)
        x2 = self.conv2(X_VGG_1)
        X_VGG_2 = torch.cat((VGG_2, x2), dim=1)

        VGG_3 = self.VGGconv3(VGG_2)
        x3 = self.conv3(X_VGG_2)
        X_VGG_3 = torch.cat((VGG_3, x3), dim=1)
        X_MSP_input = self.conv_last(X_VGG_3)
        x_MSP = self.MSP(X_MSP_input)

        y1 = self.deconv1(x_MSP)
        y2 = self.deconv2(y1)
        y3 = self.deconv3(y2)
        return y3
