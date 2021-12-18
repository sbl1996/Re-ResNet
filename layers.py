import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAvgPool(nn.Module):
    def __init__(self, keep_dim=False):
        super().__init__()
        self.keep_dim = keep_dim

    def forward(self, x):
        return x.mean((2, 3), keepdim=self.keep_dim)


class AntiAliasing(nn.Module):

    def __init__(self, channels, kernel_size=3, stride=2):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride

        assert self.kernel_size == 3
        assert stride == 2
        a = torch.tensor([1., 2., 1.])

        filt = (a[:, None] * a[None, :]).clone().detach()
        filt = filt / torch.sum(filt)
        self.filt = filt[None, None, :, :].repeat((self.channels, 1, 1, 1)).cuda()

    def forward(self, x):
        return F.conv2d(x, self.filt, stride=2, padding=1, groups=x.shape[1])


def calc_same_padding(kernel_size, dilation=(1, 1)):
    kh, kw = kernel_size
    dh, dw = dilation
    ph = (kh + (kh - 1) * (dh - 1) - 1) // 2
    pw = (kw + (kw - 1) * (dw - 1) - 1) // 2
    return ph, pw


def Conv2d(in_channels, out_channels, kernel_size, stride=1,
           padding='same', dilation=1, groups=1, bias=None,
           norm=None, act=None, anti_alias=False, avd=False):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(padding, int):
        padding = (padding, padding)
    if padding == 'same':
        padding = calc_same_padding(kernel_size, dilation)

    assert not (avd and anti_alias)

    avd = avd and stride == (2, 2)
    anti_alias = anti_alias and stride == (2, 2)

    if avd or anti_alias:
        assert norm is not None and act is not None
        stride = (1, 1)

    layers = []
    if bias is None:
        bias = norm is None

    conv = nn.Conv2d(
        in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=bias)

    if norm is not None and act is not None:
        layers.append(NormAct(out_channels))
    else:
        if norm is not None:
            layers.append(Norm(out_channels))
        if act is not None:
            layers.append(Act())
    layers = [conv] + layers

    if avd:
        layers.append(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))

    if anti_alias:
        layers.append(AntiAliasing(channels=out_channels, kernel_size=3, stride=2))

    if len(layers) == 1:
        return layers[0]
    else:
        return nn.Sequential(*layers)


def Act(type='def'):
    if type == 'def':
        return Act('relu')
    elif type == 'relu':
        return nn.ReLU(inplace=True)
    elif type == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise NotImplementedError("Activation not implemented: %s" % type)


def Norm(channels):
    return nn.BatchNorm2d(channels)


def NormAct(channels):
    return nn.Sequential(
        Norm(channels),
        Act(),
    )


@torch.jit.script
class SpaceToDepthJit(object):
    def __call__(self, x: torch.Tensor):
        # assuming hard-coded that block_size==4 for acceleration
        N, C, H, W = x.size()
        x = x.view(N, C, H // 4, 4, W // 4, 4)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * 16, H // 4, W // 4)  # (N, C*bs^2, H//bs, W//bs)
        return x

class SpaceToDepthJit2(object):
    def __call__(self, x: torch.Tensor):
        N, C, H, W = x.size()
        x = x.view(N, C, H // 2, 2, W // 2, 2)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * 4, H // 2, W // 2)  # (N, C*bs^2, H//bs, W//bs)
        return x

class SpaceToDepthModule(nn.Module):
    def __init__(self, stride):
        super().__init__()
        if stride == 4:
            self.op = SpaceToDepthJit()
        else:
            self.op = SpaceToDepthJit2()

    def forward(self, x):
        return self.op(x)