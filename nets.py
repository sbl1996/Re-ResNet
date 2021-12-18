from math import log, floor
import torch
import torch.nn as nn

from layers import Conv2d, GlobalAvgPool, SpaceToDepthModule, Act, Norm, NormAct


def get_shortcut_vd(in_channels, out_channels, stride,
                    norm='def', identity=True):
    if stride != 1 or in_channels != out_channels:
        shortcut = []
        if stride != 1:
            shortcut.append(nn.AvgPool2d(kernel_size=(2, 2), stride=2, ceil_mode=True, count_include_pad=False))
        shortcut.append(
            Conv2d(in_channels, out_channels, kernel_size=1, norm=norm))
        shortcut = nn.Sequential(*shortcut)
    else:
        shortcut = nn.Identity() if identity else None
    return shortcut


def _make_layer(block, in_channels, channels, blocks, stride, **kwargs):
    layers = [block(in_channels, channels, stride=stride,
                    start_block=True, **kwargs)]
    in_channels = channels * block.expansion
    for i in range(1, blocks - 1):
        layers.append(block(in_channels, channels, stride=1,
                            exclude_bn0=i == 1, **kwargs))
    layers.append(block(in_channels, channels, stride=1,
                        end_block=True, **kwargs))
    return nn.Sequential(*layers)


def _get_kwargs(kwargs, i, n=4):
    d = {}
    for k, v in kwargs.items():
        if isinstance(v, tuple):
            if len(v) == n:
                d[k] = v[i]
            else:
                d[k] = v
        else:
            d[k] = v
    return d



class _IResNet(nn.Module):

    def __init__(self, stem, block, layers, num_classes, channels, strides, **kwargs):
        super().__init__()

        assert len(layers) == len(channels) == len(strides)

        self.n_stages = len(strides)

        self.stem = stem
        c_in = stem.out_channels

        for i, (c, n, s) in enumerate(zip(channels, layers, strides)):
            layer = _make_layer(
                block, c_in, c, n, s, **_get_kwargs(kwargs, i, self.n_stages))
            c_in = c * block.expansion
            setattr(self, "layer" + str(i+1), layer)

        self.avgpool = GlobalAvgPool()
        self.fc = nn.Linear(c_in, num_classes)

    def forward(self, x):
        x = self.stem(x)

        for i in range(self.n_stages):
            layer = getattr(self, "layer" + str(i+1))
            x = layer(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x


class SELayer(nn.Module):

    def __init__(self, in_channels, reduction=None, groups=1, se_channels=None,
                 min_se_channels=32, act='def', mode=0, **kwargs):
        super().__init__(**kwargs)
        self.pool = GlobalAvgPool(keep_dim=True)
        if mode == 0:
            # σ(f_{W1, W2}(y))
            channels = se_channels or min(max(in_channels // reduction, min_se_channels), in_channels)
            self.fc = nn.Sequential(
                Conv2d(in_channels, channels, kernel_size=1, bias=False, act=act),
                Conv2d(channels, in_channels, 1, groups=groups, act='sigmoid'),
            )
        elif mode == 1:
            # σ(w ⊙ y)
            assert groups == 1
            self.fc = Conv2d(in_channels, in_channels, 1,
                             groups=in_channels, bias=False, act='sigmoid')
        elif mode == 2:
            # σ(Wy)
            assert groups == 1
            self.fc = Conv2d(in_channels, in_channels, 1, bias=False, act='sigmoid')
        else:
            raise ValueError("Not supported mode: {}" % mode)

    def forward(self, x):
        s = self.pool(x)
        s = self.fc(s)
        return x * s


class ECALayer(nn.Module):

    def __init__(self, channels=None, kernel_size=None):
        super().__init__()
        if channels is None:
            assert kernel_size is not None
        else:
            gamma, b = 2, 1
            t = int(abs((log(channels, 2) + b) / gamma))
            kernel_size = t if t % 2 else t + 1
        self.avg_pool = GlobalAvgPool(keep_dim=True)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, bias=False, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        y = self.avg_pool(x)

        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = torch.sigmoid(y)
        return x * y.expand_as(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride,
                 start_block=False, end_block=False, exclude_bn0=False,
                 se_reduction=4, se_last=False, se_mode=0, eca=False,
                 anti_alias=False, avd=False):
        super().__init__()
        self.eca = None
        self.se = None
        self.se_last = se_last

        out_channels = channels * self.expansion

        if not start_block:
            if exclude_bn0:
                self.act0 = Act()
            else:
                self.norm_act0 = NormAct(in_channels)

        self.conv1 = Conv2d(in_channels, channels, kernel_size=1,
                            norm='def', act='def')

        self.conv2 = Conv2d(channels, channels, kernel_size=3, stride=stride,
                            norm='def', act='def', anti_alias=anti_alias, avd=avd)

        if se_reduction and not self.se_last:
            self.se = SELayer(channels, se_channels=out_channels // se_reduction, mode=se_mode)

        self.conv3 = Conv2d(channels, out_channels, kernel_size=1)

        if start_block:
            self.bn3 = Norm(out_channels)

        if se_reduction and self.se_last:
            self.se = SELayer(out_channels, se_channels=out_channels // se_reduction, mode=se_mode)

        if eca:
            self.eca = ECALayer(kernel_size=3)

        if end_block:
            self.norm_act3 = NormAct(out_channels)

        self.shortcut = get_shortcut_vd(in_channels, out_channels, stride)

        self.start_block = start_block
        self.end_block = end_block
        self.exclude_bn0 = exclude_bn0

    def forward(self, x):
        identity = self.shortcut(x)

        if not self.start_block:
            if self.exclude_bn0:
                x = self.act0(x)
            else:
                x = self.norm_act0(x)

        x = self.conv1(x)

        x = self.conv2(x)
        if self.se is not None and not self.se_last:
            x = self.se(x)

        x = self.conv3(x)

        if self.start_block:
            x = self.bn3(x)

        if self.se is not None and self.se_last:
            x = self.se(x)

        if self.eca is not None:
            x = self.eca(x)

        x = x + identity

        if self.end_block:
            x = self.norm_act3(x)
        return x

def SpaceToDepthStem(channels=64, stride=4):
    layers = [
        SpaceToDepthModule(stride),
        Conv2d(3 * stride * stride, channels, 3, stride=1, norm='def', act='def')
    ]
    stem = nn.Sequential(*layers)
    stem.out_channels = channels
    return stem


def ResNetvdStem(channels=64, pool=True, norm_act=True):
    layers = [
        Conv2d(3, channels // 2, kernel_size=3, stride=2,
               norm='def', act='def'),
        Conv2d(channels // 2, channels // 2, kernel_size=3,
               norm='def', act='def'),
        Conv2d(channels // 2, channels, kernel_size=3, norm='def', act='def')
        if norm_act else Conv2d(channels // 2, channels, kernel_size=3),
    ]
    if pool:
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
    stem = nn.Sequential(*layers)
    stem.out_channels = channels
    return stem


class ReResNet(_IResNet):

    def __init__(self, layers, num_classes=1000, channels=(64, 64, 128, 256, 512), pool=False,
                 se_reduction=(0, 0, 0, 0), se_mode=0, se_last=True, eca=False, strides=(2, 2, 2, 2),
                 anti_alias=False, avd=False, light_stem=False):
        stem_channels, *channels = channels
        if light_stem:
            stem = SpaceToDepthStem(stem_channels, stride=4 // strides[0])
        else:
            stem = ResNetvdStem(stem_channels, pool=pool)
        super().__init__(stem, Bottleneck, layers, num_classes, channels,
                         strides=strides, anti_alias=anti_alias, avd=avd,
                         se_mode=se_mode, se_last=se_last, se_reduction=se_reduction, eca=eca)


# 32.3M 5.64G 1015
# def resnet68(**kwargs):
#     return ResNet(Bottleneck, [3, 4, 12, 3], **kwargs)

# 25.6M 4.33G 1262
# def net0(**kwargs):
#     return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

# 25.6M 4.37G 1162
def net1(**kwargs):
    return ReResNet(layers=(3, 4, 6, 3), **kwargs)

# 25.6M 5.41G 1100
def net2(**kwargs):
    return ReResNet(layers=(3, 4, 6, 3), anti_alias=(False, True, True, True), **kwargs)

# 25.6M 5.41G 1099
def net21(**kwargs):
    return ReResNet(layers=(3, 4, 6, 3), avd=(False, True, True, True), **kwargs)

# 25.6M 5.10G 1209
def net3(**kwargs):
    return ReResNet(layers=(3, 4, 6, 3), anti_alias=(False, True, True, True),
                    light_stem=True, strides=(1, 2, 2, 2), **kwargs)

# 30.7M 5.11G 1105
def net4(**kwargs):
    return ReResNet(layers=(3, 4, 6, 3), anti_alias=(False, True, True, True),
                    light_stem=True, strides=(1, 2, 2, 2), se_reduction=(4, 8, 8, 8), **kwargs)

# 30.7M 5.42G 1017
def net41(**kwargs):
    return ReResNet(layers=(3, 4, 6, 3), anti_alias=(False, True, True, True),
                    light_stem=False, strides=(2, 2, 2, 2), se_reduction=(4, 8, 8, 8), **kwargs)

# 30.7M 5.38G 1043
def net42(**kwargs):
    return ReResNet(layers=(3, 4, 6, 3), anti_alias=(False, True, True, True),
                    light_stem=False, pool=True, strides=(1, 2, 2, 2), se_reduction=(4, 8, 8, 8), **kwargs)

# 35.7M 5.42G 1014
def net43(**kwargs):
    return ReResNet(layers=(3, 4, 6, 3), anti_alias=(False, True, True, True),
                    light_stem=False, strides=(2, 2, 2, 2), se_reduction=(4, 4, 4, 4), **kwargs)

# 30.7M 5.15G 1057
def net44(**kwargs):
    return ReResNet(layers=(3, 4, 6, 3), anti_alias=(False, True, True, True),
                    light_stem=True, strides=(2, 2, 2, 2), se_reduction=(4, 8, 8, 8), **kwargs)

# 28.1M 5.42G 1079
def net411(**kwargs):
    return ReResNet(layers=(3, 4, 6, 3), anti_alias=(False, True, True, True),
                    light_stem=False, strides=(2, 2, 2, 2), se_reduction=(4, 4, 4, 4),
                    se_last=False, **kwargs)

# 45.7M 5.43G 1018
def net412(**kwargs):
    return ReResNet(layers=(3, 4, 6, 3), anti_alias=(False, True, True, True),
                    light_stem=False, strides=(2, 2, 2, 2), se_reduction=(4, 8, 8, 8),
                    se_mode=2, **kwargs)

# 25.6M 5.41G 1017
def net413(**kwargs):
    return ReResNet(layers=(3, 4, 6, 3), anti_alias=(False, True, True, True),
                    light_stem=False, strides=(2, 2, 2, 2), eca=True, **kwargs)
