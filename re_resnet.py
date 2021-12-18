from tensorflow.keras.layers import Layer

from hanser.models.modules import DropPath
from hanser.models.layers import Conv2d, Act, Identity, Norm, NormAct
from hanser.models.common.modules import get_shortcut_vd
from hanser.models.attention import SELayer, ECALayer

from hanser.models.imagenet.iresnet.resnet import _IResNet
from hanser.models.imagenet.stem import SpaceToDepthStem, ResNetvdStem


class Bottleneck(Layer):
    expansion = 4

    def __init__(self, in_channels, channels, stride,
                 start_block=False, end_block=False, exclude_bn0=False,
                 se_reduction=4, se_last=False, se_mode=0, eca=False,
                 anti_alias=False, avd=False, drop_path=0):
        super().__init__()
        self.se = None
        self.eca = None
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
                            norm='def', act='def', anti_alias=anti_alias, avd=avd, avd_first=False)

        if se_reduction and not self.se_last:
            self.se = SELayer(channels, se_channels=out_channels // se_reduction, mode=se_mode)

        self.conv3 = Conv2d(channels, out_channels, kernel_size=1)

        if start_block:
            self.bn3 = Norm(out_channels)

        if se_reduction and self.se_last:
            self.se = SELayer(out_channels, se_channels=out_channels // se_reduction, mode=se_mode)

        if eca:
            self.eca = ECALayer(out_channels)

        self.drop_path = DropPath(drop_path) if drop_path else Identity()

        if end_block:
            self.norm_act3 = NormAct(out_channels)

        self.shortcut = get_shortcut_vd(in_channels, out_channels, stride)

        self.start_block = start_block
        self.end_block = end_block
        self.exclude_bn0 = exclude_bn0

    def call(self, x):
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

        x = self.drop_path(x)
        x = x + identity

        if self.end_block:
            x = self.norm_act3(x)
        return x


class ReResNet(_IResNet):

    def __init__(self, layers, num_classes=1000, channels=(64, 64, 128, 256, 512), pool=False,
                 se_reduction=(0, 0, 0, 0), se_mode=0, se_last=True, eca=False, strides=(2, 2, 2, 2),
                 anti_alias=False, avd=False, light_stem=False, dropout=0):
        stem_channels, *channels = channels
        if light_stem:
            stem = SpaceToDepthStem(stem_channels, stride=4 // strides[0])
        else:
            stem = ResNetvdStem(stem_channels, pool=pool)
        super().__init__(stem, Bottleneck, layers, num_classes, channels,
                         strides=strides, anti_alias=anti_alias, avd=avd, dropout=dropout,
                         se_mode=se_mode, se_last=se_last, se_reduction=se_reduction, eca=eca)


def re_resnet_50(**kwargs):
    return ReResNet(layers=(3, 4, 6, 3), anti_alias=(False, True, True, True),
                    light_stem=False, strides=(2, 2, 2, 2), se_reduction=(4, 8, 8, 8), **kwargs)
