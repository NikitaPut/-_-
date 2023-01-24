import torch
from torch import nn
from core.modelling import registry


def calc_padding_size(kernel_size:int, padding=None):
    if padding is None:
        padding = kernel_size // 2 if isinstance(kernel_size, int) else [s // 2 for s in kernel_size]
    return padding


class ConvolutionBlock(nn.Module):
    def __init__(self, ch_in:int, ch_out:int, kernel:int = 1, stride:int = 1, groups:int = 1, padding = None, use_bias = True):
        super(ConvolutionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=ch_in,
                              out_channels=ch_out,
                              kernel_size=kernel,
                              stride=stride,
                              padding=calc_padding_size(kernel, padding),
                              groups=groups,
                              bias=use_bias)
        self.bn = nn.BatchNorm2d(ch_out)
        self.act = nn.ReLU() # nn.Mish()
        # self.act = nn.Hardtanh(0., 6.)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class BottleneckBlock(nn.Module):
    def __init__(self, ch_in:int, ch_out:int, expansion:float = 0.5):
        super(BottleneckBlock, self).__init__()
        self.ch_hidden = int(ch_in * expansion) # TODO: or ch_out ?
        self.conv1 = ConvolutionBlock(ch_in, self.ch_hidden, kernel=1)
        self.conv2 = ConvolutionBlock(self.ch_hidden, ch_out, kernel=3)

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return x + out


class CspBlock(nn.Module):
    def __init__(self, ch_in:int, ch_out:int, bnecks:int, expansion:float = 0.5):
        super(CspBlock, self).__init__()
        self.ch_hidden = int(ch_in * expansion) # TODO: or ch_out ?
        self.conv1 = ConvolutionBlock(ch_in, self.ch_hidden, kernel=1)
        self.conv1_2 = ConvolutionBlock(self.ch_hidden, self.ch_hidden, kernel=1)
        self.conv2 = ConvolutionBlock(ch_in, self.ch_hidden, kernel=1)
        self.bneck = nn.Sequential(*[BottleneckBlock(self.ch_hidden, self.ch_hidden, 1.0) for _ in range(bnecks)])
        self.conv3 = ConvolutionBlock(2*self.ch_hidden, ch_out, 1)
        
    def forward(self, x):
        x1 = self.conv1_2(self.bneck(self.conv1(x)))
        # x1 = self.bneck(self.conv1(x))
        x2 = self.conv2(x)
        return self.conv3(torch.cat((x1, x2), dim=1))
        

class CSP(nn.Module):
    def __init__(self, img_size):
        super(CSP, self).__init__()
        self.target = None
        self.img_size = img_size
        self.features_x2 = nn.Sequential(
            ConvolutionBlock(3, 32, kernel=3),
            ConvolutionBlock(32, 64, kernel=3, stride=2), # x2
            # BottleneckBlock(64, 64, 0.5)
        )
        self.features_x4 = nn.Sequential(
            ConvolutionBlock(64, 128, kernel=3, stride=2), #x4
            CspBlock(128, 128, 2),
        )
        self.features_x8 = nn.Sequential(
            ConvolutionBlock(128, 256, kernel=3, stride=2), #x8
            CspBlock(256, 256, 2),
        )
        self.features_x16 = nn.Sequential(
            ConvolutionBlock(256, 512, kernel=3, stride=2), #x16
            CspBlock(512, 512, 4),
        )
        self.features_x32 = nn.Sequential(
            ConvolutionBlock(512, 1024, kernel=3, stride=2), #x32
            CspBlock(1024, 1024, 4),
        )

    def export_rebuild(self, target):
        self.target = target
        return

    def forward(self, x):
        out_x2 = self.features_x2(x)
        out_x4 = self.features_x4(out_x2)
        out_x8 = self.features_x8(out_x4)
        out_x16 = self.features_x16(out_x8)
        out_x32 = self.features_x32(out_x16)
        return {'out_x32':out_x32, 'out_x16':out_x16, 'out_x8':out_x8, 'out_x4':out_x4}


@registry.BACKBONES.register('CSP')
def build_CSP(cfg, pretrained=True, freeze=False):
    model = CSP(cfg.INPUT.IMAGE_SIZE)
    if pretrained:
        print("No pretrained weights are available for CSP")
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    return model