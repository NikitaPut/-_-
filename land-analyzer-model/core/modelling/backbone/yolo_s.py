# This file contains modules common to various models
import os.path
import math

import urllib.request

import torch
import torch.nn as nn

from core.modelling import registry
from core.utils.model_zoo import load_state_dict_from_url

def silu(input):
    return input * torch.sigmoid(input)

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return silu(input)


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
        self.sp = nn.Softplus()
        self.tanh = nn.Tanh()
    def forward(self, x):
        return x * self.tanh(self.sp(x))


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True) # TODO: was False
        self.bn = nn.BatchNorm2d(c2)
        # self.act = nn.Hardtanh(0., 6.) if act else nn.Identity()
        self.act = Mish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.Hardtanh(0., 6., inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        self.conv2 = Conv(c1, c1 * 4, 3, 2, 1, c1, act)  # ch_in, ch_out, kernel, stride, padding, groups

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        k = self.conv2(x)
        k = self.conv(k)
        return k #self.conv(self.conv2(x))


class YOLOs(nn.Module):
    def __init__(self):
        super(YOLOs, self).__init__()
        self.target = None
        self.module_list = nn.ModuleList()
        self.module_list.append(Focus(3, 32))
        self.module_list.append(Conv(32, 64, 3, 2)) # 1-P2/4
        self.module_list.append(C3(64, 64, n=2))
        self.module_list.append(Conv(64, 128, 3, 2)) # 1-P3/8
        self.module_list.append(C3(128, 128, n=5))
        self.module_list.append(Conv(128, 256, 3, 2)) # 5-P4/16
        self.module_list.append(C3(256, 256, n=5))
        self.module_list.append(Conv(256, 384, 3, 1))  # 5-P5/32


    def export_rebuild(self, target):
        self.target = target
        return

    def forward(self, x):
        x = self.module_list[0](x)
        x = self.module_list[1](x)
        out_x4 = self.module_list[2](x)

        x = self.module_list[3](out_x4)
        out_x8 = self.module_list[4](x)

        x = self.module_list[5](out_x8)
        out_x16 = self.module_list[6](x)

        out_x32 = self.module_list[7](out_x16)

        # x = self.module_list[0](x)
        # f1 = self.module_list[1](x)
        # x = self.module_list[2](f1)
        # f2 = self.module_list[3](x)
        # x = self.module_list[4](f2)
        # f3 = self.module_list[5](x)
        # x = self.module_list[6](f3)
        # f4 = self.module_list[7](x)

        # return {'out4': f4, 'out3': f3,'out2': f2,'out1': f1}
        return {'out_x32':out_x32, 'out_x16':out_x16, 'out_x8':out_x8, 'out_x4':out_x4}

@registry.BACKBONES.register('YOLOs')
def build_YOLOs(cfg, pretrained=True, freeze = False):
    model = YOLOs()

    # model_url = '/media/nikolai/Data/DevProjects/ce_yolo/ce_yolo/yolov5/runs/exp17/weights/best.pt'
    model_url = 'https://drive.google.com/uc?export=download&id=1xbs39pjBALjtCKykqoeoWs6ZuOCiXCqh'
    checkpoint_filename = 'yolo.pt'
    if pretrained:
        # print("Loading pretrained weights for YOLOs")
        if not os.path.isfile(checkpoint_filename):
            print('Loading pretrained YOLOs weights...')
            urllib.request.urlretrieve(model_url, checkpoint_filename)
            print('Loaded!')
        state_dict = torch.load(checkpoint_filename, map_location=torch.device(cfg.MODEL.DEVICE))

        # # state_dict = load_state_dict_from_url("https://drive.google.com/uc?export=download&id=17d7SoyF5e4sRfCxrSl4-8au0GfRJ0pc5")
        # # load_state_dict_from_url(model_url)
        # state_dict = { k.replace('model', 'module_list'): v for k, v in state_dict.items()}
        # new_dict = {}
        # for k, v in state_dict.items():
        #     if int(k.split('.')[1]) < 7:
        #         new_dict[k] = v
        # state_dict = new_dict
        # torch.save(state_dict, 'yolo_dict.pt')

        model.load_state_dict(state_dict, strict=False)
        # input(model.module_list[6].state_dict())

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    return model