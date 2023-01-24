import torch
import torch.nn.functional as F
from collections import OrderedDict
from torch import nn, flatten
from torch.nn import Linear, Sequential, Module
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3
from torch.hub import load_state_dict_from_url
from core.modelling import registry


# simply define a silu function
def silu(input):
    return input * torch.sigmoid(input)

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return silu(input)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels, activation):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation())

    def forward(self, x):
        for mod in self:
            x = mod(x)
        return x

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, activation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            activation()
        ]
        super(ASPPConv, self).__init__(*modules)

class DecodeBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, activation):
        modules = [
            # ASPPConv(in_channels, out_channels, 1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation()

        ]
        super(DecodeBlock, self).__init__(*modules)

#takes x4, x8, x16, x32 scaled features as input
class DeepLabV3_Custom(nn.Module):
    def __init__(self, depth_list=[], hidden_depth=64, num_classes=1, image_size=224, dropout_strength=0.2, activation=nn.ReLU):
        super(DeepLabV3_Custom, self).__init__()
        self.target = None

        self.img_size = image_size
        self.depth_list = depth_list
        self.hidden_depth = hidden_depth
        self.num_classes = num_classes

        self.decode7 = ASPPPooling(self.depth_list[3], self.hidden_depth, activation)
        self.decode6 = ASPPConv(self.depth_list[3], self.hidden_depth, 3, activation)
        self.decode5 = ASPPConv(self.depth_list[3], self.hidden_depth, 1, activation)

        self.decode4 = DecodeBlock(self.depth_list[3], self.hidden_depth, activation)
        self.decode3 = DecodeBlock(self.depth_list[2], self.hidden_depth, activation)
        self.decode2 = DecodeBlock(self.depth_list[1], self.hidden_depth, activation)
        self.decode1 = DecodeBlock(self.depth_list[0], self.hidden_depth, activation)

        self.project_top = nn.Sequential(
            nn.Conv2d(4 * self.hidden_depth, self.hidden_depth, 1, bias=False),
            nn.BatchNorm2d(self.hidden_depth),
            activation())

        self.project_1 = nn.Sequential(
            nn.Conv2d(2 * self.hidden_depth, self.hidden_depth, 1, bias=False),
            nn.BatchNorm2d(self.hidden_depth),
            activation())

        self.project_2 = nn.Sequential(
            nn.Conv2d(2 * self.hidden_depth, self.hidden_depth, 1, bias=False),
            nn.BatchNorm2d(self.hidden_depth),
            activation())

        self.conv = nn.Sequential(
            nn.Conv2d(self.hidden_depth, self.hidden_depth, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_depth),
            activation())

        self.out_conv = nn.Sequential(
            nn.Conv2d(self.hidden_depth, self.num_classes, 1))

        self.gather = nn.Conv2d(self.hidden_depth, 1, 1)
        torch.nn.init.constant_(self.gather.bias, 0.0)

        self.drop1 = nn.Dropout2d(p=dropout_strength)
        self.drop2 = nn.Dropout2d(p=dropout_strength)
        self.drop3 = nn.Dropout2d(p=dropout_strength)
        self.drop4 = nn.Dropout2d(p=dropout_strength)

    def resize_op(self, x, target_size, mode):
        # input(self.export)
        if self.target == None:
            return F.interpolate(x, size=target_size, mode=mode)
        elif self.target == 'tensorrt':
            return F.interpolate(x, size=target_size, mode=mode)
        elif self.target == 'ti':
            # TODO: check it !
            # scale = target_size // x.shape[-1]
            # return F.interpolate(x, scale_factor=scale, mode=mode, recompute_scale_factor=True)
            return F.interpolate(x, size=target_size, mode=mode)
        else:
            print("Unknown target mode: {}".format(self.target))
            exit(-1)

    def add_scalar_op(self):
        return

    def export_rebuild(self, target):
        self.target = target
        return

    def _forward_impl(self, features):
        #calculate SPP for the top level features
        x = self.decode7(features["out_x32"])
        f7 = self.resize_op(x, target_size=(self.img_size[1] // 16, self.img_size[0] // 16), mode='bilinear')

        x = self.decode6(features["out_x32"])
        f6=x

        x = self.decode5(features["out_x32"])
        f5=x

        #decode features from each level, concat and project
        f4 = self.decode4(features["out_x32"])

        #project features from top
        top_features = torch.cat([f4, f5, f6, f7], dim=1)
        # top_features = self.drop1(top_features)
        top_proj = self.project_top(top_features)

        x = self.decode3(features["out_x16"])

        # x = self.drop2(x)
        curr_features = x + top_proj
        top_proj = self.resize_op(curr_features, target_size=(self.img_size[1] // 8, self.img_size[0] // 8), mode='bilinear')

        x = self.decode2(features["out_x8"])
        # x = self.drop3(x)
        curr_features = x + top_proj
        top_proj = self.resize_op(curr_features, target_size=(self.img_size[1] // 4, self.img_size[0] // 4), mode='bilinear')

        #x4
        x = self.decode1(features["out_x4"])
        # x = self.drop4(x)
        features = x + top_proj

        # Output
        x = self.conv(features)
        x = self.resize_op(x, target_size=(self.img_size[1], self.img_size[0]), mode='bilinear')
        outputs = self.out_conv(x)

        return outputs


    def forward(self, x):
        return self._forward_impl(x)


@registry.HEADS.register('DeepLabV3_Custom')
def build_DeepLabV3_Custom(cfg, pretrained=True, freeze = False):
    activation = nn.ReLU

    num_classes = len(cfg.MODEL.HEAD.CLASS_LABELS)

    model = DeepLabV3_Custom(cfg.MODEL.HEAD.INPUT_DEPTH, 
                             cfg.MODEL.HEAD.HIDDEN_DEPTH, 
                             num_classes,
                             cfg.INPUT.IMAGE_SIZE,
                             cfg.MODEL.HEAD.DROPOUT,
                             activation)
    # if pretrained:
        # print("No pretrained weights available for DeepLabV3_Custom")
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    return model