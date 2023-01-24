from core.modelling import registry
from core.utils.model_zoo import load_state_dict_from_url
from torchvision.models.mobilenet import MobileNetV2

# Returns 4 features from x4, x8, x16, x32 scales
class MNETV2(MobileNetV2):
    def __init__(self):
        super().__init__()

    def export_rebuild(self, target):
        # self.target = target
        # if self.target == 'ti':
        return

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method     [56, 28, 14, 7] [24, 32, 96, 320]
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass [3,6,13, 17]
        x = self.features[0](x)
        x = self.features[1](x)
        x = self.features[2](x)
        out_x4 = self.features[3](x)

        x = self.features[4](out_x4)
        x = self.features[5](x)
        out_x8 = self.features[6](x)

        x = self.features[7](out_x8)
        x = self.features[8](x)
        x = self.features[9](x)
        x = self.features[10](x)
        x = self.features[11](x)
        x = self.features[12](x)
        out_x16 = self.features[13](x)

        x = self.features[14](out_x16)
        x = self.features[15](x)
        x = self.features[16](x)
        out_x32 = self.features[17](x)

        return {'out_x32':out_x32, 'out_x16':out_x16, 'out_x8':out_x8, 'out_x4':out_x4}

    def forward(self, x):
        return self._forward_impl(x)

@registry.BACKBONES.register('MNETV2')
def build_MNETV2(cfg, pretrained=True, freeze=False):
    model = MNETV2()
    model_url = 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'
    if pretrained:
        print("Loading pretrained weights for MNETV2")
        model.load_state_dict(load_state_dict_from_url(model_url))
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    return model