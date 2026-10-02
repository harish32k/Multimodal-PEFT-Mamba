import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model

from .base_segmenter import BaseSegmenter
from .base_segmenter_peft import BaseSegmenter_PEFT
from .customized_model_peft import ReMamber_PEFT
from .utils import load_ckpt, update_mamba_config
from .dense_alignment import DenseMixtureOfConvolution


class Conv2d_BN(nn.Module):
    """Convolution with BN module."""
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bn_weight_init=1,
        norm_layer=nn.BatchNorm2d,
        act_layer=None,
    ):
        super().__init__()

        self.conv = torch.nn.Conv2d(in_ch,
                                    out_ch,
                                    kernel_size,
                                    stride,
                                    pad,
                                    dilation,
                                    groups,
                                    bias=False)
        self.bn = norm_layer(out_ch)
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))

        self.act_layer = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        """foward function"""
        x = self.conv(x)
        x = self.bn(x)
        x = self.act_layer(x)

        return x


class ResBlock(nn.Module):
    """Residual block for convolutional local feature."""
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.Hardswish,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = Conv2d_BN(in_features,
                               hidden_features,
                               act_layer=act_layer)
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            3,
            1,
            1,
            bias=False,
            groups=hidden_features,
        )
        self.norm = norm_layer(hidden_features)
        self.act = act_layer()
        self.conv2 = Conv2d_BN(hidden_features, out_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        initialization
        """
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def forward(self, x):
        """foward function"""
        identity = x
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.norm(feat)
        feat = self.act(feat)
        feat = self.conv2(feat)
        return feat
        # return identity + feat


class ConvDecoderAdapter(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_factor=4):
        super().__init__()
        hidden = in_channels // reduction_factor
        self.down = nn.Conv2d(in_channels,  hidden, 1)
        self.d_moc= DenseMixtureOfConvolution(hidden)
        self.up   = nn.Conv2d(hidden, out_channels, 1)
        self.act  = nn.ReLU()

    def forward(self, x):
        x = self.down(x)
        x = self.act(x)
        x = self.d_moc(x)
        x = self.up(x)
        return x

class ConvDecoder_PEFT(nn.Module):
    def __init__(self, in_dim, use_adapters: bool = True, **kwargs):
        super().__init__()
        print(use_adapters)
        self.use_adapters = use_adapters
        hidden = in_dim * 4

        # — base decoder —
        self.mixer3 = ResBlock(in_dim*8,          out_features=hidden)
        self.mixer2 = ResBlock(in_dim*4 + hidden, out_features=hidden)
        self.mixer1 = ResBlock(in_dim*2 + hidden, out_features=hidden)

        self.neck   = nn.Sequential(
            nn.Conv2d(hidden + in_dim, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(hidden, 1,     3, padding=1),
        )

        # — adapters (real or Identity) + per‑adapter alpha —
        if use_adapters:
            self.adapters = nn.ModuleList([
                ConvDecoderAdapter(in_dim*8, hidden),
                ConvDecoderAdapter(hidden + in_dim*4, hidden),
                ConvDecoderAdapter(hidden + in_dim*2, hidden),
            ])
            self.alphas = nn.ParameterList([
                nn.Parameter(torch.tensor(1.0)),
                nn.Parameter(torch.tensor(1.0)),
                nn.Parameter(torch.tensor(1.0)),
            ])
        else:
            self.adapters = None
            self.alphas = None
    def forward(self, feats, *args):
        out0, out1, out2, out3 = feats

        # stage 1
        m3_in = out0
        x     = self.mixer3(m3_in)
        if self.use_adapters:
            x = x + self.alphas[0] * self.adapters[0](m3_in)
        x     = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        # stage 2
        m2_in = torch.cat([x, out1], dim=1)
        x     = self.mixer2(m2_in)
        if self.use_adapters:
            x = x + self.alphas[1] * self.adapters[1](m2_in)
        x     = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        # stage 3
        m1_in = torch.cat([x, out2], dim=1)
        x     = self.mixer1(m1_in)
        if self.use_adapters:
            x = x + self.alphas[2] * self.adapters[2](m1_in)
        x     = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        # neck
        return self.neck(torch.cat([x, out3], dim=1))

    def set_use_adapters(self, flag: bool):
        """Toggle adapter addition on/off."""
        self.use_adapters = flag

    def set_freeze_decoder(self, freeze: bool):
        """
        Freeze or unfreeze all mixer + neck parameters.
        - freeze=True  ⇒ requires_grad=False
        - freeze=False ⇒ requires_grad=True
        """
        for name, p in self.named_parameters():
            if name.startswith("mixer") or name.startswith("neck"):
                p.requires_grad = not freeze

    def unfreeze_adapters(self):
        """Unfreeze all adapter modules & their a's."""
        for m in self.adapters:
            for p in m.parameters():
                p.requires_grad = True
        for a in self.alphas:
            a.requires_grad = True

    def unfreeze_neck(self):
        """Unfreeze only the neck block."""
        for p in self.neck.parameters():
            p.requires_grad = True


class ConvSegmentor_PEFT(BaseSegmenter_PEFT):
    def __init__(self, backbone, img_size=256, patch_size=4, embed_dim=256, **kwargs):
        super().__init__(backbone)
        print("kwargs", kwargs)
        if kwargs["use_adapters"]:
            print("Finetuning half decoder with adapters")
        else:
            print("Training the full decoder")
        res = img_size // patch_size
        self.decoder = ConvDecoder_PEFT(in_dim=embed_dim, resolution=res, use_adapters=kwargs["use_adapters"])


@register_model
def ReMamber_Conv_PEFT(img_size, model_size="base", **kwargs):
    # This function remains unchanged
    config_dict = update_mamba_config(model_size)
    backbone = ReMamber_PEFT(**config_dict)
    backbone, ret = load_ckpt(backbone, model_size)
    model = ConvSegmentor_PEFT(backbone, img_size=img_size, embed_dim=config_dict['dims'][0], patch_size=config_dict['patch_size'], use_adapters=kwargs["use_adapters"])
    return model, ret[0]

# def load_remamber_with_adapters(model_path):
#     # Initialize the new model with adapters
#     model = ReMamber(**config_dict)  # Assuming config_dict is defined
    
#     # Load the original ReMamber weights
#     checkpoint = torch.load(model_path, map_location='cpu')
    
#     # Load the weights into our model, ignoring the newly added adapter parameters
#     missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
    
#     # Verify that the missing keys are only our adapter parameters
#     adapter_params = [k for k in missing_keys if 'dense_aligners' in k]
    
#     # Log the missing and unexpected keys
#     print(f"Missing keys (should be adapter params): {missing_keys}")
#     print(f"Unexpected keys: {unexpected_keys}")
    
#     # Initialize only the adapter parameters with appropriate initialization
#     for m in model.modules():
#         if isinstance(m, DenseAligner) or isinstance(m, DenseMixtureOfConvolution) or isinstance(m, CrossAligner):
#             for n, p in m.named_parameters():
#                 if "weight" in n:
#                     nn.init.kaiming_normal_(p, mode='fan_out', nonlinearity='relu')
#                 elif "bias" in n:
#                     nn.init.zeros_(p)
    
#     return model