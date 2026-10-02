import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

import einops

from vmamba_model.vmamba import SS2D, VSSM, LayerNorm2d, Linear2d

from vmamba_model.vmamba import VSSBlock as OriginalVSSBlock
from .utils import ImageTextCorr
from .dense_alignment import DenseMixtureOfConvolution

class Twister(nn.Module):
    def __init__(
            self,         
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3, # < 2 means no conv 
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            channel_first=False,
            # ======================
            input_res=32,
            **kwargs,
        ):
        super().__init__()
        self.input_res = input_res
        self.ss2d = SS2D(d_model, d_state, ssm_ratio, dt_rank, act_layer, d_conv, conv_bias, dropout, bias, dt_min, dt_max, dt_init, dt_scale, dt_init_floor, initialize, forward_type, channel_first, **kwargs)
        self.ss2d.in_proj = Linear2d(d_model * 3, self.ss2d.in_proj.weight.shape[0], bias=bias)
        self.input_res = min(input_res, 16)
        forward_type_1d = "v052d"
        self.ss1d = SS2D(self.input_res**2, d_state, ssm_ratio, dt_rank, act_layer, d_conv, conv_bias, dropout, bias, dt_min, dt_max, dt_init, dt_scale, dt_init_floor, initialize, forward_type_1d, channel_first, **kwargs)
    
    def forward(self, x: torch.Tensor, **kwargs):
        img, global_cond, local_cond = x
        B, C, H, W = img.shape
        if global_cond is not None:
            x = torch.cat([img, global_cond, local_cond], dim=1) # b l 3c
        else:
            x = torch.cat([img, local_cond], dim=-1)
        x_prepaired = x
        x_mix = F.interpolate(x, size=(self.input_res,self.input_res), mode='bilinear')
        x_mix = x_mix.view(B, -1, self.input_res**2)
        x_mix = x_mix.permute(0, 2, 1).unsqueeze(-2).contiguous() # b, hw, 1, c
        x_mix = self.ss1d(x_mix)
        x_mix = x_mix.squeeze(-2).permute(0, 2, 1).contiguous() # b, c, hw
        x_mix = F.interpolate(x_mix.view(B,-1,self.input_res,self.input_res), size=(H,W), mode='bilinear')

        x = x_mix + x_prepaired
        
        out = self.ss2d(x)

        out = [out, global_cond, local_cond]
        return out

class VSSBlock(nn.Module):
    def __init__(
        self,
        forward_coremm='SS2D',
        **kwargs,
    ):
        super().__init__()
        norm_layer = kwargs['norm_layer']
        dim = kwargs['dim']
        drop_path = kwargs['drop_path']
        self.ln_1 = norm_layer(dim)
        self.forward_coremm = forward_coremm
        if not forward_coremm:
            raise
        elif forward_coremm == 'SS2D':
            self.self_attention = SS2D(
                d_model=dim,
                d_state=kwargs['ssm_d_state'],
                dt_rank=kwargs['ssm_dt_rank'],
                act_layer=kwargs['ssm_act_layer'],
                d_conv=kwargs['ssm_conv'],
                conv_bias=kwargs['ssm_conv_bias'],
                dropout=kwargs['ssm_drop_rate'],
                initialize=kwargs['ssm_init'],
                **kwargs,
            )
        elif forward_coremm == 'Twister':
            self.self_attention = Twister(
                d_model=dim,
                d_state=kwargs['ssm_d_state'],
                ssm_ratio=kwargs['ssm_ratio'],
                dt_rank=kwargs['ssm_dt_rank'],
                act_layer=kwargs['ssm_act_layer'],
                d_conv=kwargs['ssm_conv'],
                conv_bias=kwargs['ssm_conv_bias'],
                dropout=kwargs['ssm_drop_rate'],
                initialize=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'],
                channel_first=kwargs['channel_first'],
            )
        else:
            raise
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        if isinstance(input, torch.Tensor):
            out = self.ln_1(input)
            out = self.self_attention(out)
            out = input + self.drop_path(out)
            x = out
        else:
            # input should be a list (img and global / local conditions)
            out = [self.ln_1(i) if i is not None else None for i in input]
            out = self.self_attention(out)
            out = [i + self.drop_path(o) if i is not None else None for i, o in zip(input, out)]
            x = out
        return x


class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        depth,
        downsample=None,
        forward_coremm="Twister",
        **kwargs,
    ):
        super().__init__()
        # dim = kwargs['dim']
        drop_path = 0.
        dim = kwargs['dim']
        use_checkpoint = kwargs['use_checkpoint']
        norm_layer = kwargs['norm_layer']


        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                forward_coremm=forward_coremm,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                **kwargs,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, channel_first=kwargs['channel_first'])
        else:
            self.downsample = None

    def forward(self, x, l_feat, l_mask):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        # x: b w h c
        inner = x
        if self.downsample is not None:
            x = self.downsample(x)

        return x, inner




class BasicConvAdapter(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden_dim = dim // reduction
        self.down = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.ReLU()
        self.up = nn.Conv2d(hidden_dim, dim, kernel_size=1)

    def forward(self, x):
        return self.up(self.act(self.down(x)))

class CrossTwister(nn.Module):
    """
    Mamba Twister fusion that projects both image and text into a shared embed_dim,
    then applies global token + local fusion inside a VSSLayer Twister block.
    """
    def __init__(self, img_dim, txt_dim, embed_dim, cfg):
        super().__init__()
        # store config
        self.cfg = cfg
        # 1) project image channels -> embed_dim via 1x1 conv
        # image is already projected outside, img_dim = embed_dim
        self.img_proj = nn.Identity() #nn.Conv2d(img_dim, embed_dim, kernel_size=1)
        # 2) project text embeddings -> embed_dim via linear
        self.txt_proj = nn.Linear(txt_dim, embed_dim)
        # ReMamber norm & act lookups
        _NORMLAYERS = dict(ln=nn.LayerNorm, ln2d=LayerNorm2d, bn=nn.BatchNorm2d)
        _ACTLAYERS  = dict(silu=nn.SiLU, gelu=nn.GELU, relu=nn.ReLU, sigmoid=nn.Sigmoid)
        norm_layer    = _NORMLAYERS[cfg['norm_layer'].lower()]
        ssm_act_layer = _ACTLAYERS[cfg['ssm_act_layer'].lower()]
        # 3) one-layer VSS Twister at embed_dim
        self.vss = VSSLayer(
            dim             = embed_dim,
            depth           = 1,
            use_checkpoint  = cfg['use_checkpoint'],
            norm_layer      = norm_layer,
            ssm_act_layer   = ssm_act_layer,
            downsample      = None,
            channel_first   = cfg['channel_first'],
            # SSM params from config
            ssm_d_state     = cfg['ssm_d_state'],
            ssm_ratio       = 1, #cfg['ssm_ratio'],
            ssm_dt_rank     = 1, #cfg['ssm_dt_rank'],
            ssm_conv        = cfg['ssm_conv'],
            ssm_conv_bias   = cfg['ssm_conv_bias'],
            ssm_drop_rate   = cfg['ssm_drop_rate'],
            ssm_init        = cfg['ssm_init'],
            forward_type    = cfg['forward_type'],
            # dummy MLP args
            mlp_ratio       = cfg['mlp_ratio'],
            mlp_act_layer   = cfg['mlp_act_layer'],
            mlp_drop_rate   = cfg['mlp_drop_rate'],
            gmlp            = cfg['gmlp'],
            forward_coremm  = 'Twister',
        )
        # global guidance MLP mapped from embed_dim (post txt_proj)
        #self.global_mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())
        # using projected text output as input for global mlp
        self.global_mlp = nn.Sequential(nn.Identity(), nn.ReLU())
        # local fusion expects embed_dim for text and visual dims
        hidden_dim = 128
        self.local_fuser = ImageTextCorr(
            visual_dim = embed_dim,
            text_dim   = embed_dim,
            hidden_dim = hidden_dim,
            out_dim    = embed_dim,
        )

    def forward(self, img, l_feat, l_mask, pooler_out=None):
        """
        img:       [B, img_dim, H, W]
        l_feat:    [B, txt_dim, L] or [B, L, txt_dim]
        l_mask:    [B, L]
        pooler_out:[B, txt_dim] opt override
        """
        B, _, H, W = img.shape
        # 1) project image -> embed_dim
        img_e = self.img_proj(img)  # [B, embed_dim, H, W]
        # 2) ensure l_feat in [B, L, txt_dim]
        if l_feat.dim()==3 and l_feat.shape[1]!=l_feat.shape[-1]:
            l_feat = l_feat.transpose(1,2)
        # 3) project each text token -> embed_dim
        txt_seq = self.txt_proj(l_feat)  # [B, L, embed_dim]
        # override pooling via projected pooler_out if given
        if pooler_out is not None:
            pool = self.txt_proj(pooler_out)  # [B, embed_dim]
        else:
            pool = txt_seq[:,0]               # token0
        # 4) global cond -> broadcast to spatial
        g = self.global_mlp(pool)           # [B, embed_dim]
        global_cond = g.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        # 5) local fusion map from embedded img & txt
        local = self.local_fuser(img_e, txt_seq.transpose(1,2), l_mask)
        local_cond = einops.rearrange(local, 'b h w c -> b c h w', h=H)
        # 6) Twister on (img_e, global_cond, local_cond)
        fused, _ = self.vss((img_e, global_cond, local_cond), txt_seq, l_mask)
        # fused[0] has shape [B, embed_dim, H, W]
        return fused[0]


# ============== DenseAligner (uses CrossTwister) ================
class DenseAligner(nn.Module):
    """
    Adapter that downsamples image to a reduced dim, applies D-MoC,
    fuses via CrossTwister, then upsamples back to original dim.
    """
    def __init__(self, dim, text_dim=768, reduction_factor=8, cfg=None):
        super().__init__()
        self.cfg = cfg
        # 1) image normalization + down-projection
        self.norm = nn.LayerNorm(dim)
        reduced = dim // reduction_factor
        self.img_down = nn.Linear(dim, reduced)
        self.act = nn.ReLU()    
        # 2) dense mixture of convolution at reduced dim
        self.d_moc = DenseMixtureOfConvolution(reduced)
        # 3) cross-modal Twister fusion at reduced dim
        self.cross_twister = CrossTwister(
            img_dim   = reduced,
            txt_dim   = text_dim,
            embed_dim = reduced,
            cfg       = cfg,
        )
        # 4) up-projection back to original dim
        self.img_up = nn.Linear(reduced, dim)
        self.alpha = nn.Parameter(torch.tensor(1.0))  # Initialize with 1.0 (no effect initially)


    def forward(self, x, l_feat, l_mask, pooler_out=None):
        B, C, H, W = x.shape
        # 1) Norm and down-project image
        xi = einops.rearrange(x, 'b c h w -> b h w c')
        xi = self.norm(xi)
        xi = einops.rearrange(xi, 'b h w c -> b (h w) c')
        xi = self.img_down(xi)
        xi = self.act(xi)
        xi = einops.rearrange(xi, 'b (h w) c -> b c h w', h=H, w=W)
        # 2) Dense mixture of convolutions
        xdm = self.d_moc(xi)
        # 3) Cross-modal fusion via Twister
        xf = self.cross_twister(xdm, l_feat, l_mask, pooler_out)
        # 4) Residual at reduced scale
        xf = xf + xi
        # 5) Up-project back and final residual
        xu = einops.rearrange(xf, 'b c h w -> b (h w) c')
        xu = self.img_up(xu)
        xu = einops.rearrange(xu, 'b (h w) c -> b c h w', h=H, w=W)
        return x + self.alpha * xu



class ImageAdapter(nn.Module):
    """
    Lightweight image adapter:
    - Normalizes and downsamples the image.
    - Applies a Dense Mixture of Convolutions.
    - Residual connection at reduced scale.
    - Upsamples back and scales with a learnable alpha.
    """
    def __init__(self, dim, reduction_factor=4):
        super().__init__()
        reduced = dim // reduction_factor

        self.norm = nn.LayerNorm(dim)
        self.img_down = nn.Linear(dim, reduced)
        self.act = nn.ReLU()

        self.d_moc = DenseMixtureOfConvolution(reduced)

        self.img_up = nn.Linear(reduced, dim)

        # Learnable residual scale
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, l_feat=None, l_mask=None, pooler_out=None):
        """
        Args:
            x: [B, C, H, W] input image features
            (l_feat, l_mask, pooler_out): unused in this variant
        Returns:
            [B, C, H, W] scaled residual output
        """
        B, C, H, W = x.shape

        # 1. Normalize and down-project
        xi = einops.rearrange(x, 'b c h w -> b h w c')
        xi = self.norm(xi)
        xi = einops.rearrange(xi, 'b h w c -> b (h w) c')
        xi = self.img_down(xi)  # [B, H*W, reduced]
        xinit = xi
        xi = self.act(xi)
        xi = einops.rearrange(xi, 'b (h w) c -> b c h w', h=H, w=W)

        # 2. Apply DenseMixtureOfConvolution
        xdm = self.d_moc(xi)  # [B, reduced, H, W]

        # 3. Residual at reduced scale
        xf = xdm + einops.rearrange(xinit, 'b (h w) c -> b c h w', h=H, w=W)

        # 4. Up-project to original dim
        xu = einops.rearrange(xf, 'b c h w -> b (h w) c')
        xu = self.img_up(xu)
        xu = einops.rearrange(xu, 'b (h w) c -> b c h w', h=H, w=W)

        # 5. Return scaled residual output
        return self.alpha * xu



class ReMamber_PEFT(VSSM):
    """ReMamber with PEFT-style adapters over VSSBlocks and full multimodal pipeline"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classifier = None
        self.VSSBlockType = OriginalVSSBlock

        dims = kwargs['dims']
        use_checkpoint = kwargs['use_checkpoint']
        norm_layer_key = kwargs['norm_layer']

        self.block_adapters = nn.ModuleList()
        self.text_guidencee = nn.ModuleList()
        self.local_text_fusion = nn.ModuleList()
        self.image_adapters = nn.ModuleList()
        self.multimodal_blocks = nn.ModuleList()

        # Normalization & activation layer maps
        _NORMLAYERS = dict(ln=nn.LayerNorm, ln2d=LayerNorm2d, bn=nn.BatchNorm2d)
        norm_layer = _NORMLAYERS[norm_layer_key.lower()]
        _ACTLAYERS = dict(silu=nn.SiLU, gelu=nn.GELU, relu=nn.ReLU, sigmoid=nn.Sigmoid)
        ssm_act_layer = _ACTLAYERS[kwargs['ssm_act_layer'].lower()]

        for i_layer in range(self.num_layers):
            # Build multimodal VSS layer (your Twister-augmented version)
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=2,
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                ssm_act_layer=ssm_act_layer,
                downsample=None,
                channel_first=self.channel_first,
                # ===== SSM params
                ssm_d_state=kwargs['ssm_d_state'],
                ssm_ratio=kwargs['ssm_ratio'],
                ssm_dt_rank=kwargs['ssm_dt_rank'],
                ssm_conv=kwargs['ssm_conv'],
                ssm_conv_bias=kwargs['ssm_conv_bias'],
                ssm_drop_rate=kwargs['ssm_drop_rate'],
                ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'],
                # ===== MLP
                mlp_ratio=kwargs['mlp_ratio'],
                mlp_act_layer=kwargs['mlp_act_layer'],
                mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'],
                forward_coremm='Twister',
            )
            self.multimodal_blocks.append(layer)

            # Adapter list over each VSSBlock inside this layer
            adapter_list = nn.ModuleList()
            vss_layer = self.layers[i_layer]
            
            for j, block in enumerate(vss_layer.blocks):
                #print("Creating block:", j)
                if isinstance(block, self.VSSBlockType) and j % 2 == 0:  # Apply to every other block
                    dim = block.norm.normalized_shape[0]
                    #print("dim:",dim)
                    adapter = DenseAligner(dim=dim, cfg={**kwargs, "channel_first": self.channel_first})
                else:
                    adapter = nn.Identity()  # Or use a no-op wrapper if needed
                adapter_list.append(adapter)

            self.block_adapters.append(adapter_list)

            # Text -> channel-wise guidence MLP
            self.text_guidencee.append(
                nn.Sequential(nn.Linear(768, dims[i_layer]), nn.ReLU())
            )

            # Spatial multimodal fuser
            self.local_text_fusion.append(
                ImageTextCorr(
                    visual_dim=dims[i_layer],
                    text_dim=768,
                    hidden_dim=512,
                    out_dim=dims[i_layer],
                )
            )

            # Final adapter over fused image feature
            self.image_adapters.append(ImageAdapter(dims[i_layer]))

    def forward_layer(self, x, layer, adapters, l_feat, l_mask, pooler_out):
        blocks = layer[0]
        for block, adapter in zip(blocks, adapters):
            x_in = x
            block_out = block(x_in)
            #x = adapter(block_out, l_feat, l_mask, pooler_out)  # Adapter adds residual inside
            if isinstance(adapter, nn.Identity):
                adapter_feats = x_in  # no-op
            else:
                adapter_feats = adapter(x_in, l_feat, l_mask, pooler_out)

            x = block_out + (adapter_feats - x_in)  

        inner = x  # Before downsampling
        if len(layer) > 1:
            x = layer[1](x)  # Downsample
        return x, inner

    def forward(self, x, l_feat, l_mask, pooler_out=None):
        x = self.patch_embed(x)
        outs = []

        for i, layer in enumerate(self.layers):
            x, inner = self.forward_layer(x, layer, self.block_adapters[i], l_feat, l_mask, pooler_out)
            _, c, h, w = inner.shape
            out = inner

            pooling_text = l_feat[..., 0] if pooler_out is None else pooler_out
            text_guidence = self.text_guidencee[i](pooling_text)
            text_guidence = einops.repeat(text_guidence, "b c -> b c h w", h=h, w=w)

            local_text = self.local_text_fusion[i](out, l_feat, l_mask)
            local_text = einops.rearrange(local_text, 'b h w c -> b c h w', h=h)

            mm_input = (out, text_guidence, local_text)
            ret = self.multimodal_blocks[i](mm_input, None, None)[0]
            img_feat = ret[0]

            adapter_out = self.image_adapters[i](out)
            img_feat = img_feat + adapter_out

            if len(layer) > 1:
                x = layer[1](img_feat)  # Downsample final fused feature
            else:
                x = img_feat

            outs.append(img_feat)

        return outs
