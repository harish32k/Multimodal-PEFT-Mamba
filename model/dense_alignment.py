import torch
import torch.nn as nn
import torch.nn.functional as F
import einops



class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DenseMixtureOfConvolution(nn.Module):
    """
    Dense Mixture of Convolutions (D-MoC) core:
      - 1x1 branch
      - 3x3 branch (dense: sees input + branch1)
      - 5x5 branch (dense: sees input + branch1 + branch2)
      - concat branches + add input residual
    """
    def __init__(self, in_channels):
        super().__init__()
        # split channels so sum equals in_channels
        ch1 = in_channels // 2
        ch2 = in_channels // 4
        ch3 = in_channels - ch1 - ch2
        

        self.branch1 = BasicConv2d(in_channels, ch1, kernel_size=1)
        self.branch2 = BasicConv2d(in_channels + ch1, ch2, kernel_size=3, padding=1)
        self.branch3 = BasicConv2d(in_channels + ch1 + ch2, ch3, kernel_size=5, padding=2)

    def forward(self, x):
        # x: (B, C, H, W)
        b1 = self.branch1(x)
        b2 = self.branch2(torch.cat([x, b1], dim=1))
        b3 = self.branch3(torch.cat([x, b1, b2], dim=1))
        out = torch.cat([b1, b2, b3], dim=1)
        # residual add (channels match in_channels)
        return out + x


'''import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

class DenseMixtureOfConvolution(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1x1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv3x3 = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(dim * 3, dim, kernel_size=5, padding=2)
        
    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(torch.cat([x, x1], dim=1))
        x3 = self.conv5x5(torch.cat([x, x1, x2], dim=1))
        return torch.cat([x1, x2, x3], dim=1) + x

class CrossAligner(nn.Module):
    def __init__(self, dim, text_dim=768, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(text_dim, dim)
        self.v = nn.Linear(text_dim, dim)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x, text_feat):
        B, C, H, W = x.shape
        
        # Reshape x for attention
        x_flat = einops.rearrange(x, 'b c h w -> b (h w) c')
        
        # Multi-head attention
        q = self.q(x_flat).reshape(B, H*W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(text_feat).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(text_feat).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, H*W, C)
        out = self.proj(out)
        
        # Reshape back to 2D
        out = einops.rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)
        return out

class DenseAligner(nn.Module):
    def __init__(self, dim, text_dim=768, reduction_factor=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down_proj = nn.Linear(dim, dim // reduction_factor)
        self.d_moc = DenseMixtureOfConvolution(dim // reduction_factor)
        self.cross_aligner = CrossAligner(dim // reduction_factor, text_dim)
        self.up_proj = nn.Linear(dim // reduction_factor, dim)
        self.act = nn.ReLU()
        
    def forward(self, x, text_feat):
        # Take 2D feature map input
        B, C, H, W = x.shape
        residual = x
        
        # Layer norm expects [B, H, W, C]
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        x = einops.rearrange(x, 'b h w c -> b (h w) c')
        
        # Down projection
        x = self.down_proj(x)
        x = self.act(x)
        
        # Reshape to 2D for convolution operations
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        
        # Dense mixture of convolution
        x_dense = self.d_moc(x)
        
        # Cross attention with text features
        x_cross = self.cross_aligner(x_dense, text_feat)
        x_cross = x_cross + x_dense
        
        # Reshape back for up projection
        x_cross = einops.rearrange(x_cross, 'b c h w -> b (h w) c')
        
        # Up projection
        x_out = self.up_proj(x_cross)
        
        # Reshape back to 2D
        x_out = einops.rearrange(x_out, 'b (h w) c -> b c h w', h=H, w=W)
        
        return x_out + residual'''
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import einops

# class DenseMixtureOfConvolution(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         # Calculate channel divisions for each branch (approximately equal parts)
#         self.c1 = dim // 3
#         self.c2 = dim // 3
#         self.c3 = dim - self.c1 - self.c2  # Ensure we use all channels
        
#         # 1×1 branch - processes the input directly
#         self.conv1x1 = nn.Conv2d(dim, self.c1, kernel_size=1)
        
#         # 3×3 branch - gets input from 1×1 branch
#         self.conv3x3_reduce = nn.Conv2d(self.c1, self.c2, kernel_size=1)
#         self.conv3x3 = nn.Conv2d(self.c2, self.c2, kernel_size=3, padding=1)
        
#         # 5×5 branch - gets input from concatenated 1×1 and 3×3 branches
#         self.conv5x5_reduce = nn.Conv2d(self.c1 + self.c2, self.c3, kernel_size=1)
#         self.conv5x5 = nn.Conv2d(self.c3, self.c3, kernel_size=5, padding=2)
        
#     def forward(self, x):
#         # First branch: 1×1 convolution
#         x1 = self.conv1x1(x)
        
#         # Second branch: 3×3 convolution with input from first branch
#         x2 = self.conv3x3_reduce(x1)
#         x2 = self.conv3x3(x2)
        
#         # Third branch: 5×5 convolution with input from concatenated previous branches
#         x3_input = torch.cat([x1, x2], dim=1)
#         x3 = self.conv5x5_reduce(x3_input)
#         x3 = self.conv5x5(x3)
        
#         # Concatenate all branch outputs
#         out = torch.cat([x1, x2, x3], dim=1)
        
#         # Add residual connection
#         return out + x

# class CrossAligner(nn.Module):
#     def __init__(self, dim, text_dim=768, num_heads=8):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
        
#         # Projections for query, key, value
#         self.q = nn.Linear(dim, dim)
#         self.k = nn.Linear(text_dim, dim)
#         self.v = nn.Linear(text_dim, dim)
#         self.proj = nn.Linear(dim, dim)
        
#     def forward(self, x, text_feat):
#         B, C, H, W = x.shape
        
#         # Handle different text_feat shapes
#         # If text_feat is [B, C, L], transpose it to [B, L, C]
#         #if text_feat.shape[1] == 768 and len(text_feat.shape) == 3:
#         #    text_feat = text_feat.transpose(1, 2)  # Now [B, L, C]
#         text_feat = text_feat.transpose(1, 2)  # Now [B, L, C]
        
#         # Reshape image features for attention
#         x_flat = einops.rearrange(x, 'b c h w -> b (h w) c')
        
#         # Compute query from image features
#         q = self.q(x_flat)  # [B, H*W, dim]
        
#         # Compute key and value from text features
#         k = self.k(text_feat)  # [B, L, dim]
#         v = self.v(text_feat)  # [B, L, dim]
        
#         # Reshape for multi-head attention
#         q = q.view(B, H*W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, H*W, head_dim]
#         k = k.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, L, head_dim]
#         v = v.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, L, head_dim]
        
#         # Attention
#         attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, H*W, L]
#         attn = attn.softmax(dim=-1)
        
#         # Compute output
#         out = (attn @ v).transpose(1, 2).reshape(B, H*W, -1)  # [B, H*W, dim]
#         out = self.proj(out)
        
#         # Reshape back to 2D spatial format
#         out = einops.rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)
        
#         return out

# class DenseAligner(nn.Module):
#     def __init__(self, dim, text_dim=768, reduction_factor=4):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
        
#         # Down projection
#         reduced_dim = dim // reduction_factor
#         self.down_proj = nn.Linear(dim, reduced_dim)
        
#         # Dense Mixture of Convolution
#         self.d_moc = DenseMixtureOfConvolution(reduced_dim)
        
#         # Cross Aligner for text-vision interaction
#         self.cross_aligner = CrossAligner(reduced_dim, text_dim)
        
#         # Up projection
#         self.up_proj = nn.Linear(reduced_dim, dim)
        
#         # Activation function
#         self.act = nn.ReLU()
        
#     def forward(self, x, text_feat):
#         B, C, H, W = x.shape
#         residual = x
        
#         # Debug information
#         # print(f"DenseAligner input shapes: x={x.shape}, text_feat={text_feat.shape}")
        
#         # Layer norm (channel-last format)
#         x = einops.rearrange(x, 'b c h w -> b h w c')
#         x = self.norm(x)
        
#         # Down projection
#         x = einops.rearrange(x, 'b h w c -> b (h w) c')
#         x = self.down_proj(x)
#         x = self.act(x)
        
#         # Reshape to 2D for convolution operations
#         x = einops.rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        
#         # Dense mixture of convolution
#         x_dense = self.d_moc(x)
        
#         # Cross attention with text features
    
#         x_cross = self.cross_aligner(x_dense, text_feat)
#         x_cross = x_cross + x_dense
        
#         # Reshape back for up projection
#         x_cross = einops.rearrange(x_cross, 'b c h w -> b (h w) c')
        
#         # Up projection
#         x_out = self.up_proj(x_cross)
        
#         # Reshape back to 2D and add residual
#         x_out = einops.rearrange(x_out, 'b (h w) c -> b c h w', h=H, w=W)
        
#         return x_out + residual