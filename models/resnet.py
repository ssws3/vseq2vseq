import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

class Conditioner(nn.Module):
    def __init__(self, dim, dim_out, use_conditioning_norm, kernel_size, **kwargs):
        super().__init__()

        self.spatial_conv = nn.Conv2d(dim, dim_out, kernel_size, **kwargs)
        self.conditioning_conv = nn.Conv2d(dim, dim_out, kernel_size, **kwargs)

        self.conditioning_block = ConditioningBlock(dim_out) if use_conditioning_norm else None
        
    def forward(self, hidden_states, conditioning_hidden_states, num_frames):        
        hidden_states = self.spatial_conv(hidden_states)

        conditioning_hidden_states = self.conditioning_conv(conditioning_hidden_states)

        if self.conditioning_block: 
            hidden_states = self.conditioning_block(hidden_states, conditioning_hidden_states, num_frames) if num_frames > 1 else hidden_states

        return hidden_states, conditioning_hidden_states
    
class ConditionalNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.gamma = nn.Sequential(
            nn.Linear(dim, dim), 
            nn.SiLU()
        )
        self.beta = nn.Sequential(
            nn.Linear(dim, dim), 
            nn.SiLU()
        )

        # Zero initialization
        nn.init.zeros_(self.gamma[0].weight)
        nn.init.ones_(self.gamma[0].bias)
        
        nn.init.zeros_(self.beta[0].weight)
        nn.init.zeros_(self.beta[0].bias)

    def forward(self, x, y):
        batch, channel, frames, height, width = y.shape

        y = rearrange(y, "b c f h w -> (b f) (h w) c")

        gamma = self.gamma(y)
        beta = self.beta(y)

        gamma = rearrange(gamma, "(b f) (h w) c -> b c f h w", b=batch, c=channel, f=frames, h=height, w=width)
        beta = rearrange(beta, "(b f) (h w) c -> b c f h w", b=batch, c=channel, f=frames, h=height, w=width)

        return gamma * x + beta
    
class ConditioningBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conditioning_norm = ConditionalNorm(dim)

    def forward(self, hidden_states, conditioning_hidden_states, num_frames=1):
        batch_frames_h, _, _, _ = hidden_states.shape
        batch_frames_c, _, _, _ = conditioning_hidden_states.shape

        batch_h = batch_frames_h // num_frames
        hidden_states = rearrange(hidden_states, '(b f) c h w -> b c f h w', b=batch_h, f=num_frames)

        num_frames_c = batch_frames_c // batch_h
        conditioning_hidden_states = rearrange(conditioning_hidden_states, '(b f) c h w -> b c f h w', b=batch_h, f=num_frames_c)

        identity = hidden_states

        init_conditioning_hidden_states = conditioning_hidden_states[:, :, -1, :, :].unsqueeze(2).repeat(1, 1, num_frames, 1, 1)

        hidden_states = self.conditioning_norm(init_conditioning_hidden_states, hidden_states)

        hidden_states = identity + hidden_states
        
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        return hidden_states

class TemporalBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv = nn.Sequential(
            nn.GroupNorm(min(dim // 4, 32), dim), 
            nn.SiLU(), 
            nn.Conv3d(dim, dim, (3, 3, 3), padding=(1, 1, 1))
        )

        nn.init.zeros_(self.conv[-1].weight)
        nn.init.zeros_(self.conv[-1].bias)

    def forward(self, hidden_states, num_frames=1):
        hidden_states = (
            hidden_states[None, :].reshape((-1, num_frames) + hidden_states.shape[1:]).permute(0, 2, 1, 3, 4)
        )

        identity = hidden_states

        hidden_states = self.conv(hidden_states)

        hidden_states = identity + hidden_states
        
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(
            (hidden_states.shape[0] * hidden_states.shape[2], -1) + hidden_states.shape[3:]
        )
        return hidden_states
    
class TemporalConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim=None, dropout=0.0):
        super().__init__()

        out_dim = out_dim or in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_dim), 
            nn.SiLU(), 
            nn.Conv3d(in_dim, out_dim, (3, 1, 1), padding=(1, 0, 0))
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv3 = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv4 = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )

        nn.init.zeros_(self.conv4[-1].weight)
        nn.init.zeros_(self.conv4[-1].bias)

    def forward(self, hidden_states, num_frames=1):
        hidden_states = (
            hidden_states[None, :].reshape((-1, num_frames) + hidden_states.shape[1:]).permute(0, 2, 1, 3, 4)
        )

        identity = hidden_states

        hidden_states = self.conv1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.conv3(hidden_states)
        hidden_states = self.conv4(hidden_states)

        hidden_states = identity + hidden_states

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(
            (hidden_states.shape[0] * hidden_states.shape[2], -1) + hidden_states.shape[3:]
        )        
        return hidden_states
    
class Downsample2D(nn.Module):
    def __init__(self, channels, use_conv=False, out_channels=None, use_conditioning_norm=True, padding=1, name="conv"):
        super().__init__()

        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        self.conv = Conditioner(self.channels, self.out_channels, use_conditioning_norm, 3, stride=stride, padding=padding)

    def forward(self, hidden_states, conditioning_hidden_states, num_frames):
        assert hidden_states.shape[1] == self.channels

        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels
        
        hidden_states, conditioning_hidden_states = self.conv(hidden_states, conditioning_hidden_states=conditioning_hidden_states, num_frames=num_frames)

        return hidden_states, conditioning_hidden_states   

class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        pre_norm=True,
        eps=1e-6,
        output_scale_factor=1.0,
        up=False,
        down=False,
        conv_shortcut_bias: bool = True,
        use_conditioning_norm: bool = True
    ):
        super().__init__()

        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.silu1 = nn.SiLU()

        self.c_norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.c_silu1 = nn.SiLU()

        self.conv1 = Conditioner(in_channels, out_channels, use_conditioning_norm, kernel_size=3, stride=1, padding=1)
        
        self.time_emb_silu = nn.SiLU()
        self.time_emb_proj = torch.nn.Linear(temb_channels, out_channels)

        self.time_c_emb_silu = nn.SiLU()
        self.time_c_emb_proj = torch.nn.Linear(temb_channels, out_channels)
        
        self.dropout = torch.nn.Dropout(dropout)
        self.norm2 = torch.nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.silu2 = nn.SiLU()

        self.c_dropout = torch.nn.Dropout(dropout)
        self.c_norm2 = torch.nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.c_silu2 = nn.SiLU()

        self.conv2 = Conditioner(out_channels, out_channels, use_conditioning_norm, kernel_size=3, stride=1, padding=1)

        self.upsample = self.downsample = None

        if self.up:
            self.upsample = Upsample2D(in_channels, use_conv=False, use_conditioning_norm=use_conditioning_norm)
        elif self.down:
            self.downsample = Downsample2D(in_channels, use_conv=False, use_conditioning_norm=use_conditioning_norm, padding=1, name="op")

        self.conv_shortcut = None
        if self.in_channels != self.out_channels:
            self.conv_shortcut = Conditioner(
                in_channels, out_channels, use_conditioning_norm, kernel_size=1, stride=1, padding=0, bias=conv_shortcut_bias
            )

    def forward(self, input_tensor, conditioning_input_tensor, h_emb, c_emb, num_frames=1):
        hidden_states = input_tensor
        conditioning_hidden_states = conditioning_input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.silu1(hidden_states)

        conditioning_hidden_states = self.c_norm1(conditioning_hidden_states)
        conditioning_hidden_states = self.c_silu1(conditioning_hidden_states)

        if self.upsample is not None:
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
                conditioning_input_tensor = conditioning_input_tensor.contiguous()
                conditioning_hidden_states = conditioning_hidden_states.contiguous()

            input_tensor, conditioning_input_tensor = self.upsample(input_tensor, conditioning_hidden_states=conditioning_input_tensor, num_frames=num_frames)
            hidden_states, conditioning_hidden_states = self.upsample(hidden_states, num_frames=num_frames, conditioning_hidden_states=conditioning_hidden_states)
        elif self.downsample is not None:
            input_tensor, conditioning_input_tensor = self.downsample(input_tensor, conditioning_hidden_states=conditioning_input_tensor, num_frames=num_frames)
            hidden_states, conditioning_hidden_states = self.downsample(hidden_states, num_frames=num_frames, conditioning_hidden_states=conditioning_hidden_states)

        hidden_states, conditioning_hidden_states = self.conv1(hidden_states, num_frames=num_frames, conditioning_hidden_states=conditioning_hidden_states)

        h_emb = self.time_emb_silu(h_emb)
        h_emb = self.time_emb_proj(h_emb)[:, :, None, None]

        c_emb = self.time_c_emb_silu(c_emb)
        c_emb = self.time_c_emb_proj(c_emb)[:, :, None, None]

        hidden_states = hidden_states + h_emb
        conditioning_hidden_states = conditioning_hidden_states + c_emb

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.silu2(hidden_states)

        conditioning_hidden_states = self.c_norm2(conditioning_hidden_states)
        conditioning_hidden_states = self.c_silu2(conditioning_hidden_states)

        hidden_states = self.dropout(hidden_states)
        conditioning_hidden_states = self.c_dropout(conditioning_hidden_states)

        hidden_states, conditioning_hidden_states = self.conv2(hidden_states, num_frames=num_frames, conditioning_hidden_states=conditioning_hidden_states)

        if self.conv_shortcut is not None:
            input_tensor, conditioning_input_tensor = self.conv_shortcut(input_tensor, conditioning_hidden_states=conditioning_input_tensor, num_frames=num_frames)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        output_conditioning_hidden_states = (conditioning_input_tensor + conditioning_hidden_states) / self.output_scale_factor

        return output_tensor, output_conditioning_hidden_states
    
class Upsample2D(nn.Module):
    def __init__(self, channels, use_conv=False, out_channels=None, use_conditioning_norm=True, name="conv"):
        super().__init__()

        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.name = name

        self.conv = Conditioner(self.channels, self.out_channels, use_conditioning_norm, 3, padding=1)

    def forward(self, hidden_states, conditioning_hidden_states, output_size=None, num_frames=1):
        assert hidden_states.shape[1] == self.channels

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # TODO(Suraj): Remove this cast once the issue is fixed in PyTorch
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)
            conditioning_hidden_states = conditioning_hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()
            conditioning_hidden_states = conditioning_hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
            conditioning_hidden_states = F.interpolate(conditioning_hidden_states, scale_factor=2.0, mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")
            conditioning_hidden_states = F.interpolate(conditioning_hidden_states, size=output_size, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)
            conditioning_hidden_states = conditioning_hidden_states.to(dtype)

        hidden_states, conditioning_hidden_states = self.conv(hidden_states, conditioning_hidden_states=conditioning_hidden_states, num_frames=num_frames)

        return hidden_states, conditioning_hidden_states
