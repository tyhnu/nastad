import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np

from .actionformer_proj import get_sinusoid_encoding
from ..bricks import ConvModule, AffineDropPath
from ..builder import PROJECTIONS
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

try:
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn_no_out_proj,selective_scan_fn_ops

    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False

try:
    from flash_attn import flash_attn_qkvpacked_func

    FLASHATTN_AVAILABLE = True
except ImportError:
    FLASHATTN_AVAILABLE = False
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None


@PROJECTIONS.register_module()
class NASProjFlashRetrain(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            super_arch=(2, 10, 5),  # (#convs, stem, branch)
            choice=None,
            conv_cfg=None,  # kernel_size proj_pdrop
            norm_cfg=None,
            use_abs_pe=False,  # use absolute position embedding
            max_seq_len=2304,
            input_pdrop=0.0,  # drop out the input feature
            mamba_kernel_size=4,  # kernel size of causal conv1d in mamba
            channel_expand=2,  # expand ratio for mamba
            num_head=4,  # number of heads in transformer
            drop_path_rate=0.3,
            input_noise=0.0,
            all_one=True,

    ):
        super().__init__()
        assert (
            MAMBA_AVAILABLE
        ), "Please install mamba-ssm to use this module. Check: https://github.com/OpenGVLab/video-mamba-suite"
        assert (
            FLASHATTN_AVAILABLE
        ), "Please install flash-attention-2 to use this module. Check: https://github.com/Dao-AILab/flash-attentio"

        assert len(super_arch) == 3

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.arch = super_arch
        self.kernel_size = conv_cfg["kernel_size"]
        self.scale_factor = 2  # as default
        self.with_norm = norm_cfg is not None
        self.use_abs_pe = use_abs_pe
        self.max_seq_len = max_seq_len
        self.input_pdrop = nn.Dropout1d(p=input_pdrop) if input_pdrop > 0 else None
        self.input_noise = input_noise
        self.all_one = all_one
        self.choice = choice


        if isinstance(self.in_channels, (list, tuple)):
            assert isinstance(self.out_channels, (list, tuple)) and len(self.in_channels) == len(self.out_channels)
            self.proj = nn.ModuleList([])
            for n_in, n_out in zip(self.in_channels, self.out_channels):
                self.proj.append(
                    ConvModule(
                        n_in,
                        n_out,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
            in_channels = out_channels = sum(self.out_channels)
        else:
            self.proj = None

        if self.use_abs_pe:
            pos_embed = get_sinusoid_encoding(self.max_seq_len, out_channels) / (out_channels ** 0.5)
            self.register_buffer("pos_embed", pos_embed, persistent=False)


        # embedding network using convs
        self.embed = nn.ModuleList()
        for i in range(self.arch[0]):
            self.embed.append(
                ConvModule(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type="relu"),
                )
            )

        # stem network
        self.stem = nn.ModuleList()
        for i in range(self.arch[1]):
            opt = HybridSearchableBlock(
                n_embd=out_channels,  # dimension of the input features
                stride=1,  # downsampling stride for the current layer
                kernel_size=mamba_kernel_size,  # conv kernel size
                expand=channel_expand,  # expand ratio for mamba
                num_head=num_head,  # number of heads in transformer
                drop_path_rate=drop_path_rate  # drop path rate
            )

            self.stem.append(opt)
            
        # main branchwith pooling
        self.branch = nn.ModuleList()
        for _ in range(self.arch[2]):
            opt = HybridSearchableBlock(
                n_embd=out_channels,  # dimension of the input features
                stride=2,  # downsampling stride for the current layer
                kernel_size=mamba_kernel_size,  # conv kernel size
                expand=channel_expand,  # expand ratio for mamba
                num_head=num_head,  # number of heads in transformer
                drop_path_rate=drop_path_rate  # drop path rate
            )
            self.branch.append(opt)
        self.apply(self.__init_weights__)

    def __init_weights__(self, m):
        # set nn.Linear bias term to 0
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            if m.bias is not None:
                if not getattr(m.bias, "_no_reinit", False):
                    torch.nn.init.constant_(m.bias, 0.0)

    def forward(self, x, mask, choice=None):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, sequence length (bool)

        choice=self.choice
        # feature projection
        if self.proj is not None:
            x = torch.cat([proj(s, mask)[0] for proj, s in zip(self.proj, x.split(self.in_channels, dim=1))], dim=1)

        # drop out input if needed
        if self.input_pdrop is not None:
            x = self.input_pdrop(x)

        # embedding network
        for idx in range(len(self.embed)):
            x, mask = self.embed[idx](x, mask)

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert x.shape[-1] <= self.max_seq_len, "Reached max length."
            pe = self.pos_embed
            # add pe to x
            x = x + pe[:, :, : x.shape[-1]] * mask.unsqueeze(1).to(x.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if x.shape[-1] >= self.max_seq_len:
                pe = F.interpolate(self.pos_embed, x.shape[-1], mode="linear", align_corners=False)
            else:
                pe = self.pos_embed
            # add pe to x
            x = x + pe[:, :, : x.shape[-1]] * mask.unsqueeze(1).to(x.dtype)

        # stem  
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask, choice[idx])

        out_feats = (x,)
        out_masks = (mask,)

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask, choice[idx + self.arch[1]])
            out_feats += (x,)
            out_masks += (mask,)

        return out_feats, out_masks, choice


class HybridSearchableBlock(nn.Module):
    def __init__(
            self,
            n_embd,  # dimension of the input features
            stride=1,  # downsampling stride for the current layer
            kernel_size=4,  # conv kernel size
            expand=2,  # expand ratio for mamba
            num_head=4,  # number of heads in transformer
            drop_path_rate=0.3,  # drop path rate
    ):
        super().__init__()

        # normalization
        self.norm = nn.LayerNorm(n_embd, eps=1e-6)

        # Selective block with mamba and self-attn
        self.block = SelectiveBlock(n_embd, d_conv=kernel_size, expand=expand, num_head=num_head)


        # downsampling
        if stride > 1:
            assert stride == 2
            self.downsample = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        else:
            self.downsample = None

        # drop path
        if drop_path_rate > 0.0:
            self.drop_path = AffineDropPath(n_embd, drop_prob=drop_path_rate, transpose=True)
        else:
            self.drop_path = nn.Identity()

    def forward(self, x, mask, chioce=0):
        x = x.permute(0, 2, 1)
        x = x + self.drop_path(self.block(self.norm(x),chioce))
        x = x.permute(0, 2, 1)
        x = x * mask.unsqueeze(1).to(x.dtype)

        if self.downsample is not None:
            mask = self.downsample(mask.float()).bool()
            x = self.downsample(x) * mask.unsqueeze(1).to(x.dtype)
        return x, mask


class SelectiveBlock(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            stride=1,
            bias=False,
            num_head=4,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.activation = "silu"
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2 *2 , bias=bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.act = nn.SiLU()
        self.qkv_proj = nn.Conv1d(self.d_inner, self.d_inner * 3, kernel_size=3, padding=1, groups=self.d_inner)

        self.bc = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False)  ###B,C,x_t/Q,K
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(
            min=dt_init_floor
        )
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32), "n -> d n", d=self.d_inner).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))  # Keep in fp32
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner * 2, self.d_model, bias=bias)

        # downsampling
        if stride > 1:
            assert stride == 2
            self.downsample = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        else:
            self.downsample = None

        self.num_heads = num_head

    def forward(self, hidden_states, choice=0):
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        xz_f, xz_b = torch.chunk(xz, 2, dim=1)

        xz = torch.cat([xz_f, xz_b.flip([-1])], dim=0)

        x_f, x_b = torch.chunk(xz, 2, dim=1)

        A = -torch.exp(self.A_log.float())

        if conv_state is not None:
            conv_state.copy_(F.pad(x_f, (self.d_conv - x.shape[-1], 0))) 

        if causal_conv1d_fn is None:  
            x = self.act(self.conv1d(x_f)[..., :seqlen]) 
        else:  
            x = causal_conv1d_fn(
                x=x_f, 
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),  
                bias=self.conv1d.bias, 
                activation=self.activation,  
            )


        if choice == 0:  

            B, _, L = x.shape

            qkv = self.qkv_proj(x).transpose(1, 2).reshape(B, L, 3, self.num_heads, -1)
            x = flash_attn_qkvpacked_func(qkv, deterministic=True, causal=True)
            x = x.reshape(B, L, -1).transpose(1, 2)  # (B, D, L)

            out = x * F.silu(x_b)

        elif choice == 1: 
            bc= self.bc(rearrange(x, "b d l -> (b l) d"))
            dt, B, C = torch.split(bc, [self.dt_rank, self.d_state, self.d_state], dim=-1)  
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()  
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous() 

            dt = self.dt_proj.weight @ dt.t() 
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)  

            assert self.activation in ["silu", "swish"]

            out = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=x_b,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )

            if ssm_state is not None:
                out, last_state = out  
                ssm_state.copy_(last_state)  


        elif choice == 2:
            bc = self.bc(rearrange(x, "b d l -> (b l) d"))
            dt, B, C = torch.split(bc, [self.dt_rank, self.d_state, self.d_state], dim=-1)  
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous() 
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()  

            dt = self.dt_proj.weight @ dt.t() 
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)  

            assert self.activation in ["silu", "swish"]


            out_m = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=x_b,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )

            if ssm_state is not None:
                out, last_state = out  
                ssm_state.copy_(last_state) 

            B, _, L = x.shape

            qkv = self.qkv_proj(x).transpose(1, 2).reshape(B, L, 3, self.num_heads, -1)
            out_t = flash_attn_qkvpacked_func(qkv, deterministic=True, causal=True)
            out_t = out_t.reshape(B, L, -1).transpose(1, 2)  # (B, D, L)

            out_t = out_t * F.silu(x_b)
            out= out_m + out_t


        else:
            raise ValueError(f"Invalid choice: {self.choice}")
        out = out.chunk(2)
        out = torch.cat([out[0], out[1].flip([-1])], dim=1)
        out = rearrange(out, "b d l -> b l d")


        out = F.linear(out, self.out_proj.weight, self.out_proj.bias)

        return out

class Identity(nn.Module):
    def __init__(
            self,
    ):
        super().__init__()

    def forward(self, x, mask):
        return x, mask
