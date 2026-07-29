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
class NASProjMixer(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            super_arch=(2, 2, 5),  # (#convs, #stem transformers, #branch transformers)
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
            min_stem_layers=10

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
        self.opt_list = ['T', 'M']
        self.input_pdrop = nn.Dropout1d(p=input_pdrop) if input_pdrop > 0 else None
        self.input_noise = input_noise
        self.max_stem_layers = super_arch[1]
        self.min_stem_layers = min_stem_layers

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

        ###feature projection######
        # 1.build a supernet which contain 3 types of sub-blocks
        # 2.define an set sample config function to generate which block is selected
        # 3.according to the sampled block, forward the feature
        # position embedding (1, C, T), rescaled by 1/sqrt(n_embed)

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

        # stem network using (vanilla) transformer
        self.stem = nn.ModuleList()
        for i in range(self.max_stem_layers):
            opt = HybridCausalBlock(
                n_embd=out_channels,  # dimension of the input features
                stride=1,  # downsampling stride for the current layer
                kernel_size=mamba_kernel_size,  # conv kernel size
                expand=channel_expand,  # expand ratio for mamba
                num_head=num_head,  # number of heads in transformer
                seq_len=self.max_seq_len,
                drop_path_rate=drop_path_rate,  # drop path rate
                drop=0.,
                act_layer=nn.GELU,
                mlp_ratio=4.,
            )

            self.stem.append(opt)
            # main branch using transformer with pooling

        self.branch = nn.ModuleList()
        for _ in range(self.arch[2]):
            opt = HybridCausalBlock(
                n_embd=out_channels,  # dimension of the input features
                stride=2,  # downsampling stride for the current layer
                kernel_size=mamba_kernel_size,  # conv kernel size
                expand=channel_expand,  # expand ratio for mamba
                num_head=num_head,  # number of heads in transformer
                seq_len=self.max_seq_len,
                drop_path_rate=drop_path_rate,  # drop path rate
                drop=0.,
                act_layer=nn.GELU,
                mlp_ratio=4.,
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
        if choice is None:
            stem_length = np.random.randint(self.min_stem_layers, self.max_stem_layers + 1)

            stem_length_array = np.array([stem_length], dtype=np.int64)

            choice_1 = np.random.randint(3, size=stem_length)
            choice_2 = np.random.randint(3, size=self.arch[2])
            choice = np.concatenate((stem_length_array, choice_1, choice_2))
        else:
            stem_length = choice[0]
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

        # stem transformer
        for idx in range(stem_length):
            # print(choice[idx])
            # print(self.stem[idx][choice[idx]])
            x, mask = self.stem[idx](x, mask, choice[idx + 1])

        out_feats = (x,)
        out_masks = (mask,)

        # main branch with downsampling
        for idx in range(len(self.branch)):
            choice_idx = 1 + stem_length + idx
            x, mask = self.branch[idx](x, mask, choice[choice_idx])
            out_feats += (x,)
            out_masks += (mask,)

        return out_feats, out_masks, choice


class HybridCausalBlock(nn.Module):
    def __init__(
            self,
            n_embd,  # dimension of the input features
            stride=1,  # downsampling stride for the current layer
            kernel_size=4,  # conv kernel size
            expand=2,  # expand ratio for mamba
            num_head=4,  # number of heads in transformer
            drop_path_rate=0.3,  # drop path rate
            seq_len=192,
            drop=0.,
            act_layer=nn.GELU,
            mlp_ratio=4.,
    ):
        super().__init__()

        # normalization
        self.norm = nn.LayerNorm(n_embd, eps=1e-6)
        # self.norm2 = nn.LayerNorm(n_embd, eps=1e-6)

        # hybrid block with mamba and self-attn
        self.block = MixtureCausalBlock(n_embd, d_conv=kernel_size, expand=expand, num_head=num_head,seq_len=seq_len)
        # self.mlp = Mlp(in_features=n_embd, hidden_features=int(n_embd * mlp_ratio), act_layer=act_layer, drop=drop)

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

    def forward(self, x, mask,chioce=0):
        x = x.permute(0, 2, 1)
        x = x + self.drop_path(self.block(self.norm(x),chioce))
        x = x.permute(0, 2, 1)
        x = x * mask.unsqueeze(1).to(x.dtype)

        if self.downsample is not None:
            mask = self.downsample(mask.float()).bool()
            x = self.downsample(x) * mask.unsqueeze(1).to(x.dtype)
            # print(x.shape)
        return x, mask


class MixtureCausalBlock(nn.Module):
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
            mlp_ratio=4.,
            num_head=4,
            seq_len=192,
            qkv_bias=True,
            choice=0,  # branch selector: 0-SSM, 1-Linear Attention, 2-Hybrid (SSM+LinearAttention)
    ):
        super().__init__()

        self.d_model = d_model
        self.seq_len=seq_len
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
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32), "n -> d n", d=self.d_inner).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))  # Keep in fp32
        self.D._no_weight_decay = True
        # self.fusion_layer = nn.Linear(self.d_inner * 4, self.self.d_inner * 2, bias=bias)
        self.out_proj = nn.Linear(self.d_inner * 2, self.d_model, bias=bias)

        # downsampling
        if stride > 1:
            assert stride == 2
            self.downsample = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        else:
            self.downsample = None

        # --- linear attention branch components ---

        # self.choice = choice
        self.num_heads = num_head
        self.register_buffer("mask", torch.tril(torch.ones(self.seq_len, self.seq_len)).view(1, 1, self.seq_len, self.seq_len))  # causal mask sized to seq_len

        # self.mlp_ratio = mlp_ratio
        # self.d_k = d_model // num_head

        # self.seq_len = 192  # adjust to the actual sequence length
        # self.elu = nn.ELU()
        # self.lepe = nn.Conv1d(self.d_inner, self.d_inner, 3, padding=1, groups=d_model)  # 1D LePE
        # self.rope = RoPE(self.seq_len, self.d_inner)  # RoPE positional encoding (import from mlla.py)
        # self.register_buffer("causal_mask", torch.tril(torch.ones(self.seq_len, self.seq_len)).view(1, 1, self.seq_len, self.seq_len)) # causal mask sized to seq_len

    def forward(self, hidden_states, choice=0):
        batch, seqlen, dim = hidden_states.shape

        head_dim = dim // self.num_heads


        conv_state, ssm_state = None, None
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        xz_f, xz_b = torch.chunk(xz, 2, dim=1)
        # print(xz_f.shape)
        xz = torch.cat([xz_f, xz_b.flip([-1])], dim=0)
        # print(xz.shape)
        x_f, x_b = torch.chunk(xz, 2, dim=1)

        A = -torch.exp(self.A_log.float())
        # x, z = xz.chunk(2, dim=1)  # split into the two halves of the input projection
        if conv_state is not None:
            # pad the conv state so the convolution stays valid for short sequences
            conv_state.copy_(F.pad(x_f, (self.d_conv - x.shape[-1], 0)))  # update conv state (B D W)

            # run the 1D causal conv followed by the activation
        if causal_conv1d_fn is None:  # fused causal conv kernel unavailable
            x = self.act(self.conv1d(x_f)[..., :seqlen])  # regular conv + activation
        else:  # use the fused causal conv kernel
            x = causal_conv1d_fn(
                x=x_f,  # input
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),  # reshape the conv weight
                bias=self.conv1d.bias,  # conv bias
                activation=self.activation,  # activation function (silu / swish)
            )
        # x_qk=self.qk_proj(rearrange(x, "b d l -> (b l) d"))
        # dt, qk = torch.split(x_qk, [self.dt_rank, 2*self.d_inner], dim=-1)  # split into dt, B, C
        # print("x_dbl.shape:",x_dbl.shape)

        if choice == 0:  # SSM branch
            # x_t, z_t = torch.chunk(x, 2, dim=1)
            B, D, L = x.shape
            # print("x.shape:",x.shape)
            qkv = self.qkv_proj(x).transpose(1, 2).reshape(B, L, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            # x = flash_attn_qkvpacked_func(qkv, deterministic=True, causal=True)
            q, k, v = qkv[0], qkv[1], qkv[2]
            # print(q.shape)#[32, 4, 192, 128]

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
            # print(scores.shape)#[32, 4, 192, 192]
            # print(self.mask.shape)
            scores = scores.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
            # causal_mask = self._make_causal_mask(L).unsqueeze(0).unsqueeze(0)

            # scores = scores.masked_fill(causal_mask == 0, -1e9)
            attn = torch.softmax(scores, dim=-1)

            x_t = torch.matmul(attn, v).transpose(1, 2)
            x = x_t.contiguous().view(B, D, -1)


            # x = x.reshape(B, L, -1).transpose(1, 2)  # (B, D, L)

            out = x * F.silu(x_b)

        elif choice == 1:  # Linear Attention branch
            bc= self.bc(rearrange(x, "b d l -> (b l) d"))
            dt, B, C = torch.split(bc, [self.dt_rank, self.d_state, self.d_state], dim=-1)  # split into dt, B, C
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()  # reshape B
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()  # reshape C

            dt = self.dt_proj.weight @ dt.t()  # project the time-step dt
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)  # reshape to (b, d, l)

            assert self.activation in ["silu", "swish"]

            # run the selective state-space scan
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
            # update the SSM state cache if present
            if ssm_state is not None:
                out, last_state = out  # unpack the updated state
                ssm_state.copy_(last_state)  # cache the updated state

            # reshape the output y from (B, D, L) -> (B, L, D)
            # y = rearrange(y, "b d l -> b l d")
            # project back to the original feature dim via the output layer
            # out = out * F.silu(z_t)

            # y = attn_out.transpose(1, 2).reshape(batch, seqlen, -1)
        elif choice == 2:  # hybrid branch: SSM + linear attention
            bc = self.bc(rearrange(x, "b d l -> (b l) d"))
            dt, B, C = torch.split(bc, [self.dt_rank, self.d_state, self.d_state], dim=-1)  # split into dt, B, C
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()  # reshape B
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()  # reshape C

            dt = self.dt_proj.weight @ dt.t()  # project the time-step dt
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)  # reshape to (b, d, l)

            assert self.activation in ["silu", "swish"]

            # run the selective state-space scan
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
            # update the SSM state cache if present
            if ssm_state is not None:
                out, last_state = out  # unpack the updated state
                ssm_state.copy_(last_state)  # cache the updated state

            B, D, L = x.shape
            # print("x.shape:",x.shape)
            qkv = self.qkv_proj(x).transpose(1, 2).reshape(B, L, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            # print(q.shape)#[32, 4, 192, 128]

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
            # print(scores.shape)#[32, 4, 192, 192]
            # print(self.mask.shape)
            scores = scores.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
            # causal_mask = self._make_causal_mask(L).unsqueeze(0).unsqueeze(0)

            # scores = scores.masked_fill(causal_mask == 0, -1e9)
            attn = torch.softmax(scores, dim=-1)

            x_t = torch.matmul(attn, v).transpose(1, 2)
            out_t = x_t.contiguous().view(B, D, -1)

            out_t = out_t * F.silu(x_b)
            out= out_m + out_t
            # out=torch.cat(out_m, out_t,dim=1)
            # out = F.linear(out)

        else:
            raise ValueError(f"Invalid choice: {self.choice}")
        out = out.chunk(2)
        out = torch.cat([out[0], out[1].flip([-1])], dim=1)
        out = rearrange(out, "b d l -> b l d")
        # print(out.shape)

        out = F.linear(out, self.out_proj.weight, self.out_proj.bias)

        return out


class Identity(nn.Module):
    def __init__(
            self,
    ):
        super().__init__()

    def forward(self, x, mask):
        return x, mask
