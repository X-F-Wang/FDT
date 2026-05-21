import numpy as np

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
import smm_cuda

import models
from models import register
from utils import make_coord


class FEM(nn.Module):
    r"""Mixing pixels with their surrounding pixels.
    ”“”64*3
    In the paper, it is referred to as FEM

    """

    def __init__(self, planes: int, mix_margin: int = 1) -> None:
        super(FEM, self).__init__()

        assert planes % 8 == 0

        self.planes = planes
        self.mix_margin = nn.Parameter(torch.tensor(mix_margin, dtype=torch.float), requires_grad=True)

        self.mask = nn.Parameter(torch.zeros((self.planes, 1, 3, 3)), requires_grad=False)

        self.mask[3::4, 0, 0, mix_margin] = 1.
        self.mask[2::4, 0, -1, mix_margin] = 1.
        self.mask[1::4, 0, mix_margin, 0] = 1.
        self.mask[0::4, 0, mix_margin, -1] = 1.

        self.mask[4::8, 0, 0, 2] = 1.
        self.mask[5::8, 0, 2, 0] = 1.
        self.mask[6::8, 0, 2, 2] = 1.
        self.mask[7::8, 0, 0, 0] = 1.


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: b c h w

        Returns:
            b c h w -> b c h w
        """
        m = int(self.mix_margin.item())
        x = F.conv2d(input=F.pad(x, pad=(m, m, m, m), mode='circular'),
                     weight=self.mask, bias=None, stride=(1, 1), padding=(0, 0),
                     dilation=(m, m), groups=self.planes)

        return x


class ACM(nn.Module):
    #ACM
    def __init__(self, channels, num_tokens=64, n_iter=3,
                 init_ema_decay=0.99, init_topk=3,
                 init_soft_tau=0.07, init_alpha_base=0.5,
                 init_amp_quantile=0.90, eps=1e-6):
        super().__init__()
        self.channels = channels
        self.num_tokens = num_tokens
        self.n_iter = n_iter
        self.topk = max(1, init_topk)
        self.eps = eps

        # trainable parameters (raw, unconstrained)
        self.logit_ema_decay = nn.Parameter(torch.logit(torch.tensor(init_ema_decay)))
        self.log_soft_tau = nn.Parameter(torch.log(torch.tensor(init_soft_tau)))
        self.logit_alpha_base = nn.Parameter(torch.logit(torch.tensor(init_alpha_base)))
        self.logit_amp_quantile = nn.Parameter(torch.logit(torch.tensor(init_amp_quantile)))

        # centers and initted flag
        self.register_buffer('means', torch.randn(num_tokens, channels))
        self.register_buffer('initted', torch.tensor(False))

        # per-center learnable scale
        self.log_gamma = nn.Parameter(torch.zeros(num_tokens))  # gamma = exp(log_gamma)

        # post conv
        self.post_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )


    def constrained_params(self):
        ema_decay = torch.sigmoid(self.logit_ema_decay).clamp(0.90, 0.999)
        soft_tau = torch.exp(self.log_soft_tau).clamp(0.01, 0.5)
        alpha_base = torch.sigmoid(self.logit_alpha_base).clamp(0.1, 0.9)
        amp_threshold_quantile = torch.sigmoid(self.logit_amp_quantile).clamp(0.7, 0.99)
        gamma = torch.exp(self.log_gamma).clamp(max=10.0)  # 防止过大
        return ema_decay, soft_tau, alpha_base, amp_threshold_quantile, gamma


    def regularization_loss(self):
        ema_decay, soft_tau, alpha_base, amp_q, gamma = self.constrained_params()
        reg = 0.0

        reg += (soft_tau - 0.1).pow(2).mean()

        reg += (alpha_base - 0.5).pow(2).mean()

        reg += (torch.log(gamma + 1e-6)).pow(2).mean()
        return reg * 1e-3

    def _ema_update(self, moving_avg, new, decay):
        moving_avg.data.mul_(decay).add_(new, alpha=(1. - decay))

    def forward(self, amp):
        B, C, Hf, Wf = amp.shape
        device = amp.device


        ema_decay, soft_tau, alpha_base, amp_threshold_quantile, gamma = self.constrained_params()

        tokens = amp.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        N = tokens.shape[1]
        M = self.num_tokens


        if not bool(self.initted):
            pad_n = (M - (N % M)) % M
            if pad_n > 0:
                pad_tokens = tokens[:, N - pad_n:N, :].flip(dims=[1])
                tokens_padded = torch.cat([tokens, pad_tokens], dim=1)
            else:
                tokens_padded = tokens
            Np = tokens_padded.shape[1]
            group_size = Np // M
            grouped = tokens_padded.view(B, M, group_size, C)
            means_init = grouped.mean(dim=2).mean(dim=0)
            means_init = F.normalize(means_init, dim=-1, eps=self.eps)
            self.means.data.copy_(means_init)
            self.initted.data.copy_(torch.tensor(True, device=device))

        means = self.means.detach()
        means = F.normalize(means, dim=-1, eps=self.eps)


        if self.training:
            with torch.no_grad():
                for _ in range(self.n_iter):
                    toks_n = F.normalize(tokens, dim=-1, eps=self.eps)
                    sims = torch.einsum('bnc,mc->bnm', toks_n, means)
                    p = torch.softmax(sims / soft_tau, dim=-1)
                    weighted = torch.einsum('bnm,bnc->mc', p, tokens)
                    counts = p.sum(dim=(0,1)).unsqueeze(-1).clamp_min(self.eps)
                    new_means = weighted / counts
                    new_means = F.normalize(new_means, dim=-1, eps=self.eps)
                    means = new_means
                self._ema_update(self.means, means, ema_decay)


        token_amp = tokens.norm(dim=-1)
        try:
            q = torch.tensor(amp_threshold_quantile, device=device, dtype=token_amp.dtype)
            thresh = torch.quantile(token_amp, q, dim=1, keepdim=True)
        except Exception:
            k = max(1, int((1.0 - amp_threshold_quantile.item()) * N))
            topk_vals, _ = torch.topk(token_amp, k=k, dim=1, largest=True, sorted=True)
            thresh = topk_vals[:, -1:].detach()

        s = 20.0
        gate = torch.sigmoid(s * (token_amp - thresh))
        alpha_token = (alpha_base * gate).unsqueeze(-1)


        with torch.no_grad():
            toks_n = F.normalize(tokens, dim=-1, eps=self.eps)
            sims = torch.einsum('bnc,mc->bnm', toks_n, means)
            sims = sims * gamma.unsqueeze(0).unsqueeze(0)
            if self.topk <= 1:
                buckets = sims.argmax(dim=-1)
                assigned = means[buckets.view(-1)].view(B, N, C)
                fused_tokens = tokens + alpha_token * assigned
            else:
                vals, idx = torch.topk(sims, k=self.topk, dim=-1, largest=True, sorted=True)
                weights = torch.softmax(vals, dim=-1)
                idx_flat = idx.contiguous().view(-1)
                means_expanded = means[idx_flat].view(B, N, self.topk, C)
                assigned = (weights.unsqueeze(-1) * means_expanded).sum(dim=2)
                fused_tokens = tokens + alpha_token * assigned

        fused = fused_tokens.view(B, Hf, Wf, C).permute(0, 3, 1, 2).contiguous()
        fused = self.post_conv(fused)
        return fused

class SMM_QmK(Function):
    @staticmethod
    def forward(ctx, A, B, index):
        ctx.save_for_backward(A, B, index)
        return smm_cuda.SMM_QmK_forward_cuda(A.contiguous(), B.contiguous(), index.contiguous())

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        A, B, index = ctx.saved_tensors
        grad_A, grad_B = smm_cuda.SMM_QmK_backward_cuda(
            grad_output.contiguous(), A.contiguous(), B.contiguous(), index.contiguous()
        )
        return grad_A, grad_B, None


class SMM_AmV(Function):
    @staticmethod
    def forward(ctx, A, B, index):
        ctx.save_for_backward(A, B, index)
        return smm_cuda.SMM_AmV_forward_cuda(A.contiguous(), B.contiguous(), index.contiguous())

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        A, B, index = ctx.saved_tensors
        grad_A, grad_B = smm_cuda.SMM_AmV_backward_cuda(
            grad_output.contiguous(), A.contiguous(), B.contiguous(), index.contiguous()
        )
        return grad_A, grad_B, None



def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x



class ProgressiveFocusedAttention(nn.Module):
    def __init__(self, dim, layer_id, window_size, num_heads, num_topk, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.layer_id = layer_id
        self.window_size = window_size
        self.num_heads = num_heads
        self.num_topk = num_topk


        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.eps = 1e-20


        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)



        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        trunc_normal_(self.relative_position_bias_table, std=.02)


        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        self.register_buffer('relative_position_index', relative_coords.sum(-1))


        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, pfa_values, pfa_indices):
        B_, N, C = x.shape


        q = self.q_proj(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)


        q = q * self.scale


        if pfa_indices is not None:
            topk = pfa_indices.shape[-1]
            q = q.contiguous().view(B_ * self.num_heads, N, C // self.num_heads)
            k = k.contiguous().view(B_ * self.num_heads, N, C // self.num_heads).transpose(-2, -1)
            smm_index = pfa_indices.view(B_ * self.num_heads, N, topk).int()
            attn = SMM_QmK.apply(q, k, smm_index).view(B_, self.num_heads, N, topk)


            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)].view(
                self.window_size * self.window_size, self.window_size * self.window_size, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
            relative_position_bias = torch.gather(relative_position_bias, dim=-1, index=pfa_indices)
            attn = attn + relative_position_bias
        else:

            attn = (q @ k.transpose(-2, -1))  # <--- 这里的 k 需要转置
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)].view(
                self.window_size * self.window_size, self.window_size * self.window_size, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
            attn = attn + relative_position_bias


        attn = self.softmax(attn)


        if pfa_values is not None:
            attn = attn * pfa_values
            attn = (attn + self.eps) / (attn.sum(dim=-1, keepdim=True) + self.eps)


        if self.num_topk < self.window_size * self.window_size:
            topk_values, topk_indices = torch.topk(attn, self.num_topk, dim=-1, largest=True, sorted=False)
            attn = topk_values
            if pfa_indices is not None:
                pfa_indices = torch.gather(pfa_indices, dim=-1, index=topk_indices)
            else:
                pfa_indices = topk_indices


        if pfa_indices is not None:
            topk = pfa_indices.shape[-1]
            attn = attn.view(B_ * self.num_heads, N, topk)
            v = v.contiguous().view(B_ * self.num_heads, N, C // self.num_heads)
            smm_index = pfa_indices.view(B_ * self.num_heads, N, topk).int()
            x = SMM_AmV.apply(attn, v, smm_index).view(B_, self.num_heads, N, C // self.num_heads)
        else:
            x = (attn @ v)


        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)

        return x, attn, pfa_indices


class PCSA(nn.Module):
    def __init__(self, dim, depth, window_size=8, num_heads=4, num_topk_list=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.num_heads = num_heads

        if num_topk_list is None:
            base_topk = window_size * window_size
            self.num_topk_list = [max(base_topk // (2 ** i), 4) for i in range(depth)]
        else:
            self.num_topk_list = num_topk_list

        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                ProgressiveFocusedAttention(
                    dim=dim,
                    layer_id=i,
                    window_size=window_size,
                    num_heads=num_heads,
                    num_topk=self.num_topk_list[i]
                )
            )

        self.norm = nn.LayerNorm(dim)

        self.output_proj = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        B, C, H_orig, W_orig = x.shape

        pad_h = (self.window_size - H_orig % self.window_size) % self.window_size
        pad_w = (self.window_size - W_orig % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        H, W = x.shape[2:]

        x = x.permute(0, 2, 3, 1).contiguous()

        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        x_windows = self.norm(x_windows)

        pfa_values = None
        pfa_indices = None

        for layer in self.layers:
            x_windows, pfa_values, pfa_indices = layer(
                x_windows, pfa_values, pfa_indices
            )

        x_windows = x_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x_windows, self.window_size, H, W)

        x = x.permute(0, 3, 1, 2).contiguous()

        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H_orig, :W_orig]
        x = self.output_proj(x)

        return x

class APFN(nn.Module):
    def __init__(self, dim):
        super(APFN, self).__init__()
        self.fpre = FEM(planes=64)
        self.amp_fuse = ACM(
            channels=64,
            num_tokens=64,
            n_iter=3,
            init_ema_decay=0.99,
            init_topk=3,
            init_soft_tau=0.07,
            init_alpha_base=0.45,
            init_amp_quantile=0.90
        )
        self.pha_fuse = PCSA(
            dim=64,
            depth=3,
            window_size=8,
            num_heads=4,
            num_topk_list=[]
        )
        self.post = FEM(planes=)

    def forward(self, x):
        B, C, H, W = x.shape
        x_pre = self.fpre(x) + 1e-8
        msF = torch.fft.rfft2(x_pre, norm='backward')
        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)

        amp_fuse = self.amp_fuse(msF_amp)
        amp_fuse = amp_fuse + msF_amp

        pha_fuse = self.pha_fuse(msF_pha)
        pha_fuse = pha_fuse + msF_pha

        real = amp_fuse * torch.cos(pha_fuse) + 1e-8
        imag = amp_fuse * torch.sin(pha_fuse) + 1e-8
        out_complex = torch.complex(real, imag) + 1e-8
        out = torch.abs(torch.fft.irfft2(out_complex, s=(H, W), norm='backward'))

        out = self.post(out)
        x_skip = self.post(x_pre)
        out = out + x_skip
        out = torch.nan_to_num(out, nan=1e-5, posinf=1e-5, neginf=1e-5)
        return out

@register('FDT')
class FDT(nn.Module):
    def __init__(
            self,
            encoder_spec,
            imnet_spec,
            pb_spec=None,
            pe_spec=None,
            base_dim=192,
            head=8,
            r=3,
            imnet_num=1,
            conv_num=1,
            is_cell=True,
            local_attn=True,
            verbose=False,
    ):
        super().__init__()

        self.dim = base_dim
        self.head = head
        self.r = r
        self.imnet_num = imnet_num
        self.conv_num = conv_num
        self.is_cell = is_cell
        self.local_attn = local_attn
        self.verbose = verbose

        self.encoder = models.make(encoder_spec)
        self.filter = APFN(dim=64)
        self.conv_ch = nn.Conv2d(self.encoder.out_dim, self.dim, kernel_size=3, padding=1)

        self.conv_vs = nn.ModuleList([
            nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1) for _ in range(self.conv_num)
        ])

        if self.local_attn:
            self.conv_qs = nn.ModuleList([
                nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)
                for _ in range(self.conv_num)
            ])

            self.conv_ks = nn.ModuleList([
                nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)
                for _ in range(self.conv_num)
            ])

            self.is_pb = True if pb_spec else False

            if self.is_pb:
                self.pb_encoder = models.make(pb_spec, args={'head': self.head}).cuda()
        else:
            self.r = 0

        self.r_area = (2 * self.r + 1) ** 2

        imnet_in_dim = self.dim * self.r_area + 2 if self.is_cell else self.dim * self.r_area

        self.imnets = nn.ModuleList([
            models.make(
                imnet_spec,
                args={'in_dim': imnet_in_dim}
            ) for _ in range(self.imnet_num)
        ])

    def gen_feat(self):
        if self.prev_feat is None:
            self.feat = self.encoder(self.inp)
            #self.feat = self.conv_ch(self.feat)
            self.feat = self.filter(self.feat)
            self.modulator = self.feat
            self.feat = self.conv_ch(self.feat)
            self.init_feat = self.feat.clone()
        else:
            self.feat = self.prev_feat.clone()

        if self.local_attn:
            self.feat_q = self.conv_qs[self.conv_idx](self.feat)
            self.feat_k = self.conv_ks[self.conv_idx](self.feat)

        self.feat_v = self.conv_vs[self.conv_idx](self.feat)
        return self.feat


    def query_rgb(self, sample_coord, cell=None):
        feat = self.feat

        bs, q_sample, _ = sample_coord.shape

        coord_lr = make_coord(feat.shape[-2:], flatten=False).cuda().permute(2, 0, 1). \
            unsqueeze(0).expand(bs, 2, *feat.shape[-2:])

        # b, q, 1, 2
        sample_coord_ = sample_coord.clone()
        sample_coord_ = sample_coord_.unsqueeze(2)

        # field radius (global: [-1, 1])
        rh = 2 / feat.shape[-2]
        rw = 2 / feat.shape[-1]

        r = self.r

        # b, 2, h, w -> b, 2, q, 1 -> b, q, 1, 2
        sample_coord_k = F.grid_sample(
            coord_lr, sample_coord_.flip(-1), mode='nearest', align_corners=False
        ).permute(0, 2, 3, 1)

        if self.local_attn:
            dh = torch.linspace(-r, r, 2 * r + 1).cuda() * rh
            dw = torch.linspace(-r, r, 2 * r + 1).cuda() * rw
            # 1, 1, r_area, 2
            delta = torch.stack(torch.meshgrid(dh, dw, indexing='ij'), axis=-1).view(1, 1, -1, 2)

            # Q - b, c, h, w -> b, c, q, 1 -> b, q, 1, c -> b, q, 1, h, c -> b, q, h, 1, c
            sample_feat_q = F.grid_sample(
                self.feat_q, sample_coord_.flip(-1), mode='bilinear', align_corners=False
            ).permute(0, 2, 3, 1)
            sample_feat_q = sample_feat_q.reshape(
                bs, q_sample, 1, self.head, self.dim // self.head
            ).permute(0, 1, 3, 2, 4)

            # b, q, 1, 2 -> b, q, 49, 2
            sample_coord_k = sample_coord_k + delta

            # K - b, c, h, w -> b, c, q, 49 -> b, q, 49, c -> b, q, 49, h, c -> b, q, h, c, 49
            sample_feat_k = F.grid_sample(
                self.feat_k, sample_coord_k.flip(-1), mode='nearest', align_corners=False
            ).permute(0, 2, 3, 1)
            sample_feat_k = sample_feat_k.reshape(
                bs, q_sample, self.r_area, self.head, self.dim // self.head
            ).permute(0, 1, 3, 4, 2)

        sample_feat_v = F.grid_sample(
            self.feat_v, sample_coord_k.flip(-1), mode='nearest', align_corners=False
        ).permute(0, 2, 3, 1)

        # b, q, 49, 2
        rel_coord = sample_coord_ - sample_coord_k
        rel_coord[..., 0] *= feat.shape[-2]
        rel_coord[..., 1] *= feat.shape[-1]

        # b, 2 -> b, q, 2
        rel_cell = cell.clone()
        rel_cell = rel_cell.unsqueeze(1).repeat(1, q_sample, 1)
        rel_cell[..., 0] *= feat.shape[-2]
        rel_cell[..., 1] *= feat.shape[-1]

        if self.local_attn:
            # b, q, h, 1, r_area -> b, q, r_area, h
            # print(sample_feat_q.shape,sample_feat_k.shape)
            attn = torch.matmul(sample_feat_q, sample_feat_k).reshape(
                bs, q_sample, self.head, self.r_area
            ).permute(0, 1, 3, 2) / np.sqrt(self.dim // self.head)

            if self.is_pb:
                _, pb = self.pb_encoder(rel_coord)
                attn = F.softmax(torch.add(attn, pb), dim=-2)
            else:
                attn = F.softmax(attn, dim=-2)

            attn = attn.reshape(bs, q_sample, self.r_area, self.head, 1)
            sample_feat_v = sample_feat_v.reshape(
                bs, q_sample, self.r_area, self.head, self.dim // self.head
            )
            sample_feat_v = torch.mul(sample_feat_v, attn).reshape(bs, q_sample, self.r_area, -1)

        feat_in = sample_feat_v.reshape(bs, q_sample, -1)

        if self.is_cell:
            feat_in = torch.cat([feat_in, rel_cell], dim=-1)

        pred = self.imnets[self.im_idx](feat_in)

        if self.prev_pred is None:
            self.prev_pred = pred
        else:
            pred = pred + self.prev_pred * 0.75
            self.prev_pred = pred

        pred = pred + F.grid_sample(self.inp, sample_coord_.flip(-1), mode='bilinear', \
                                    padding_mode='border', align_corners=False)[:, :, :, 0].permute(0, 2, 1)

        if self.local_attn and self.verbose:
            return pred, attn[0, :, :, :, 0].reshape(-1, 2 * r + 1, 2 * r + 1, self.head)

        return pred

    def cascaded_forward(self, inp, coords, sample_coord, cell):
        preds = []

        self.im_idx = 0
        self.conv_idx = 0
        self.prev_feat = None
        self.prev_pred = None

        for idx in range(len(coords)):
            bs, h, w, _ = coords[idx].shape

            self.gen_feat()
            if idx < len(coords) - 1:
                # b, c, q, 1
                coord = coords[idx].clone()
                coord = coord.reshape(bs, -1, 2).unsqueeze(2)
                prev_feat = F.grid_sample(
                    self.init_feat, coord.flip(-1), mode='bilinear', align_corners=False
                )
                self.prev_feat = prev_feat.reshape(bs, -1, h, w)

            pred = self.query_rgb(sample_coord, cell)

            self.im_idx += 1
            self.conv_idx += 1

            preds.append(pred)

        return preds

    def forward(self, inp, coords, sample_coord, cell):
        self.inp = inp

        return self.cascaded_forward(inp, coords, sample_coord, cell)

    def chop_forward(self, inp, coords, cell, eval_bsize=100000, visual_indices=None):  # 用于demo map
        self.inp = inp

        bs = coords[-1].shape[0]
        hr_coord = coords[-1].clone()
        hr_coord = hr_coord.reshape(bs, -1, 2)
        n = hr_coord.shape[1]

        self.prev_feat = None
        self.prev_pred = None
        prev_pred = None

        preds = []
        attns = []

        self.im_idx = 0
        self.conv_idx = 0

        for idx in range(len(coords)):
            self.gen_feat()
            if idx < len(coords) - 1:
                h, w = coords[idx].shape[1:3]
                coord = coords[idx].clone()
                coord = coord.reshape(bs, -1, 2).unsqueeze(2)
                prev_feat = F.grid_sample(
                    self.init_feat, coord.flip(-1), mode='bilinear', align_corners=False
                )
                self.prev_feat = prev_feat.reshape(bs, -1, h, w)

            if visual_indices is not None:
                self.verbose = True
                sample_coord = hr_coord[:, visual_indices, :]

                pred, attn = self.query_rgb(sample_coord, cell)

                attns.append(attn)
            else:
                cur_preds = []
                prev_preds = []
                ql = 0

                while ql < n:
                    qr = min(ql + eval_bsize, n)

                    sample_coord = hr_coord[:, ql:qr, :]

                    if prev_pred is None:
                        self.prev_pred = None
                    else:
                        self.prev_pred = prev_pred[:, ql:qr, :]

                    pred = self.query_rgb(sample_coord, cell)
                    cur_preds.append(pred)
                    prev_preds.append(self.prev_pred)

                    ql = qr

                cur_pred = torch.cat(cur_preds, dim=1)
                prev_pred = torch.cat(prev_preds, dim=1)
                preds.append(cur_pred)

            self.im_idx += 1
            self.conv_idx += 1

        if len(attns) > 0:
            return attns
        else:
            return preds[-1]
