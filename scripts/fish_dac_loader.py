"""
Standalone Fish DAC (Firefly GAN VQ-FSQ) loader.

Reconstructs the FireflyArchitecture model from fishaudio/fish-speech source
components so we can load the raw .pth checkpoint without requiring the full
fish-speech package (which has heavy deps like pyaudio).

Architecture: ConvNeXtEncoder → DownsampleFiniteScalarQuantize → HiFiGANGenerator
Config from:  fishaudio/fish-speech-1.5/firefly_gan_vq.yaml
Source:       fishaudio/fish-speech (Apache-2.0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from math import prod
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
from vector_quantize_pytorch import GroupedResidualFSQ

# ─── Utility functions ───────────────────────────────────────────────


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv1D") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


def unpad1d(x: torch.Tensor, paddings: tuple[int, int]):
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


def get_extra_padding_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad1d(x: torch.Tensor, paddings: tuple[int, int], mode: str = "zeros", value: float = 0.0):
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


# ─── Conv primitives ────────────────────────────────────────────────


class FishConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, dilation=dilation, groups=groups)
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.dilation = dilation

    def forward(self, x):
        pad = self.kernel_size - self.stride
        extra_padding = get_extra_padding_for_conv1d(x, self.kernel_size, self.stride, pad)
        x = pad1d(x, (pad, extra_padding), mode="constant", value=0)
        return self.conv(x).contiguous()

    def weight_norm(self, name="weight", dim=0):
        self.conv = weight_norm(self.conv, name=name, dim=dim)
        return self

    def remove_parametrizations(self, name="weight"):
        self.conv = remove_parametrizations(self.conv, name)
        return self


class FishTransConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                                       stride=stride, dilation=dilation)
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
        x = self.conv(x)
        pad = self.kernel_size - self.stride
        padding_right = math.ceil(pad)
        padding_left = pad - padding_right
        x = unpad1d(x, (padding_left, padding_right))
        return x.contiguous()

    def weight_norm(self, name="weight", dim=0):
        self.conv = weight_norm(self.conv, name=name, dim=dim)
        return self

    def remove_parametrizations(self, name="weight"):
        self.conv = remove_parametrizations(self.conv, name)
        return self


# ─── Blocks ──────────────────────────────────────────────────────────


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6,
                 mlp_ratio=4.0, kernel_size=7, dilation=1):
        super().__init__()
        self.dwconv = FishConvNet(dim, dim, kernel_size=kernel_size, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, int(mlp_ratio * dim))
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0 else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, apply_residual=True):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1)
        x = self.drop_path(x)
        if apply_residual:
            x = input + x
        return x


class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            FishConvNet(channels, channels, kernel_size, stride=1, dilation=d).weight_norm()
            for d in dilation
        ])
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList([
            FishConvNet(channels, channels, kernel_size, stride=1, dilation=d).weight_norm()
            for d in dilation
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.silu(x)
            xt = c1(xt)
            xt = F.silu(xt)
            xt = c2(xt)
            x = xt + x
        return x


class ParallelBlock(nn.Module):
    def __init__(self, channels, kernel_sizes=(3, 7, 11),
                 dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5))):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResBlock1(channels, k, d) for k, d in zip(kernel_sizes, dilation_sizes)
        ])

    def forward(self, x):
        return torch.stack([block(x) for block in self.blocks], dim=0).mean(dim=0)


# ─── Encoder ─────────────────────────────────────────────────────────


class ConvNeXtEncoder(nn.Module):
    def __init__(self, input_channels=3, depths=(3, 3, 9, 3),
                 dims=(96, 192, 384, 768), drop_path_rate=0.0,
                 layer_scale_init_value=1e-6, kernel_size=7):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            FishConvNet(input_channels, dims[0], kernel_size=7),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(len(depths) - 1):
            mid_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv1d(dims[i], dims[i + 1], kernel_size=1),
            )
            self.downsample_layers.append(mid_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(len(depths)):
            stage = nn.Sequential(*[
                ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j],
                              layer_scale_init_value=layer_scale_init_value, kernel_size=kernel_size)
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

        self.norm = LayerNorm(dims[-1], eps=1e-6, data_format="channels_first")
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(len(self.downsample_layers)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x)


# ─── Decoder ─────────────────────────────────────────────────────────


class HiFiGANGenerator(nn.Module):
    def __init__(self, *, hop_length=512, upsample_rates=(8, 8, 2, 2, 2),
                 upsample_kernel_sizes=(16, 16, 8, 2, 2),
                 resblock_kernel_sizes=(3, 7, 11),
                 resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
                 num_mels=128, upsample_initial_channel=512,
                 pre_conv_kernel_size=7, post_conv_kernel_size=7,
                 post_activation: Callable = partial(nn.SiLU, inplace=True)):
        super().__init__()
        assert prod(upsample_rates) == hop_length
        self.conv_pre = FishConvNet(num_mels, upsample_initial_channel,
                                    pre_conv_kernel_size, stride=1).weight_norm()
        self.num_upsamples = len(upsample_rates)
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(FishTransConvNet(
                upsample_initial_channel // (2**i),
                upsample_initial_channel // (2**(i + 1)),
                k, stride=u,
            ).weight_norm())
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2**(i + 1))
            self.resblocks.append(ParallelBlock(ch, resblock_kernel_sizes, resblock_dilation_sizes))
        self.activation_post = post_activation()
        self.conv_post = FishConvNet(ch, 1, post_conv_kernel_size, stride=1).weight_norm()
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.silu(x, inplace=True)
            x = self.ups[i](x)
            x = self.resblocks[i](x)
        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x


# ─── FSQ Quantizer ───────────────────────────────────────────────────


@dataclass
class FSQResult:
    z: torch.Tensor
    codes: torch.Tensor
    latents: torch.Tensor


class DownsampleFiniteScalarQuantize(nn.Module):
    def __init__(self, input_dim=512, n_codebooks=9, n_groups=1,
                 levels=(8, 5, 5, 5), downsample_factor=(2, 2),
                 downsample_dims=None):
        super().__init__()
        if downsample_dims is None:
            downsample_dims = [input_dim for _ in range(len(downsample_factor))]
        all_dims = (input_dim,) + tuple(downsample_dims)

        self.residual_fsq = GroupedResidualFSQ(
            dim=all_dims[-1], levels=list(levels),
            num_quantizers=n_codebooks, groups=n_groups,
        )
        self.downsample_factor = downsample_factor
        self.downsample = nn.Sequential(*[
            nn.Sequential(
                FishConvNet(all_dims[idx], all_dims[idx + 1], kernel_size=factor, stride=factor),
                ConvNeXtBlock(dim=all_dims[idx + 1]),
            ) for idx, factor in enumerate(downsample_factor)
        ])
        self.upsample = nn.Sequential(*[
            nn.Sequential(
                FishTransConvNet(all_dims[idx + 1], all_dims[idx], kernel_size=factor, stride=factor),
                ConvNeXtBlock(dim=all_dims[idx]),
            ) for idx, factor in reversed(list(enumerate(downsample_factor)))
        ])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, z) -> FSQResult:
        original_shape = z.shape
        z = self.downsample(z)
        quantized, indices = self.residual_fsq(z.mT)
        result = FSQResult(z=quantized.mT, codes=indices.mT, latents=z)
        result.z = self.upsample(result.z)
        diff = original_shape[-1] - result.z.shape[-1]
        left = diff // 2
        right = diff - left
        if diff > 0:
            result.z = F.pad(result.z, (left, right))
        elif diff < 0:
            result.z = result.z[..., -left:right]
        return result

    def encode(self, z):
        z = self.downsample(z)
        _, indices = self.residual_fsq(z.mT)
        indices = rearrange(indices, "g b l r -> b (g r) l")
        return indices

    def decode(self, indices):
        indices = rearrange(indices, "b (g r) l -> g b l r", g=self.residual_fsq.groups)
        z_q = self.residual_fsq.get_output_from_indices(indices)
        z_q = self.upsample(z_q.mT)
        return z_q


# ─── Spectrogram ─────────────────────────────────────────────────────


class LinearSpectrogram(nn.Module):
    def __init__(self, n_fft=2048, win_length=2048, hop_length=512, center=False, mode="pow2_sqrt"):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.mode = mode
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def forward(self, y):
        if y.ndim == 3:
            y = y.squeeze(1)
        y = F.pad(
            y.unsqueeze(1),
            ((self.win_length - self.hop_length) // 2, (self.win_length - self.hop_length + 1) // 2),
            mode="reflect",
        ).squeeze(1)
        spec = torch.stft(y, self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                          window=self.window, center=self.center, pad_mode="reflect",
                          normalized=False, onesided=True, return_complex=True)
        spec = torch.view_as_real(spec)
        if self.mode == "pow2_sqrt":
            spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
        return spec


class LogMelSpectrogram(nn.Module):
    def __init__(self, sample_rate=44100, n_fft=2048, win_length=2048,
                 hop_length=512, n_mels=128, center=False, f_min=0.0, f_max=None):
        super().__init__()
        import torchaudio.functional as AF
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or float(sample_rate // 2)
        self.spectrogram = LinearSpectrogram(n_fft, win_length, hop_length, center)
        fb = AF.melscale_fbanks(
            n_freqs=self.n_fft // 2 + 1, f_min=self.f_min, f_max=self.f_max,
            n_mels=self.n_mels, sample_rate=self.sample_rate,
            norm="slaney", mel_scale="slaney",
        )
        self.register_buffer("fb", fb, persistent=False)

    def forward(self, x, return_linear=False, sample_rate=None):
        import torchaudio.functional as AF
        if sample_rate is not None and sample_rate != self.sample_rate:
            x = AF.resample(x, orig_freq=sample_rate, new_freq=self.sample_rate)
        linear = self.spectrogram(x)
        x = torch.matmul(linear.transpose(-1, -2), self.fb).transpose(-1, -2)
        x = torch.log(torch.clamp(x, min=1e-5))
        if return_linear:
            return x, torch.log(torch.clamp(linear, min=1e-5))
        return x


# ─── Top-level Architecture ─────────────────────────────────────────


class FireflyArchitecture(nn.Module):
    def __init__(self, backbone, head, quantizer, spec_transform):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.quantizer = quantizer
        self.spec_transform = spec_transform
        self.downsample_factor = math.prod(self.quantizer.downsample_factor)

    def encode(self, audios, audio_lengths=None):
        audios = audios.float()
        mels = self.spec_transform(audios)
        if audio_lengths is not None:
            mel_lengths = audio_lengths // self.spec_transform.hop_length
            mel_masks = sequence_mask(mel_lengths, mels.shape[2])
            mels = mels * mel_masks[:, None, :].float()
        encoded_features = self.backbone(mels)
        return self.quantizer.encode(encoded_features)

    def decode(self, indices, feature_lengths=None):
        z = self.quantizer.decode(indices)
        x = self.head(z)
        return x

    def from_indices(self, indices):
        z = self.quantizer.decode(indices)
        return self.head(z)


# ─── Loader ──────────────────────────────────────────────────────────

# Config from fishaudio/fish-speech-1.5/firefly_gan_vq.yaml
FISH_SPEECH_15_CONFIG = dict(
    spec_transform=dict(sample_rate=44100, n_mels=160, n_fft=2048, hop_length=512, win_length=2048),
    backbone=dict(input_channels=160, depths=[3, 3, 9, 3], dims=[128, 256, 384, 512],
                  drop_path_rate=0.2, kernel_size=7),
    head=dict(hop_length=512, upsample_rates=[8, 8, 2, 2, 2],
              upsample_kernel_sizes=[16, 16, 4, 4, 4],
              resblock_kernel_sizes=[3, 7, 11],
              resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
              num_mels=512, upsample_initial_channel=512,
              pre_conv_kernel_size=13, post_conv_kernel_size=13),
    quantizer=dict(input_dim=512, n_groups=8, n_codebooks=1,
                   levels=[8, 5, 5, 5], downsample_factor=[2, 2]),
)


def load_fish_dac(checkpoint_path: str, device: str = "cpu") -> FireflyArchitecture:
    """Load Fish DAC model from a .pth state dict checkpoint."""
    cfg = FISH_SPEECH_15_CONFIG

    spec_transform = LogMelSpectrogram(**cfg["spec_transform"])
    backbone = ConvNeXtEncoder(**cfg["backbone"])
    head = HiFiGANGenerator(**cfg["head"])
    quantizer = DownsampleFiniteScalarQuantize(**cfg["quantizer"])

    model = FireflyArchitecture(backbone, head, quantizer, spec_transform)

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    if device == "mps" and torch.backends.mps.is_available():
        model = model.to("mps")
    elif device == "cuda" and torch.cuda.is_available():
        model = model.to(device)

    return model
