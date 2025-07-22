#!/usr/bin/env python3
# 
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Conveniance wrapper to perform STFT and iSTFT"""

import torch


def spectro(x, n_fft=512, hop_length=None, pad=0):
    *other, length = x.shape
    x = x.reshape(-1, length)
    is_mps_xpu = x.device.type in ['mps', 'xpu']
    if is_mps_xpu:
        x = x.cpu()
    z = torch.stft(x,
                    n_fft * (1 + pad),
                    hop_length or n_fft // 4,
                    window=torch.hann_window(n_fft).to(x),
                    win_length=n_fft,
                    normalized=True,
                    center=True,
                    return_complex=True,
                    pad_mode='reflect')
    _, freqs, frame = z.shape
    return z.view(*other, freqs, frame)


def ispectro(z, hop_length=None, length=None, pad=0):
    *other, freqs, frames = z.shape
    n_fft = 2 * freqs - 2
    z = z.view(-1, freqs, frames)
    win_length = n_fft // (1 + pad)
    is_mps_xpu = z.device.type in ['mps', 'xpu']
    if is_mps_xpu:
        z = z.cpu()
    x = torch.istft(z,
                    n_fft,
                    hop_length,
                    window=torch.hann_window(win_length).to(z.real),
                    win_length=win_length,
                    normalized=True,
                    length=length,
                    center=True)
    _, length = x.shape
    return x.view(*other, length)
