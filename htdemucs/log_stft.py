#!/usr/bin/env python3

from typing import List, Tuple, Optional, Dict

import torch
import torchaudio

from einops import rearrange, reduce, repeat


class Log1pSTFT(torch.nn.Module):
    def __init__(self, channels = 2, n_fft: int = 4096, hop_length: int = None):
        super(Log1pSTFT, self).__init__()

        self.n_fft = n_fft
        if hop_length is None:
            hop_length = n_fft // 4
        self.hop_length = hop_length
        self.window = torch.hann_window(n_fft, periodic=False)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = rearrange(x, 'b c t -> (b c) t', b=batch_size, c=self.channels)

        freq = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window,
            return_complex=True, normalized=False, center=True, onesided=True
        )
        freq = torch.view_as_real(freq)
        freq = rearrange(freq, '(b c) freq len h -> b len (c freq h)', b=batch_size, c=self.channels, h=2)

        sgn_freq = torch.sign(freq)
        log_freq = torch.log1p(torch.abs(freq))
        output = sgn_freq * log_freq

        return output

    def restore(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = rearrange(x, 'b len (c freq h) -> (b c) freq len h', b=batch_size, c=self.channels, h=2)

        sgn_x = torch.sign(x)
        exp_x = torch.expm1(torch.abs(x))
        x = sgn_x * exp_x
        
        x = torch.view_as_complex(x)
        x = torch.istft(
            x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window,
            normalized=False, center=True, onesided=True
        )
        
        return rearrange(x, '(b c) t -> b c t', b=batch_size, c=self.channels)

if __name__ == "__main__":
    # Example usage
    x = torch.randn(4, 2, 2**20)  # (batch_size, channels, time)
    
    stft = Log1pSTFT(n_fft=2048, hop_length=512)
    y = stft(x)
    print(y.shape)  # Should print the shape of the log STFT output
    x_restore = stft.restore(y)
    print(x_restore.shape)

    assert x.shape == x_restore.shape, "Restored tensor shape does not match original"
    print(torch.abs(x - x_restore).max())  # Should be close to zero
