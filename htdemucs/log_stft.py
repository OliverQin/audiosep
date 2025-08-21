#!/usr/bin/env python3

import torch

from einops import rearrange

@torch.compile()
class Log1pSTFT(torch.nn.Module):
    def __init__(self, channels = 2, n_fft: int = 4096, hop_length: int = None):
        super(Log1pSTFT, self).__init__()

        self.n_fft = n_fft
        if hop_length is None:
            hop_length = n_fft // 4
        self.hop_length = hop_length
        self.window = torch.hann_window(n_fft, periodic=False)
        self.channels = channels

    @torch.compiler.disable()
    def _in_stft(self, x: torch.Tensor) -> torch.Tensor:
        spct = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window,
            return_complex=True, normalized=False, center=True, onesided=True
        )
        return torch.view_as_real(spct)

    @torch.compiler.disable()
    def _in_istft(self, x: torch.Tensor) -> torch.Tensor:
        if x.stride()[-1] != 1:
            x = x.contiguous()
        x = torch.view_as_complex(x)
        return torch.istft(
            x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window,
            normalized=False, center=True, onesided=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = rearrange(x, 'b c len -> (b c) len', b=batch_size, c=self.channels)

        spct = self._in_stft(x)
        spct = rearrange(spct, '(b c) freq len h -> b (c freq h) len', b=batch_size, c=self.channels, h=2)

        sgn_spct = torch.sign(spct)
        log_spct = torch.log1p(torch.abs(spct))
        output = sgn_spct * log_spct

        return output

    def restore(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = rearrange(x, 'b (c freq h) len -> (b c) freq len h', b=batch_size, c=self.channels, h=2)

        sgn_x = torch.sign(x)
        exp_x = torch.expm1(torch.abs(x))
        x = sgn_x * exp_x
        
        x = self._in_istft(x)
        
        return rearrange(x, '(b c) len -> b c len', b=batch_size, c=self.channels)

if __name__ == "__main__":
    # Example usage
    x = torch.randn(4, 2, 2**20)  # (batch_size, channels, time)
    
    stft = Log1pSTFT(n_fft=2048, hop_length=512)
    y = stft(x)
    print(y.shape)  # Should print the shape of the log STFT output
    x_restore = stft.restore(y)
    print(x_restore.shape)

    assert x.shape == x_restore.shape, "Restored tensor shape does not match original"
    print("Mean L1 of x", torch.abs(x).mean())
    print("Max difference after restoration:", torch.abs(x - x_restore).max())  # Should be close to zero
