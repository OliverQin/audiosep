#!/usr/bin/env python3

import os
import time
import soundfile
import torch

from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from pathlib import Path
from htdemucs import *

from tqdm import tqdm

def _dump_batch(batch, out_dir="debug_out", sr=44100):
    os.makedirs(out_dir, exist_ok=True)
    mix   = batch["mix"]    # [B, C, T], float32 tensor
    music = batch["music"]
    noise = batch["noise"]

    B = mix.size(0)
    for i in range(B):
        for name, tensor in [("mix", mix[i]), ("music", music[i]), ("noise", noise[i])]:
            wav = tensor.detach().cpu().numpy().T  # (T, C)
            fn  = f"{i:06d}_{name}.wav"
            soundfile.write(os.path.join(out_dir, fn), wav, sr, subtype="FLOAT")

def l1_loss(pred, tgt):
    return (pred - tgt).abs().mean()

def hann_weighted_l1_loss(pred: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if pred.shape != tgt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs tgt {tgt.shape}")
    
    B, C, L = pred.shape
    window = torch.hann_window(L, periodic=False, device=pred.device, dtype=pred.dtype, requires_grad=False)
    window += eps
    window = window.view(1, 1, L)

    diff = (pred - tgt).abs()
    weighted_diff = diff * window

    return weighted_diff.sum() / (window.sum() * B * C)

@torch.compile(mode="default")
def high_freq_loss(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    B, C, L = pred.shape

    pred = pred.reshape(-1, L)
    tgt = tgt.reshape(-1, L)

    win = 2048
    cut = 400
    mask = torch.hann_window(win, device=pred.device, dtype=pred.dtype, requires_grad=False)
    pred_freq = torch.stft(pred, n_fft=win, hop_length=win//4, win_length=win, window=mask, center=False, normalized=True, onesided=True, return_complex=True)
    tgt_freq = torch.stft(tgt, n_fft=win, hop_length=win//4, win_length=win, window=mask, center=False, normalized=True, onesided=True, return_complex=True)
    
    pred_freq = torch.view_as_real(pred_freq)
    tgt_freq = torch.view_as_real(tgt_freq)

    _, n_freq, n_frames, _2  = pred_freq.shape
    assert n_freq == (win // 2 + 1), f"Expected n_freq={win // 2 + 1}, got {n_freq}"

    out_win = torch.hann_window(n_frames, device=pred.device, dtype=pred.dtype, requires_grad=False)
    out_win /= out_win.sum()
    out_win = out_win.view(1, 1, n_frames, 1)
    pred_freq = pred_freq[:, cut:, :, :]
    tgt_freq = tgt_freq[:, cut:, :, :]

    return (out_win * (pred_freq - tgt_freq).abs()).sum() / (B * C * 2 * (n_freq - cut) * n_frames)

@torch.compile(mode="default")
def low_freq_loss(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    B, C, L = pred.shape

    pred = pred.reshape(-1, L)
    tgt = tgt.reshape(-1, L)

    win = 2048
    cut = 450
    mask = torch.hann_window(win, device=pred.device, dtype=pred.dtype, requires_grad=False)
    pred_freq = torch.stft(pred, n_fft=win, hop_length=win//4, win_length=win, window=mask, center=False, normalized=True, onesided=True, return_complex=True)
    tgt_freq = torch.stft(tgt, n_fft=win, hop_length=win//4, win_length=win, window=mask, center=False, normalized=True, onesided=True, return_complex=True)
    
    pred_freq = torch.view_as_real(pred_freq)
    tgt_freq = torch.view_as_real(tgt_freq)

    _, n_freq, n_frames, _2  = pred_freq.shape
    assert n_freq == (win // 2 + 1), f"Expected n_freq={win // 2 + 1}, got {n_freq}"

    out_win = torch.hann_window(n_frames, device=pred.device, dtype=pred.dtype, requires_grad=False)
    out_win /= out_win.sum()
    out_win = out_win.view(1, 1, n_frames, 1)
    pred_freq = pred_freq[:, :cut, :, :]
    tgt_freq = tgt_freq[:, :cut, :, :]

    return (out_win * (pred_freq - tgt_freq).abs()).sum() / (B * C * 2 * (n_freq - cut) * n_frames)


def si_sdr(estimate, target, eps=1e-8):
    # estimate/target: (B, C, T)
    B = target.size(0)
    t = target.view(B, -1)
    e = estimate.view(B, -1)
    t_energy = (t ** 2).sum(dim=1, keepdim=True) + eps
    proj = ((e * t).sum(dim=1, keepdim=True) / t_energy) * t
    noise = e - proj
    ratio = (proj ** 2).sum(dim=1) / ((noise ** 2).sum(dim=1) + eps)
    return 10 * torch.log10(ratio + eps)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    sdr_music_all, sdr_noise_all = [], []
    for batch in tqdm(loader, desc="Eval", leave=False):
        mix   = batch["mix"].to(device)     # (B,C,T)
        music = batch["music"].to(device)
        noise = batch["noise"].to(device)

        out = model(mix)                    # (B,2,C,T)
        pred_music = out[:, 0]              # (B,C,T)
        pred_noise = out[:, 1]

        sdr_music_all.append(si_sdr(pred_music, music).cpu())
        sdr_noise_all.append(si_sdr(pred_noise, noise).cpu())

    return {
        "SI-SDR_music": torch.cat(sdr_music_all).mean().item(),
        "SI-SDR_noise": torch.cat(sdr_noise_all).mean().item()
    }

def load_model_ckpt(model, optimizer, scaler, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scaler.load_state_dict(checkpoint["scaler_state"])
    return 

def train_loop(model, train_ds, eval_ds, save_dir="ckpt",
               epochs=10, batch_size=8, lr=3e-4, betas=(0.9, 0.999),
               num_workers=4, device="cuda", grad_clip=None, amp=True):

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    eval_loader  = DataLoader(eval_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, drop_last=True)

    opt = optim.Adam(model.parameters(), lr=lr, betas=betas)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # load_model_ckpt(model, opt, scaler, "ckpt_short_loss_v1/epoch_004.pt", device)
    for epoch in range(1, epochs + 1):
        train_ds.reset_seed()
        model.train()
        running_loss = 0.0
        running_freq_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

        for batch in pbar:
            mix   = batch["mix"].to(device)     # (B,C,T)
            music = batch["music"].to(device)
            noise = batch["noise"].to(device)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp):
                out = model(mix)                # (B,2,C,T)
                pred_music = out[:, 0]
                pred_noise = out[:, 1]

                l1_loss = hann_weighted_l1_loss(pred_music, music) + hann_weighted_l1_loss(pred_noise, noise)
                freq_loss = 5e3 * (low_freq_loss(pred_music, music) + low_freq_loss(pred_noise, noise))
                loss = l1_loss + freq_loss

            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            running_loss += loss.item()
            running_freq_loss += freq_loss.item()
            pbar.set_postfix(loss=running_loss / (pbar.n or 1), freq_loss=running_freq_loss / (pbar.n or 1))

        eval_ds.reset_seed()
        metrics = evaluate(model, eval_loader, device)
        print(f"[Epoch {epoch}] loss={running_loss/len(train_loader):.4f} | "
              f"freq_loss={running_freq_loss/len(train_loader):.4f} | "
              f"SI-SDR_music={metrics['SI-SDR_music']:.2f} | "
              f"SI-SDR_noise={metrics['SI-SDR_noise']:.2f}")

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "scaler_state": scaler.state_dict(),
            "metrics": metrics
        }, os.path.join(save_dir, f"epoch_{epoch:03d}.pt"))

if __name__ == '__main__':
    music_dirs = ["./dataset/flute_recorder_sopranino/wavs", "./dataset/flute_recorder_sopranino2/wavs", "./dataset/flute_recorder_homebrew/wavs"]
    noise_dirs = ["./dataset/classical_nowoodwind/wavs", "./dataset/baroque_live_nowoodwind/wavs"]

    model = HTDemucs(['music', 'accompany'], t_layers=7)

    train_ds = BaroqueNoiseDataset(
        music_dirs, noise_dirs,
        rirs_path='./rirs',
        seg_len=2**18,
        mode='train',
        swap_channels_prob=0.5,
        speed_change_range=(0.8, 1.2),
    )

    eval_ds = BaroqueNoiseDataset(
        music_dirs, noise_dirs,
        rirs_path=None,
        seg_len=2**18,
        mode='eval',
        swap_channels_prob=0.0,
        speed_change_range=None,  # No speed change in eval
    )

    train_loop(model, train_ds, eval_ds, save_dir="ckpt_short_low_v1",
               epochs=50, batch_size=4, lr=3e-4, betas=(0.9, 0.999),
               num_workers=8, device="cuda", grad_clip=None, amp=True)




