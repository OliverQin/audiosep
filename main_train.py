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

    # load_model_ckpt(model, opt, scaler, "ckpt_recorder_v4/epoch_029.pt", device)
    for epoch in range(1, epochs + 1):
        train_ds.reset_seed()
        model.train()
        running_loss = 0.0
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
                loss = l1_loss(pred_music, music) + l1_loss(pred_noise, noise) + 0.5 * l1_loss(pred_music + pred_noise, mix)

            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (pbar.n or 1))

        eval_ds.reset_seed()
        metrics = evaluate(model, eval_loader, device)
        print(f"[Epoch {epoch}] loss={running_loss/len(train_loader):.4f} | "
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
    music_dirs = ["./dataset/flute_recorder/wavs", "./dataset/flute_recorder_fake/wavs", "./dataset/flute_recorder_homebrew/wavs"]
    noise_dirs = ["./dataset/classical_nowoodwind/wavs"]

    model = HTDemucs(['music', 'accompany'], t_layers=7)

    music_aug = Augmenter(gain_db_range=(-3, 3), swap_channels_prob=0.5, eq_prob=0.2, rnd_noise_prob=0.02)
    noise_aug = Augmenter(gain_db_range=(-3, 3), swap_channels_prob=0.5, eq_prob=0.2, rnd_noise_prob=0.0)

    train_ds = BaroqueNoiseDataset(
        music_dirs, noise_dirs,
        rirs_path='./rirs',
        seg_len=2**19,
        mode='train',
        seed=int(time.time()),
        music_augment=music_aug,
        noise_augment=noise_aug
    )

    eval_ds = BaroqueNoiseDataset(
        music_dirs, noise_dirs,
        rirs_path=None,
        seg_len=2**19,
        mode='eval',
        music_augment=None,
        noise_augment=None
    )

    train_loop(model, train_ds, eval_ds, save_dir="ckpt_recorder_v7",
               epochs=45, batch_size=4, lr=3e-4, betas=(0.9, 0.999),
               num_workers=4, device="cuda", grad_clip=None, amp=True)




