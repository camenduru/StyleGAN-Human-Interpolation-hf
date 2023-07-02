#!/usr/bin/env python

from __future__ import annotations

import functools
import pickle
import sys

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

sys.path.insert(0, 'StyleGAN-Human')

TITLE = 'StyleGAN-Human (Interpolation)'
DESCRIPTION = 'https://github.com/stylegan-human/StyleGAN-Human'


def load_model(file_name: str, device: torch.device) -> nn.Module:
    path = hf_hub_download('public-data/StyleGAN-Human', f'models/{file_name}')
    with open(path, 'rb') as f:
        model = pickle.load(f)['G_ema']
    model.eval()
    model.to(device)
    with torch.inference_mode():
        z = torch.zeros((1, model.z_dim)).to(device)
        label = torch.zeros([1, model.c_dim], device=device)
        model(z, label, force_fp32=True)
    return model


def generate_z(z_dim: int, seed: int, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.random.RandomState(seed).randn(
        1, z_dim)).to(device).float()


@torch.inference_mode()
def generate_interpolated_images(seed0: int, psi0: float, seed1: int,
                                 psi1: float, num_intermediate: int,
                                 model: nn.Module,
                                 device: torch.device) -> list[np.ndarray]:
    seed0 = int(np.clip(seed0, 0, np.iinfo(np.uint32).max))
    seed1 = int(np.clip(seed1, 0, np.iinfo(np.uint32).max))

    z0 = generate_z(model.z_dim, seed0, device)
    z1 = generate_z(model.z_dim, seed1, device)
    vec = z1 - z0
    dvec = vec / (num_intermediate + 1)
    zs = [z0 + dvec * i for i in range(num_intermediate + 2)]
    dpsi = (psi1 - psi0) / (num_intermediate + 1)
    psis = [psi0 + dpsi * i for i in range(num_intermediate + 2)]

    label = torch.zeros([1, model.c_dim], device=device)

    res = []
    for z, psi in zip(zs, psis):
        out = model(z, label, truncation_psi=psi, force_fp32=True)
        out = (out.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(
            torch.uint8)
        out = out[0].cpu().numpy()
        res.append(out)
    return res


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = load_model('stylegan_human_v2_1024.pkl', device)
fn = functools.partial(generate_interpolated_images,
                       model=model,
                       device=device)

gr.Interface(
    fn=fn,
    inputs=[
        gr.Slider(label='Seed 1',
                  minimum=0,
                  maximum=100000,
                  step=1,
                  value=0,
                  randomize=True),
        gr.Slider(label='Truncation psi 1',
                  minimum=0,
                  maximum=2,
                  step=0.05,
                  value=0.7),
        gr.Slider(label='Seed 2',
                  minimum=0,
                  maximum=100000,
                  step=1,
                  value=1,
                  randomize=True),
        gr.Slider(label='Truncation psi 2',
                  minimum=0,
                  maximum=2,
                  step=0.05,
                  value=0.7),
        gr.Slider(label='Number of Intermediate Frames',
                  minimum=0,
                  maximum=21,
                  step=1,
                  value=7),
    ],
    outputs=gr.Gallery(label='Output Images', type='numpy'),
    title=TITLE,
    description=DESCRIPTION,
).queue(max_size=10).launch()
