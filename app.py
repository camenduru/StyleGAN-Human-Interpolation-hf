#!/usr/bin/env python

from __future__ import annotations

import argparse
import functools
import os
import pickle
import sys

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

sys.path.insert(0, 'StyleGAN-Human')

TITLE = 'StyleGAN-Human (Interpolation)'
DESCRIPTION = '''This is an unofficial demo for https://github.com/stylegan-human/StyleGAN-Human.

Related App: [StyleGAN-Human](https://huggingface.co/spaces/hysts/StyleGAN-Human)
'''
ARTICLE = None

TOKEN = os.environ['TOKEN']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    parser.add_argument('--allow-flagging', type=str, default='never')
    parser.add_argument('--allow-screenshot', action='store_true')
    return parser.parse_args()


def load_model(file_name: str, device: torch.device) -> nn.Module:
    path = hf_hub_download('hysts/StyleGAN-Human',
                           f'models/{file_name}',
                           use_auth_token=TOKEN)
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
def generate_interpolated_images(
        seed0: int, psi0: float, seed1: int, psi1: float,
        num_intermediate: int, model: nn.Module,
        device: torch.device) -> tuple[list[np.ndarray], np.ndarray]:
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
    concatenated = np.hstack(res)
    return res, concatenated


def main():
    args = parse_args()
    device = torch.device(args.device)

    model = load_model('stylegan_human_v2_1024.pkl', device)

    func = functools.partial(generate_interpolated_images,
                             model=model,
                             device=device)
    func = functools.update_wrapper(func, generate_interpolated_images)

    gr.Interface(
        func,
        [
            gr.inputs.Number(default=0, label='Seed 1'),
            gr.inputs.Slider(
                0, 2, step=0.05, default=0.7, label='Truncation psi 1'),
            gr.inputs.Number(default=1, label='Seed 2'),
            gr.inputs.Slider(
                0, 2, step=0.05, default=0.7, label='Truncation psi 2'),
            gr.inputs.Slider(0,
                             21,
                             step=1,
                             default=7,
                             label='Number of Intermediate Frames'),
        ],
        [
            gr.outputs.Carousel(gr.outputs.Image(type='numpy'),
                                label='Output Images'),
            gr.outputs.Image(type='numpy', label='Concatenated'),
        ],
        title=TITLE,
        description=DESCRIPTION,
        article=ARTICLE,
        theme=args.theme,
        allow_screenshot=args.allow_screenshot,
        allow_flagging=args.allow_flagging,
        live=args.live,
    ).launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
