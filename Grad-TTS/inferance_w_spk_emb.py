# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import argparse
import datetime as dt
import json
import os
import sys
import tempfile

import numpy as np

# import params
import params_multi_w_spk_emb as params
import torch
import torchaudio as ta
from model import GradTTS
from scipy.io.wavfile import write
from text import cmudict, text_to_sequence
from text.symbols import symbols
from utils import intersperse

from TTS.tts.utils.speakers import SpeakerManager

sys.path.append("./hifi-gan/")
from env import AttrDict
from models import Generator as HiFiGAN

# HIFIGAN_CONFIG = "./checkpts/hifigan-config.json"
# HIFIGAN_CHECKPT = "./checkpts/hifigan.pt"

HIFIGAN_CONFIG = "./checkpts/UNIVERSAL_V1/config.json"
HIFIGAN_CHECKPT = "./checkpts/UNIVERSAL_V1/hifigan.pt"


def get_spk_emb(enc_manager, filepath):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = os.path.join(tmpdir, "temp.wav")
        audio, sr = ta.load(filepath)
        audio = ta.functional.resample(audio, orig_freq=sr, new_freq=16000)
        ta.save(tmpfile, audio, 16000, encoding="PCM_S", bits_per_sample=16)
        spk_emb = (
            torch.tensor(enc_manager.compute_embedding_from_clip(tmpfile))
            .to(torch.float32)
            .unsqueeze(0)
            .cuda()
        )
    return spk_emb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=True,
        help="path to a file with texts to synthesize",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        required=True,
        help="path to a checkpoint of Grad-TTS",
    )
    parser.add_argument(
        "-t",
        "--timesteps",
        type=int,
        required=False,
        default=10,
        help="number of timesteps of reverse diffusion",
    )
    parser.add_argument(
        "-r",
        "--ref_wav",
        type=str,
        required=False,
        default=None,
        help="speaker id for multispeaker model",
    )
    args = parser.parse_args()

    enc_manager = SpeakerManager(
        encoder_model_path=params.spk_enc_model_path,
        encoder_config_path=params.spk_enc_model_config_path,
        use_cuda=True,
    )

    spk = get_spk_emb(enc_manager, filepath=args.ref_wav)
    print("Initializing Grad-TTS...")
    generator = GradTTS(
        len(symbols) + 1,
        params.n_spks,
        params.spk_emb_dim,
        params.n_enc_channels,
        params.filter_channels,
        params.filter_channels_dp,
        params.n_heads,
        params.n_enc_layers,
        params.enc_kernel,
        params.enc_dropout,
        params.window_size,
        params.n_feats,
        params.dec_dim,
        params.beta_min,
        params.beta_max,
        params.pe_scale,
    )
    generator.load_state_dict(
        torch.load(args.checkpoint, map_location=lambda loc, storage: loc)
    )
    _ = generator.cuda().eval()
    print(f"Number of parameters: {generator.nparams}")

    print("Initializing HiFi-GAN...")
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(
        torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)["generator"]
    )
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()

    with open(args.file, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f.readlines()]
    cmu = cmudict.CMUDict("./resources/cmu_dictionary")

    with torch.no_grad():
        for i, text in enumerate(texts):
            print(f"Synthesizing {i} text...", end=" ")
            x = torch.LongTensor(
                intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))
            ).cuda()[None]
            x_lengths = torch.LongTensor([x.shape[-1]]).cuda()

            t = dt.datetime.now()
            y_enc, y_dec, attn = generator.forward(
                x,
                x_lengths,
                n_timesteps=args.timesteps,
                temperature=1.5,
                stoc=False,
                spk=spk,
                length_scale=0.91,
            )
            t = (dt.datetime.now() - t).total_seconds()
            print(f"Grad-TTS RTF: {t * 22050 / (y_dec.shape[-1] * 256)}")

            audio = (
                vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768
            ).astype(np.int16)

            write(f"./out/sample_{i}.wav", 22050, audio)

    print("Done. Check out `out` folder for samples.")
