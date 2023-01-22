import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import params
import torch
from model import GradTTS
from scipy.io.wavfile import write
from text import cmudict, ja_text_to_sequence, text_to_sequence
from text.symbols import symbols
from utils import intersperse

sys.path.append("./hifi-gan/")
from env import AttrDict
from models import Generator as HiFiGAN

HIFIGAN_CONFIG = "./checkpts/UNIVERSAL_V1/config.json"
HIFIGAN_CHECKPT = "./checkpts/UNIVERSAL_V1/hifigan.pt"


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
        "-s",
        "--speaker_id",
        type=int,
        required=False,
        default=None,
        help="speaker id for multispeaker model",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=False,
        default="out_jsut",
        help="output dir",
    )
    args = parser.parse_args()

    if not isinstance(args.speaker_id, type(None)):
        assert (
            params.n_spks > 1
        ), "Ensure you set right number of speakers in `params.py`."
        spk = torch.LongTensor([args.speaker_id]).cuda()
    else:
        spk = None

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
    token_list_path = (
        "/data/Speech-Backbones/Grad-TTS/resources/filelists/jsut/tokens.txt"
    )

    with torch.no_grad():
        for i, text in enumerate(texts):
            print(f"Synthesizing {i} text...", end=" ")
            x = torch.LongTensor(
                ja_text_to_sequence(text, transcript_token_list=token_list_path)
                # intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))
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
            os.makedirs(args.output_dir, exist_ok=True)
            write(f"{args.output_dir}/sample_{i}.wav", 22050, audio)

    print("Done. Check out `out` folder for samples.")
