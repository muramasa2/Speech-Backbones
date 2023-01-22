# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import os

import numpy as np
import params_multi_w_spk_emb as params
import torch
from model import GradTTS
from text.symbols import symbols
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import plot_tensor, save_plot

from data import TextMelSpekEmbDataset, TextMelSpkEmbBatchCollate
from TTS.tts.utils.speakers import SpeakerManager

train_filelist_path = params.train_filelist_path
valid_filelist_path = params.valid_filelist_path
cmudict_path = params.cmudict_path
add_blank = params.add_blank
n_spks = params.n_spks
spk_emb_dim = params.spk_emb_dim

log_dir = params.log_dir
n_epochs = params.n_epochs
batch_size = params.batch_size
out_size = params.out_size
learning_rate = params.learning_rate
random_seed = params.seed

nsymbols = len(symbols) + 1 if add_blank else len(symbols)
n_enc_channels = params.n_enc_channels
filter_channels = params.filter_channels
filter_channels_dp = params.filter_channels_dp
n_enc_layers = params.n_enc_layers
enc_kernel = params.enc_kernel
enc_dropout = params.enc_dropout
n_heads = params.n_heads
window_size = params.window_size

n_feats = params.n_feats
n_fft = params.n_fft
sample_rate = params.sample_rate
hop_length = params.hop_length
win_length = params.win_length
f_min = params.f_min
f_max = params.f_max

dec_dim = params.dec_dim
beta_min = params.beta_min
beta_max = params.beta_max
pe_scale = params.pe_scale
train_wav_to_spk_emb_path = params.train_wav_to_spk_emb_path
valid_wav_to_spk_emb_path = params.valid_wav_to_spk_emb_path


def train_one_epoch(iteration, epoch, model, loader, train_dataset, batch_size):
    model.train()

    dur_losses = []
    prior_losses = []
    diff_losses = []
    with tqdm(loader, total=len(train_dataset) // batch_size) as progress_bar:
        for batch_idx, batch in enumerate(progress_bar):
            model.zero_grad()
            x, x_lengths = batch["x"].cuda(), batch["x_lengths"].cuda()
            y, y_lengths = batch["y"].cuda(), batch["y_lengths"].cuda()
            spk_emb = batch["spk_emb"].cuda()

            dur_loss, prior_loss, diff_loss = model.compute_loss(
                x, x_lengths, y, y_lengths, spk=spk_emb, out_size=out_size
            )

            loss = sum([dur_loss, prior_loss, diff_loss])
            loss.backward()
            optimizer.step()

            dur_losses.append(dur_loss.item())
            prior_losses.append(prior_loss.item())
            diff_losses.append(diff_loss.item())

            if batch_idx % 5 == 0:
                msg = f"Epoch: {epoch}, iteration: {iteration} | dur_loss: {dur_loss.item()}, prior_loss: {prior_loss.item()}, diff_loss: {diff_loss.item()}"
                progress_bar.set_description(msg)

            iteration += 1

    log_msg = "Epoch %d: duration loss = %.3f " % (epoch, np.mean(dur_losses))
    log_msg += "| prior loss = %.3f " % np.mean(prior_losses)
    log_msg += "| diffusion loss = %.3f\n" % np.mean(diff_losses)

    logger.add_scalars(
        "duration_loss", {"train": np.mean(dur_losses)}, global_step=epoch
    )
    logger.add_scalars(
        "prior_loss", {"train": np.mean(prior_losses)}, global_step=epoch
    )
    logger.add_scalars(
        "diffusion_loss",
        {"train": np.mean(diff_losses)},
        global_step=epoch,
    )

    with open(f"{log_dir}/train.log", "a") as f:
        f.write(log_msg)


def validate(iteration, epoch, model, loader, test_dataset, batch_size):
    model.eval()

    losses = []
    dur_losses = []
    prior_losses = []
    diff_losses = []
    with torch.no_grad():
        with tqdm(loader, total=len(test_dataset) // batch_size) as progress_bar:
            for batch in progress_bar:
                model.zero_grad()
                x, x_lengths = batch["x"].cuda(), batch["x_lengths"].cuda()
                y, y_lengths = batch["y"].cuda(), batch["y_lengths"].cuda()
                spk_emb = batch["spk_emb"].cuda()

                dur_loss, prior_loss, diff_loss = model.compute_loss(
                    x, x_lengths, y, y_lengths, spk=spk_emb, out_size=out_size
                )

                loss = sum([dur_loss, prior_loss, diff_loss])

                losses.append(loss.item())
                dur_losses.append(dur_loss.item())
                prior_losses.append(prior_loss.item())
                diff_losses.append(diff_loss.item())

                iteration += 1

        log_msg = "[Eval] Epoch %d: duration loss = %.3f " % (
            epoch,
            np.mean(dur_losses),
        )
        log_msg += "| prior loss = %.3f " % np.mean(prior_losses)
        log_msg += "| diffusion loss = %.3f\n" % np.mean(diff_losses)

        logger.add_scalars(
            "duration_loss", {"val": np.mean(dur_losses)}, global_step=epoch
        )
        logger.add_scalars(
            "prior_loss", {"val": np.mean(prior_losses)}, global_step=epoch
        )
        logger.add_scalars(
            "diffusion_loss",
            {"val": np.mean(diff_losses)},
            global_step=epoch,
        )
        with open(f"{log_dir}/train.log", "a") as f:
            f.write(log_msg)

        return np.mean(losses)


def check_early_stopping(
    patience: int,
    best_epoch: int,
    epoch: int = None,
) -> bool:
    if epoch - best_epoch > patience:
        log_msg = f"[Early stopping] loss has not been improved {epoch - best_epoch} epochs continuously. \
                    The training was stopped at {epoch}epoch"
        with open(f"{log_dir}/train.log", "a") as f:
            f.write(log_msg)
        return True
    else:
        return False


if __name__ == "__main__":
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print("Initializing logger...")
    logger = SummaryWriter(log_dir=log_dir)

    print("Initializing data loaders...")
    batch_collate = TextMelSpkEmbBatchCollate()
    train_dataset = TextMelSpekEmbDataset(
        train_filelist_path,
        cmudict_path,
        add_blank,
        n_fft,
        n_feats,
        sample_rate,
        hop_length,
        win_length,
        f_min,
        f_max,
        train_wav_to_spk_emb_path,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        collate_fn=batch_collate,
        drop_last=True,
        num_workers=4,
        shuffle=True,
    )
    test_dataset = TextMelSpekEmbDataset(
        valid_filelist_path,
        cmudict_path,
        add_blank,
        n_fft,
        n_feats,
        sample_rate,
        hop_length,
        win_length,
        f_min,
        f_max,
        valid_wav_to_spk_emb_path,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        collate_fn=batch_collate,
        drop_last=True,
        num_workers=4,
        shuffle=True,
    )
    print("Initializing model...")

    model = GradTTS(
        nsymbols,
        n_spks,
        spk_emb_dim,
        n_enc_channels,
        filter_channels,
        filter_channels_dp,
        n_heads,
        n_enc_layers,
        enc_kernel,
        enc_dropout,
        window_size,
        n_feats,
        dec_dim,
        beta_min,
        beta_max,
        pe_scale,
    ).cuda()
    print(
        "Number of encoder + duration predictor parameters: %.2fm"
        % (model.encoder.nparams / 1e6)
    )
    print("Number of decoder parameters: %.2fm" % (model.decoder.nparams / 1e6))
    print("Total parameters: %.2fm" % (model.nparams / 1e6))

    print("Initializing optimizer...")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    print("Logging test batch...")
    test_batch = test_dataset.sample_test_batch(size=params.test_size)
    for item in test_batch:
        mel, spk_emb = item["y"], item["spk_emb"]
        # i = int(spk.cpu())
        logger.add_image(
            f"image_1/ground_truth",
            plot_tensor(mel.squeeze()),
            global_step=0,
            dataformats="HWC",
        )
        save_plot(mel.squeeze(), f"{log_dir}/original_1.png")

    print("Start training...")
    iteration = 0
    best_val_loss = None
    best_epoch = 0
    patience = 100
    for epoch in range(1, n_epochs + 1):
        train_one_epoch(
            iteration, epoch, model, train_loader, train_dataset, batch_size
        )
        val_loss = validate(
            iteration, epoch, model, test_loader, test_dataset, batch_size
        )

        if best_val_loss is None:
            best_val_loss = val_loss

        elif best_val_loss > val_loss:
            print("Synthesis...")
            for i, item in enumerate(test_batch):
                x = item["x"].to(torch.long).unsqueeze(0).cuda()
                x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
                spk_emb = item["spk_emb"].to(torch.float32).unsqueeze(0).cuda()

                y_enc, y_dec, attn = model(x, x_lengths, n_timesteps=50, spk=spk_emb)
                logger.add_image(
                    f"image_{i}/generated_enc",
                    plot_tensor(y_enc.squeeze().cpu()),
                    global_step=epoch,
                    dataformats="HWC",
                )
                logger.add_image(
                    f"image_{i}/generated_dec",
                    plot_tensor(y_dec.squeeze().cpu()),
                    global_step=epoch,
                    dataformats="HWC",
                )
                logger.add_image(
                    f"image_{i}/alignment",
                    plot_tensor(attn.squeeze().cpu()),
                    global_step=epoch,
                    dataformats="HWC",
                )
                save_plot(y_enc.squeeze().cpu(), f"{log_dir}/generated_enc_{i}.png")
                save_plot(y_dec.squeeze().cpu(), f"{log_dir}/generated_dec_{i}.png")
                save_plot(attn.squeeze().cpu(), f"{log_dir}/alignment_{i}.png")

            print("Save weight...")
            ckpt = model.state_dict()
            torch.save(ckpt, f=f"{log_dir}/grad_{epoch}.pt")
            best_epoch = epoch
            best_val_loss = val_loss
        if check_early_stopping(patience, best_epoch, epoch):
            break
    os.symlink(
        "best_loss.pt",
        f"{log_dir}/grad_{best_epoch}.pt",
    )
