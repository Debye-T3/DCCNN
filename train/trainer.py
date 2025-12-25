import time  # measure time
import os
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt  # plots
import numpy as np
import pandas as pd  # dataframes
import torch  # Neural Network Framework
from torch.amp import GradScaler, autocast
from tqdm import tqdm  # progress bar

from modules.loss_functions.loss_function import loss_function


def train_model(model, train_loader, val_loader, epochs, learning_rate, alpha, device, early_stopping_patience):
    """
    Main training loop.
    """

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    metrics = pd.DataFrame(
        columns=[
            "Epoch",
            "Loss",
            "Mae",
            "Msssim",
            "Val-Loss",
            "Val-Mae",
            "Val-Msssim",
            "Learning-Rate",
            "Time",
        ]
    )
    lrs = []

    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    print("Training started!")

    scheduler = None
    epochs_no_improve = 0
    best_avg_val_loss = float("inf")
    cosine = False

    warmup_epochs = min(10, epochs)
    warmup_lrs = np.linspace(1e-7, learning_rate, num=max(warmup_epochs, 1))

    cosine_counter = 0
    cosine_epochs = 0

    for epoch in range(epochs):
        start = time.time()

        if epoch < warmup_epochs:
            warmup_lr = warmup_lrs[epoch]
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr

        model.train()
        total_loss = 0.0
        total_mae = 0.0
        total_msssim = 0.0
        progress = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}]", leave=False)

        current_lr = optimizer.param_groups[0]["lr"]
        lrs.append(current_lr)

        for low_res_image, high_res_image in progress:

            low_res_image, high_res_image = low_res_image.to(device), high_res_image.to(device)
            optimizer.zero_grad()

            autocast_ctx = autocast(device_type=device.type) if use_amp else nullcontext()
            with autocast_ctx:
                output = model(low_res_image)
                loss, mae, msssim = loss_function(output, high_res_image, alpha)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_mae += mae.item()
            total_msssim += msssim.item()
            progress.set_postfix(loss=loss.item())

        duration = time.time() - start

        avg_loss = total_loss / max(1, len(train_loader))
        avg_mae = total_mae / max(1, len(train_loader))
        avg_msssim = total_msssim / max(1, len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_msssim = 0.0
        with torch.no_grad():
            for low_res_image, high_res_image in val_loader:
                low_res_image, high_res_image = low_res_image.to(device), high_res_image.to(device)
                autocast_ctx = autocast(device_type=device.type) if use_amp else nullcontext()
                with autocast_ctx:
                    output = model(low_res_image)
                    loss, mae, msssim = loss_function(output, high_res_image, alpha)
                val_loss += loss.item()
                val_mae += mae.item()
                val_msssim += msssim.item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        avg_val_mae = val_mae / max(1, len(val_loader))
        avg_val_msssim = val_msssim / max(1, len(val_loader))

        if cosine and scheduler is not None:
            cosine_counter += 1
            metrics.loc[len(metrics)] = [
                epoch + 1,
                avg_loss,
                avg_mae,
                avg_msssim,
                avg_val_loss,
                avg_val_mae,
                avg_val_msssim,
                current_lr,
                duration,
            ]
            print(
                f"[{epoch + 1:3d}/{epochs}] Loss: {avg_loss:.4f} | Val-Loss: {avg_val_loss:.4f} | "
                f"Time: {duration:.2f}s | Learning-Rate: {current_lr:.2e} | "
                f"CosineAnnealing [{cosine_counter}/{cosine_epochs}]"
            )
            scheduler.step()
            if cosine_counter >= cosine_epochs:
                print("Ready with CosineAnnealing!")
                break
            continue

        if avg_val_loss < best_avg_val_loss - 1e-4:
            best_avg_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            if epoch >= warmup_epochs:
                epochs_no_improve += 1

        metrics.loc[len(metrics)] = [
            epoch + 1,
            avg_loss,
            avg_mae,
            avg_msssim,
            avg_val_loss,
            avg_val_mae,
            avg_val_msssim,
            current_lr,
            duration,
        ]

        print(
            f"[{epoch + 1:3d}/{epochs}] Loss: {avg_loss:.4f} | Val-Loss: {avg_val_loss:.4f} | "
            f"Time: {duration:.2f}s | Learning-Rate: {current_lr:.2e} | Patience: {epochs_no_improve}"
        )

        if epochs_no_improve >= early_stopping_patience:
            cosine_epochs = max(int(0.3 * (epoch + 1)), 10)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cosine_epochs - 1, eta_min=1e-7
            )
            cosine = True
            cosine_counter = 0
            print(f"Switching to CosineAnnealing with {cosine_epochs} epochs!")

    # Plot LR curve
    plot_dir = Path("results/lr_finder")
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(lrs)
    plt.xlabel("Training steps")
    plt.ylabel("Learning rate")
    plt.yscale("log")
    plt.title("Learning rate Plot")
    plt.grid(True)
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = plot_dir / f"lr_plot_{timestamp}.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"Learning rate plot saved in {plot_path}")

    return model, metrics
