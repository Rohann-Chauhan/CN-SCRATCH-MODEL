import os
import torch
import torch.nn as nn
from tqdm import tqdm

from config import (
    device,
    epochs,
    learning_rate
)
from model import GPTModel
from dataset import create_dataloader_v1
from utils import causal_mask


def train_model(token_ids, vocab_size):
    # 1. Prepare data loaders
    train_loader, val_loader = create_dataloader_v1(token_ids)
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))

    # 2. Build the model architecture
    model = GPTModel(vocab_size).to(device)

    # 3. Setup optimizer, scheduler, and loss function
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    loss_fn = nn.CrossEntropyLoss()

    # 4. Initialize tracking variables
    best_val_loss = float("inf")
    patience = 0
    max_patience = 3
    ckpt_path = "checkpoint.pt"
    start_epoch = 0

    # 5. Load existing progress if available
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            
        start_epoch = checkpoint["epoch"] + 1
        print("\nLoaded existing checkpoint.")
        print(f"Resuming training from Epoch {start_epoch + 1}...")

    # 6. Core training and validation loops
    for epoch in range(start_epoch, epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        
        # --- TRAINING STEP ---
        model.train()
        total_train_loss = 0
        
        train_bar = tqdm(train_loader, desc="Training")
        
        for x, y in train_bar:
            x = x.to(device)
            y = y.to(device)

            # Mask ensures the model can only look at past words, not future ones
            mask = causal_mask(x.size(1)).to(device)

            # Ask the model for its predictions
            logits = model(x, mask)

            # Flatten inputs to calculate the error (loss)
            loss = loss_fn(
                logits.view(-1, vocab_size),
                y.view(-1)
            )

            # Update weights based on errors
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients to stop extreme mathematical explosions
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)

        # --- VALIDATION STEP ---
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                mask = causal_mask(x.size(1)).to(device)
                logits = model(x, mask)

                loss = loss_fn(
                    logits.view(-1, vocab_size),
                    y.view(-1)
                )
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        
        # Gradually lower the learning speed (learning rate)
        scheduler.step()

        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # 7. Quality check and file saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience = 0
            torch.save(model.state_dict(), "best_model.pt")
            print("⭐ New lowest error! Saved best_model.pt")
        else:
            patience += 1
            print(f"⚠️ No improvement. Early stopping timer: ({patience}/{max_patience})")
            
            if patience >= max_patience:
                print("🛑 Stopping early because model stopped learning.")
                break

        # Save an exact backup snapshot in case the computer crashes
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict()
            },
            ckpt_path
        )
        print("💾 Checkpoint saved.")

    print("\n🎉 Training Complete!")
    return model