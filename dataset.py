import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    batch_size,
    max_seq_len,
    stride
)
class GPTDataset(Dataset):
# iSKA KAAM BASICALLY DAYTA KO LOAD KARNE KA HAI 
    def __init__(
        self,
        token_ids,
        max_length,
        stride
    ):
        self.input_ids = []
        self.target_ids = []

        # Use a sliding window to slice token_ids into input and target chunks
        for i in range(0, len(token_ids) - max_length, stride):

            input_chunk = token_ids[i : i + max_length]
            # Target is shifted by exactly 1 token to the right for next-token prediction
            target_chunk = token_ids[i + 1 : i + max_length + 1]

            self.input_ids.append(
                torch.tensor(input_chunk, dtype=torch.long)
            )
            self.target_ids.append(
                torch.tensor(target_chunk, dtype=torch.long)
            )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(token_ids):
    """
    Splits token IDs into 90% training and 10% validation sets, 
    then initializes and returns PyTorch DataLoaders.
    """
    split_idx = int(0.9 * len(token_ids))

    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]

    # Instantiate datasets
    train_dataset = GPTDataset(
        train_ids,
        max_seq_len,
        stride
    )

    val_dataset = GPTDataset(
        val_ids,
        max_seq_len,
        stride
    )

    # Instantiate loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,      # Shuffle training data to break sequential bias
        drop_last=True     # Drop incomplete batches at the end
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,     # Do not shuffle validation data
        drop_last=True
    )

    return train_loader, val_loader