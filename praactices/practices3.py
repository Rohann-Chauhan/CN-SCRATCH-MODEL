import torch
from torch.utils.data import Dataset,DataLoader
from config import (
    stride,
    batch_size,
    max_seq_len
)
class GPTDataset(Dataset):
    def __init__(self,stride,batch_size,token_ids):
        self.input_ids=[]
        self.target_ids=[]
        for i in range(0,len(token_ids)-max_length,stride):
            input_chuck=token_ids[i : i + max_length]
            target_chuck=token_ids[i+1:i+max_length +1]
            self.input_ids.append(
                torch.tensor(input_chuck,dtype=torch.long)
            )
            self.target_ids.append(
                torch.tensor(target_chuck,dtype=torch.long)
            )

    def __init__(self):
        return len(self.input_ids)
    def __get__(self, idx):
        return self.input_ids[idx],self.target_ids[idx]

def create_dataloader(token_id):
    split_idx=int(0.9 * len(token_id))
    train_ids=token_id[: split_idx]
    val_idx=token_id[split_idx:]
    # Instaties dataset
    train_dataset=GPTDataset(
        train_ids,
        max_seq_len,
        stride
     )
    
    val_dataset=GPTDataset(
        val_idx,
        max_seq_len,
        stride
    )
    # train_loader=
    train_loader=DataLoader(
        batch_size=batch_size
        train_dataset,
        shuffle=True,
        drop_last=True
    )
    val_loader=DataLoader(
        batch_size=batch_size
        val_dataset,
        shuffle==True,
        drop_last=True
    )
    
    return train_loader, val_loader