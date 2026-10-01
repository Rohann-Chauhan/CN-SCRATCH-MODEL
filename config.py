import torch
device=torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
d_model =128
num_heads =4
num_layers=2
d_ff =512
dropout =0.1
max_seq_len=128
stride=64
batch_size=4
#told epoch=100
epochs=21
learning_rate=3e-4