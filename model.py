import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    d_model,
    max_seq_len,
    dropout,
    num_heads,
    d_ff,
    num_layers
)


# ==================================
# INPUT EMBEDDING
# ==================================
class InputEmbedding(nn.Module):
    """
    Converts token IDs into dense, meaningful vector spaces (word meanings).
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)


# ==================================
# POSITIONAL ENCODING
# ==================================
class PositionalEncoding(nn.Module):
    """
    Injects structural order into the model using static sine and cosine wave equations.
    """
    def __init__(self, d_model, max_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register buffer ensures the matrix moves to the GPU automatically but isn't updated by gradients
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ==================================
# MULTI HEAD ATTENTION
# ==================================
class MultiheadAttentionBlock(nn.Module):
    """
    Allows tokens to dynamically focus on relevant past words simultaneously across multiple views.
    """
    def __init__(self, d_model, h, dropout):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.shape[-1]
        
        # Matrix multiplication to calculate relational importance scores
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # Causal mask forces the model to ignore future context
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores = F.softmax(attention_scores, dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask=None):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Reshape to split the sequence context among distinct focus heads
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiheadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )

        # Stitch all split focus heads back into a unified feature vector
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)


# ==================================
# FEED FORWARD
# ==================================
class FeedForward(nn.Module):
    """
    A non-linear network block allowing the model to reflect and process patterns.
    """
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(F.gelu(self.linear_1(x))))


# ==================================
# LAYER NORM
# ==================================
class LayerNormalization(nn.Module):
    """
    Normalizes layer values to zero-mean and unit variance to stabilize mathematical training.
    """
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


# ==================================
# RESIDUAL
# ==================================
class ResidualConnection(nn.Module):
    """
    Wraps layers inside a bypass bridge to allow raw gradients to flow without degrading.
    """
    def __init__(self, d_model, dropout):
        super().__init__()
        self.norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # Pre-Layer Normalization architecture pattern
        return x + self.dropout(sublayer(self.norm(x)))


# ==================================
# DECODER BLOCK
# ==================================
class DecoderBlock(nn.Module):
    """
    An individual Transformer layer combining self-attention and feed-forward operations.
    """
    def __init__(self, d_model, h, d_ff, dropout):
        super().__init__()
        self.attention = MultiheadAttentionBlock(d_model, h, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.residuals = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, mask=None):
        x = self.residuals[0](x, lambda ctx: self.attention(ctx, ctx, ctx, mask))
        x = self.residuals[1](x, self.feed_forward)
        return x


# ==================================
# GPT MODEL
# ==================================
class GPTModel(nn.Module):
    """
    The master structure organizing embeddings, blocks, and language projection.
    """
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = InputEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        self.layers = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        
        self.norm = LayerNormalization(d_model)
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        logits = self.projection(x)
        return logits