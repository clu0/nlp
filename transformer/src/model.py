"""
You could say this is the fun part

We will just do a decoder only model for now.
"""
import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FeedForward(nn.Module):
    def __init__(self, n_embd: int, n_inner: int, dropout: float = 0.5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(n_embd, n_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_inner, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_inner: int,
        block_size: int,
        feed_forward_dropout: float = 0.5,
        attn_dropout: float = 0.5,
    ):
        super().__init__()
        assert n_embd // n_head == int(n_embd / n_head), f"n_embd {n_embd} must be divisible by n_head {n_head}"
        self.ln1 = nn.LayerNorm(n_embd)
        self.feed_forward = FeedForward(n_embd, n_inner, feed_forward_dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, attn_dropout, batch_first=True)
        # attention mask
        zero_one_mask = torch.tril(torch.ones(block_size, block_size)).to(device)
        zero_mask = torch.zeros(block_size, block_size).to(device)
        self.causal_mask = zero_mask.masked_fill(zero_one_mask == 0, float("-inf"))
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized_x = self.ln1(x)
        x = x + self.attn.forward(normalized_x, normalized_x, normalized_x, attn_mask=self.causal_mask)[0]
        x = x + self.feed_forward(self.ln2(x))
        return x
    

class Decoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        block_size: int,
        n_layer: int,
        n_embd: int,
        n_head: int,
        n_inner: int,
        feed_forward_dropout: float = 0.5,
        attn_dropout: float = 0.5,
        **kwargs
    ):
        super().__init__()
        self.token_embeddings_table = nn.Embedding(n_vocab, n_embd)
        self.timestep_embeddings = nn.Parameter(torch.randn(block_size, n_embd))
        self.blocks = nn.Sequential(
            *[
                DecoderBlock(n_embd, n_head, n_inner, block_size, feed_forward_dropout, attn_dropout)
                for _ in range(n_layer)
            ]
        )
        self.ln_final = nn.LayerNorm(n_embd)
        self.logits = nn.Linear(n_embd, n_vocab)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings_table(x)
        x += self.timestep_embeddings
        x = self.blocks(x)
        x = self.ln_final(x)
        return self.logits(x)
