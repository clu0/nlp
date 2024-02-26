"""
You could say this is the fun part

We will just do a decoder only model for now.
"""
from typing import Optional
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


def get_causal_mask(T: int) -> torch.Tensor:
    """
    Get a mask with lower triangular part filled with zeros and upper triangular part filled with -inf
    """
    zero_one_mask = torch.tril(torch.ones(T, T))
    zero_mask = torch.zeros(T, T)
    causal_mask = zero_mask.masked_fill(zero_one_mask == 0, float("-inf"))
    return causal_mask.to(device)

class DecoderBlock(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_inner: int,
        feed_forward_dropout: float = 0.5,
        attn_dropout: float = 0.5,
        cross_attn: bool = True,
    ):
        super().__init__()
        assert n_embd // n_head == int(n_embd / n_head), f"n_embd {n_embd} must be divisible by n_head {n_head}"
        self.ln1 = nn.LayerNorm(n_embd)
        self.feed_forward = FeedForward(n_embd, n_inner, feed_forward_dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, attn_dropout, batch_first=True)
        self.cross_attn = cross_attn
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # attention mask
        causal_mask: Optional[torch.Tensor] = None
        if self.cross_attn:
            causal_mask = get_causal_mask(x.size(1))

        normalized_x = self.ln1(x)
        x = x + self.attn.forward(normalized_x, normalized_x, normalized_x, attn_mask=causal_mask)[0]
        x = x + self.feed_forward(self.ln2(x))
        return x


class CrossAttentionDecoderBlock(nn.Module):
    """
    instead of the vanilla decoder block above, also takes as input the embeddings from an encoder
    and creates the KVs in multi-head attention using the input embeddings
    """
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_inner: int,
        feed_forward_dropout: float = 0.5,
        attn_dropout: float = 0.5,
    ):
        super().__init__()
        assert n_embd // n_head == int(n_embd / n_head), f"n_embd {n_embd} must be divisible by n_head {n_head}"
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, attn_dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.cross_attn = nn.MultiheadAttention(n_embd, n_head, attn_dropout, batch_first=True)
        self.ln3 = nn.LayerNorm(n_embd)
        self.feed_forward = FeedForward(n_embd, n_inner, feed_forward_dropout)
    
    def forward(self, x: torch.Tensor, embd: torch.Tensor) -> torch.Tensor:
        causal_mask = get_causal_mask(x.size(1))

        # self attention
        x_ln = self.ln1(x)
        x += self.attn(x_ln, x_ln, x_ln, attn_mask=causal_mask)[0]

        # cross attention
        x_ln = self.ln2(x)
        x += self.cross_attn.forward(x_ln, embd, embd)[0]

        # feed forward
        x += self.feed_forward(self.ln3(x))
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
        # the reason that timestep embeddings is a table is because
        # we want to be able to input contexts of sizes <= block_size
        self.timestep_embeddings_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[
                DecoderBlock(n_embd, n_head, n_inner, feed_forward_dropout, attn_dropout)
                for _ in range(n_layer)
            ]
        )
        self.ln_final = nn.LayerNorm(n_embd)
        self.logits = nn.Linear(n_embd, n_vocab)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings_table(x)
        x += self.timestep_embeddings_table(torch.arange(x.size(1)).to(device))
        x = self.blocks(x)
        x = self.ln_final(x)
        return self.logits(x)

class Encoder(nn.Module):
    """
    Basically the same as the bare bones decoder above, but without causal attention mask,
    and instead of returning the logits, it just returns the final embeddings
    """
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
        # the reason that timestep embeddings is a table is because
        # we want to be able to input contexts of sizes <= block_size
        self.timestep_embeddings_table = nn.Embedding(block_size, n_embd)
        # important: cross_attn is false
        self.blocks = nn.Sequential(
            *[
                DecoderBlock(n_embd, n_head, n_inner, feed_forward_dropout, attn_dropout, cross_attn=False)
                for _ in range(n_layer)
            ]
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings_table(x)
        x += self.timestep_embeddings_table(torch.arange(x.size(1)).to(device))
        return self.blocks(x)


class CrossAttentionDecoder(nn.Module):
    """
    a decoder with cross attention that takes as input the encoder embeddings
    """
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
        self.timestep_embeddings_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.ModuleList()
        self.blocks.extend(
            [
                CrossAttentionDecoderBlock(n_embd, n_head, n_inner, feed_forward_dropout, attn_dropout)
                for _ in range(n_layer)
            ]
        )

        self.ln_final = nn.LayerNorm(n_embd)
        self.logits = nn.Linear(n_embd, n_vocab)

    def forward(self, x: torch.Tensor, embd: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings_table(x)
        x += self.timestep_embeddings_table(torch.arange(x.size(1)).to(device))
        for block in self.blocks:
            x = block(x, embd)
        x = self.ln_final(x)
        return self.logits(x)