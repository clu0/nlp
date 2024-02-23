"""
reads a text dataset from a file path, does a train/test split if specified,
and returns batches
"""
from typing import Tuple, List
import tiktoken
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NaiveEncoder:
    """
    do a character based encoding of the text, as in Andrej's tutorial
    
    reason why we're not using the tiktoken encoder by default is because the cl100k_base encoder for instance
    has a vocab size of 100k. That means we need to keep a trainable work embedding parameter of size 100k x d_embed
    which is a lot for our purposes
    """
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.n_vocab: int = len(chars)
    
    def encode(self, string: str) -> List[int]:
        return [self.stoi[c] for c in string]
    
    def decode(self, inds: List[int]) -> str:
        return "".join([self.itos[ind] for ind in inds])


def encoded_tensor_from_path(filepath: str, encoder) -> torch.Tensor:
    with open(filepath, "r") as f:
        text = f.read()
    return torch.tensor(encoder.encode(text), dtype=torch.long, device=device)

class TextData:
    def __init__(
        self,
        text_filepath: str,
        batch_size: int = 16,
        block_size: int = 32,
        val_frac: float = 0.1,
        encode_scheme: str = "naive",
        **kwargs
    ):
        """
        Since the text datasets are quite small in memory
        just read the text from the file, encode it,
        and load the whole thing into a torch tensor
        """
        with open(text_filepath, "r") as f:
            text = f.read()
        encoder = NaiveEncoder(text) if encode_scheme == "naive" else tiktoken.get_encoding("cl100k_base")
        text_tensor = encoded_tensor_from_path(text_filepath, encoder)
        if val_frac > 0:
            n_train = int(len(text_tensor) * (1 - val_frac))
            self.train_data: torch.Tensor = text_tensor[:n_train]
            self.val_data: torch.Tensor = text_tensor[n_train:]
        else:
            self.train_data = text_tensor
        self.batch_size = batch_size
        self.block_size = block_size
        self.n_vocab: int = encoder.n_vocab
    
    def get_batch(self, split: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        return random batches from the training tensor of block_size length each
        """
        data: torch.Tensor = self.train_data if split == "train" else self.val_data
        start_inds = torch.randint(high=len(data) - self.block_size, size=(self.batch_size,))
        xs = torch.stack([data[i : i + self.block_size] for i in start_inds])
        ys = torch.stack([data[i + 1 : i + 1 + self.block_size] for i in start_inds])
        return xs, ys
        