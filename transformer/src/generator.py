from typing import Dict, Any

import torch
import torch.nn.functional as F

from src.model import Decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextGenerator:
    def __init__(
        self,
        model_checkpoint: str,
        block_size: int,
        decoder_args: Dict[str, Any],
        n_max_token: int = 2000,
    ):
        # load model
        self.decoder = Decoder(**decoder_args).to(device)
        self.decoder.eval()
        self.decoder.load_state_dict(torch.load(model_checkpoint))
        
        self.block_size = block_size
        self.n_max_token = n_max_token
    
    def generate(self, context: torch.Tensor) -> torch.Tensor:
        """
        context of shape (batch_size, block_size)
        """
        context = context.to(device)
        with torch.no_grad():
            for _ in range(self.n_max_token):
                local_context = context[:, -self.block_size:]
                logits = self.decoder(local_context)
                # sample from logits of last token
                # logits originally of shape (B, T, n_vocab)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx = torch.multinomial(probs, num_samples=1)
                context = torch.cat([context, idx], dim=-1)
        return context.to("cpu")
