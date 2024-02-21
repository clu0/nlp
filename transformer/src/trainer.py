from typing import Dict, Any, Optional, Tuple
import os
import sys
import datetime
from time import time

import torch
from torch.nn import functional as F
import torch.optim as optim
from torch.optim import Optimizer

from model import Decoder
from dataset import TextData
from logger import Logger, HumanOutputFormat, CSVOutputFormat
from utils import compute_norms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DecoderTrainer:
    def __init__(
        self,
        decoder_args: Dict[str, Any],
        dataset_args: Dict[str, Any],
        log_save_dir: str,
        model_save_dir: str,
    ):
        self.decoder = Decoder(**decoder_args)
        self.dataset = TextData(**dataset_args)
        self.log_save_dir = log_save_dir
        self.model_save_dir = model_save_dir
        self.decoder_args = decoder_args
        self.dataset_args = dataset_args
        self.loss: Optional[torch.Tensor] = None
        self.optimizer: Optional[Optimizer] = None
        self.logger: Optional[Logger] = None
        

    def setup_logging_and_model_dirs(self) -> Tuple[str, str]:
        time_now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_save_dir = os.path.join(
            self.log_save_dir, time_now
        )
        model_save_dir = os.path.join(
            self.model_save_dir, time_now
        )
        os.mkdir(log_save_dir, exist_ok=True)
        os.mkdir(model_save_dir, exist_ok=True)
        return log_save_dir, model_save_dir
    
    def setup_logger(self, log_save_dir: str):
        output_formats = [
            HumanOutputFormat(sys.stdout),
            HumanOutputFormat(os.path.join(log_save_dir, f"log.txt")),
            CSVOutputFormat(
                os.path.join(log_save_dir, f"progress.csv")
            ),
        ]
        logger = Logger(output_formats)
        logger.log(f"training args: \n{self.decoder_args}")
        logger.log(f"dataset args: \n{self.dataset_args}")
        self.logger = logger
    
    def load_and_setup_train_model(self, model_checkpoint: Optional[str] = None):
        if model_checkpoint is not None:
            self.decoder.load_state_dict(torch.load(model_checkpoint))
            self.logger.log(f"loaded model {model_checkpoint}")
        self.decoder.to(device=device)
        self.decoder.train()
        self.logger.log(f"model setup on {device}")
    
    def compute_loss(self, x: torch.Tensor, y: torch.Tensor):
        logits: torch.Tensor = self.decoder(x)
        # F.cross_entropy only supports shape of (N, C)
        # need to reshape from shape (B, T, n_vocab)
        B, T, n_vocab = logits.shape
        logits = logits.view(B * T, n_vocab)
        y = y.view(-1)
        self.loss = F.cross_entropy(logits, y)
    
    def set_optimizer(self, lr: float):
        self.optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
    
    def run_train_batch(self):
        self.decoder.zero_grad()
        x, y = self.dataset.get_batch()
        self.compute_loss(x, y)
        self.loss.backward()
        self.logger.logkv("train_loss", self.loss.item())
        self.logger.logkv_mean("train_loss_avg", self.loss.item())
        weight_norm, grad_norm = compute_norms(self.decoder)
        self.logger.logkv("weight_norm", weight_norm)
        self.logger.logkv("grad_norm", grad_norm)
        self.logger.logkv_mean("weight_norm_avg", weight_norm)
        self.logger.logkv_mean("grad_norm_avg", grad_norm)
        self.optimizer.step()
    
    def save_model(self, save_dir: str, prefix: str, iteration: int):
        save_path = os.path.join(save_dir, f"{prefix}_iter-{iteration}.pt")
        torch.save(self.decoder.state_dict(), save_path)
    
    def log_val_loss(self, n_val_iter: int):
        with torch.no_grad():
            self.decoder.eval()
            val_losses = torch.zeros(n_val_iter)
            for i in range(n_val_iter):
                x, y = self.dataset.get_batch(split="val")
                self.compute_loss(x, y)
                val_losses[i] = self.loss
            mean_val_loss = val_losses.mean().item()
            self.logger.logkv("val_loss", mean_val_loss)
            self.logger.logkv_mean("val_loss_avg", mean_val_loss)
            self.decoder.train()
    
    def train(
        self,
        n_iter: int,
        lr: float,
        n_val_iter: int,
        log_interval: int,
        save_interval: int,
        model_checkpoint: Optional[str] = None,
        past_n_iter: Optional[int] = None,
    ):
        log_save_dir, model_save_dir = self.setup_logging_and_model_dirs()
        model_save_prefix = "".join(
            ["mode"] + [f"_{k}-{v}_" for k, v in self.decoder_args.items()] + [f"_lr-{lr}"]
        )
        self.setup_logger(log_save_dir)
        self.load_and_setup_train_model(model_checkpoint)
        self.set_optimizer(lr)

        # training loop
        start_time = time()
        for i in range(n_iter):
            if past_n_iter is not None:
                i += past_n_iter
            self.logger.logkv("iteration", i)
            self.run_train_batch()
            
            if (i + 1) % log_interval == 0:
                self.logger.log(f"finished epoch {i}, took {time() - start_time} seconds")
                self.log_val_loss(n_val_iter)
                self.logger.dumpkvs()
            if (i + 1) % save_interval == 0:
                self.save_model(model_save_dir, model_save_prefix, i + 1)
        
        # edge case: if save_interval is not a divisor of n_iter, save at the end
        if n_iter % save_interval != 0:
            self.save_model(model_save_dir, model_save_prefix, n_iter)

        self.logger.log(f"training finished, took {time() - start_time} seconds")
        self.logger.close()
