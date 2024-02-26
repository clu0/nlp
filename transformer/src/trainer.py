from typing import Dict, Any, Optional, Tuple, Type
from abc import ABC, abstractmethod
import os
import sys
import datetime
from time import time

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim import Optimizer

from src.model import Decoder
from src.dataset import TextData
from src.logger import Logger, HumanOutputFormat, CSVOutputFormat
from src.utils import compute_norms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(ABC):
    def __init__(
        self,
        args: Dict[str, Any]
    ):
        self.args = args
        self.logger: Optional[Logger] = None
        self.log_save_dir: Optional[str] = None
        self.model_save_dir: Optional[str] = None
    
    def setup_logging_and_model_dirs(self):
        time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_save_dir = os.path.join(
            self.args["log_save_dir"], time_now
        )
        self.model_save_dir = os.path.join(
            self.args["model_save_dir"], time_now
        )
        os.makedirs(self.log_save_dir, exist_ok=True)
        os.makedirs(self.model_save_dir, exist_ok=True)

        self.setup_model_filepaths()
    
    @abstractmethod
    def setup_model_filepaths(self):
        pass
    
    def get_model_save_prefix(self, prefix: str = "model") -> str:
        keys_to_include = {
            "batch_size",
            "block_size",
            "encode_scheme",
            "lr",
            "n_embd",
            "n_head",
            "n_inner",
            "n_layer",
        }
        model_save_prefix = prefix
        for k, v in self.args.items():
            if k in keys_to_include:
                model_save_prefix += f"_{k}-{str(v)}"
        return model_save_prefix

    def setup_logger(self):
        output_formats = [
            HumanOutputFormat(sys.stdout),
            HumanOutputFormat(os.path.join(self.log_save_dir, f"log.txt")),
            CSVOutputFormat(
                os.path.join(self.log_save_dir, f"progress.csv")
            ),
        ]
        logger = Logger(output_formats)
        logger.log(f"training args: \n{self.args}")
        logger.log(f"cwd: {os.getcwd()}, files in cwd: {os.listdir()}")
        self.logger = logger

    @abstractmethod
    def compute_loss(self, x: torch.Tensor, y: torch.Tensor):
        pass

    def load_and_setup_train_model(
        self,
        model_name: str,
        model_class: Type[nn.Module],
        args: Dict[str, Any],
        model_checkpoint: Optional[str] = None
    ):
        model = model_class(**args)
        if model_checkpoint is not None:
            model.load_state_dict(torch.load(model_checkpoint))
            self.logger.log(f"loaded model {model_checkpoint}")
        model.to(device=device)
        model.train()
        self.logger.log(f"model setup on {device}")
        setattr(self, model_name, model)
    
    @abstractmethod
    def setup_models(self):
        pass

    def set_adam_optimizer(self, optimizer_name: str, model_name: str, lr: float):
        model: nn.Module = getattr(self, model_name)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        setattr(self, optimizer_name, optimizer)

    @abstractmethod
    def setup_optimizers(self, lr: float):
        pass
    
    @abstractmethod
    def compute_loss(self, x: torch.Tensor, y: torch.Tensor):
        pass

    @abstractmethod
    def run_train_batch(self, gradient_clip: Optional[float] = None):
        pass

    def save_model(self, model_name: str, prefix: str, iteration: int):
        save_path = os.path.join(self.model_save_dir, f"{prefix}_iter-{iteration}.pt")
        model: nn.Module = getattr(self, model_name)
        torch.save(model.state_dict(), save_path)
    
    @abstractmethod
    def save_models(self, iteration: int):
        pass

    @abstractmethod
    def log_val_loss(self, n_val_iter: int):
        pass

    def train(
        self,
        n_iter: int,
        lr: float,
        n_val_iter: int,
        log_interval: int,
        save_interval: int,
        model_checkpoint: Optional[str] = None,
        past_n_iter: Optional[int] = None,
        gradient_clip: Optional[float] = None,
    ):
        self.setup_logging_and_model_dirs()
        self.setup_logger()
        self.setup_models(model_checkpoint=model_checkpoint)
        self.setup_optimizers(lr)

        # training loop
        start_time = time()
        for i in range(n_iter):
            if past_n_iter is not None:
                i += past_n_iter
            self.logger.logkv("iteration", i)
            self.run_train_batch(gradient_clip=gradient_clip)
            
            if (i + 1) % log_interval == 0:
                self.logger.log(f"finished epoch {i}, took {time() - start_time} seconds")
                self.log_val_loss(n_val_iter)
                self.logger.dumpkvs()
            if (i + 1) % save_interval == 0:
                self.save_models(i)
        
        # edge case: if save_interval is not a divisor of n_iter, save at the end
        if n_iter % save_interval != 0:
            self.save_models(n_iter)

        self.logger.log(f"training finished, took {time() - start_time} seconds")
        self.logger.close()


class DecoderTrainer(Trainer):
    def __init__(
        self,
        args: Dict[str, Any],
    ):
        super().__init__(args)
        self.dataset = TextData(**args)
        args["n_vocab"] = self.dataset.n_vocab
        self.decoder = Decoder(**args)
        self.model_name = "decoder"
        self.model_class = Decoder
        self.model_save_prefix: Optional[str] = None
        self.loss: Optional[torch.Tensor] = None
        self.optimizer: Optional[Optimizer] = None

    def setup_model_filepaths(self):
        self.model_save_prefix = self.get_model_save_prefix(prefix=self.model_name)
    
    def setup_models(self, model_checkpoint: Optional[str] = None):
        self.load_and_setup_train_model(
            model_name=self.model_name,
            model_class=self.model_class,
            args=self.args,
            model_checkpoint=model_checkpoint)
        
    def compute_loss(self, x: torch.Tensor, y: torch.Tensor):
        logits: torch.Tensor = self.decoder(x)
        # F.cross_entropy only supports shape of (N, C)
        # need to reshape from shape (B, T, n_vocab)
        B, T, n_vocab = logits.shape
        logits = logits.view(B * T, n_vocab)
        y = y.view(-1)
        self.loss = F.cross_entropy(logits, y)
    
    def setup_optimizers(self, lr: float):
        self.set_adam_optimizer("optimizer", "decoder", lr)
    
    def run_train_batch(self, gradient_clip: Optional[float] = None):
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
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), gradient_clip)
        self.optimizer.step()
    
    def save_models(self, iteration: int):
        self.save_model(self.model_save_dir, self.model_save_prefix, iteration)
    
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
    
