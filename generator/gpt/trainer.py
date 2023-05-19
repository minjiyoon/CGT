import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
import time

logger = logging.getLogger(__name__)

class TrainerConfig:
    batch_size = 64
    block_size = 4
    short_seq_num = 25
    # optimization parameters
    max_epochs = 100
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:
    def __init__(self, args, config, model, data_loader):
        self.config = config
        self.args = args
        self.model = model
        self.data_loader = data_loader

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        start_time = time.time()
        self.tokens = 0 # counter used for learning rate decay
        best_loss = float('inf')
        for epoch in range(config.max_epochs):
            model.train()
            with tqdm(self.data_loader, unit="batch") as t_data_loader:
                for batch in t_data_loader:
                    x, y, lbl = batch["query"].to(self.device), batch["predict"].to(self.device), batch["label"].to(self.device)

                    with torch.set_grad_enabled(True):
                        logits, loss = model(x, lbl, y)
                        loss = loss.mean()

                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    if config.lr_decay:
                        # number of tokens processed this step
                        self.tokens += config.block_size * config.short_seq_num * config.batch_size
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(3e-5 / config.learning_rate, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    t_data_loader.set_description(f"Epoch {epoch}")
                    t_data_loader.set_postfix(loss=loss.item())
                    time.sleep(0.1)

        # Save the trained model
        self.save_checkpoint()
