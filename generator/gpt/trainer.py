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
    # optimization parameters
    max_epochs = 10
    batch_size = 64
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
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, args):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.args = args

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        #logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self, log_prefix="train"):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        start_time = time.time()

        def run_epoch(split):
            is_train = (split == 'train')
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            if is_train:
                data.reset()

            losses = []
            batch_num = len(data) // config.batch_size
            for it in range(batch_num):
                x, y, lbl = data.get(it * config.batch_size, (it + 1) * config.batch_size)

                x = x.to(self.device)
                y = y.to(self.device)
                lbl = lbl.to(self.device).long()

                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x, lbl, y)
                    loss = loss.mean()
                    losses.append(loss.item())

                if is_train:
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    if config.lr_decay:
                        # number of tokens processed this step
                        self.tokens += data.block_size * data.short_seq_num * config.batch_size
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(3e-5/config.learning_rate, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

            if not is_train:
                test_loss = float(np.mean(losses))
                report_dict = {"ETA (m)": (time.time() - start_time) * (config.max_epochs - (epoch+1)) / (epoch+1) / (60), \
                               "TIME (m)":(time.time() - start_time) / (60)}
                report_dict.update({log_prefix:{"epoch": epoch, "val_loss": test_loss}})
                return test_loss

        # counter used for learning rate decay
        self.tokens = 0
        best_loss = float('inf')
        for epoch in range(config.max_epochs):
            run_epoch('train')

            if self.test_dataset is not None:
                test_loss = run_epoch('val')
                if test_loss < best_loss:
                    best_loss = test_loss
                    self.save_checkpoint()

        if self.test_dataset is None:
            self.save_checkpoint()
