# import torch
# import argparse


# class Args:
#     def __init__(self) -> None:
#         parser = argparse.ArgumentParser()

#         parser.add_argument("--do_train", default=False, action="store_true")
#         parser.add_argument("--do_eval", default=False, action="store_true")
#         parser.add_argument("--do_test", default=False, action="store_true")
#         parser.add_argument("--dataset", default="wiki.txt", type=str)
#         parser.add_argument("--output_dir", default="output/Bert", type=str)
#         parser.add_argument("--cache_dir", default="cached", type=str)
#         parser.add_argument("--batch_size", default=42, type=int)
#         parser.add_argument("--n_gpu", default=torch.cuda.device_count(), type=int)
#         parser.add_argument("--learning_rate", default=5e-5, type=float)
#         parser.add_argument("--weight_decay", default=0.0, type=float)
#         parser.add_argument("--adam_epsilon", default=1e-8, type=float)
#         parser.add_argument("--max_grad_norm", default=1.0, type=float)
#         parser.add_argument("--seed", default=42, type=int)
#         parser.add_argument("--max_steps", default=-1, type=int)
#         parser.add_argument("--warmup_steps", default=0, type=int)
#         parser.add_argument("--log_steps", default=1000, type=int)
#         parser.add_argument("--save_steps", default=50000, type=int)
#         parser.add_argument("--num_train_epochs", default=5, type=int)
#         parser.add_argument("--gradient_accumulation_steps", default=1, type=int)

#         self.parser = parser

#     def get_args(self) -> argparse.ArgumentParser:
#         return self.parser.parse_args()


# args = Args().get_args()


# def main():
#     print("> Starting the program")

import os
import sys
import torch
import numpy as np
from tqdm.auto import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from dataset import WikipediaCorpus

from torch.nn import DataParallel
from torch.utils.data import random_split
from model import Word2Vec

# Dataset
dataset = WikipediaCorpus("../wiki_t.txt")

# Split to train and val
train_size = int(0.7 * len(dataset))
eval_size = int(0.1 * len(dataset))
test_size = len(dataset) - (train_size + eval_size)
train_dataset, eval_dataset, test_dataset = random_split(
    dataset, [train_size, eval_size, test_size]
)

# Model
model = Word2Vec(54234, 500)

# if args.n_gpu > 1:
#     model = DataParallel(model)
model.to("cpu")

dl = DataLoader(
    dataset,
    drop_last=False,
    collate_fn=dataset.collate_fn,
    batch_size=1,
    sampler=RandomSampler(dataset),
)

t_total = len(dl)
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [
            p
            for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0,
    },
    {
        "params": [
            p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=5e-5,
    eps=1e-8,
)

# Scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=t_total,
)

global_step = 0
epochs_trained = 0

# Loss
train_loss, log_loss = 0.0, 0.0
checkpoint_idx = 0

# Best model configs
best_model = None
best_ppl = np.inf
best_epoch = 0

model.zero_grad()

# Iterator
train_iterator = trange(epochs_trained, int(1), desc="Epoch")

for epoch_num in train_iterator:
    epoch_iterator = tqdm(dl, desc="Steps")
    epoch_loss = 0.0
    for step, batch in enumerate(epoch_iterator):
        # Skip past any already trained steps if resuming training

        model.train()
        word_pos, ctx, neg = batch
        # input_ids = input_ids.to(self.args.device)
        # attention_masks = attention_masks.to(self.args.device)
        # token_type_ids = token_type_ids.to(self.args.device)
        # labels = labels.to(self.args.device)
        loss = model(word_pos, ctx, neg)

        epoch_loss += loss
        train_loss += loss

        loss.backward()

        optimizer.step()
        scheduler.step()
        model.zero_grad()

        global_step += 1

    checkpoint_idx += 1
    epoch_iterator.close()

train_iterator.close()
train_loss = train_loss / global_step
print("Best epoch: {} --- PPL: {}".format(best_epoch, best_ppl))
print(">>> End of Training <<<")

# if __name__ == "__main__":
#     main()