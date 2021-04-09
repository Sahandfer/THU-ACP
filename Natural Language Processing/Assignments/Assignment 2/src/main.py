import torch
import argparse
import numpy as np
import torch.optim as optim
from tqdm.auto import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler
from src.dataset import WikipediaCorpus
from torch.nn import DataParallel
from torch.utils.data import random_split
from src.model import Word2Vec


class Args:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser()

        parser.add_argument("--do_train", default=False, action="store_true")
        parser.add_argument("--do_eval", default=False, action="store_true")
        parser.add_argument("--do_test", default=False, action="store_true")
        parser.add_argument("--use_cuda", default=False, action="store_true")
        parser.add_argument("--dataset", default="wiki_t.txt", type=str)
        parser.add_argument("--output_dir", default="output", type=str)
        parser.add_argument("--cache_dir", default="cached", type=str)
        parser.add_argument("--batch_size", default=128, type=int)
        parser.add_argument("--window_size", default=2, type=int)
        parser.add_argument("--neg_sample_size", default=5, type=int)
        parser.add_argument("--embedding_size", default=100, type=int)
        parser.add_argument("--n_gpu", default=torch.cuda.device_count(), type=int)
        parser.add_argument("--learning_rate", default=5e-5, type=float)
        parser.add_argument("--weight_decay", default=0.0, type=float)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--max_grad_norm", default=1.0, type=float)
        parser.add_argument("--max_steps", default=-1, type=int)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--log_steps", default=1000, type=int)
        parser.add_argument("--save_steps", default=1000, type=int)
        parser.add_argument("--num_train_epochs", default=1, type=int)
        parser.add_argument("--gradient_accumulation_steps", default=1, type=int)

        self.parser = parser

    def get_args(self) -> argparse.ArgumentParser:
        return self.parser.parse_args()


args = Args().get_args()
args.device = "cuda" if args.use_cuda else "cpu"


def main():
    print("> Starting the program")

    # Dataset
    dataset = WikipediaCorpus("wiki_t.txt")

    # Split to train and val
    train_size = int(0.7 * len(dataset))
    eval_size = int(0.1 * len(dataset))
    test_size = len(dataset) - (train_size + eval_size)
    train_dataset, eval_dataset, test_dataset = random_split(
        dataset, [train_size, eval_size, test_size]
    )

    # Model
    model = Word2Vec(len(dataset.vocab_dict), args.embedding_size)

    if args.n_gpu > 1:
        model = DataParallel(model)
    model.to(args.device)

    dl = DataLoader(
        dataset,
        drop_last=False,
        collate_fn=dataset.collate_fn,
        batch_size=args.batch_size,
        sampler=RandomSampler(dataset),
    )

    optimizer = optim.SparseAdam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dl))

    global_step = 0
    epochs_trained = 0

    # Loss
    train_loss, log_loss = 0.0, 0.0
    checkpoint_idx = 0

    model.zero_grad()

    # Iterator
    train_iterator = trange(epochs_trained, args.num_train_epochs, desc="Epoch")

    for _ in train_iterator:
        epoch_iterator = tqdm(dl, desc="Steps")
        for _, batch in enumerate(epoch_iterator):
            model.train()
            word_pos, ctx, neg = batch
            word_pos = word_pos.to(args.device)
            ctx = ctx.to(args.device)
            neg = neg.to(args.device)
            loss = model(word_pos, ctx, neg)

            train_loss += loss

            loss.backward()

            optimizer.step()
            scheduler.step()
            model.zero_grad()

            global_step += 1

            if global_step % args.save_steps == 0:
                print(f"Loss at {global_step} --> {train_loss/global_step}")
                model.save(args.output_dir, f"checkpoint_{checkpoint_idx}")
                checkpoint_idx += 1
        model.save(args.output_dir, "word_embedding")
        epoch_iterator.close()

    train_iterator.close()
    train_loss = train_loss / global_step
    print("Best epoch: {}".format(train_loss))
    print(">>> End of Training <<<")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n > Shutdown")