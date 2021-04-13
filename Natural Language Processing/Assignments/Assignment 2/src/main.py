import torch
import argparse
from src.train import train
from src.test import test
from src.model import Word2Vec
from src.dataset import VocabularyDict, WikipediaCorpus


class Args:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser()

        parser.add_argument("--do_train", default=False, action="store_true")
        parser.add_argument("--do_test", default=False, action="store_true")
        parser.add_argument("--use_cuda", default=False, action="store_true")
        parser.add_argument("--dataset", default="wiki_t.txt", type=str)
        parser.add_argument("--output_dir", default="output", type=str)
        parser.add_argument("--cache_dir", default="cached", type=str)
        parser.add_argument("--batch_size", default=128, type=int)
        parser.add_argument("--window_size", default=5, type=int)
        parser.add_argument("--min_discard", default=10, type=int)
        parser.add_argument("--neg_sample_size", default=5, type=int)
        parser.add_argument("--embedding_size", default=100, type=int)
        parser.add_argument("--learning_rate", default=0.001, type=float)
        parser.add_argument("--weight_decay", default=0.0, type=float)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--max_steps", default=-1, type=int)
        parser.add_argument("--log_loss", default=10000, type=int)
        parser.add_argument("--num_train_epochs", default=5, type=int)

        self.parser = parser

    def get_args(self) -> argparse.ArgumentParser:
        return self.parser.parse_args()


args = Args().get_args()
args.device = "cuda" if args.use_cuda else "cpu"
args.device = torch.device(args.device)


def main():
    print("> Starting the program")

    # Vocabulary Dictionary
    vocab_dict = VocabularyDict(
        filename=args.dataset,
        min_discard=args.min_discard,
        cache_dir=args.cache_dir,
    )
    # Dataset
    dataset = WikipediaCorpus(
        filename=args.dataset,
        vocab_dict=vocab_dict,
        window_size=args.window_size,
        neg_sample_size=args.neg_sample_size,
    )

    # Model
    vocab_size = len(vocab_dict.word_count)
    model = Word2Vec(vocab_size, args.embedding_size)
    model.to(args.device)

    if args.do_train:
        train(model, dataset, args)

    if args.do_test:
        test()


if __name__ == "__main__":
    main()