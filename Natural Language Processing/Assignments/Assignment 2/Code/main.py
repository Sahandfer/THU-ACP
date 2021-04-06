import torch
import argparse


class Args:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser()

        parser.add_argument("--do_train", default=False, action="store_true")
        parser.add_argument("--do_eval", default=False, action="store_true")
        parser.add_argument("--do_test", default=False, action="store_true")
        parser.add_argument("--dataset", default="wiki.txt", type=str)
        parser.add_argument("--output_dir", default="output/Bert", type=str)
        parser.add_argument("--cache_dir", default="cached", type=str)
        parser.add_argument("--batch_size", default=42, type=int)
        parser.add_argument("--n_gpu", default=torch.cuda.device_count(), type=int)
        parser.add_argument("--learning_rate", default=5e-5, type=float)
        parser.add_argument("--weight_decay", default=0.0, type=float)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--max_grad_norm", default=1.0, type=float)
        parser.add_argument("--seed", default=42, type=int)
        parser.add_argument("--max_steps", default=-1, type=int)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--log_steps", default=1000, type=int)
        parser.add_argument("--save_steps", default=50000, type=int)
        parser.add_argument("--num_train_epochs", default=5, type=int)
        parser.add_argument("--gradient_accumulation_steps", default=1, type=int)

        self.parser = parser

    def get_args(self) -> argparse.ArgumentParser:
        return self.parser.parse_args()


args = Args().get_args()


def main():
    print("> Starting the program")


if __name__ == "__main__":
    main()