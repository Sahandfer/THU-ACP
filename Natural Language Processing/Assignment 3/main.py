import os
import re
import nltk
import argparse
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from src.test import test
from src.train import train
from torchtext.legacy import data
from torchtext.vocab import Vectors
from src.model import BiLSTM, LSTMTree
from nltk.tree import ParentedTree as PT


class Args:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--batch_size", default=64, type=int)
        parser.add_argument("--num_epochs", default=10, type=int)
        parser.add_argument("--num_layers", default=1, type=int)
        parser.add_argument("--hidden_dim", default=300, type=int)
        parser.add_argument("--learning_rate", default=0.001, type=float)
        parser.add_argument("--dropout", default=0.2, type=float)
        parser.add_argument("--use_glove", default=False, action="store_true")
        parser.add_argument("--use_improved", default=False, action="store_true")

        self.parser = parser

    def get_args(self):
        return self.parser.parse_args()


def read_data():
    data_dir = "trees/"
    files = ["train.txt", "dev.txt", "test.txt"]
    out_files = ["train.csv", "dev.csv", "test.csv"]
    for i, file in enumerate(files):
        sentences = []
        labels = []
        for line in tqdm(open(data_dir + file)):
            tokenizer = nltk.RegexpTokenizer(r"\w+")
            # Remove digits
            sentence = re.sub("\d+", "", line)
            # Remove non-words
            sentence = tokenizer.tokenize(sentence)
            # Re-create the sentence
            sentence = " ".join(sentence)
            sentences.append(sentence)

            # Target label is the first digit in the line
            label = line[1]
            labels.append(label)

        # Save files to CSV for torchtext tabular dataset.
        df = pd.DataFrame(list(zip(sentences, labels)), columns=["sentence", "label"])
        df.to_csv(data_dir + out_files[i], index=False)


def preprocess_data(args):
    # The input data has two attributes -> text and label (sentiment)
    SENTENCE = data.Field(
        lower=True,
        sequential=True,
        batch_first=True,
        eos_token="<eos>",
        init_token="<sos>",
        include_lengths=True,
        tokenize=nltk.word_tokenize,
    )
    LABEL = data.Field(sequential=False, use_vocab=False, is_target=True)

    # Split data into training, validation, and testing sets
    fields = [("sentence", SENTENCE), ("label", LABEL)]
    train_set, val_set, test_set = data.TabularDataset.splits(
        path="./trees",
        format="csv",
        train="train.csv",
        validation="dev.csv",
        test="test.csv",
        fields=fields,
        skip_header=True,
    )

    print(
        f">>> Created Sets -> {len(train_set)} Training - {len(val_set)} Validation - {len(test_set)} Test"
    )

    # Build the vocabulary from pre-trained embeddings
    SENTENCE.build_vocab(
        train_set, vectors=Vectors(name="glove.6B.300d.txt", cache="./")
    )
    LABEL.build_vocab(train_set)
    num_words, embed_dim = SENTENCE.vocab.vectors.size()
    print(f">>> Created Vocabulary Dictionary")
    print(f"| # Words: {num_words} --- Embedding Dim: {embed_dim} |")

    # Create batch iterators for the datasets
    train_iter = data.BucketIterator(
        train_set,
        batch_size=args.batch_size,
    )
    val_iter = data.BucketIterator(
        val_set,
        batch_size=args.batch_size,
    )
    test_iter = data.BucketIterator(
        test_set,
        batch_size=args.batch_size,
    )

    print(f">>> Created Batch Iterators with size {args.batch_size}")

    return SENTENCE, LABEL, train_iter, val_iter, test_iter


def get_trees(vocab_dict):
    data_dir = "trees/"
    files = ["train.txt", "dev.txt", "test.txt"]
    trees_list = []
    for file in files:
        trees = []
        for line in tqdm(open(data_dir + file)):
            tree = PT.fromstring(line)
            trees.append(tree)

        for tree in trees:
            for leaf in tree.treepositions("leaves"):
                val = vocab_dict.get(tree[leaf], -1)
                tree[leaf] = val if val != -1 else 0
            for subtree in tree.subtrees():
                subtree.set_label(int(subtree.label()))

        trees_list.append(trees)

    return trees_list


def main():
    args = Args().get_args()
    if not (os.path.exists("trees/dev.csv")):
        read_data()  # Create CSV files if they don't exists
    SENTENCE, LABEL, train_iter, val_iter, test_iter = preprocess_data(args)
    # Sentiment Classification Model (Bi_LSTM)
    if args.use_improved:
        # Read the Sentiment Trees
        train_iter, val_iter, test_iter = get_trees(SENTENCE.vocab.stoi)
        # The Tree LSTM model
        model = LSTMTree(args, SENTENCE, LABEL)
    else:
        # The Bidirectional LSTM model
        model = BiLSTM(args, SENTENCE, LABEL)
    # Cross Entropy loss
    criterion = nn.CrossEntropyLoss()
    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # Training
    _, model = train(
        model,
        train_iter,
        val_iter,
        optimizer,
        criterion,
        args.num_epochs,
        args.use_improved,
    )
    # Testing
    test(model, test_iter, args.use_improved)


if __name__ == "__main__":
    main()