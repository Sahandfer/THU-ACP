import torch
from torchtext.legacy import data
from torchtext.legacy import datasets
from torchtext.vocab import Vectors


def preprocess(args):
    # The input data has two attributes -> text and label (sentiment)
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL = data.Field(sequential=False, dtype=torch.long)

    # Split data into training, validation, and testing sets
    train, val, test = datasets.SST.splits(
        TEXT, LABEL, fine_grained=True, train_subtrees=False
    )
    print(
        f">>> Created Sets -> {len(train)} Training - {len(val)} Validation - {len(test)} Test"
    )

    # Build the vocabulary from pre-trained embeddings
    TEXT.build_vocab(train, vectors=Vectors(name="vector.txt", cache="../data"))
    LABEL.build_vocab(train)
    num_words, embed_dim = TEXT.vocab.vectors.size()
    print(f">>> Created Vocabulary Dictionary")
    print(f"| # Words: {num_words} --- Embedding Dim: {embed_dim} |")

    # Create batch iterators for the datasets
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_size=args.batch_size
    )
    print(f">>> Created Batch Iterators with size {args.batch_size}")

    return TEXT, train_iter, val_iter, test_iter