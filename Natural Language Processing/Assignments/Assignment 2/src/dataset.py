import os
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim import corpora, utils
from smart_open import open
from src.utils import load_dict, load_pickle, save_pickle
from torch.utils.data import Dataset


class WikipediaCorpus(Dataset):
    def __init__(self, file_name, window_size=2, neg_sample_size=5, cache_dir="cached"):
        self.window_size = window_size
        self.neg_sample_size = neg_sample_size
        self.cache_dir = cache_dir

        # Create vocabulary dictionary
        self.vocab_dict = self.get_vocab_dict()
        self.word_to_id = self.vocab_dict.token2id
        self.words = list(self.word_to_id.values())

        # Create n-grams
        # self.ngram_files = os.listdir(f"{cache_dir}/ngrams")
        self.input_file = open(file_name, encoding="utf8")
        self.embed_size = len(self.vocab_dict)
        self.input_len = 1600000
        print(
            f"--- Created dataset -> {len(self.vocab_dict) -1} words and {self.input_len} ngrams ---"
        )

    def get_vocab_dict(self):
        print(">>> Getting vocabulary dictionary")
        return load_dict(self.cache_dir, "vocab_dict")

    def get_neg_sample(self, target):
        neg_samples = []
        while len(neg_samples) < self.neg_sample_size:
            neg_sample = random.sample(self.words, 1)
            if neg_sample != target:
                neg_samples.append(neg_sample[0])
        return neg_samples

    def __len__(self):
        return self.input_len

    def __getitem__(self, idx):
        while True:
            line = self.input_file.readline()
            if not line:
                self.input_file.seek(0, 0)
                line = self.input_file.readline()

            if len(line) > 1:
                words = utils.simple_preprocess(line)

                if len(words) > 1:
                    word_ids = [
                        self.word_to_id[w] for w in words if w in self.word_to_id
                    ]

                    tem = [
                        (u, v, self.get_neg_sample(v))
                        for i, u in enumerate(word_ids)
                        for j, v in enumerate(
                            word_ids[
                                max(i - self.window_size, 0) : i + self.window_size
                            ]
                        )
                        if u != v
                    ]
                    if tem:
                        return tem
        # ngrams = load_pickle(f"{self.cache_dir}/ngrams", self.ngram_files[idx])
        # word_pos = [ngram[0] for ngram in ngrams]
        # ctx_pos = [ngram[1] for ngram in ngrams]
        # neg_pos = [self.get_neg_sample(pos) for pos in ctx_pos]
        # return {"center": word_pos, "context": ctx_pos, "neg": neg_pos}

    @staticmethod
    def collate_fn(batches):
        all_u = [u for batch in batches for u, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _ in batch if len(batch) > 0]
        all_neg_v = [
            neg_v for batch in batches for _, _, neg_v in batch if len(batch) > 0
        ]

        return (
            torch.LongTensor(all_u),
            torch.LongTensor(all_v),
            torch.LongTensor(all_neg_v),
        )
        # word_pos = torch.LongTensor(batch[0]["center"])
        # ctx_pos = torch.LongTensor(batch[0]["context"])
        # neg_pos = torch.LongTensor(batch[0]["neg"])
        # return word_pos, ctx_pos, neg_pos
