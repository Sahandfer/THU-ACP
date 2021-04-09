import os
import torch
import random
import numpy as np
from tqdm import tqdm
from gensim import corpora, utils
from smart_open import open
from src.utils import load_dict, load_pickle, save_dict, save_pickle
from torch.utils.data import Dataset


def process_data(file_name):
    data = []
    for line in tqdm(open(file_name)):
        line = utils.simple_preprocess(line)
        data.append(line)
    return data


def get_neg_samples():
    return 0


class WikipediaCorpus(Dataset):
    def __init__(self, file_name, window_size=2, neg_sample_size=2, cache_dir="cached"):
        self.vocab_exists = os.path.isfile(f"{cache_dir}/vocab_dict")
        self.ngrams_exists = os.path.isfile(f"{cache_dir}/ngrams.pickle")

        self.window_size = window_size
        self.neg_sample_size = neg_sample_size
        self.cache_dir = cache_dir

        # Create vocabulary dictionary
        if not (self.vocab_exists and self.ngrams_exists):
            self.data = process_data(file_name)
        self.vocab_dict = self.get_vocab_dict()
        self.id_to_word = self.vocab_dict.id2token
        self.word_to_id = self.vocab_dict.token2id
        self.word_counts = self.vocab_dict.cfs

        # Create n-grams
        self.create_ngrams()
        self.input_len = len(self.ngrams)
        print(
            f"--- Created dataset -> {len(self.vocab_dict)} words and {self.input_len} ngrams ---"
        )

    def get_vocab_dict(self):
        print(">>> Getting vocabulary dictionary")
        if self.vocab_exists:
            vocab_dict = load_dict(self.cache_dir, "vocab_dict")
        else:
            vocab_dict = corpora.Dictionary(self.data)
            once_ids = [token for token, freq in vocab_dict.cfs.items() if freq < 6]
            vocab_dict.filter_tokens(once_ids)
            vocab_dict.compactify()
            save_dict(self.cache_dir, "vocab_dict", vocab_dict)

        return vocab_dict

    def create_ngrams(self):
        print(">>> Getting N-grams")
        self.ngrams = load_pickle(self.cache_dir, "ngrams")
        if not self.ngrams:
            self.ngrams = []
            idx = 0
            num_lines = 0
            for words in tqdm(self.data):
                num_lines += 1
                for i in range(len(words)):
                    if self.word_to_id.get(words[i], -1) > -1:
                        word_id = self.word_to_id[words[i]]
                        left_words = words[max(i - self.window_size, 0) : i]
                        right_words = words[i + 1 : i + 1 + self.window_size]
                        ctx_words = left_words + right_words
                        ctx_word_ids = [
                            self.word_to_id[word]
                            for word in ctx_words
                            if self.word_to_id.get(word, -1) > -1
                        ]
                        ngrams = [[word_id, ctx_id] for ctx_id in ctx_word_ids]
                        self.ngrams += ngrams
                        if num_lines >= 500000:
                            num_lines = 0
                            save_pickle(self.cache_dir, f"ngrams_{idx}", self.ngrams)
                            idx += 1
                            self.ngrams = []

            save_pickle(self.cache_dir, "ngrams", self.ngrams)

    def get_neg_sample(self, target):
        num_words = list(self.word_to_id.values())
        neg_samples = []
        while len(neg_samples) < self.neg_sample_size:
            neg_sample = random.sample(num_words, 1)
            if neg_sample != target:
                neg_samples.append(neg_sample)
        return neg_samples

    def __len__(self):
        return self.input_len

    def __getitem__(self, idx):
        ngram = self.ngrams[idx]
        word_pos = ngram[0]
        ctx_pos = ngram[1]
        neg_pos = self.get_neg_sample(ctx_pos)
        return {"center": word_pos, "context": ctx_pos, "neg": neg_pos}

    @staticmethod
    def collate_fn(batch):
        word_pos = torch.LongTensor([pos["center"] for pos in batch])
        ctx_pos = torch.LongTensor([pos["context"] for pos in batch])
        neg_pos = torch.LongTensor([pos["neg"] for pos in batch])
        return word_pos, ctx_pos, neg_pos
