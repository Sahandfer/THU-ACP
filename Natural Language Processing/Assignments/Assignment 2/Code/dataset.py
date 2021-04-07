import os
import random
import numpy as np
from tqdm import tqdm
from gensim import corpora, utils
from smart_open import open
from util import load_dict, load_pickle, save_dict, save_pickle
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
        print(f"--- Created dataset -> {self.input_len} words ---")

    def get_vocab_dict(self):
        print(">>> Getting vocabulary dictionary")
        if self.vocab_exists:
            vocab_dict = load_dict(f"{self.cache_dir}/vocab_dict")
        else:
            vocab_dict = corpora.Dictionary(self.data)
            once_ids = [token for token, freq in vocab_dict.cfs.items() if freq == 1]
            vocab_dict.filter_tokens(once_ids)
            vocab_dict.compactify()
            save_dict(vocab_dict, f"{self.cache_dir}/vocab_dict")

        return vocab_dict

    def create_ngrams(self):
        print(">>> Getting N-grams")
        self.ngrams = load_pickle(f"{self.cache_dir}/ngrams")
        if not self.ngrams:
            self.ngrams = {}
            for words in tqdm(self.data):
                for i in range(len(words)):
                    if self.word_to_id.get(words[i], 0):
                        left_words = words[max(i - self.window_size, 0) : i]
                        right_words = words[i + 1 : i + 1 + self.window_size]
                        self.ngrams[words[i]] = left_words + right_words
            save_pickle(f"{self.cache_dir}/ngrams", self.ngrams)
        print(f"--- Create N-grams ---")

    def get_neg_sample(self):
        return random.sample(list(self.word_to_id.keys()), self.neg_sample_size)

    def __len__(self):
        return self.input_len

    def __getitem__(self, idx):
        return self.ngrams[idx]

    @staticmethod
    def collate_fn(self, batch):
        return batch