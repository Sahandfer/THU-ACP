import os
import pickle
import numpy as np
from tqdm import tqdm, trange
from gensim import corpora
from smart_open import open
from nltk.corpus import stopwords
from torch.utils.data import Dataset


def read_data(file_name):
    return open(file_name, encoding="utf-8")


def get_stop_words():
    stop_words = set(stopwords.words("english"))
    stop_words.update([".", ",", ":", ";", "(", ")", "#", "--", "...", '"'])
    return stop_words


def get_vocab_dict(input_file):
    print(">>> Getting vocabulary dictionary")
    if os.path.isfile("vocab_dict"):
        vocab_dict = load_dict()
    else:
        vocab_dict = corpora.Dictionary(line.lower().split() for line in input_file)
        stop_words = get_stop_words()
        # Remove common stop words
        stop_ids = [
            vocab_dict.token2id[stopword]
            for stopword in stop_words
            if stopword in vocab_dict.token2id
        ]
        # Remove words with few occurences
        few_ids = [
            tokenid for tokenid, docfreq in vocab_dict.dfs.items() if docfreq == 6
        ]
        vocab_dict.filter_tokens(stop_ids + few_ids)
        # Reset in the dict index (because some entries were removed)
        vocab_dict.compactify()
        save_dict(vocab_dict)

    return vocab_dict


def save_dict(vocab_dict):
    vocab_dict.save("vocab_dict")
    print(f"**Saved dictionary**")


def load_dict():
    print(f"**Loading cached dictionary**")
    return corpora.Dictionary.load("vocab_dict")


def get_ngrams(input_file, window_size):
    print(">>> Getting N-grams")
    ngrams = load_pickle("ngrams")
    if not ngrams:
        ngrams = {}
        for sent in input_file:
            words = sent.split()
            for i in range(len(words)):
                left_words = words[max(i - window_size, 0) : i]
                right_words = words[i + 1 : i + 1 + window_size]
                ngrams[words[i]] = left_words + right_words
        save_pickle("ngrams", ngrams)

    return ngrams


def get_neg_samples():
    return 0


def get_lookup_table(vocab_dict):
    print(">>> Getting lookup table")
    word_to_id = load_pickle("word_2_id")
    if not word_to_id:
        word_to_id = {value: key for key, value in vocab_dict.items()}
        save_pickle("word_2_id", word_to_id)
    return word_to_id


def load_pickle(file_name):
    try:
        with open(f"{file_name}.pickle", "rb") as f:
            print(f"**Loading cached {file_name}**")
            return pickle.load(f)
    except Exception:
        return {}


def save_pickle(file_name, data):
    with open(f"{file_name}.pickle", "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print(f"**Saved {file_name}**")


class WikipediaCorpus(Dataset):
    def __init__(self, file_name, window_size, cache_dir="cached"):
        self.data = read_data(file_name)
        self.vocab_dict = get_vocab_dict(self.data)
        self.lookup = get_lookup_table(self.vocab_dict)
        self.ngrams = get_ngrams(self.data, window_size)
        self.window_size = window_size
        self.input_len = len(self.ngrams)

        print(f"**Created dataset -> {self.input_len} words")

    def __len__(self):
        return self.input_len

    def __getitem__(self, idx):
        return self.ngrams[idx]

    @staticmethod
    def collate_fn(self, batch):
        return batch