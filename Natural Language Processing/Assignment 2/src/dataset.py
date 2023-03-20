import os
import math
import torch
import random
import numpy as np
from tqdm import tqdm
from gensim import utils
from smart_open import open
from nltk.corpus import stopwords
from torch.utils.data import Dataset
from src.utils import load_pickle, save_pickle


class VocabularyDict:
    def __init__(self, filename, min_discard=6, cache_dir="cached"):
        self.num_lines = 0
        self.num_words = 0
        self.word_count = {}
        self.word_to_id = {}
        self.id_to_word = {}
        self.neg_samples = []
        self.cache_dir = cache_dir
        self.min_discard = min_discard
        self.filename = filename

        # Load cache if available
        if os.path.isdir(self.cache_dir):
            self.load_state()
        else:
            self.process_file()
            self.create_dict()
            self.sub_sample()
            self.neg_sample()
            self.save_state()

    def process_file(self):
        print(f">>> Reading {self.filename}")
        stop_words = set(stopwords.words("english"))
        for line in tqdm(open(self.filename)):
            # Remove punctuations
            line = utils.simple_preprocess(line)
            # Remove stop words
            line = [w for w in line if w not in stop_words]
            # Remove words like aa, aaa, aaaa
            line = [w for w in line if set(w) != set(w[0])]

            self.num_lines += 1
            self.num_words += len(line)

            for word in line:
                self.word_count[word] = self.word_count.get(word, 0) + 1

        print(f"*** Read file -> {self.num_lines} lines and {self.num_words} words***")

    def create_dict(self):
        print(">>> Creating dictionary")
        idx = 0
        discarded = []
        for word, count in tqdm(self.word_count.items()):
            if count >= self.min_discard:
                self.word_to_id[word] = idx
                self.id_to_word[idx] = word
                idx += 1
            else:
                discarded.append(word)

        # Discard barely frequent words
        for word in discarded:
            self.num_words -= self.word_count[word]
            self.word_count.pop(word, None)
        print("*** Created dictionary ***")

    def sub_sample(self):
        print(">>> Subsampling data")
        discarded = []
        word_count = self.word_count
        num_words = self.num_words
        ratios = {w: word_count[w] / num_words for w in word_count}
        for word in tqdm(word_count):
            ratio = ratios[word]
            word_prob = (math.sqrt(ratio / 1e-3) + 1) * (1e-3 / ratio)
            rand_prob = random.random()
            if rand_prob > word_prob:
                discarded.append(word)

        # Discard words with low probability ratio
        for word in discarded:
            self.num_words -= self.word_count[word]
            self.word_count.pop(word, None)
            w_id = self.word_to_id.pop(word, None)
            self.id_to_word.pop(w_id, None)

        self.reset_keys()
        print("*** Subsampled data ***")

    def reset_keys(self):
        # Reset indexes since some words are removed from the dictionary
        self.id_to_word = {idx: val for idx, val in enumerate(self.id_to_word.values())}
        self.word_to_id = {val: idx for idx, val in self.id_to_word.items()}

    def neg_sample(self):
        # Create the negative sample table
        print(">>> Negative sampling data")
        word_count = self.word_count
        for w in word_count:
            self.neg_samples += [w] * word_count[w]
        random.shuffle(self.neg_samples)
        print("*** Negative sampled data ***")

    def save_state(self):
        print(">>> Saving state")
        stats = {"num_lines": self.num_lines, "num_words": self.num_words}
        save_dir = self.cache_dir
        save_pickle(save_dir, "stats", stats)
        save_pickle(save_dir, "word_count", self.word_count)
        save_pickle(save_dir, "word_to_id", self.word_to_id)
        save_pickle(save_dir, "id_to_word", self.id_to_word)
        save_pickle(save_dir, "neg_samples", self.neg_samples)
        print("*** Saved state ***")

    def load_state(self):
        print(">>> Loading state")
        save_dir = self.cache_dir
        stats = load_pickle(save_dir, "stats")
        self.num_lines = stats["num_lines"]
        self.num_words = stats["num_words"]
        self.word_count = load_pickle(save_dir, "word_count")
        self.word_to_id = load_pickle(save_dir, "word_to_id")
        self.id_to_word = load_pickle(save_dir, "id_to_word")
        self.neg_samples = np.array(load_pickle(save_dir, "neg_samples"))
        print("*** Loaded state ***")


class WikipediaCorpus(Dataset):
    def __init__(self, filename, vocab_dict, window_size=2, neg_sample_size=5):
        self.filename = filename
        self.vocab_dict = vocab_dict
        self.window_size = window_size
        self.neg_sample_size = neg_sample_size

        self.neg_idx = 0
        self.input_file = open(self.filename)
        self.input_len = self.vocab_dict.num_lines

    def __len__(self):
        return self.input_len

    def __getitem__(self, _):
        # Read a line from the corpus as a one set of the batch
        line = self.input_file.readline()
        if not line:
            self.input_file.seek(0)
            line = self.input_file.readline()
        words = utils.simple_preprocess(line)
        # Return all the bigram pairs + negative samples for the line
        return self.create_ngrams(words)

    def get_neg_samples(self):
        # Read consequent negative samples from the table
        samples = self.vocab_dict.neg_samples
        neg_samples = samples[self.neg_idx : self.neg_idx + self.neg_sample_size]
        if len(neg_samples) != self.neg_sample_size:
            self.neg_idx = 0
            neg_samples = np.concatenate(
                [
                    neg_samples,
                    samples[
                        self.neg_idx : self.neg_idx
                        + self.neg_sample_size
                        - len(neg_samples)
                    ],
                ]
            )
        else:
            self.neg_idx += self.neg_sample_size

        return neg_samples

    def create_ngrams(self, words):
        ngrams = []
        word_ids = [
            self.vocab_dict.word_to_id[word]
            for word in words
            if word in self.vocab_dict.word_to_id
        ]
        for i in range(len(word_ids)):
            word_id = word_ids[i]
            left_words = word_ids[max(i - self.window_size, 0) : i]
            right_words = word_ids[i + 1 : i + 1 + self.window_size]
            # Words within window size of center words
            ctx_words = left_words + right_words
            neg_samples = self.get_neg_samples()
            neg_sample_ids = [self.vocab_dict.word_to_id[word] for word in neg_samples]
            ngrams += [[word_id, ctx_id] + neg_sample_ids for ctx_id in ctx_words]

        return ngrams

    @staticmethod
    def collate_fn(batch):
        word_pos = torch.LongTensor([n[0] for b in batch for n in b if len(b) > 0])
        ctx_pos = torch.LongTensor([n[1] for b in batch for n in b if len(b) > 0])
        neg_pos = torch.LongTensor([n[2:] for b in batch for n in b if len(b) > 0])
        return word_pos, ctx_pos, neg_pos
