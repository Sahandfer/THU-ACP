import os
import numpy as np
import random
import math
import pandas as pd
from tqdm import tqdm
from gensim import corpora, utils
from smart_open import open
from src.utils import save_dict, save_pickle

from nltk.corpus import stopwords


def read_file(file_name="wiki_t.txt"):
    print(f">>> Reading {file_name}")
    stop_words = set(stopwords.words("english"))
    data = []
    for line in tqdm(open(file_name)):
        # Remove punctuations
        line = utils.simple_preprocess(line)
        # Remove stop words
        line = [w for w in line if w not in stop_words]
        # Remove words like aa, aaa, aaaa
        line = [w for w in line if set(w) != set(w[0])]
        data.append(line)
    return data


def create_vocab_dict(data, save_vocab=False):
    print(">>> Creating vocabulary dictionary")
    vocab_dict = corpora.Dictionary(data)
    few_ids = [token for token, freq in vocab_dict.cfs.items() if freq < 6]
    vocab_dict.filter_tokens(few_ids)
    vocab_dict.compactify()
    if save_vocab:
        save_dict("cached", "vocab_dict", vocab_dict)
    return vocab_dict


def sub_sampling(vocab_dict):
    print(">>> Subsampling")
    data = []
    words_count = vocab_dict.cfs
    num_words = sum(words_count.values())
    words_count_ratio = {w: words_count[w] / num_words for w in words_count}
    for word in tqdm(words_count):
        ratio = words_count_ratio[word]
        word_prob = (math.sqrt(ratio / 1e-3) + 1) * (1e-3 / ratio)
        rand_prob = random.random()
        if rand_prob < word_prob:
            data += [vocab_dict[word]] * words_count[word]

    return [data]


def create_neg_samples(vocab_dict):
    print(">>> Creating negative samples list")
    neg_samples = []
    words_count = vocab_dict.cfs
    for w in words_count:
        neg_samples += [w] * words_count[w]
    random.shuffle(neg_samples)
    save_pickle("cached", "neg_samples", neg_samples)

    return neg_samples


def get_neg_samples(samples, idx=0, neg_sample_size=5):
    neg_samples = samples[idx : idx + neg_sample_size]
    if len(neg_samples) != neg_sample_size:
        idx = 0
        neg_samples += samples[idx : idx + neg_sample_size - len(neg_samples)]
    else:
        idx += neg_sample_size

    return neg_samples, idx


def create_ngrams(data, vocab_dict, neg_table, window_size=2):
    ngrams = []
    neg_idx = 0
    idx = 0
    for words in tqdm(data):
        for i in range(len(words)):
            word_id = vocab_dict.token2id.get(words[i])
            if word_id != None:
                left_words = words[max(i - window_size, 0) : i]
                right_words = words[i + 1 : i + 1 + window_size]
                ctx_words = left_words + right_words
                ctx_word_ids = [
                    vocab_dict.token2id[word]
                    for word in ctx_words
                    if vocab_dict.token2id.get(word) != None
                ]
                neg_samples, neg_idx = get_neg_samples(neg_table, neg_idx)
                ngrams += [[word_id, ctx_id] + neg_samples for ctx_id in ctx_word_ids]
                if len(ngrams) >= 1000000:
                    save_pickle("cached/ngrams", f"ngrams_{idx}", ngrams)
                    idx += 1
                    ngrams = []

    print("Finished")
    save_pickle("cached/ngrams", f"ngrams_{idx}", ngrams)


def preprocess():
    data = read_file()
    v_d = create_vocab_dict(data)
    print(v_d)
    sampled_data = sub_sampling(v_d)
    vocab_dict = create_vocab_dict(sampled_data, True)
    print(vocab_dict)
    neg_table = create_neg_samples(vocab_dict)
    create_ngrams(data, vocab_dict, neg_table)


if __name__ == "__main__":
    preprocess()