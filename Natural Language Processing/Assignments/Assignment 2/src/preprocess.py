import os
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from gensim import corpora, utils
from smart_open import open
from src.utils import save_dict, save_pickle


def create_vocab_dict(file_name="wiki_t.txt"):
    data = []
    n = 0
    for line in tqdm(open(file_name)):
        line = utils.simple_preprocess(line)
        data.append(line)
        n += 1
    print(n)
    vocab_dict = corpora.Dictionary(data)
    few_ids = [token for token, freq in vocab_dict.cfs.items() if freq < 2]
    vocab_dict.filter_tokens(few_ids)
    vocab_dict.compactify()
    save_dict("cached", "vocab_dict", vocab_dict)


def temp():
    window_size = 2
    file_reader = pd.read_csv("tt.txt", delimiter="\n", chunksize=1, header=None)
    vocab_dict = corpora.Dictionary([[]])
    ngrams = []
    idx = 0
    while True:
        try:
            data = file_reader.get_chunk()[0].to_list()
            for j in range(len(data)):
                line = data[j]
                words = utils.simple_preprocess(line)
                vocab_dict.add_documents([words])

                for i in range(len(words)):
                    word_id = vocab_dict.token2id.get(words[i], -1)
                    if word_id != -1:
                        left_words = words[max(i - window_size, 0) : i]
                        right_words = words[i + 1 : i + 1 + window_size]
                        ctx_words = left_words + right_words
                        ctx_word_ids = [
                            vocab_dict.token2id[word]
                            for word in ctx_words
                            if vocab_dict.token2id.get(word, -1) != -1
                        ]
                        ngrams += [[word_id, ctx_id] for ctx_id in ctx_word_ids]
            if len(ngrams) >= 1000000:
                save_pickle("cached/ngrams", f"ngrams_{idx}", ngrams)
                idx += 1
                ngrams = []

        except Exception:
            save_dict("cached", "vocab_dict", vocab_dict)
            save_pickle("cached/ngrams", f"ngrams_{idx}", ngrams)
            break


if __name__ == "__main__":
    create_vocab_dict()

    # if j:
    #     neg_words = []
    #     while len(neg_words) < 2:
    #         neg_sample = random.sample(vocabs, 1)
    #         if neg_sample != [-1]:
    #             neg_words.append(neg_sample[0])
    #     prev_ngram = []
