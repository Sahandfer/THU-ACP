import pickle
import torch
import pandas as pd
from src.utils import load_dict
from nltk.metrics.spearman import spearman_correlation as spc


def read_test_data():
    df = pd.read_csv("wordsim353/combined.csv")
    words = df["Word 1"].to_list()
    ctx = df["Word 2"].to_list()
    scores = df["Human (mean)"].tolist()
    scores = [score / 10 for score in scores]

    return words, ctx, scores


def calc_sim(embed, vocab_dict, word, ctx_word):
    try:
        pdist = torch.nn.PairwiseDistance()
        word_pos = vocab_dict.token2id[word]
        ctx_pos = vocab_dict.token2id[ctx_word]

        word_embed = embed(torch.LongTensor([word_pos]))
        ctx_embed = embed(torch.LongTensor([ctx_pos]))

        return float(pdist(word_embed, ctx_embed))
    except Exception:
        return -100


def test():
    words, ctx, scores = read_test_data()
    with open(f"output/word_embedding.pickle", "rb") as f:
        embed = pickle.load(f)
    embed = torch.nn.Embedding.from_pretrained(torch.Tensor(embed))
    vocab_dict = load_dict("cached", "vocab_dict")
    ranks = [calc_sim(embed, vocab_dict, words[i], ctx[i]) for i in range(len(words))]

    s = {}
    r = {}
    for i in range(len(ranks)):
        if ranks[i] != -100:
            r[ctx[i]] = ranks[i]
            s[ctx[i]] = scores[i]

    print(spc(r, s))


if __name__ == "__main__":
    test()
