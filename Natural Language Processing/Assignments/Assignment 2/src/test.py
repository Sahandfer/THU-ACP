import pickle
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import spearmanr
from nltk.metrics.spearman import spearman_correlation as spc


def read_wordsim():
    df = pd.read_csv("wordsim353/combined.csv")
    words = df["Word 1"].to_list()
    ctx = df["Word 2"].to_list()
    scores = df["Human (mean)"].tolist()

    return words, ctx, scores


def get_visual_samples():
    countries = [
        "china",
        "usa",
        "france",
        "germany",
        "spain",
        "portugal",
    ]

    cities = [
        "beijing",
        "shanghai",
        "washington",
        "paris",
        "berlin",
        "madrid",
        "lisbon",
    ]

    companies = [
        "facebook",
        "apple",
        "amazon",
        "google",
        "github",
        "yahoo",
        "netflix",
    ]

    return countries + cities + companies


def get_analogy_pairs():

    pairs = [
        ["England", "London", "France", "Paris"],
    ]

    return pairs


class Test:
    def __init__(self):
        self.embed_dim = 300
        with open(f"output/word_embedding_{self.embed_dim}", "rb") as f:
            word_embed = pickle.load(f)
        self.word_embed = torch.nn.Embedding.from_pretrained(torch.Tensor(word_embed))

        with open(f"cached/word_to_id", "rb") as f:
            self.word_to_id = pickle.load(f)

        with open(f"cached/id_to_word", "rb") as f:
            self.id_to_word = pickle.load(f)

        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    def calc_sim(self, word_1, word_2):
        try:
            word_pos = self.word_to_id[word_1]
            ctx_pos = self.word_to_id[word_2]

            word_embed = self.word_embed(torch.LongTensor([word_pos]))
            ctx_embed = self.word_embed(torch.LongTensor([ctx_pos]))

            sim_score = self.cos_sim(word_embed, ctx_embed)

            return sim_score.item()
        except Exception:
            return ""

    def calc_spearman(self, words, ctx_words, scores):
        ranks = []
        human = []
        for i in range(len(words)):
            score = self.calc_sim(words[i], ctx_words[i])
            if score:
                ranks.append(score)
                human.append(scores[i])

        corr = spearmanr(ranks, human).correlation
        print(f">>> Spearman Correlation -> {corr*100}%")

    def find_analogy(self, w_1, w_2, w_3):
        try:
            w_1 = w_1.lower()
            w_2 = w_2.lower()
            w_3 = w_3.lower()

            w_1_pos = self.word_to_id[w_1]
            w_2_pos = self.word_to_id[w_2]
            w_3_pos = self.word_to_id[w_3]

            w_1_embed = self.word_embed(torch.LongTensor([w_1_pos]))
            w_2_embed = self.word_embed(torch.LongTensor([w_2_pos]))
            w_3_embed = self.word_embed(torch.LongTensor([w_3_pos]))

            max_sim = -1000
            w_4_pos = None

            for w in self.id_to_word:
                if w in [w_1_pos, w_2_pos, w_3_pos]:
                    continue
                w_4_embed = self.word_embed(torch.LongTensor([w]))
                sim = self.cos_sim(w_2_embed - w_1_embed, w_4_embed - w_3_embed)

                if sim > max_sim:
                    max_sim = sim
                    w_4_pos = w

            return self.id_to_word[w_4_pos]
        except Exception:
            return ""

    def calc_analogy_score(self, pairs):
        correct = 0
        total = 0

        for pair in tqdm(pairs):
            pred = self.find_analogy(pair[0], pair[1], pair[2])
            gt = pair[3].lower()
            if pred:
                print("words", pair[0], pair[1], pair[2])
                print("wanted ", gt)
                print("got ", pred)
                if pred == gt:
                    correct += 1
                total += 1

        acc = correct / total
        print(f">>> Analogy Accuracy -> {acc*100}%")

    def plot_embeddings(self, words):
        pca = PCA(n_components=2)
        word_ids = [self.word_to_id[w] for w in words]
        word_embeds = [
            self.word_embed(torch.LongTensor([i])).numpy()[0] for i in word_ids
        ]

        embeds = pca.fit_transform(word_embeds)
        plt.figure(figsize=(10, 12), dpi=100)
        plt.plot(embeds[:, 0], embeds[:, 1], ".")
        for i in range(len(embeds)):
            plt.annotate(words[i], xy=embeds[i])
        plt.show()


def test():
    words, ctx, scores = read_wordsim()
    samples = get_visual_samples()
    pairs = get_analogy_pairs()

    test = Test()

    # Calculate Spearman's Correlation
    test.calc_spearman(words, ctx, scores)

    # Visualize Word Embeddings
    test.plot_embeddings(samples)

    # Calculate Analogy Accuracy
    # test.calc_analogy_score(pairs)


if __name__ == "__main__":
    test()
