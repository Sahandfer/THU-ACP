import nltk
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn

from tqdm import tqdm
from scipy.stats import spearmanr
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize


def read_scws():
    corpus = open("ratings.txt")

    words_1 = []
    words_2 = []
    sents_1 = []
    sents_2 = []
    scores = []

    for line in corpus:
        s = line.split("\t")

        words_1.append(s[1].lower())
        words_2.append(s[3].lower())
        sents_1.append(s[5])
        sents_2.append(s[6])

        score = np.average([float(i) for i in s[7:]])
        scores.append(score)

    return words_1, words_2, sents_1, sents_2, scores


# Word Sense Disambiguation
class WSD:
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

    def process_sent(self, sentence):
        # Processing and POS-tagging of a given line (sentence)
        stop_words = set(stopwords.words("english"))
        tokens = nltk.word_tokenize(sentence)
        tokens = [token for token in tokens if token not in stop_words]
        tokens = [token for token in tokens if set(token) != set(token[0])]
        pos_tags = nltk.pos_tag(tokens)
        words = [w.lower() for w, tag in pos_tags if tag[0] in ["N", "V", "J", "R"]]

        return words

    def get_sense_vectors(self, word, w_embeds):
        sense_vectors = {}
        init_embed = w_embeds[word]
        # For each sense of a word in WordNet
        for sense in wn.lemmas(word):
            # Find the gloss
            gloss = [sense.synset().definition()]
            gloss.extend(sense.synset().examples())
            vectors = []
            # For each sentence in the gloss (definition) of the word
            for sentence in gloss:
                # Retreive the word vectors from the base embeddigns
                words = self.process_sent(sentence)
                for w in words:
                    try:
                        w_embed = self.word_embed(
                            torch.LongTensor([self.word_to_id[w]])
                        )
                    except Exception:
                        continue
                    # Find the cosine similarity between this word and the initial word
                    sim = self.cos_sim(w_embed, init_embed).item()
                    # If the similarity exceeds the threshold, the word vector is recorded
                    if sim > 0.05:
                        vectors.append(w_embed)
            if len(vectors) == 0:
                continue
            # The sense vector for each sense its the average of its vectors
            sense_vector = torch.mean(torch.stack(vectors), dim=0)
            sense_vectors[sense] = sense_vector
        return sense_vectors

    def get_disambigution(self, word, sense_vectors, ctx_vec):
        sense_vecs = sense_vectors[word]
        if not sense_vecs:
            return [None, 0]

        cos_sims = {}
        # Find cosine similarities between context and different senses
        for sense, sense_vec in sense_vecs.items():
            sim_score = self.cos_sim(sense_vec, ctx_vec)
            cos_sims[sense] = sim_score

        # Sort to find highest similarity
        cos_sims_sorted = sorted(cos_sims.items(), key=lambda x: x[1])
        if not len(cos_sims_sorted):
            return [None, 0]

        # The highest similarity is the disambigution sense
        dis_sense, dis_sim = cos_sims_sorted.pop()

        # Use second highest similarity to find score margin
        second_sim = 0
        if len(cos_sims_sorted):
            second_sim = cos_sims_sorted.pop()[1]
        score_margin = dis_sim - second_sim

        return [dis_sense, score_margin]

    def get_wsd(self, sentence):
        sense_vectors = {}
        sense_vectors_sorted = {}
        words = self.process_sent(sentence)

        w_embeds = {}
        # For each word in the sentence, retrieve the word vectors from the base embedding
        for w in words:
            try:
                w_embeds[w] = self.word_embed(torch.LongTensor([self.word_to_id[w]]))
            except Exception:
                continue

        # For each word that was found in the original vocabulary, find the sense vectors
        for w in w_embeds:
            sense_vecs = self.get_sense_vectors(w, w_embeds)
            sense_vectors[w] = sense_vecs
            sense_vectors_sorted[w] = len(sense_vecs)

        # Sort senses based on their length
        sense_vectors_sorted = sorted(sense_vectors_sorted.items(), key=lambda x: x[1])
        # Initialize context vector
        ctx_vec = torch.mean(torch.stack(list(w_embeds.values())), dim=0)
        # Resolving word disambiguation
        for w, _ in sense_vectors_sorted:
            dis_sense, score_margin = self.get_disambigution(w, sense_vectors, ctx_vec)
            if score_margin > 0.1:
                # Update the word vector if the score margin exceeds the threshold
                w_embeds[w] = sense_vectors[w][dis_sense]
                # Update the context vector since a word vector is updated
                ctx_vec = torch.mean(torch.stack(list(w_embeds.values())), dim=0)

        return w_embeds


def main():
    words_1, words_2, sents_1, sents_2, scores = read_scws()
    wsd = WSD()

    new_ranks = []
    base_ranks = []
    human = []

    new_error = 0
    base_error = 0

    num_sent = 0

    for i in tqdm(range(len(words_1))):
        word_1 = words_1[i]
        word_2 = words_2[i]
        sent_1 = sents_1[i]
        sent_2 = sents_2[i]
        score = scores[i]

        try:
            # Word vectors from the base model
            w1_b_embed = wsd.word_embed(torch.LongTensor([wsd.word_to_id[word_1]]))
            w2_b_embed = wsd.word_embed(torch.LongTensor([wsd.word_to_id[word_2]]))
            # Cosine similarity (base model)
            b_sim_score = abs(wsd.cos_sim(w1_b_embed, w2_b_embed).item())

            # Word vectors from new model (sense embedding)
            word_embed_1 = wsd.get_wsd(sent_1)
            word_embed_2 = wsd.get_wsd(sent_2)
            embed_1 = word_embed_1[word_1]
            embed_2 = word_embed_2[word_2]
            # Cosine similarity (new model)
            n_sim_score = abs(wsd.cos_sim(embed_1, embed_2).item())

            # Calculate error
            b_err = abs(score / 10 - b_sim_score)
            n_err = abs(score / 10 - n_sim_score)
            base_error += b_err
            new_error += n_err

            new_ranks.append(n_sim_score)
            base_ranks.append(b_sim_score)
            human.append(score)

            num_sent += 1

        except Exception:
            continue

    print(f">>> New Model's Accuracy -> {(1 - (new_error/num_sent))*100}%")
    print(f">>> Base Model's Accuracy -> {(1 - (base_error/num_sent))*100}%")
    new_corr = spearmanr(new_ranks, human).correlation
    print(f">>> New Model's Spearman Correlation -> {new_corr*100}%")
    base_corr = spearmanr(base_ranks, human).correlation
    print(f">>> Base Model's Spearman Correlation -> {base_corr*100}%")


if __name__ == "__main__":
    main()
