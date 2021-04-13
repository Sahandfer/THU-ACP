import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import save_pickle


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.word_embed = nn.Embedding(vocab_size, embed_size, sparse=True)
        self.ctx_embed = nn.Embedding(vocab_size, embed_size, sparse=True)

        self.init_weights()

    def init_weights(self):
        val_range = 0.5 / self.embed_size
        self.word_embed.weight.data.uniform_(-val_range, val_range)
        self.ctx_embed.weight.data.uniform_(0, 0)

    def forward(self, word_pos, ctx_pos, neg_ctx_pos):
        # Embedding rows
        word_embed = self.word_embed(word_pos)  # center word
        ctx_embed = self.ctx_embed(ctx_pos)  # neighbor word
        neg_ctx_embed = self.ctx_embed(neg_ctx_pos)  # negative sample word

        # Similarity score (target)
        sim_score = torch.sum(torch.mul(word_embed, ctx_embed), dim=1)
        target_loss = F.logsigmoid(sim_score)

        # Similarity score (negative)
        neg_sim_score = torch.bmm(neg_ctx_embed, word_embed.unsqueeze(2)).squeeze()
        neg_loss = torch.sum(F.logsigmoid(-neg_sim_score), dim=1)

        # Total loss
        loss = target_loss + neg_loss

        return -loss

    def save(self, output_dir, file_name):
        embedding = self.word_embed.weight.data.cpu().numpy()
        save_pickle(output_dir, file_name, embedding)
