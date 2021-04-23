import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class SentimentClassifier(nn.Module):
    def __init__(self, args, SENTENCE, LABEL):
        super(SentimentClassifier, self).__init__()
        self.args = args
        # Word Embedding
        if args.use_glove:
            embed = SENTENCE.vocab.vectors
            self.embed = nn.Embedding.from_pretrained(embed)
        else:
            input_size, embed_dim = SENTENCE.vocab.vectors.size()
            self.embed = nn.Embedding(input_size, embed_dim)
            val_range = 0.5 / embed_dim
            self.embed.weight.data.uniform_(-val_range, val_range)

        # Bi-LSTM Model
        self.lstm = nn.LSTM(
            input_size=self.embed.embedding_dim,
            hidden_size=self.args.hidden_dim,
            num_layers=self.args.num_layers,
            dropout=self.args.dropout,
            bidirectional=True,
            batch_first=True,
        )
        # The classification layer added after LSTM
        num_classes = len(LABEL.vocab.itos)
        self.layers = nn.Sequential(
            nn.Dropout(self.args.dropout),
            nn.Linear(self.args.hidden_dim * 2, num_classes),
        )

    def forward(self, x, l):
        # x is the input and l is the sentence length
        w_v = self.embed(x)
        # This saves times since elements in each have different sizes
        packed_embed = pack_padded_sequence(
            w_v, l, batch_first=True, enforce_sorted=False
        )
        # Get the last hidden state
        output, (last_h, _) = self.lstm(packed_embed)
        # Concat the normal and reverse layer's output
        output = torch.cat((last_h[-2, :, :], last_h[-1, :, :]), dim=1)
        # Pass output to dropout and FC layer for classification
        output = self.layers(output)
        return output