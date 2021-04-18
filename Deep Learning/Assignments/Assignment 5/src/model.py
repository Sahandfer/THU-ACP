import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class SentimentClassifier(nn.Module):
    def __init__(self, config, embed):
        super(SentimentClassifier, self).__init__()
        self.config = config
        # Word Embedding
        self.embed = nn.Embedding.from_pretrained(embed)
        # Bi-LSTM Model
        self.lstm = nn.LSTM(
            input_size=embed.shape[1],
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            bidirectional=True,
            batch_first=True,
        )
        # The classification layer added after LSTM
        self.layers = nn.Sequential(
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim * 2, self.config.num_classes),
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