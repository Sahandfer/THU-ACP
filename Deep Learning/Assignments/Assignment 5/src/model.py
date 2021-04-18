import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SentimentClassifier(nn.Module):
    def __init__(self, config, embed):
        super(SentimentClassifier, self).__init__()
        self.config = config
        self.embed = nn.Embedding.from_pretrained(embed)

        self.lstm = nn.LSTM(
            input_size=embed.shape[1],
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            bidirectional=True,
            batch_first=True,
        )

        self.layers = nn.Sequential(
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim * 2, self.config.num_classes),
        )

    def forward(self, x, l):
        w_v = self.embed(x)
        packed_embed = pack_padded_sequence(
            w_v, l, batch_first=True, enforce_sorted=False
        )
        output, (last_h, last_c) = self.lstm(packed_embed)

        last_h = torch.cat((last_h[-2, :, :], last_h[-1, :, :]), dim=1)
        output = self.layers(last_h)
        return output