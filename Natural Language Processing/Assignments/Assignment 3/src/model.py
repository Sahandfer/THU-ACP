import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class BiLSTM(nn.Module):
    def __init__(self, args, SENTENCE, LABEL):
        super(BiLSTM, self).__init__()
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
        print(len(x), l)
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


"""
Tree-Structured LSTM model from the following paper:
    Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks (2015)
    Kai Sheng Tai, Richard Socher*, Christopher D. Manning
"""


class LSTMTree(nn.Module):
    def __init__(self, args, SENTENCE, LABEL):
        super(LSTMTree, self).__init__()
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

        hidden_dim = self.args.hidden_dim
        num_classes = len(LABEL.vocab.itos)

        self.W_i = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_u = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.U_i = nn.Linear(2 * hidden_dim, hidden_dim, bias=True)
        self.U_o = nn.Linear(2 * hidden_dim, hidden_dim, bias=True)
        self.U_u = nn.Linear(2 * hidden_dim, hidden_dim, bias=True)

        self.F_1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.F_2 = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.layers = nn.Sequential(
            nn.Dropout(self.args.dropout),
            nn.Linear(hidden_dim, num_classes, bias=True),
        )

        self.outputs = []
        self.labels = []

    # Traverse through all nodes of a given tree to find and update all c and h
    def traverse_tree(self, node):
        if node.height() == 2:  # A leaf node
            x = self.embed(torch.LongTensor([node[0]]))
            i = torch.sigmoid(self.W_i(x))
            o = torch.sigmoid(self.W_o(x))
            u = torch.tanh(self.W_u(x))
            c = i * u
        else:
            left_h, left_c = self.traverse_tree(node[0])  # Left side of the node
            right_h, right_c = self.traverse_tree(node[1])  # Right side of the node
            x = torch.cat((left_h, right_h), 1)
            i = torch.sigmoid(self.U_i(x))
            o = torch.sigmoid(self.U_o(x))
            u = torch.tanh(self.U_u(x))
            c = (
                i * u
                + torch.sigmoid(self.F_1(left_h)) * left_c
                + torch.sigmoid(self.F_2(right_h)) * right_c
            )

        h = o * torch.tanh(c)
        output = self.layers(h)

        self.outputs.append(output)
        self.labels.append(torch.LongTensor([node.label()]))

        return h, c

    def forward(self, x):
        self.outputs, self.labels = [], []
        self.traverse_tree(x)
        self.outputs = torch.cat(self.outputs)
        self.labels = torch.cat(self.labels)

        return self.outputs, self.labels