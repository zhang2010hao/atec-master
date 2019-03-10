from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


class ESIM(nn.Module):
    def __init__(self, hidden_size, embeds_dim, linear_size, vocab, embs, dropout=0.1):
        super(ESIM, self).__init__()
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.embeds_dim = embeds_dim

        self.embeds = nn.Embedding(len(vocab), self.embeds_dim)
        if embs:
            embvecs, embwords = embs
            self.embeds.weight.data.copy_(torch.from_numpy(np.asarray(embvecs)))
        self.embeds.weight.requires_grad = False

        self.bn_embeds = nn.BatchNorm1d(self.embeds_dim)
        self.lstm1 = nn.LSTM(self.embeds_dim, self.hidden_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(self.hidden_size * 8, self.hidden_size, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size * 8),
            nn.Linear(self.hidden_size * 8, linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(linear_size),
            nn.Dropout(self.dropout),
            nn.Linear(linear_size, linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(linear_size),
            nn.Dropout(self.dropout),
            nn.Linear(linear_size, 2),
            nn.Softmax(dim=-1)
        )

    def soft_attention_align(self, x1, x2, mask1, mask2):
        '''
        x1: batch_size * seq_len * dim
        x2: batch_size * seq_len * dim
        '''
        # attention: batch_size * seq_len * seq_len
        attention = torch.matmul(x1, x2.transpose(1, 2))
        # mask1 = mask1.float().masked_fill_(mask1, float('-inf'))
        # mask2 = mask2.float().masked_fill_(mask2, float('-inf'))

        # weight: batch_size * seq_len * seq_len
        weight1 = F.softmax(attention, dim=-1)
        x1_align = torch.matmul(weight1, x2)
        weight2 = F.softmax(attention.transpose(1, 2), dim=-1)
        x2_align = torch.matmul(weight2, x1)
        # x_align: batch_size * seq_len * hidden_size

        return x1_align, x2_align

    def submul(self, x1, x2):
        mul = x1 * x2
        sub = x1 - x2
        return torch.cat([sub, mul], -1)

    def apply_multiple(self, x):
        # input: batch_size * seq_len * (2 * hidden_size)
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], 1)

    def forward(self, q1_inputs, q2_inputs, q1_char_inputs=None, q2_char_inputs=None, q1_lens=None, q2_lens=None):
        q1_lens_tmp, q1_indices = torch.sort(q1_lens, descending=True)
        q1_input_tmp = q1_inputs[q1_indices]

        q2_lens_tmp, q2_indices = torch.sort(q2_lens, descending=True)
        q2_input_tmp = q2_inputs[q2_indices]

        # batch_size * seq_len
        mask1, mask2 = q1_input_tmp.eq(0), q2_input_tmp.eq(0)

        # embeds: batch_size * seq_len => batch_size * seq_len * dim
        x1 = self.bn_embeds(self.embeds(q1_input_tmp).transpose(1, 2).contiguous()).transpose(1, 2)
        x2 = self.bn_embeds(self.embeds(q2_input_tmp).transpose(1, 2).contiguous()).transpose(1, 2)

        # batch_size * seq_len * dim => batch_size * seq_len * hidden_size
        x1_packed = pack_padded_sequence(x1, q1_lens_tmp, batch_first=True)
        o1_packed, _ = self.lstm1(x1_packed)
        o1, _ = pad_packed_sequence(o1_packed, batch_first=True)

        x2_packed = pack_padded_sequence(x2, q2_lens_tmp, batch_first=True)
        o2_packed, _ = self.lstm1(x2_packed)
        o2, _ = pad_packed_sequence(o2_packed, batch_first=True)

        # Attention
        # batch_size * seq_len * hidden_size
        q1_align, q2_align = self.soft_attention_align(o1, o2, mask1, mask2)

        # Compose
        # batch_size * seq_len * (8 * hidden_size)
        q1_combined = torch.cat([o1, q1_align, self.submul(o1, q1_align)], -1)
        q2_combined = torch.cat([o2, q2_align, self.submul(o2, q2_align)], -1)

        # batch_size * seq_len * (2 * hidden_size)
        q1_compose, _ = self.lstm2(q1_combined)
        q2_compose, _ = self.lstm2(q2_combined)

        # Aggregate
        # input: batch_size * seq_len * (2 * hidden_size)
        # output: batch_size * (4 * hidden_size)
        q1_rep = self.apply_multiple(q1_compose)
        q2_rep = self.apply_multiple(q2_compose)

        # Classifier
        x = torch.cat([q1_rep, q2_rep], -1)
        similarity = self.fc(x)
        return similarity
