import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.embedding_layer import TokenEmbedder


class Rnn():
    def __init__(self, input_size, hidden_size, rnn_type='lstm', num_layers=1, bias=False, batch_first=True, dropout=0,
                 bidirectional=False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

    def get_rnn(self):
        if self.rnn_type == 'lstm':
            rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=self.batch_first,
                          num_layers=self.num_layers, bidirectional=self.bidirectional, bias=self.bias,
                          dropout=self.dropout)
        elif self.rnn_type == 'gru':
            rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=self.batch_first,
                         num_layers=self.num_layers, bidirectional=self.bidirectional, bias=self.bias,
                         dropout=self.dropout)
        else:
            raise ValueError('Invalid rnn type, support: rnn,gru')
        return rnn


class Atten(nn.Module):
    def __init__(self, method, hidden_size, atten_size):
        super(Atten, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat', 'perceptron']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        self.atten_size = atten_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, atten_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, atten_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(atten_size))
        elif self.method == 'perceptron':
            self.attn_q = torch.nn.Linear(self.hidden_size, atten_size)
            self.attn_k = torch.nn.Linear(self.hidden_size, atten_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(atten_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(
            torch.cat((hidden.expand(hidden.size(0), encoder_output.size(1), hidden.size(2)), encoder_output), 2))
        return torch.sum(self.v * energy, dim=2)

    def perceptron_score(self, hidden, encoder_output):
        energy_q = self.attn_q(hidden.expand(hidden.size(0), encoder_output.size(1), hidden.size(2)))
        energy_k = self.attn_k(encoder_output)
        energy = energy_q + energy_k
        energy = energy.tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)
        elif self.method == 'perceptron':
            attn_energies = self.perceptron_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        # attn_energies = attn_energies

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class Model(nn.Module):
    def __init__(self, vocab, emb_size, hidden_size, atten_size, n_labels, embs=None, bidirectional=False):
        super(Model, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.embs = TokenEmbedder(emb_size, vocab, embs=embs)
        rnn_1 = Rnn(emb_size, hidden_size, bidirectional=bidirectional, dropout=0)
        rnn_2 = Rnn(emb_size, hidden_size, bidirectional=bidirectional, dropout=0)
        self.q1_lstm = rnn_1.get_rnn()
        self.q2_lstm = rnn_2.get_rnn()
        self.attn = Atten(method='perceptron', hidden_size=hidden_size, atten_size=atten_size)
        self.linear = nn.Linear(2 * hidden_size, n_labels)

    def forward(self, q1_inputs, q2_inputs, q1_lens=None, q2_lens=None):
        q1_embs = self.embs(q1_inputs)
        q2_embs = self.embs(q2_inputs)
        # if self.bidirectional:
        #     q1_packed = pack_padded_sequence(q1_embs, q1_lens, batch_first=True)
        #     q1_outs, q1_hidden = self.q1_lstm(q1_packed)
        #     q1_outs, _ = pad_packed_sequence(q1_outs, batch_first=True)
        #     q1_outs = q1_outs[:, :, :self.hidden_size] + q1_outs[:, :, self.hidden_size:]
        #
        #     q2_pached = pack_padded_sequence(q2_embs, q2_lens, batch_first=True)
        #     q2_outs, q2_hidden = self.q2_lstm(q2_pached, q1_hidden)
        #     q2_outs, _ = pad_packed_sequence(q2_outs, batch_first=True)
        #     q2_outs = q2_outs[:, :, :self.hidden_size] + q2_outs[:, :, self.hidden_size:]
        # else:
        q1_outs, q1_hidden = self.q1_lstm(q1_embs)
        q2_outs, q2_hidden = self.q2_lstm(q2_embs, q1_hidden)

        batch_size = q1_inputs.size(0)
        batch_ids = [i for i in range(batch_size)]
        q2_tmp = q2_outs[batch_ids, q2_lens - 1, :].unsqueeze(1)
        # q1_tmp = q1_outs[:, : q1_lens.cpu().numpy()[0], :]
        attn_weights = self.attn(q2_tmp, q1_outs)
        q1_atten = attn_weights.bmm(q1_outs)

        concat_input = torch.cat((q1_atten.squeeze(1), q2_tmp.squeeze(1)), 1)
        output = self.linear(concat_input)
        output = F.softmax(output, dim=1)

        return output
