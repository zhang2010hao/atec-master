import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.embedding_layer import TokenEmbedder
from torch.autograd import Variable


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
        if self.method not in ['dot', 'general', 'concat', 'perceptron', 'cosine']:
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
        elif self.method == 'cosine':
            self.cos = torch.nn.CosineSimilarity(dim=1)

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

    def cosine_score(self, hidden, encoder_output):
        energy = self.cos()

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
    def __init__(self, vocab, emb_size, hidden_size, w_size, n_labels, embs=None, bidirectional=True):
        super(Model, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.w_size = w_size
        self.bidirectional = bidirectional

        self.embs = TokenEmbedder(emb_size, vocab, embs=embs)
        rnn_1 = Rnn(emb_size, hidden_size, bidirectional=bidirectional, dropout=0.1)
        rnn_2 = Rnn(emb_size, hidden_size, bidirectional=bidirectional, dropout=0.1)
        self.q1_lstm = rnn_1.get_rnn()
        self.q2_lstm = rnn_2.get_rnn()

        self.W_1 = nn.Parameter(Variable(torch.randn(w_size, hidden_size)))
        self.W_2 = nn.Parameter(Variable(torch.randn(w_size, hidden_size)))
        self.W_3 = nn.Parameter(Variable(torch.randn(w_size, hidden_size)))
        self.W_4 = nn.Parameter(Variable(torch.randn(w_size, hidden_size)))
        self.W_5 = nn.Parameter(Variable(torch.randn(w_size, hidden_size)))
        self.W_6 = nn.Parameter(Variable(torch.randn(w_size, hidden_size)))
        self.W_7 = nn.Parameter(Variable(torch.randn(w_size, hidden_size)))
        self.W_8 = nn.Parameter(Variable(torch.randn(w_size, hidden_size)))

        self.cos = torch.nn.CosineSimilarity(dim=1)

        self.linear = nn.Linear(2 * hidden_size, n_labels)

    def forward(self, q1_inputs, q2_inputs, q1_lens=None, q2_lens=None):
        """
        这里的输入都是单个句子
        :param q1_inputs:
        :param q2_inputs:
        :param q1_lens:
        :param q2_lens:
        :return:
        """
        q1_len = q1_lens.cpu().numpy()[0]
        q2_len = q2_lens.cpu().numpy()[0]

        # [p_1, p_2,...,p_M]
        q1_embs = self.embs(q1_inputs[:, :q1_len])
        # [q_1, q_2,...,q_N]
        q2_embs = self.embs(q2_inputs[:, :q2_len])
        # [hp_1, hp_2,...,hp_M]
        q1_outs, q1_hidden = self.q1_lstm(q1_embs)
        # [hq_1, hq_2,...,hq_N]
        q2_outs, q2_hidden = self.q2_lstm(q2_embs)

        q1_pos_ = q1_outs[:, :, :self.hidden_size]

        #### full 1 正向交互
        w_1 = self.W_1.expand(q1_len, self.w_size, self.hidden_size)
        q1_pos = q1_pos_.permute(1, 0, 2).expand(q1_len, self.w_size, self.hidden_size)

        q2_pos_tail = q2_outs[:, -1, :self.hidden_size]
        q2_pos_tail = q2_pos_tail.expand(q1_len, self.w_size, self.hidden_size)

        m_pos_full_1 = w_1 * q1_pos
        m_pos_full_1 = m_pos_full_1.view(-1, self.hidden_size)
        m_pos_full_2 = w_1 * q2_pos_tail
        m_pos_full_2 = m_pos_full_2.view(-1, self.hidden_size)
        m_pos_full = self.cos(m_pos_full_1, m_pos_full_2)
        m_pos_full = m_pos_full.view(q1_len, self.w_size)

        q1_neg_ = q1_outs[:, :, self.hidden_size:]
        #### full 2 反向交互
        w_2 = self.W_2.expand(q1_len, self.w_size, self.hidden_size)
        q1_neg = q1_neg_.permute(1, 0, 2).expand(q1_len, self.w_size, self.hidden_size)

        q2_neg_head = q2_outs[:, 0, self.hidden_size:]
        q2_neg_head = q2_neg_head.expand(q1_len, self.w_size, self.hidden_size)

        m_neg_full_1 = w_2 * q1_neg
        m_neg_full_1 = m_neg_full_1.view(-1, self.hidden_size)
        m_neg_full_2 = w_2 * q2_neg_head
        m_neg_full_2 = m_neg_full_2.view(-1, self.hidden_size)
        m_neg_full = self.cos(m_neg_full_1, m_neg_full_2)
        m_neg_full = m_neg_full.view(q1_len, self.w_size)

        q2_pos_ = q2_outs[:, :, :self.hidden_size]
        #### max 1 正向交互
        w_3 = self.W_3.expand(q1_len, q2_len, self.w_size, self.hidden_size)
        q1_max_pos = q1_pos_.permute(1, 0, 2).unsqueeze(dim=1).expand(q1_len, q2_len, self.w_size, self.hidden_size)
        q2_max_pos = q2_pos_.permute(1, 0, 2).unsqueeze(dim=0).expand(q1_len, q2_len, self.w_size, self.hidden_size)

        m_pos_max_1 = w_3 * q1_max_pos
        m_pos_max_1 = m_pos_max_1.view(-1, self.hidden_size)
        m_pos_max_2 = w_3 * q2_max_pos
        m_pos_max_2 = m_pos_max_2.view(-1, self.hidden_size)
        m_pos_max = self.cos(m_pos_max_1, m_pos_max_2)
        m_pos_max = m_pos_max.view(q1_len, q2_len, self.w_size)
        m_pos_max, _ = torch.max(m_pos_max, dim=1)

        q2_neg_ = q2_outs[:, :, self.hidden_size:]
        #### max 2 反向交互
        w_4 = self.W_4.expand(q1_len, q2_len, self.w_size, self.hidden_size)
        q1_max_neg = q1_neg_.permute(1, 0, 2).unsqueeze(dim=1).expand(q1_len, q2_len, self.w_size, self.hidden_size)
        q2_max_neg = q2_neg_.permute(1, 0, 2).unsqueeze(dim=0).expand(q1_len, q2_len, self.w_size, self.hidden_size)

        m_neg_max_1 = w_4 * q1_max_neg
        m_neg_max_1 = m_neg_max_1.view(-1, self.hidden_size)
        m_neg_max_2 = w_4 * q2_max_neg
        m_neg_max_2 = m_neg_max_2.view(-1, self.hidden_size)
        m_neg_max = self.cos(m_neg_max_1, m_neg_max_2)
        m_neg_max = m_neg_max.view(q1_len, q2_len, self.w_size)
        m_neg_max, _ = torch.max(m_neg_max, dim=1)

        #### attention 正向交互
        q1_atten_pos = q1_pos_.permute(1, 0, 2).expand(q1_len, q2_len, self.hidden_size)
        q2_atten_pos = q2_pos_.expand(q1_len, q2_len, self.hidden_size)
        alpha_pos = self.cos(q1_atten_pos.contiguous().view(-1, self.hidden_size),
                             q2_atten_pos.contiguous().view(-1, self.hidden_size))
        alpha_pos = alpha_pos.view(q1_len, q2_len)
        alphs_pos_sum = torch.sum(alpha_pos, dim=1)
        alpha_pos_tmp = alpha_pos.unsqueeze(2).expand(q1_len, q2_len, self.hidden_size)
        q2_pos_atten = alpha_pos_tmp * q2_atten_pos
        q2_pos_atten = torch.sum(q2_pos_atten, dim=1)
        q2_pos_atten = q2_pos_atten / alphs_pos_sum.unsqueeze(1).expand(q1_len, self.hidden_size)

        w_5 = self.W_5.expand(q1_len, self.w_size, self.hidden_size)
        q1_atten_pos_f = q1_pos_.permute(1, 0, 2).expand(q1_len, self.w_size, self.hidden_size)
        q2_atten_pos_f = q2_pos_atten.unsqueeze(1).expand(q1_len, self.w_size, self.hidden_size)
        q1_atten_pos_f = w_5 * q1_atten_pos_f
        q1_atten_pos_f = q1_atten_pos_f.view(-1, self.hidden_size)
        q2_atten_pos_f = w_5 * q2_atten_pos_f
        q2_atten_pos_f = q2_atten_pos_f.view(-1, self.hidden_size)
        m_pos_atten = self.cos(q1_atten_pos_f, q2_atten_pos_f)
        m_pos_atten = m_pos_atten.view(q1_len, self.w_size)

        #### attention 反向交互
        q1_atten_neg = q1_neg_.permute(1, 0, 2).expand(q1_len, q2_len, self.hidden_size)
        q2_atten_neg = q2_neg_.expand(q1_len, q2_len, self.hidden_size)
        alpha_neg = self.cos(q1_atten_neg.contiguous().view(-1, self.hidden_size),
                             q2_atten_neg.contiguous().view(-1, self.hidden_size))
        alpha_neg = alpha_neg.view(q1_len, q2_len)
        alphs_neg_sum = torch.sum(alpha_neg, dim=1)
        alpha_neg_tmp = alpha_neg.unsqueeze(2).expand(q1_len, q2_len, self.hidden_size)
        q2_neg_atten = alpha_neg_tmp * q2_atten_neg
        q2_neg_atten = torch.sum(q2_neg_atten, dim=1)
        q2_neg_atten = q2_neg_atten / alphs_neg_sum.unsqueeze(1).expand(q1_len, self.hidden_size)

        w_6 = self.W_6.expand(q1_len, self.w_size, self.hidden_size)
        q1_atten_neg_f = q1_neg_.permute(1, 0, 2).expand(q1_len, self.w_size, self.hidden_size)
        q2_atten_neg_f = q2_neg_atten.unsqueeze(1).expand(q1_len, self.w_size, self.hidden_size)
        q1_atten_neg_f = w_5 * q1_atten_neg_f
        q1_atten_neg_f = q1_atten_neg_f.view(-1, self.hidden_size)
        q2_atten_neg_f = w_5 * q2_atten_neg_f
        q2_atten_neg_f = q2_atten_neg_f.view(-1, self.hidden_size)
        m_neg_atten = self.cos(q1_atten_neg_f, q2_atten_neg_f)
        m_neg_atten = m_neg_atten.view(q1_len, self.w_size)

        #### attention max 正向交互
        _, pos_indics = torch.max(alpha_pos, dim=1)
        q2_pos_atten_max = q2_atten_pos[range(q1_len), pos_indics, :]

        # q2的正反向输出
        q2_pos = q2_outs[:, :, :self.hidden_size]
        q2_pos = q2_outs.permute(1, 0, 2).expand(q2_len, self.w_size, self.hidden_size)
        q2_neg = q2_outs[:, :, self.hidden_size:]
        q2_neg = q2_outs.permute(1, 0, 2).expand(q2_len, self.w_size, self.hidden_size)

        m_neg_full_1 = w_2 * q1_neg
        m_neg_full_2 = w_1 * q2_head
        m_neg_full = self.cos(m_pos_full_1, m_pos_full_2)

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
