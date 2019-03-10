import torch.nn as nn
import torch
import numpy as np

class TokenEmbedder(nn.Module):
    def __init__(self, emb_dim, token2id, embs=None, oov='<OOV>', pad='<PAD>'):
        super(TokenEmbedder, self).__init__()
        if embs is not None:
            embvecs, embwords = embs
            emb_dim = len(embvecs[0])

        self.padid = token2id[pad]
        self.oovid = token2id[oov]
        self.word_num = len(token2id)
        self.embeddings = nn.Embedding(self.word_num, emb_dim, padding_idx=self.padid)
        self.embeddings.weight.data.uniform_(-1, 1)

        if embs is not None:
            weight = self.embeddings.weight
            weight.data[:len(embwords)].copy_(torch.from_numpy(np.asarray(embvecs)))
            print("embedding shape: {}".format(weight.size()))

        self.embeddings.weight.requires_grad = True

    def forward(self, input):
        return self.embeddings(input)