import os


class Tokenizer(object):
    def __init__(self, vocab_file, char_vocab_file, pad='<PAD>', oov='<OOV>'):
        self.vocab_file = vocab_file
        self.char_vocab_file = char_vocab_file
        self.pad = pad
        self.oov = oov
        self.init_vocab()

    def init_vocab(self):
        if not os.path.exists(self.vocab_file) or not os.path.exists(self.char_vocab_file):
            raise ValueError("vocab file not exist, please input correct path")
        else:
            with open(self.vocab_file, 'r') as fr:
                self.vocab = {}
                for line in fr.readlines():
                    w = line.strip()
                    if w != '':
                        self.vocab[w] = len(self.vocab)

            with open(self.char_vocab_file, 'r') as ch_fr:
                self.char_vocab = {}
                for line in ch_fr.readlines():
                    w = line.strip()
                    if w != '':
                        self.char_vocab[w] = len(self.char_vocab)

    def tokenize(self, line, delimiter=' '):
        arr = line.split(delimiter)

        return arr

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab[self.oov])

        return ids

    def convert_word_to_char_ids(self, word):
        ids = []
        for ch in word:
            if ch in self.char_vocab:
                ids.append(self.char_vocab[ch])
            else:
                ids.append(self.char_vocab[self.oov])

        return ids


