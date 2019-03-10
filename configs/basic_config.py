import os
import codecs

seed = 2019
# n_gpus 为整数或者数组，当为整数时，小于1表示使用cpu，大于等于1时表示使用gpu：0到n_gpus-1,
# 当为数组时使用数组对应的gpu
n_gpus = -1

ROOT_DIR = '/home/zhanghao/projectfile/nlp/atec/atec-master'
raw_path = ROOT_DIR + '/resource/data/atec_train_cut.csv'
vocab_file = ROOT_DIR + '/resource/data/vocab.txt'
char_vocab_file = ROOT_DIR + '/resource/data/char_vocab.txt'
data_dir = ROOT_DIR + '/resource/data'
train_dir = ROOT_DIR + '/resource/data/train.csv'
valid_dir = ROOT_DIR + '/resource/data/valid.csv'
model_path = ROOT_DIR + '/resource/model'
word_emb_path = ROOT_DIR + '/resource/data/words.vec'
word_strokes_emb_path = ROOT_DIR + '/resource/data/words_strokes.vec'
stop_words_file = ROOT_DIR + '/resource/data/stop_dict.txt'
flag_words = ['<PAD>', '<OOV>']

train_batch_size = 32
eval_batch_size = 32
gradient_accumulation_steps = 1
num_train_epochs = 20

max_seq_length = 20
max_word_len = 10
emb_size = 200
char_emb_size = 200
hidden_size = 300
char_hidden_size = 100
atten_size = 100
w_size = 20
linear_size = 50

task_name = 'qqp'
labels = [str(i) for i in range(2)]

gpus=[0]