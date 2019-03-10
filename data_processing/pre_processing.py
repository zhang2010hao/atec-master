import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from data_processing.tokenizer import Tokenizer
import configs.basic_config as args
import os
import csv
import random
from collections import Counter
from tqdm import tqdm
import operator
from data_processing.dataProcessor import MyPro
from data_processing.inputs import InputFeature
import numpy as np


def train_val_split(X, y, valid_size=0.2, random_state=2018, shuffle=True):
    """
    训练集验证集分割
    :param X: sentences
    :param y: labels
    :param random_state: 随机种子
    """
    train, valid = [], []
    bucket = [[] for _ in args.labels]

    for data_x, data_y in tqdm(zip(X, y), desc='bucket'):
        bucket[int(data_y)].append((data_x, data_y))

    del X, y

    for bt in tqdm(bucket, desc='split'):
        N = len(bt)
        if N == 0:
            continue
        test_size = int(N * valid_size)

        if shuffle:
            random.seed(random_state)
            random.shuffle(bt)

        valid.extend(bt[:test_size])
        train.extend(bt[test_size:])

    if shuffle:
        random.seed(random_state)
        random.shuffle(valid)
        random.shuffle(train)

    return train, valid


def sent_label_split(line):
    """
    句子处理成单词
    :param line: 原始行
    :return: 单词， 标签
    """
    line = line.strip('\n').split('\t')
    label = line[-1]
    sent = line[0].split(' ') + line[1].split(' ')
    return sent, label


def build_vocab(sentenses, vocab_size=None, min_freq=1, stop_word=None):
    """
    建立词典
    :param vocab_size: 词典大小
    :param min_freq: 最小词频限制
    :param stop_list: 停用词 @type：file_path
    :return: vocab
    """
    counter = Counter()
    print('Building vocab')
    for line in tqdm(sentenses, desc='build vocab'):
        sent_pairs = line.split('\t')
        if len(sent_pairs) != 2:
            continue
        counter.update(sent_pairs[0].split(' '))
        counter.update(sent_pairs[1].split(' '))

    # 去停用词
    if stop_word:
        stop_list = {}
        with open(os.path.join(args.ROOT_DIR, args.stop_words_file), 'r') as fr:
            for i, line in enumerate(fr):
                word = line.strip('\n')
                if stop_list.get(word) is None:
                    stop_list[word] = i
        counter = {k: v for k, v in counter.items() if k not in stop_list}
    counter = sorted(counter.items(), key=operator.itemgetter(1))

    # 去低频词
    vocab = [w[0] for w in counter if w[1] >= min_freq]

    # 按最大词汇量截取词典
    if vocab_size:
        if vocab_size - 3 < len(vocab):
            vocab = vocab[:vocab_size - 2]

    char_vocab = set()
    for word in vocab:
        for ch in word:
            char_vocab.add(ch)
    char_vocab = args.flag_words + list(char_vocab)
    print('char_vocab_size is %d' % len(char_vocab))
    with open(args.char_vocab_file, 'w') as ch_fw:
        for ch in char_vocab:
            ch_fw.write(ch + '\n')

    vocab = args.flag_words + vocab
    print('Vocab_size is %d' % len(vocab))

    with open(args.vocab_file, 'w') as fw:
        for w in vocab:
            fw.write(w + '\n')
    print("Vocab write down at {}".format(args.vocab_file))


def prepare_data(min_freq=1, vocab_size=None, re_gen_data=True):
    """
    数据预处理,分训练集和验证集
    :param re_gen_data: 是否重新生成数据集合
    :return:
    """

    # 如果训练集和验证集已经存在，则不重新生成，否则重新生成
    # 这样操作可以减少此处的计算量，如果想每次都使用新的数据集
    # 可以注释掉此代码或者删除生成的数据集
    if not re_gen_data and os.path.exists(args.train_dir) and os.path.exists(args.valid_dir):
        return

    targets, sentences = [], []
    with open(args.raw_path, 'r', encoding='utf-8-sig') as fr:
        for line in fr:
            lines = line.strip().split('\t')
            if len(lines) != 3:
                print(line)
                continue
            target = lines[-1]
            sent = '\t'.join([lines[0], lines[1]])
            targets.append(target.strip())
            sentences.append(sent.strip())

    build_vocab(sentences, min_freq=min_freq, vocab_size=vocab_size, stop_word=['花呗', '借呗'])

    train, valid = train_val_split(sentences, targets)

    with open(args.train_dir, 'w') as fw:
        for sent, label in train:
            line = "\t".join([str(label), sent])
            fw.write(line + '\n')
        print("Train data write finish")

    with open(args.valid_dir, 'w') as fw:
        for sent, label in valid:
            line = "\t".join([str(label), sent])
            fw.write(line + '\n')
        print("Valid data write finish")


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, pad='<PAD>', oov='<OOV>',
                                 is_padding=True):
    # 标签转换为数字
    label_map = {label: i for i, label in enumerate(label_list)}
    pad_id = tokenizer.vocab[pad]

    features = []
    for ex_index, example in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        if len(tokens_a) >= max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
        text_a_ids = tokenizer.convert_tokens_to_ids(tokens_a)
        text_a_lens = len(text_a_ids)
        if is_padding:
            text_a_ids = text_a_ids + [pad_id] * (max_seq_length - len(text_a_ids))

        text_a_char_ids = np.zeros((max_seq_length, args.max_word_len))
        if args.char_emb_size > 0:
            for i, word in enumerate(tokens_a):
                word_char_ids = tokenizer.convert_word_to_char_ids(word)
                if len(word_char_ids) > args.max_word_len:
                    word_char_ids = word_char_ids[:args.max_word_len]
                text_a_char_ids[i, :len(word_char_ids)] = word_char_ids

        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            if len(tokens_b) >= max_seq_length:
                tokens_b = tokens_b[:max_seq_length]
            text_b_ids = tokenizer.convert_tokens_to_ids(tokens_b)
            text_b_lens = len(text_b_ids)
            if is_padding:
                text_b_ids = text_b_ids + [pad_id] * (max_seq_length - len(text_b_ids))

            text_b_char_ids = np.zeros((max_seq_length, args.max_word_len))
            if args.char_emb_size > 0:
                for i, word in enumerate(tokens_b):
                    word_char_ids = tokenizer.convert_word_to_char_ids(word)
                    text_b_char_ids[i, :len(word_char_ids)] = word_char_ids

        label_id = label_map[example.label]

        # --------------看结果是否合理-------------------------
        if ex_index < 0:
            print("*** Example ***")
            print("guid: %s" % (example.guid))
            print("tokens_a: %s" % " ".join(
                [str(x) for x in tokens_a]))
            print("tokens_b: %s" % " ".join([str(x) for x in tokens_b]))
            print("label: %s (id = %d)" % (example.label, label_id))
        # ----------------------------------------------------

        feature = InputFeature(text_a_ids=text_a_ids,
                               text_b_ids=text_b_ids,
                               text_a_lens=text_a_lens,
                               text_b_lens=text_b_lens,
                               label_id=label_id,
                               text_a_char_ids=text_a_char_ids,
                               text_b_char_ids=text_b_char_ids)
        features.append(feature)

    return features


def init_tokenizer():
    tokenizer = Tokenizer(vocab_file=args.vocab_file, char_vocab_file=args.char_vocab_file)
    return tokenizer


def init_processor():
    processors = {"qqp": MyPro}
    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()

    return processor


def create_features(mode, tokenizer):
    """构造数据特征"""
    processor = init_processor()
    examples = processor.get_examples(mode, args.data_dir)
    label_list = processor.get_labels()
    # 特征
    train_features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer)
    print("Num examples = %d", len(examples))

    return train_features


def create_batch_iter(mode, tokenizer):
    """构造迭代器"""
    processor = init_processor()
    if mode == "train":
        examples = processor.get_examples(mode, args.data_dir)
        num_train_steps = int(len(examples) / args.train_batch_size)
        batch_size = args.train_batch_size
        print("train steps number = %d", num_train_steps)
    elif mode == "dev":
        examples = processor.get_examples(mode, args.data_dir)
        num_eval_steps = int(len(examples) / args.eval_batch_size)
        batch_size = args.eval_batch_size
        print("valid steps number = %d", num_eval_steps)
    else:
        raise ValueError("Invalid mode %s" % mode)

    label_list = processor.get_labels()
    # 特征
    train_features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer)
    print("Num examples = %d", len(examples))
    print("Batch size = %d", batch_size)

    all_text_a_ids = torch.tensor([f.text_a_ids for f in train_features], dtype=torch.long)
    all_text_b_ids = torch.tensor([f.text_b_ids for f in train_features], dtype=torch.long)
    all_text_a_lens = torch.tensor([f.text_a_lens for f in train_features], dtype=torch.long)
    all_text_b_lens = torch.tensor([f.text_b_lens for f in train_features], dtype=torch.long)
    all_text_a_char_ids = torch.tensor([f.text_a_char_ids for f in train_features], dtype=torch.long)
    all_text_b_char_ids = torch.tensor([f.text_b_char_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

    # 数据集
    data = TensorDataset(all_text_a_ids, all_text_a_lens, all_text_b_ids, all_text_b_lens, all_text_a_char_ids,
                         all_text_b_char_ids, all_label_ids)

    if mode == "train":
        sampler = RandomSampler(data)
    elif mode == "dev":
        sampler = SequentialSampler(data)
    else:
        raise ValueError("Invalid mode %s" % mode)

    # 迭代器
    iterator = DataLoader(data, sampler=sampler, batch_size=batch_size)

    if mode == "train":
        return iterator, num_train_steps
    elif mode == "dev":
        return iterator, num_eval_steps
    else:
        raise ValueError("Invalid mode %s" % mode)


def load_embs(word2id):
    emb_vecs = np.random.random((len(word2id), args.emb_size))
    emb_words = np.array([w for w, id in word2id.items()])
    emb_size = args.emb_size // 2
    with open(args.word_emb_path, 'r') as fr_1:
        for line in fr_1.readlines():
            arr = line.strip().split(' ')
            if len(arr) != emb_size + 1:
                continue
            if arr[0] in word2id:
                word_id = word2id[arr[0]]
                vec = list(map(float, arr[1:]))
                emb_vecs[word_id, :emb_size] = vec

    with open(args.word_strokes_emb_path, 'r') as fr_2:
        for line in fr_2.readlines():
            arr = line.strip().split(' ')
            if len(arr) != emb_size + 1:
                continue
            if arr[0] in word2id:
                word_id = word2id[arr[0]]
                vec = list(map(float, arr[1:]))
                emb_vecs[word_id, emb_size:] = vec

    emb_vecs = emb_vecs.astype(np.float32)
    return (emb_vecs, emb_words)
