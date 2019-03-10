from gensim.models import FastText, Word2Vec
import time
from collections import Counter
import jieba

jieba.load_userdict('/home/zhanghao/projectfile/nlp/atec/atec-master/resource/data/user_dict.txt')


# from cw2vc.load_embedding import load_strokes


def load_stroke_data(path, chchar2stroke, max_len=200):
    x = []
    word2stoke = {}
    stroke2word = {}
    with open(path, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            arr = line.strip().split('\t')

            for sent in arr[:2]:
                tmp = []
                for word in sent.split(' '):
                    strarr = [chchar2stroke[ch] for ch in word if ch in chchar2stroke]
                    chstrok = ''.join(strarr)
                    if chstrok != '':
                        word2stoke[word] = chstrok
                        tmp.append(chstrok)
                tmp_line = ' '.join(tmp)
            x.append(tmp_line)

    for word, stks in word2stoke.items():
        if stks in stroke2word:
            tmp = stroke2word[stks]
        else:
            tmp = []
        tmp.append(word)
        stroke2word[stks] = tmp

    return x, stroke2word


def load_strokes():
    stroke_path = '/home/zhanghao/projectfile/nlp/ChineseCixing-master/strokes.txt'
    stroke2id = {'横': '1', '提': '1', '竖': '2', '竖钩': '2', '撇': '3', '捺': '4', '点': '4'}
    chchar2stroke = {}

    with open(stroke_path, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            line = line.strip().split(':')
            if len(line) == 2:
                arr = line[1].split(',')
                strokes = [stroke2id[stroke] if stroke in stroke2id else '5' for stroke in arr]
                chchar2stroke[line[0]] = ''.join(strokes)

    return chchar2stroke


def load_data(file_path):
    with open(file_path, 'r') as fr:
        documents = []
        for line in fr.readlines():
            arr = line.strip().split('\t')
            if len(arr) != 3:
                continue
            documents.append(arr[0])
            documents.append(arr[1])

    return documents


def cut_text(path, cut_path):
    with open(cut_path, 'w', encoding='utf-8') as fw:
        with open(path, 'r', encoding='utf-8-sig') as fr:
            for line in fr.readlines():
                line = line.strip()
                seg_list = jieba.cut(line)
                fw.write(' '.join(seg_list) + '\n')


def word2vec(documents, size, stroke2chchar):
    texts = [[word for word in document.split(' ')] for document in documents]
    # texts = [["cat", "say", "meow"], ["dog", "say", "woof"]]
    print('train word2vec Model..., time: ', time.time())
    model = FastText(texts, size=size, window=5, iter=5, min_count=1, workers=1, seed=12, sg=1)
    path = '/home/zhanghao/projectfile/nlp/atec/atec-master/resource/data/gensim_word2vec.model'
    model.save(path)
    print('caculate features, time: ', time.time())

    wordcounter = Counter()
    wordset = set()
    j = 0
    for text in texts:
        wordcounter.update(text)
        for word in text:
            wordset.add(word)
            j += 1
    print(j)
    print(len(wordset))

    with open("/home/zhanghao/projectfile/nlp/atec/atec-master/resource/data/words_strokes.vec", 'w',
              encoding='utf-8') as fw:
        i = 0
        if stroke2word:
            for stroke, words in stroke2word.items():
                if stroke in model:
                    vec = model[stroke]
                    vec = list(map(str, vec))
                    for w in words:
                        fw.write(w + ' ' + ' '.join(vec) + '\n')
                    i += 1
        else:
            for word in wordset:
                if word in model:
                    vec = model[word]
                    vec = list(map(str, vec))
                    fw.write(word + ' ' + ' '.join(vec) + '\n')
                    i += 1

        print("the num of feature is: ", i)


if __name__ == '__main__':
    path = '/home/zhanghao/projectfile/nlp/atec/atec-master/resource/data/atec_train.csv'
    cut_path = '/home/zhanghao/projectfile/nlp/atec/atec-master/resource/data/atec_train_cut.csv'
    # cut_text(path, cut_path)

    chchar2stroke = load_strokes()
    X, stroke2word = load_stroke_data(cut_path, chchar2stroke)
    # texts = load_data(cut_path)
    word2vec(X, 100, stroke2word)
