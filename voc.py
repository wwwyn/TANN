import numpy as np
from collections import defaultdict
import re
import codecs
import os
import operator
import cPickle
pad_symbol = "<pad>"
start_symbol = '<bos>'
end_symbol = '<eos>'
unk_symbol = "<unk>"
dummy_symbols = [pad_symbol, start_symbol, end_symbol, unk_symbol]

np.random.seed(1)

DIR = os.getcwd()


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


def update_vocab(symbol, idxvocab, vocabxid):
    idxvocab.append(symbol)
    vocabxid[symbol] = len(idxvocab) - 1


class Vocab(object):
    def __init__(self, path_vec, lower, zeros, id_list, frequency=15):
        self.path = path_vec
        self.word2idx = defaultdict(int)
        self.idx2word = []
        self.word_vectors = []
        self.frequency = frequency
        self.lower = lower
        self.zeros = zeros
        self.idx = 0
        self.id_list = id_list
        self.load_data()

    def load_sentences(self, path, lower, zeros):
        """
        load sentences from original dataset
        each abstract is separated by the '$'
        in each abstract, one sentence is represented by two lines, one for words, the other for tags, each line is separated by space
        """
        sentences = []
        with codecs.open(path, 'r', 'utf-8') as f:
            while 1:
                ln = f.readline()
                wordline = zero_digits(ln.strip()) if zeros else ln.strip()
                wordline = wordline.lower() if lower else wordline

                if not wordline:
                    break

                wordline = wordline.split()
                # if len(wordline) == 1 and wordline[0] == '$':
                #     continue
                tagline = f.readline().strip().split()
                assert (len(wordline) == len(tagline))

                sentences.append(wordline)
        return sentences

    def load_data(self):
        vocab_freq = defaultdict(int)
        sentences = []
        for dirpath, dirnames, filenames in os.walk(DIR):
            for filename in filenames:
                if filename.endswith('train_data'):
                    path = os.path.join(dirpath, filename)
                    for id in self.id_list:
                        if str(id) in path:
                            sentences += self.load_sentences(path, self.lower, self.zeros)
                            print(path)

        for sent in sentences:
            for word in sent:
                vocab_freq[word] += 1
        # add in dummy symbols into vocab
        for s in dummy_symbols:
            update_vocab(s, self.idx2word, self.word2idx)
        # remove low fequency words
        for w, f in sorted(vocab_freq.items(), key=operator.itemgetter(1), reverse=True):
            if f < self.frequency:
                break
            else:
                update_vocab(w, self.idx2word, self.word2idx)
        word_vec = {}
        is_in = defaultdict(int)
        with open(self.path, 'r') as f:
            line = f.readline().strip().split()
            word_dim = len(line) - 1
            word_vec[line[0]] = line[1:]
            is_in[line[0]] = 1
            for line in f.readlines():
                line = line.strip().split()
                word_vec[line[0]] = line[1:]
                is_in[line[0]] = 1
            for vi, v in enumerate(self.idx2word):
                if is_in[v]:
                    self.word_vectors.append(word_vec[v])
                else:
                    self.word_vectors.append(
                        np.random.uniform(-0.5 / word_dim, 0.5 / word_dim, [word_dim, ]))
            print 'Vocab size:', len(self.word_vectors)
            print 'word2idx:', len(self.word2idx)
            print 'idx2word:', len(self.idx2word)

            self.word_vectors = np.asarray(self.word_vectors, dtype=np.float32)


class Tag(object):
    def __init__(self):
        self.tag2idx = defaultdict(int)
        self.idx2tag = {}
        self.define_tags()

    def define_tags(self):
        self.tag2idx['O'] = 0
        self.tag2idx['B'] = 1
        self.tag2idx['M'] = 2
        self.tag2idx['E'] = 3
        self.tag2idx['S'] = 4

        self.idx2tag[0] = 'O'
        self.idx2tag[1] = 'B'
        self.idx2tag[2] = 'M'
        self.idx2tag[3] = 'E'
        self.idx2tag[4] = 'S'