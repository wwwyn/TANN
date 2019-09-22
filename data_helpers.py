import numpy as np
import codecs
np.random.seed(1)
import ast


class DataIterator(object):
    def __init__(self, df, is_train):
        """
        is_train to keep the original order of data
        num_file: per abstract sentence number
        """
        self.is_train = is_train
        self.df = df
        self.total = len(df)
        self.pos = 0
        self.shuffle()

        self.epochs = 0

    def shuffle(self):
        if self.is_train:
            self.df = self.df.sample(frac=1, random_state=1).reset_index(drop=True)
            self.pos = 0
        return

    def next_batch(self, batch_size, round=-1, classifier=False, labeled=True):
        if self.pos + batch_size - 1 >= self.total:
            self.shuffle()
        res = self.df.ix[self.pos:self.pos + batch_size - 1]

        words = map(lambda x: map(int, x.split(",")), res['words'].tolist())
        tags = map(lambda x: map(int, x.split(",")), res['tags'].tolist())
        topics = map(lambda x: map(float, x.split(",")), res['topic_vector'].tolist())

        self.pos += batch_size

        # Pad sequences with 0s so they are all the same length
        maxlen = max(res['length'].values)
        x = np.zeros([batch_size, maxlen], dtype=np.int32)
        for i, x_i in enumerate(x):
            x_i[:res['length'].values[i]] = words[i]
        if labeled == True:
            y = np.zeros([batch_size, maxlen], dtype=np.int32)
            for i, y_i in enumerate(y):
                y_i[:res['length'].values[i]] = tags[i]
        else:
            y = None

        if classifier is False:
            return x, y, res['length'].values, topics
        else:
            y_class = np.array([round] * batch_size)
            return x, y, y_class, res['length'].values, topics

    def next_test_batch(self, batch_size):
        res = self.df.ix[self.pos: self.pos + batch_size - 1]

        real_words = res['real_words'].tolist()
        words = map(lambda x: map(int, x.split(",")), res['words'].tolist())
        tags = map(lambda x: map(int, x.split(",")), res['tags'].tolist())
        topics = map(lambda x: map(float, x.split(",")), res['topic_vector'].tolist())

        self.pos += batch_size
        if self.pos >= len(self.df):
            self.pos = 0
        maxlen = max(res['length'])
        x = np.zeros([batch_size, maxlen], dtype=np.int32)
        for i, x_i in enumerate(x):
            x_i[:res['length'].values[i]] = words[i]
        y = np.zeros([batch_size, maxlen], dtype=np.int32)
        for i, y_i in enumerate(y):
            y_i[:res['length'].values[i]] = tags[i]

        return real_words, x, y, res['length'].values, topics
