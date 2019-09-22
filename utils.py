# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from voc import Tag
from nltk.stem.porter import PorterStemmer
import tensorflow as tf
#
stem = PorterStemmer().stem
# s = porter_stemmer.stem(u'symbolic' +  u'execution')


def get_stem(setlist):
    return [' '.join([stem(word) for word in phrase.split()]) for phrase in setlist]


def get_chunk(x, y):
    """
    get the chunk of the tagging sentence
    x: sentence list
    y: the label of the sentence list
    return: the chunk list of x sentence
    """
    leny = len(y)
    chunk_list = []
    output = ''
    for i in range(leny):
        tag = y[i]
        if tag == 'B':
            output += x[i]
        elif tag == 'M':
            output += ' ' + x[i]
        elif tag == 'E':
            output += ' ' + x[i]
            chunk_list.append(output)
            output = ''
        elif tag == 'S':
            chunk_list.append(x[i])
    return chunk_list


def get_phrase(sentence, y_predict, y_true):

    tag = Tag()
    id_to_tag = tag.idx2tag
    sentence = list(sentence)
    y_predict = list(y_predict)
    y_true = list(y_true)

    lens = len(sentence)
    if y_predict[0] not in ['B', 'M', 'S', 'E', 'O']:
        for i in range(lens):
            y_predict[i] = id_to_tag[y_predict[i]]
            y_true[i] = id_to_tag[y_true[i]]

    return get_chunk(sentence, y_predict), get_chunk(sentence, y_true)


def get_prf_num(predict_labels, true_labels):
    predict_labels = get_stem(predict_labels)
    true_labels = get_stem(true_labels)
    hit_num = 0
    pred_num = 0
    true_num = 0
    hit_num += len(set(true_labels) & set(predict_labels))
    pred_num += len(set(predict_labels))
    true_num += len(set(true_labels))
    return hit_num, pred_num, true_num

