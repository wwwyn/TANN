import os
import datetime
import random
import numpy as np
import tensorflow as tf
import pandas as pd
import sys
import math
import shutil
sys.path.append("..")
import codecs
from utils import *
from voc import Vocab, Tag
from adv_model import Model
import data_helpers
import logging
import cPickle
DIR = os.path.abspath(os.getcwd())
# ==================================================
print 'Generate words and characters need to be trained'

tf.flags.DEFINE_integer("vocab_size", 16169, "vocab_size(default)")
# Data parameters
tf.flags.DEFINE_integer("word_dim", 300, "word_dim")
tf.flags.DEFINE_integer("lstm_dim", 300, "lstm_dim")
tf.flags.DEFINE_integer("num_classes", 5, "num_classes")
tf.flags.DEFINE_integer("num_domain", 2, "num_domain")
tf.flags.DEFINE_boolean("embed_status", True, "embed_status")
# Model Hyperparameters[t]
tf.flags.DEFINE_float("l2_reg_lambda", 0.000, "L2 regularizaion lambda (default: 0.5)")
tf.flags.DEFINE_float("clip", 1, "gradient clip")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_string("conv_activation", "relu", "activation")

# Training parameters
tf.flags.DEFINE_string("vocab_path", os.path.join(DIR, 'model_dump/vocab.pickle'), "voc path")
tf.flags.DEFINE_string("CORPUS", '0,1', "the pair task")
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 40)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("seed", 1, "train_status")
tf.flags.DEFINE_string('log_dir', 'LOGs/log1', "log directory")
tf.flags.DEFINE_float("lr", 0.0001, "learning rate (default: 0.01)")
tf.flags.DEFINE_float("adv_rate", 0.1, "adv rate (default: 0.01)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.8)")
tf.flags.DEFINE_integer("lm_fw", 1, "fw")
tf.flags.DEFINE_integer("lm_bw", 1, "bw")
tf.flags.DEFINE_boolean("lstm_net", True, "use lstm or not")
tf.flags.DEFINE_integer("source_only", 0, "is_source_only")
tf.flags.DEFINE_float("lm_rate", 0.1, "language modeling scalar rate")
tf.flags.DEFINE_integer("topic_num", 100, "topic_num")
tf.flags.DEFINE_integer("use_gate", 1, "use gate to control the topic and local information")
tf.flags.DEFINE_integer("use_lm", 1, "use the language model or not")
tf.flags.DEFINE_integer("use_adv", 1, "use the adversarial training or not")
tf.flags.DEFINE_integer("decay_rate", 0, "decay the learning rate")
tf.flags.DEFINE_integer("lm_decay", 0, "decay the lm rate")
tf.flags.DEFINE_integer("early_stop", 3, "the epochs to early stop")
tf.flags.DEFINE_string("initializer", "he", "the initializer method")
tf.flags.DEFINE_integer("change_adv", 0, "change the adv method")
tf.flags.DEFINE_integer("change_lm", 0, "change the lm rate")
tf.flags.DEFINE_integer("change_lm_soft", 0, "change the lm rate softly")
tf.flags.DEFINE_integer("change_adv_soft", 0, "change the adv rate softly")
tf.flags.DEFINE_integer("adv_adjust", 0, "change the adv method2")
tf.flags.DEFINE_integer("adv_zero", 0, "change the adv method3")
tf.flags.DEFINE_integer("quick_evaluate", 1, "evaluate according to steps")
tf.flags.DEFINE_float("adv_gama", 0.5, "change the adv gama")
tf.flags.DEFINE_string('labeled_train', '', "train_data_dir")
tf.flags.DEFINE_string('restore_model', '', "model directory")
tf.flags.DEFINE_integer("k", 1, "train d step")
tf.flags.DEFINE_integer("lamda_type", 1, "adv_type")
tf.flags.DEFINE_float("lamda_v", 0.05, "adv_value")
tf.flags.DEFINE_integer("num_decode_steps", 1, "decoder_step")
# Misc
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
random.seed(FLAGS.seed)
np.random.seed(FLAGS.seed)
TAGS = Tag()
# init_embedding = np.random.rand(FLAGS.vocab_size, FLAGS.word_dim)
# init_embedding = init_embedding.astype(dtype=np.float32)

if FLAGS.embed_status is False:
    init_embedding = None
    vocab = cPickle.load(open(FLAGS.vocab_path))
else:
    vocab = cPickle.load(open(FLAGS.vocab_path))
    init_embedding = vocab[0]
    FLAGS.vocab_size = init_embedding.shape[0]
    print(type(init_embedding[0][0]))
if FLAGS.source_only == 1:
    FLAGS.use_lm = 0
    FLAGS.use_adv = 0
    FLAGS.use_gate = 0
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items(), reverse=True):
    print("{}={} \n".format(attr.upper(), value))
print("")
corpus_list = FLAGS.CORPUS.split(',')
assert(FLAGS.num_domain == len(corpus_list))
out_dir = os.path.abspath(os.path.join(os.path.curdir, FLAGS.log_dir))
# if os.path.exists(out_dir):
#     shutil.rmtree(out_dir)
os.mkdir(out_dir)
# define log file
log_name = os.path.join(out_dir, 'Adv_train.log')
os.mknod(log_name)
logger = logging.getLogger('record')
hdlr = logging.FileHandler(log_name)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

logger.info("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items(), reverse=True):
    logger.info("{}={} \n".format(attr.upper(), value))
logger.info("")

print("Loading data...")
logger.info("Loading data...")

f_domain = codecs.open(os.path.join(out_dir, 'domain_correct'), 'w', 'utf-8')
# Load data
train_data_iterator = []
domain_data_iterator = []
dev_data_iterator = []
test_data_iterator = []
dev_df = []
test_df = []

print("Loading data...")
logger.info("Loading data...")
for i in range(len(corpus_list)):
    paper_i = corpus_list[i]
    # TRAIN_FILE = DIR + '/data/' + str(paper_i) + '/train_mt.csv'
    TRAIN_FILE = DIR + '/data/' + str(paper_i) + '/labeled_train_mt.csv'
    DEV_FILE = DIR + '/data/' + str(paper_i) + '/dev_mt.csv'
    TEST_FILE = DIR + '/data/' + str(paper_i) + '/test_mt.csv'
    DOMAIN_TRAIN = DIR + '/data/' + str(paper_i) + '/domain_train_mt.csv'
    # labeled data is only in the source domain
    if i == 0:
        train_data_iterator.append(data_helpers.DataIterator(pd.read_csv(TRAIN_FILE), is_train=True))
    domain_data_iterator.append(data_helpers.DataIterator(pd.read_csv(DOMAIN_TRAIN), is_train=True)) #Domain
    logger.info('Domain ' + str(i) + str(domain_data_iterator[i].total))
    if i == 0:
        # only use source dev data to select model
        dev_df.append(pd.read_csv(DEV_FILE))
        dev_data_iterator.append(data_helpers.DataIterator(dev_df[0], is_train=False))
    if i == 1:
        # final test on the target data
        test_df.append(pd.read_csv(TEST_FILE))
        test_data_iterator.append(data_helpers.DataIterator(test_df[0], is_train=False))

session_conf = tf.ConfigProto(allow_soft_placement=True,
                              log_device_placement=False)
session_conf.gpu_options.allow_growth = True


def train_d_step(model, s_x, s_dom, s_seq_len, s_topic, t_x, t_dom, t_seq_len, t_topic, lr):
    feed_dict = {
        model.s_x: s_x,
        model.s_dom: s_dom,
        model.s_seq_len: s_seq_len,
        model.s_topic_input: s_topic,
        model.t_x: t_x,
        model.t_dom: t_dom,
        model.t_seq_len: t_seq_len,
        model.t_topic_input: t_topic,
        model.lr: lr,
        model.dropout_keep_prob: FLAGS.dropout_keep_prob,
    }
    _, d_num, d_step, d_loss, merge_summary_d = sess.run(
        [model.train_d_op, model.domain_correct_num, model.d_global_step, model.d_loss, model.merge_summary_domain],
        feed_dict)

    time_str = datetime.datetime.now().isoformat()
    print("d***{}: step {}, loss {:g}".format(time_str, d_step, d_loss))

    return d_step, merge_summary_d, d_num


def train_g_step(model, s_x, s_y, s_dom, s_seq_len, s_topic, t_x, t_dom, t_seq_len, t_topic, t_fw_x, t_fw_y, t_bw_x, t_bw_y, lm_seq_len, lr, beta, lamda):
    feed_dict = {
        model.s_x: s_x,
        model.s_y: s_y,
        model.s_dom: s_dom,
        model.s_seq_len: s_seq_len,
        model.s_topic_input: s_topic,
        model.t_x: t_x,
        model.t_dom: t_dom,
        model.t_seq_len: t_seq_len,
        model.t_topic_input: t_topic,
        model.lr: lr,
        model.lamda: lamda,
        model.beta: beta,
        model.t_lm_fw_x: t_fw_x,
        model.t_lm_fw_y: t_fw_y,
        model.t_lm_bw_x: t_bw_x,
        model.t_lm_bw_y: t_bw_y,
        model.t_lm_seq_len: lm_seq_len,
        model.dropout_keep_prob: FLAGS.dropout_keep_prob
    }
    _, g_step, g_loss, lr_summary, merge_summary_g = sess.run(
        [model.train_g_op, model.g_global_step, model.g_loss, model.lr_summary, model.merge_summary_g],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("g***{}: step {}, loss {:g}".format(time_str, g_step, g_loss))

    return g_step, lr_summary, merge_summary_g


def evaluate_word_PRF(all_words, pred, true, test=False):
    res = []
    all_hit = 0
    all_pred = 0
    all_true = 0
    for abstract_i in range(len(all_words)):
        predict, true_ = get_phrase(all_words[abstract_i], pred[abstract_i], true[abstract_i])
        # print(all_words[abstract_i])
        # print(pred[abstract_i])
        # print(true[abstract_i])
        # print(predict)
        # print(true_)

        hit_num, pred_num, true_num = get_prf_num(predict, true_)
        all_hit += hit_num
        all_pred += pred_num
        all_true += true_num

        res.append({'hit_num': hit_num, 'pred_num': pred_num, 'true_num': true_num})

    precision = -1.0
    recall = -1.0
    f1 = -1.0

    if all_pred != 0:
        precision = 1.0 * all_hit / all_pred
    if all_true != 0:
        recall = 1.0 * all_hit / all_true
    if precision > 0 and recall > 0:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    print 'P: ', precision
    print 'R: ', recall
    print 'F: ', f1
    if test:
        return precision, recall, f1
    else:
        return f1


def fast_all_predict(model, batch_iterator):
    """
    every batch is an abstract
    """
    y_pred, y_true = [], []
    all_words = []
    while 1:
        real_words, x_batch, y_batch, seq_len_batch, topic_vec = batch_iterator.next_test_batch(50)
        real_words = [words.split(' ') for words in real_words]
        all_words += real_words
        # infer predictions
        if FLAGS.use_gate:
            feed_dict = {
                model.t_x: x_batch,
                model.t_seq_len: seq_len_batch,
                model.dropout_keep_prob: 1.0,
                model.t_topic_input: topic_vec
            }
        else:
            feed_dict = {
                model.t_x: x_batch,
                model.t_seq_len: seq_len_batch,
                model.dropout_keep_prob: 1.0
            }

        unary_scores, transition_params = sess.run(
            [model.unary_scores, model.transition_params], feed_dict)

        y_pred_, y_true_ = [], []
        for unary_scores_, y_, seq_len_ in zip(unary_scores, y_batch, seq_len_batch):
            # remove padding
            unary_scores_ = unary_scores_[:seq_len_]

            # Compute the highest scoring sequence.
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                unary_scores_, transition_params)

            y_pred_.append(viterbi_sequence)
            y_true_.append(y_[:seq_len_].tolist())

        y_pred += y_pred_
        y_true += y_true_
        if batch_iterator.pos == 0:
            break

    return all_words, y_pred, y_true


def final_test_step(model, iterator, test=False):
    all_words, y_true, y_pred = fast_all_predict(model, iterator)
    if test:
        print 'Test:'
    else:
        print 'Dev'
    return all_words, y_pred, y_true


def get_batch_lm(batch_x, seq_x):
    row_length = len(batch_x)
    seq_length = len(batch_x[0])
    batch_x_new = np.zeros((row_length, FLAGS.num_decode_steps + 1), dtype=int)
    batch_x_new[:, 1:seq_length+ 1] = batch_x
    eos_idx = vocab[1]['<eos>']
    col = np.zeros((row_length, 1))
    col[:, :] = eos_idx
    batch_x_new[:, 0:1] = col

    batch_y_new = np.zeros((row_length, FLAGS.num_decode_steps + 1), dtype=int)
    batch_y_new[:, 0:seq_length] = batch_x
    for row, col in enumerate(seq_x):
        batch_y_new[row, col] = eos_idx

    batch_lm_x = batch_x_new
    batch_lm_y = batch_y_new

    return batch_lm_x, batch_lm_y, seq_x + 1


def get_lm_data(targetx_batch, targetseq_len_batch):
    target_seq_len_batch = np.zeros(len(targetseq_len_batch), dtype=int)
    for i in range(len(targetseq_len_batch)):
        target_seq_len_batch[i] = min(targetseq_len_batch[i], FLAGS.num_decode_steps)
    target_x_batch = targetx_batch[:, 0:FLAGS.num_decode_steps]
    lm_fw_x, lm_fw_y, seq_x = get_batch_lm(target_x_batch, target_seq_len_batch)
    batch_x_r = np.zeros((len(target_x_batch), len(target_x_batch[0])))
    for i, j in enumerate(target_seq_len_batch):
        batch_x_r[i, 0:j] = np.flip(target_x_batch[i, 0:j], 0)
    lm_bw_x, lm_bw_y, seq_x = get_batch_lm(batch_x_r, target_seq_len_batch)
    return lm_fw_x, lm_fw_y, lm_bw_x, lm_bw_y, target_seq_len_batch

# t_ul_x, _, t_ul_dom, t_ul_len, t_ul_topic = \
#     domain_data_iterator[1].next_batch(FLAGS.batch_size, round=1, classifier=True)
# lm_fw_x, lm_fw_y, lm_bw_x, lm_bw_y, seq_x = get_lm_data(t_ul_x, t_ul_len)
#
# print(t_ul_x[0])
# print(lm_fw_x[0])
# print(lm_bw_x[0])
#
# print(len(lm_fw_x[0]))
# print(len(lm_fw_y[0]))
# exit()


def get_var():
    tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    if FLAGS.use_adv:
        logger.info('all_op')
        for i in tvars:
            logger.info(i)

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)

    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    tf.set_random_seed(FLAGS.seed)
    with sess.as_default():
        # build model
        initializer = tf.contrib.layers.xavier_initializer(seed=FLAGS.seed)
    with tf.variable_scope('model', reuse=None, initializer=initializer):
        model = Model(batch_size=FLAGS.batch_size,
                      vocab_size=FLAGS.vocab_size,
                      word_dim=FLAGS.word_dim,
                      lstm_dim=FLAGS.lstm_dim,
                      num_classes=FLAGS.num_classes,
                      clip=FLAGS.clip,
                      l2_reg_lambda=FLAGS.l2_reg_lambda,
                      init_embedding=init_embedding,
                      lstm_net=FLAGS.lstm_net,
                      num_domain=FLAGS.num_domain,
                      lm_fw=FLAGS.lm_fw,
                      lm_bw=FLAGS.lm_bw,
                      topic_num=FLAGS.topic_num,
                      use_gate=FLAGS.use_gate,
                      use_lm=FLAGS.use_lm,
                      use_adv=FLAGS.use_adv,
                      filter_sizes=list(map(int, FLAGS.filter_sizes.split(','))),
                      num_filters=FLAGS.num_filters,
                      conv_activation=FLAGS.conv_activation,
                      num_decode_steps=FLAGS.num_decode_steps
                      )
    get_var()

    # Output directory for models
    logger.info("Writing to {}\n".format(out_dir))

    sess.run(tf.global_variables_initializer())

    log_dir = os.path.abspath(os.path.join(out_dir, "log"))
    adv_train_writer = tf.summary.FileWriter(log_dir + '/adv', sess.graph)

    # Checkpoint directory. TensorFlow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    filename = 'task' + str('_') + str(corpus_list[1])
    checkpoint_all = os.path.join(checkpoint_dir, filename)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    else:
        logger.info('remove previous checkpoints')
        shutil.rmtree(checkpoint_dir)
    if FLAGS.restore_model != '':
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/embedding') + \
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/encoder') + \
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/softmax') + \
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/loss/crf_layer')
        if FLAGS.use_gate:
            tvars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/gate')
        saver = tf.train.Saver(tvars)
        latest_cp = tf.train.latest_checkpoint(FLAGS.restore_model)
        saver.restore(sess, latest_cp)
    else:
        saver = tf.train.Saver()

    logger.info('Task_{}-{} Training starts'.format(corpus_list[0], corpus_list[1]))
    best_accuary = 0.0
    best_epoch = 0
    best_step = 0
    test_p, test_r, test_f = 0.0, 0.0, 0.0
    source_id = 0
    target_id = 1
    num_steps = int(math.floor(train_data_iterator[source_id].total * 1.0 / FLAGS.batch_size))
    for epoch in range(FLAGS.num_epochs):
        train_data_iterator[source_id].shuffle()
        p = float(epoch) / FLAGS.num_epochs
        for step_i in range(num_steps):
            progress = epoch + (step_i + 0.0) / num_steps
            if FLAGS.lamda_type == 1:
                if FLAGS.adv_zero == 0:
                    if epoch == 0:
                        lamda = 0
                    else:
                        lamda = 2.0 / (1.0 + np.exp(-FLAGS.adv_gama * (progress - 1))) - 1.0
                else:
                    lamda = 2.0 / (1.0 + np.exp(-FLAGS.adv_gama * progress)) - 1.0
            elif FLAGS.lamda_type == 2:
                lamda = FLAGS.lamda_v
            elif FLAGS.lamda_type == 3:
                lamda = 2. / (1. + np.exp(-10 * p)) - 1

            # train d step
            for d_k in range(int(FLAGS.k)):
                s_ul_x, _, s_ul_dom, s_ul_len, s_ul_topic = \
                    domain_data_iterator[source_id].next_batch(FLAGS.batch_size / 2, round=0, classifier=True)
                t_ul_x, _, t_ul_dom, t_ul_len, t_ul_topic = \
                    domain_data_iterator[target_id].next_batch(FLAGS.batch_size / 2, round=1, classifier=True)

                d_step, merge_summary_d, d_num = train_d_step(model, s_ul_x, s_ul_dom, s_ul_len, s_ul_topic, t_ul_x,
                                                              t_ul_dom, t_ul_len, t_ul_topic, FLAGS.lr)

                adv_train_writer.add_summary(merge_summary_d, d_step)
                f_domain.write(str(d_step) + ' ' + str(d_num) + '\n')

            # train g step
            s_l_x, s_l_y, s_l_dom, s_l_len, s_l_topic = \
                train_data_iterator[source_id].next_batch(FLAGS.batch_size, round=0, classifier=True)
            t_ul_x, _, t_ul_dom, t_ul_len, t_ul_topic = \
                domain_data_iterator[target_id].next_batch(FLAGS.batch_size, round=1, classifier=True, labeled=False)

            lm_fw_x, lm_fw_y, lm_bw_x, lm_bw_y, lm_seq_len = get_lm_data(t_ul_x, t_ul_len)
            g_step, lr_summary, merge_summary_g = train_g_step(model, s_l_x, s_l_y, s_l_dom, s_l_len, s_l_topic, t_ul_x,
                                                                t_ul_dom, t_ul_len, t_ul_topic, lm_fw_x, lm_fw_y,
                                                               lm_bw_x, lm_bw_y, lm_seq_len, FLAGS.lr, FLAGS.lm_rate, lamda)
            adv_train_writer.add_summary(merge_summary_g, g_step)
            adv_train_writer.add_summary(lr_summary, g_step)

            if (epoch == 0 and step_i == 0) or (g_step % FLAGS.evaluate_every == 0 and FLAGS.quick_evaluate):
                dev_words, yp, yt = final_test_step(model, dev_data_iterator[source_id])
                tmpacc = evaluate_word_PRF(dev_words, yp, yt)
                if best_accuary < tmpacc:
                    best_accuary = tmpacc
                    best_step = step_i
                    best_epoch = epoch
                    test_words, yp_test, yt_test = final_test_step(model, test_data_iterator[0], test=True)
                    test_p, test_r, test_f = evaluate_word_PRF(test_words, yp_test, yt_test, test=True)
                    logger.info('epoch {} step {} f_{} p_{} r_{} valid_f{}'.format(epoch, step_i, test_f, test_p, test_r, best_accuary))
                    path = saver.save(sess, checkpoint_all)
                    logger.info("Saved model checkpoint to {}\n".format(path))
            print(num_steps, best_epoch, best_step, test_p, test_r, test_f, best_accuary)
        if epoch - best_epoch >= FLAGS.early_stop:
            print('early stoping')
            break
    logger.info('final epoch_{} step_{} f_{} p_{} r_{}'.format(best_epoch, best_step, test_f, test_p, test_r))
