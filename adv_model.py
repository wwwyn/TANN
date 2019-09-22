import numpy as np
import tensorflow as tf
import sys

sys.path.append("..")
import tflearn
from tensorflow.python.ops import rnn_cell
from tensorflow.contrib.rnn import LSTMStateTuple
import math


class Model(object):
    def __init__(self, batch_size=64, vocab_size=5620,
                 word_dim=300, lstm_dim=300, num_classes=5,
                 l2_reg_lambda=0.0,
                 clip=1,
                 init_embedding=None,
                 lstm_net=False,
                 num_domain=2,
                 lm_fw=1,
                 lm_bw=1,
                 topic_num=100,
                 use_gate=1,
                 use_lm=1,
                 use_adv=1,
                 filter_sizes=[3, 4, 5],
                 num_filters=200,
                 conv_activation='relu',
                 num_decode_steps=100,
                 is_training=True):

        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.lstm_dim = lstm_dim
        self.num_classes = num_classes
        self.num_domain = num_domain
        self.l2_reg_lambda = l2_reg_lambda
        self.clip = clip
        self.lstm_net = lstm_net
        self.topic_num = topic_num
        self.use_gate = use_gate
        self.use_adv = use_adv
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.conv_activation = conv_activation
        self.hidden_size = self.lstm_dim * 2
        self.lm_fw = lm_fw
        self.lm_bw = lm_bw
        self.num_steps = num_decode_steps + 1
        with tf.variable_scope("embedding"):
            if init_embedding is None:
                self.embedding = tf.get_variable(name='embedding', shape=[vocab_size, word_dim],
                                                      dtype=np.float32)
            else:
                self.embedding = tf.get_variable(name="embedding", shape=init_embedding.shape,
                                                 initializer=tf.constant_initializer(init_embedding), trainable=True)
        self.ready()

    def get_precision(self, logits, labels, type):
        soft_logits = tf.nn.softmax(logits)
        predicted = tf.equal(tf.argmax(soft_logits, 1), tf.cast(labels, tf.int64))
        if type == 'mean':
            acc = tf.reduce_mean(tf.cast(predicted, tf.float32))
        elif type == 'sum':
            acc = tf.reduce_sum(tf.cast(predicted, tf.int32))
        return acc

    def target_bi_autoencoder(self, fw_x, fw_y, bw_x, bw_y, seq_len, size, t_lm_seq_len, encoder_fw_state, encoder_bw_state):
        with tf.variable_scope("bi-decoder"):
            lm_fw_x = tf.nn.embedding_lookup(self.embedding, fw_x)
            lm_fw_x = tf.nn.dropout(lm_fw_x, self.dropout_keep_prob)
            lm_fw_x = tf.reshape(lm_fw_x, [size, -1, self.word_dim])  # ba*se*wd

            lm_bw_x = tf.nn.embedding_lookup(self.embedding, bw_x)
            lm_bw_x = tf.nn.dropout(lm_bw_x, self.dropout_keep_prob)
            lm_bw_x = tf.reshape(lm_bw_x, [size, -1, self.word_dim])  # ba*se*wd

            fw_decoder_inputs = self._tensor_to_list(lm_fw_x, self.num_steps)
            bw_decoder_inputs = self._tensor_to_list(lm_bw_x, self.num_steps)

            decoder_initial_fw_state = encoder_fw_state
            decoder_initial_bw_state = encoder_bw_state

            if self.lm_fw:
                decode_fw_cell = rnn_cell.BasicLSTMCell(self.lstm_dim, forget_bias=1.0)
                fw_decoder_outputs, _ = tf.contrib.legacy_seq2seq.rnn_decoder(fw_decoder_inputs,
                                                                              decoder_initial_fw_state,
                                                                              decode_fw_cell,
                                                                              scope='fw_decoder')
                fw_decoder_outputs = tf.reshape(tf.transpose(tf.stack(fw_decoder_outputs, 0), [1, 0, 2]),
                                                [-1, self.lstm_dim])
            if self.lm_bw:
                decode_bw_cell = rnn_cell.BasicLSTMCell(self.lstm_dim, forget_bias=1.0)
                bw_decoder_outputs, _ = tf.contrib.legacy_seq2seq.rnn_decoder(bw_decoder_inputs,
                                                                              decoder_initial_bw_state,
                                                                              decode_bw_cell,
                                                                              scope='bw_decoder')
                bw_decoder_outputs = tf.reshape(tf.transpose(tf.stack(bw_decoder_outputs, 0), [1, 0, 2]),
                                                [-1, self.lstm_dim])
            with tf.variable_scope('lm_mlp'):
                if self.lm_fw:
                    self.fw_w = tf.get_variable(
                        shape=[self.lstm_dim, self.vocab_size],
                        name="lm_weights_fw",
                        regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))
                    self.fw_b = tf.get_variable(name="lm_bias_fw", shape=[self.vocab_size],
                                                initializer=tf.zeros_initializer)
                if self.lm_bw:
                    self.bw_w = tf.get_variable(
                        shape=[self.lstm_dim, self.vocab_size],
                        name="lm_weights_bw",
                        regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))
                    self.bw_b = tf.get_variable(name="lm_bias_bw", shape=[self.vocab_size],
                                                initializer=tf.zeros_initializer)
            mask = tf.sequence_mask(t_lm_seq_len, self.num_steps, dtype=tf.bool)
            fw_loss = 0
            bw_loss = 0
            if self.lm_fw:
                fw_labels = tf.one_hot(fw_x, self.vocab_size)
                fw_decoder_prob = tf.reshape(tf.nn.softmax(tf.matmul(fw_decoder_outputs, self.fw_w) + self.fw_b),
                                             [-1, self.num_steps, self.vocab_size])

                fw_loss = tf.reduce_sum(tf.boolean_mask(-fw_labels * tf.log(fw_decoder_prob), mask)) / tf.reduce_sum(tf.cast(t_lm_seq_len,tf.float32))
            if self.lm_bw:
                bw_labels = tf.one_hot(bw_x, self.vocab_size)
                bw_decoder_prob = tf.reshape(tf.nn.softmax(tf.matmul(bw_decoder_outputs, self.bw_w) + self.bw_b),
                                             [-1, self.num_steps, self.vocab_size])
                bw_loss = tf.reduce_sum(tf.boolean_mask(-bw_labels * tf.log(bw_decoder_prob), mask)) / tf.reduce_sum(tf.cast(t_lm_seq_len,tf.float32))

            return fw_loss, bw_loss

    def ready_one_domain(self, input_x, input_y, input_dom, topic_input, seq_len, size, d_id, reuse=False):
        with tf.variable_scope("encoder") as encoder_scope:
            if reuse:
                encoder_scope.reuse_variables()
            fw_cell = rnn_cell.BasicLSTMCell(self.lstm_dim)
            bw_cell = rnn_cell.BasicLSTMCell(self.lstm_dim)
            if self.use_gate:
                with tf.variable_scope('gate'):
                    b_z = tf.get_variable(name="bias_topic", shape=[self.hidden_size],
                                               initializer=tf.zeros_initializer)

                    u_z = tf.get_variable(
                        shape=[self.topic_num, self.hidden_size],
                        name="u_z",
                        regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))
                    w_z = tf.get_variable(
                        shape=[self.hidden_size, self.hidden_size],
                        name="w_z",
                        regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))

            (forward_output, backward_output), (encoder_fw_state, encoder_bw_state) = tf.nn.bidirectional_dynamic_rnn(
                fw_cell,
                bw_cell,
                input_x,
                dtype=tf.float32,
                sequence_length=seq_len,
                scope='layer_1'
            )
            output = tf.concat(axis=2, values=[forward_output, backward_output])
            if self.use_gate:
                topic_correlation = tf.nn.tanh(tf.nn.bias_add(tf.tensordot(output, w_z, ((2,), (0,))) + \
                                                                   tf.expand_dims(tf.matmul(topic_input, u_z), 1),b_z))
                gate_output = output * topic_correlation
            else:
                gate_output = output

        with tf.variable_scope("MLP_layer") as mlp_scope:
            if reuse:
                mlp_scope.reuse_variables()
            w_mlp = tf.get_variable(
                shape=[self.hidden_size, self.num_classes],
                name="mlp_weights",
                regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))

            b_mlp = tf.get_variable(name="mlp_bias", shape=[self.num_classes], initializer=tf.zeros_initializer)

        output = tf.reshape(gate_output, [-1, self.hidden_size])
        matricized_unary_scores = tf.matmul(output, w_mlp) + b_mlp
        self.unary_scores = tf.reshape(
            matricized_unary_scores,
            [size, -1, self.num_classes])

        # CRF log likelihood
        if d_id == 0:
            with tf.variable_scope('crf_layer') as crf_scope:
                if reuse:
                    crf_scope.reuse_variables()
                log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
                    self.unary_scores, input_y, seq_len)
                self.transition_params = transition_params
                tagging_loss = tf.reduce_mean(-log_likelihood)

        with tf.variable_scope('domain_classifier') as discriminator_scope:
            if reuse:
                discriminator_scope.reuse_variables()
            # apply convolutional filters on the LSTM output
            # b * seq * h * 1
            #      f  * h * 1 * num
            # b * (seq - f + 1) * 1  * num
            # b * 1 * 1 * num
            pooled_outputs = []
            conv_input = tf.expand_dims(gate_output, -1)
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.variable_scope("conv-filter-%d" % filter_size):
                    filter_w = tf.get_variable("filter_w", [filter_size, self.hidden_size, 1, self.num_filters],
                                               initializer=tf.contrib.layers.xavier_initializer())
                    filter_b = tf.get_variable("filter_b", [self.num_filters],
                                               initializer=tf.zeros_initializer)
                    conv = tf.nn.conv2d(conv_input, filter_w, strides=[1, 1, 1, 1], padding="VALID")
                    if self.conv_activation == "identity":
                        conv_activated = tf.nn.bias_add(conv, filter_b)
                    elif self.conv_activation == "relu":
                        conv_activated = tf.nn.relu(tf.nn.bias_add(conv, filter_b))
                    elif self.conv_activation == "tanh":
                        conv_activated = tf.nn.tanh(tf.nn.bias_add(conv, filter_b))
                    elif self.conv_activation == "elu":
                        conv_activated = tf.nn.elu(tf.nn.bias_add(conv, filter_b))

                    # max pooling over time steps
                    h = tf.reduce_max(conv_activated, axis=1, keep_dims=True)
                    # h = tf.nn.max_pool(
                    #     conv_activated,
                    #     ksize=[1, MAX_LEN - filter_size + 1, 1, 1],
                    #     strides=[1, 1, 1, 1],
                    #     padding='VALID',
                    #     name="pool")
                    pooled_outputs.append(h)

            # Combine all the pooled features
            num_filters_total = self.num_filters * len(self.filter_sizes)
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
            # Add dropout
            h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
            with tf.variable_scope('domain_mlp_layer'):
                d_w = tf.get_variable(
                    "domain_w",
                    shape=[num_filters_total, self.num_domain],
                    initializer=tf.contrib.layers.xavier_initializer())
                d_b = tf.get_variable(initializer=tf.zeros_initializer, shape=[self.num_domain], name = "domain_b")
            domain_logits = tf.matmul(h_drop, d_w) + d_b
            domain_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=domain_logits, labels=input_dom, name='domain_x_entropy'))
        if d_id == 0:
            return tagging_loss, domain_loss, topic_correlation, gate_output, domain_logits, encoder_fw_state, encoder_bw_state
        else:
            return domain_loss, topic_correlation, gate_output, domain_logits, encoder_fw_state, encoder_bw_state

    def ready(self):
        s_x = self.s_x = tf.placeholder(tf.int32, [None, None])
        s_y = self.s_y = tf.placeholder(tf.int32, [None, None])
        s_dom = self.s_dom = tf.placeholder(tf.int32, [None])
        s_seq_len = self.s_seq_len = tf.placeholder(tf.int32, [None])
        s_topic_input = self.s_topic_input = tf.placeholder(tf.float32, [None, None])

        t_x = self.t_x = tf.placeholder(tf.int32, [None, None])
        t_dom = self.t_dom = tf.placeholder(tf.int32, [None])
        t_seq_len = self.t_seq_len = tf.placeholder(tf.int32, [None])
        t_topic_input = self.t_topic_input = tf.placeholder(tf.float32, [None, None])

        self.lr = tf.placeholder(tf.float32, [])
        self.lamda = tf.placeholder(tf.float32, [])
        self.beta = tf.placeholder(tf.float32, [])
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # transform to the embedding form
        s_seq_len = tf.cast(s_seq_len, tf.int64)
        s_x = tf.nn.embedding_lookup(self.embedding, s_x)  # batch_size * (sequence*1) * word_dim
        s_x = tf.nn.dropout(s_x, self.dropout_keep_prob)
        s_size = tf.shape(s_x)[0]
        s_x = tf.reshape(s_x, [s_size, -1, self.word_dim])  # ba*se*wd

        t_seq_len = tf.cast(t_seq_len, tf.int64)
        t_x = tf.nn.embedding_lookup(self.embedding, t_x)
        t_x = tf.nn.dropout(t_x, self.dropout_keep_prob)
        t_size = tf.shape(t_x)[0]
        t_x = tf.reshape(t_x, [t_size, -1, self.word_dim])

        t_lm_fw_x = self.t_lm_fw_x = tf.placeholder(tf.int32, [None, self.num_steps])
        t_lm_fw_y = self.t_lm_fw_y = tf.placeholder(tf.int32, [None, None])
        t_lm_bw_x = self.t_lm_bw_x = tf.placeholder(tf.int32, [None, self.num_steps])
        t_lm_bw_y = self.t_lm_bw_y = tf.placeholder(tf.int32, [None, None])
        t_lm_seq_len = self.t_lm_seq_len = tf.placeholder(tf.int32, [None])


        self.s_tagging_loss, self.s_domain_loss, self.s_topic_correlation, self.s_gate_output, self.s_domain_logits, self.s_encoder_fw_state, self.s_encoder_bw_state = \
            self.ready_one_domain(s_x, s_y, s_dom, s_topic_input, s_seq_len, s_size, 0, reuse=False)

        self.t_domain_loss, self.t_topic_correlation, self.t_gate_output, self.t_domain_logits, self.t_encoder_fw_state, self.t_encoder_bw_state = \
            self.ready_one_domain(t_x, t_x, t_dom, t_topic_input, t_seq_len, t_size, 1, reuse=True)

        self.fw_loss, self.bw_loss = self.target_bi_autoencoder(t_lm_fw_x, t_lm_fw_y, t_lm_bw_x, t_lm_bw_y, t_seq_len,
                                                                t_size, t_lm_seq_len, self.t_encoder_fw_state, self.t_encoder_bw_state)

        self.lr_summary = tf.summary.scalar('learning_rate', self.lr)
        self.beta_summary = tf.summary.scalar('lm_rate', self.beta)
        self.lamda_summary = tf.summary.scalar('lamda_adv', self.lamda)

        with tf.variable_scope("d_op"):
            self.d_loss = self.s_domain_loss + self.t_domain_loss
            self.d_global_step = tf.Variable(0, name="d_global_step", trainable=False)
            self.domain_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/domain_classifier')

            print('d_op')
            for i in self.domain_params:
                print(i)
            d_grads, _ = tf.clip_by_global_norm(tf.gradients(self.d_loss, self.domain_params), self.clip)
            d_optimizer = tf.train.MomentumOptimizer(self.lr, 0.9)
            self.train_d_op = d_optimizer.apply_gradients(zip(d_grads, self.domain_params),
                                                          global_step=self.d_global_step)

            domain_logits = tf.concat([self.s_domain_logits, self.t_domain_logits], 0)
            domain_labels = tf.concat([s_dom, t_dom], 0)
            domain_acc = self.get_precision(domain_logits, domain_labels, 'mean')
            self.domain_correct_num = self.get_precision(domain_logits, domain_labels, 'sum')
            self.domain_acc_summary = tf.summary.scalar('domain_acc', domain_acc)
            self.domain_loss_summary = tf.summary.scalar('domain_loss', self.d_loss)
            self.merge_summary_domain = tf.summary.merge([self.domain_acc_summary, self.domain_loss_summary])

        with tf.variable_scope("g_op"):
            self.g_loss = self.s_tagging_loss + tf.negative(self.lamda * self.d_loss) + self.beta * (self.fw_loss + self.bw_loss)
            self.tagging_loss_summary = tf.summary.scalar('s_tagging_loss', self.s_tagging_loss)
            self.g_loss_summary = tf.summary.scalar('g_loss', self.g_loss)
            self.recon_loss_summary = tf.summary.scalar('recon_loss', self.fw_loss + self.bw_loss)

            self.g_global_step = tf.Variable(0, name="g_global_step", trainable=False)
            self.g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/embedding') + \
                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/encoder') + \
                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/MLP_layer') + \
                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/crf_layer') + \
                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/bi-decoder')

            print('g_op')
            for i in self.g_params:
                print(i)
            g_grads, _ = tf.clip_by_global_norm(tf.gradients(self.g_loss, self.g_params), self.clip)
            g_optimizer = tf.train.MomentumOptimizer(self.lr, 0.9)
            self.train_g_op = g_optimizer.apply_gradients(zip(g_grads, self.g_params), global_step=self.g_global_step)
            self.merge_summary_g = tf.summary.merge([self.g_loss_summary, self.tagging_loss_summary,
                                                     self.recon_loss_summary, self.beta_summary, self.lamda_summary])

    def _tensor_to_list(self, tensor, num_steps):
        """
        Splits the input tensor sentence into a list of 1-d
        tensors, as much as the number of time steps.
        This is necessary for seq2seq functions.
        """
        return [tf.squeeze(step, [1])
                for step in tf.split(tensor, num_steps, 1)]