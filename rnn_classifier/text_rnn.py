#!/usr/bin/env python

import random
import numpy as np
import sys
import tensorflow as tf
import time


class TextRNNClassifier(object):
    """Text RNN Classifier
    """
    def __init__(self, vocab_size, emb_dim=256, hid_dim=128, nclass=1,
                 seq_len=50, cellt='LSTM', nlayer=1, reg_lambda=0, nsample=5,
                 nlabel=1, label_repr='dense',
                 obj='softmax', lr=1e-3, init_embed=None):
        """Construct RNN network.
        nlabel > 1 only when multi-label task.
        """
        if label_repr not in ['dense', 'sparse']:
            raise Exception('invalid label_repr')
        if obj not in ['softmax', 'ss']:
            raise Exception('invalid obj')

        # prepare input and output placeholder
        self.inp_x = tf.placeholder(tf.int32, [None, seq_len], 'input_x')
        self.dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
        inp_len = tf.reduce_sum(tf.sign(self.inp_x), reduction_indices=1)
        self.label_repr = label_repr
        if label_repr == 'dense':
            self.inp_y_dense = tf.placeholder(tf.float32, [None, nclass], 'input_y')
            self.inp_y_sparse = tf.argmax(self.inp_y_dense, 1)
        if label_repr == 'sparse':
            self.inp_y_sparse = tf.placeholder(tf.int64, [None, nlabel], 'input_y')
            self.inp_y_dense = tf.reduce_sum(
                tf.one_hot(indices=self.inp_y_sparse, depth=nclass),
                reduction_indices=1)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # embedding
        if init_embed is not None:
            embedding = tf.Variable(
                tf.convert_to_tensor(init_embed, dtype=tf.float32),
                trainable=emb_trainable, name='embedding')
        else:
            embedding = tf.get_variable(
                'embedding', shape=[vocab_size, emb_dim],
                initializer=tf.random_uniform_initializer(
                    minval=-0.2,
                    maxval=0.2,
                    dtype=tf.float32))
        inp_emb = tf.nn.embedding_lookup(embedding, self.inp_x)

        # construct basic cell
        if cellt == 'LSTM':
            cell = tf.nn.rnn_cell.LSTMCell(
                num_units=hid_dim,
                initializer=tf.random_uniform_initializer(
                    minval=-1./emb_dim**0.5,
                    maxval=+1./emb_dim**0.5))
        elif cellt == 'GRU':
            cell = tf.nn.rnn_cell.GRUCell(num_units=hid_dim)
        elif cellt == 'BasicRNN':
            cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hid_dim)
        else:
            sys.stderr.write('invalid cell type')
            sys.exit(1)

        # layers
        if nlayer > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell] * nlayer)

        # dropout
        # set keep_prob = 1.0 when predicting
        cell = tf.nn.rnn_cell.DropoutWrapper(
            cell, output_keep_prob=self.dropout_prob)

        # construct rnn
        outputs, states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            sequence_length=inp_len,
            inputs=inp_emb)

        # extract res from dynamic series by given series_length
        batch_size = tf.shape(outputs)[0]  # [batch_size, seq_len, dimension]
        oup_idx = tf.range(0, batch_size) * seq_len + (inp_len - 1)
        oup_flat = tf.reshape(outputs, [-1, hid_dim])  # [batch*seq_len, dim]
        oup_rnn = tf.gather(oup_flat, oup_idx)

        # make prediction
        w = tf.get_variable(
            'W', shape=[hid_dim, nclass],
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[nclass]), name='b')
        self.scores = tf.nn.xw_plus_b(oup_rnn, w, b, name='scores')
        self.preds = tf.argmax(self.scores, 1, name='predictions')
        self.proba = tf.nn.softmax(self.scores)

        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self.scores, labels=self.inp_y_dense))

        # calculate softmax loss
        if obj == 'softmax':
            l2_loss = reg_lambda * (tf.nn.l2_loss(w) + tf.nn.l2_loss(b))
            self.total_loss = self.loss + l2_loss

        # calculate sampled softmax_loss
        if obj == 'ss':
            labels = tf.reshape(self.inp_y_sparse, [-1, 1])
            local_w_t = tf.cast(tf.transpose(w), tf.float32)
            local_b = tf.cast(b, tf.float32)
            local_inp = tf.cast(oup_rnn, tf.float32)
            self.total_loss = tf.cast(
                tf.nn.sampled_softmax_loss(
                    weights=local_w_t,
                    biases=local_b,
                    labels=labels,
                    inputs=local_inp,
                    num_sampled=nsample,
                    num_classes=nclass,
                    partition_strategy='div'),
                dtype=tf.float32)

        # bptt
        self.opt = tf.train.AdamOptimizer(lr).minimize(
            self.total_loss, global_step=self.global_step)

        # accuracy
        correct_preds = tf.equal(
            self.preds, self.inp_y_sparse)
        self.accuracy = tf.reduce_mean(
            tf.cast(correct_preds, 'float'), name='accuracy')

        # auc
        labels_c = self.inp_y_sparse
        preds_c = self.proba[:, 1]
        self.auc = tf.metrics.auc(
            labels=labels_c,
            predictions=preds_c,
            num_thresholds=1000)

        # saver and loader
        self.saver = tf.train.Saver()

    def train_step(self, sess, inp_batch_x, inp_batch_y):
        input_dict = {
            self.inp_x: inp_batch_x,
            self.dropout_prob: 0.5}
        if self.label_repr == 'dense':
            input_dict[self.inp_y_dense] = inp_batch_y
        if self.label_repr == 'sparse':
            input_dict[self.inp_y_sparse] = inp_batch_y
        sess.run(self.opt, feed_dict=input_dict)

    def eval_step(self, sess, dev_x, dev_y, metrics=None):
        if not metrics:
            metrics = ['loss']
        eval_dict = {
            self.inp_x: dev_x,
            self.dropout_prob: 1.0}
        if self.label_repr == 'dense':
            eval_dict[self.inp_y_dense] = dev_y
        if self.label_repr == 'sparse':
            eval_dict[self.inp_y_sparse] = dev_y
        eval_res = []
        for metric in metrics:
            if metric == 'loss':
                eval_res.append(sess.run(self.loss, feed_dict=eval_dict))
            if metric == 'accuracy':
                eval_res.append(sess.run(self.accuracy, feed_dict=eval_dict))
            if metric == 'auc':
                eval_res.append(sess.run(self.auc, feed_dict=eval_dict)[0])
        return eval_res

    def predict(self, sess, input_x):
        pred_dict = {
            self.inp_x: input_x,
            self.dropout_prob: 1.0}
        return sess.run(self.preds, feed_dict=pred_dict)

    def predict_proba(self, sess, input_x):
        pred_dict = {
            self.inp_x: input_x,
            self.dropout_prob: 1.0}
        return sess.run(self.proba, feed_dict=pred_dict)
