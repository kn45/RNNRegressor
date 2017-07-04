#!/usr/bin/env python

import numpy as np
import tensorflow as tf


class TextCNNClassifier(object):
    """CNN for text classifier
    """
    def __init__(self, seq_len=100, emb_dim=256, nclass=1, vocab_size=10000,
                 filter_sizes, nfilters):

        # prepare input and output placeholder
        self.inp_x = tf.placeholder(tf.int32, [None, seq_len], name='input_x')
        self.inp_y = tf.placeholder(tf.float32, [None, nclass], name='input_y')
        self.dropout_prob = tf.placeholder(tf.float32, name="dropout_prob")

        # embedding
        with tf.namescope('embedding'):
            embedding = tf.Variable(
                tf.random_uniform([vocab_size, emb_dim], -1.0, 1.0),
                name='W')
            self.emb_chars = tf.nn.embedding_lookup(embedding, self.inp_x)
            # expand 'channel' with -1 to satisfy conv2d requirement
            self.emb_chars_exp = tf.expand_dims(self.emb_chars, -1)

        # convolution Layer
        with tf.name_scope('conv-max-pool-%s', filter_size):
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                # [filter height, filter width, in channels, out channels]
                filter_shape = [filter_size, emb_dim, 1, nfilters]
                W = tf.Variable(
                    tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[nfilters]), name='b')
                conv = tf.nn.conv2d(
                    self.emb_chars_exp,
                    W,
                    strides=[1, 1, 1, 1],  # step(direction) of filter moving
                    # commonly [1, stride, stride, 1]
                    padding='VALID',  # filter move within input without pad
                    name='conv')
                # apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                # max-pooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    # max-pool mask size
                    ksize=[1, seq_len - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool')
                # shape of pooled: [batch_size, 1, 1, out_nchannel]
                pooled_outputs.append(pooled)

        # combine all the pooled features
        nfilters_total = nfilters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        # shape of h_pool: [batch_size, 1, 1, out_nchannel*nfilter_size]
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, nfilters_total])

        # add dropout
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_prob)

        # output
        with tf.name_scope('output'):
            W = tf.Variable(
                tf.truncated_normal([nfilters_total, nclass], stddev=0.1),
                name='W')
            b = tf.Variable(tf.constant(0.1, shape=[nclass]), name='b')
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')
            self.preds = tf.argmax(self.scores, 1, name='predictions')
            # self.pred_proba =

        # Calculate mean cross-entropy loss
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    self.scores, self.inp_y))
            # ignore L2 norm loss? refer to paper
            self.total_loss = loss
