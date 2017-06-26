#!/usr/bin/env python

import random
import numpy as np
import cPickle as pkl
import sparse_rnn_reg as rnnreg
import sys
import tensorflow as tf
import time


if __name__ == '__main__':
    vocab_size = max([int(x.strip('\n').split('\t')[1])
                      for x in open('worddict').readlines()]) + 1
    print 'vocab size: ', vocab_size
    init_embed = pkl.load(open('init_embedding.pkl', 'rb'))
    mdl_rnn = rnnreg.RNNRegressor(
        vocab_size=vocab_size,
        emb_dim=200,
        hid_dim=128,
        time_len=25,
        one_hot=False,
        cellt='BasicRNN',
        nlayer=1,
        init_embed=init_embed,
        keep_prob=1.0,
        emb_trainable=True,
        # lr=0.0005,
        reg_lambda=1.0)

    # with open('data_train/valid_feature.tsv') as f:
    with open('data_train/valid_feature_small.tsv') as f:
        valid_data = [x.rstrip('\n').split('\t') for x in f.readlines()]
    valid_y = np.array([x[0:1] for x in valid_data], dtype=np.float32)
    valid_x = [map(int, x[1:]) for x in valid_data]
    valid_x = mdl_rnn.padding(valid_x)

    with open('data_train/train_feature.tsv') as f:
        train_data = [x.rstrip('\n').split('\t') for x in f.readlines()]
    train_data = [[map(float, x[0:1]), map(int, x[1:])] for x in train_data]

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    for x in xrange(2000):
        train_x, train_y = mdl_rnn.get_pad_batch(train_data, 256)
        # probability of conducting evaluation
        # 1.0 means do evalutation every batch
        evals = [['train', 1.0, train_x, train_y],
                 ['valid', 1.0, valid_x, valid_y]]
        sys.stdout.write(str(x) + '    ')
        mdl_rnn.train_step(sess, train_x, train_y, evals)

    valid_res = mdl_rnn.pred(sess, valid_x)
    with open('valid_res', 'w') as fo:
        for rec in valid_res:
            print >> fo, rec[0]
    sess.close()
