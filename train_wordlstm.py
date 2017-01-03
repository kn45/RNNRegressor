#!/usr/bin/env python

import random
import numpy as np
import cPickle as pkl
import tensorflow as tf
import time
import sparse_rnn_reg as rnnreg


if __name__ == '__main__':
    vocab_size = max([int(x.strip('\n').split('\t')[1])
                      for x in open('worddict').readlines()]) + 1
    print 'vocab size: ', vocab_size
    init_embed = pkl.load(open('init_embedding.pkl', 'rb'))
    mdl_lstm = rnnreg.RNNRegressor(
        vocab_size=vocab_size,
        emb_dim=200,
        hid_dim=100,
        nclass=1,
        time_len=40,
        pad_id=0,
        one_hot=False,
        cellt='LSTM',
        nlayer=1,
        init_embed=init_embed,
        reg_lambda=1,
        lr=0.0002)

    # with open('data_train/valid_feature.tsv') as f:
    with open('data_train/valid_feature_small.tsv') as f:
        valid_data = [x.rstrip('\n').split('\t') for x in f.readlines()]
    valid_y = np.array([x[0:1] for x in valid_data], dtype=np.float32)
    valid_x = [map(int, x[1:]) for x in valid_data]
    valid_x = mdl_lstm.padding(valid_x)

    with open('data_train/train_feature.tsv') as f:
        train_data = [x.rstrip('\n').split('\t') for x in f.readlines()]
    train_data = [[map(float, x[0:1]), map(int, x[1:])] for x in train_data]

    train_x, train_y = mdl_lstm.get_pad_batch(train_data, 128)

    # probability of conducting evaluation
    # 1.0 means do evalutation every batch
    evals = [['train', 1.0, train_x, train_y],
             ['valid', 1.0, valid_x, valid_y]]

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    for x in range(500):
        print x
        mdl_lstm.train_step(sess, train_x, train_y, evals)

    sess.close()
