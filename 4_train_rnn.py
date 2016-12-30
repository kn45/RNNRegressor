#!/usr/bin/env python

import random
import numpy as np
import tensorflow as tf
import time
import rnn_regressor as rnnreg


if __name__ == '__main__':
    vocab_size = max([int(x.strip('\n').split('\t')[1])
                      for x in open('chardict').readlines()]) + 1
    print 'vocab size: ', vocab_size
    mdl_lstm = rnnreg.LSTMRegressor(
        vocab_size=vocab_size,
        emb_dim=256,
        hid_dim=100,
        nclass=1,
        time_len=40,
        pad_id=0)

    with open('data_train/valid_feature.tsv') as f:
        valid_data = [x.rstrip('\n').split('\t') for x in f.readlines()]
    valid_y = np.array([x[0:1] for x in valid_data], dtype=np.float32)
    valid_x = [map(int, x[1:]) for x in valid_data]
    valid_x = mdl_lstm.padding(valid_x)

    with open('data_train/train_feature.tsv') as f:
        train_data = [x.rstrip('\n').split('\t') for x in f.readlines()]
    train_data = [[map(float, x[0:1]), map(int, x[1:])] for x in train_data]

    train_x, train_y = mdl_lstm.get_pad_batch(train_data, 128)

    evals = [['train', 1.0, train_x, train_y],
             ['valid', 0.1, valid_x, valid_y]]

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    for x in range(100):
        print x
        mdl_lstm.train_step(sess, train_x, train_y, evals)

    sess.close()
