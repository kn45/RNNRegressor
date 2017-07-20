#!/usr/bin/env python

import numpy as np
import sys
sys.path.append('../../MLFlow/utils')
import tensorflow as tf
from text_rnn import TextRNNClassifier
import dataproc

NCLASS = 2
NWORDS = 18764
SEQ_LEN = 100

def inp_fn(data):
    inp_x = []
    inp_y = []
    for inst in data:
        flds = inst.split('\t')
        label = int(flds[0])
        feats = map(int, flds[1:])
        inp_y.append(dataproc.sparse2dense([label], ndim=NCLASS))
        inp_x.append(dataproc.zero_padding(feats, SEQ_LEN))
    return np.array(inp_x), np.array(inp_y)

# train_file = '/Users/tchen/projects/ZhihuCup/mdl_cnn1/feat_train/trnvld_feature.tsv'
train_file = './rt-polarity.shuf'
freader = dataproc.BatchReader(train_file)

mdl = TextRNNClassifier(
    seq_len=SEQ_LEN,
    emb_dim=256,
    nclass=NCLASS,
    vocab_size=NWORDS,
    reg_lambda=0.0,
    lr=1e-3)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
niter = 0
while niter < 1000:
    niter += 1
    batch_data = freader.get_batch(64)
    #print batch_data[0]
    if len(batch_data) <= 0:
        break
    train_x, train_y = inp_fn(batch_data)
    mdl.train_step(sess, train_x, train_y)
    print mdl.eval_step(sess, train_x, train_y)

train_fp.close()
sess.close()

