#!/usr/bin/env python

import numpy as np
import sys
import tensorflow as tf
from text_rnn import TextRNNClassifier
import preproc

NCLASS = 1999
NWORDS = 411721
SEQ_LEN = 150

def inp_fn(data):
    inp_x = []
    inp_y = []
    for inst in data:
        flds = inst.split('\t')
        label = int(flds[0])
        feats = map(int, flds[1:])
        inp_y.append(preproc.sparse2dense([label], ndim=NCLASS))
        inp_x.append(preproc.zero_padding(feats, SEQ_LEN))
    return np.array(inp_x), np.array(inp_y)

def get_fbatch(fp, batch_size=1):
    out = []
    for line in fp:
        out.append(line.rstrip('\n'))
        if len(out) >= batch_size:
            break
    return out

train_file = '/Users/tchen/projects/ZhihuCup/mdl_cnn1/feat_train/trnvld_feature.tsv'
train_fp = open(train_file)

mdl = TextRNNClassifier(
    seq_len=SEQ_LEN,
    emb_dim=128,
    nclass=NCLASS,
    vocab_size=NWORDS,
    reg_lambda=0.0,
    lr=1e-3)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
niter = 0
while niter < 1000:
    niter += 1
    batch_data = get_fbatch(train_fp, 64)
    #print batch_data[0]
    if len(batch_data) <= 0:
        break
    train_x, train_y = inp_fn(batch_data)
    mdl.train_step(sess, train_x, train_y)
    #print train_x[0]
    #print train_y[0][590:600]
    print mdl.eval_step(sess, train_x, train_y)

train_fp.close()
sess.close()

