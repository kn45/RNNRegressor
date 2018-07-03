#!/usr/bin/env python

import numpy as np
import sys
sys.path.append('../../MLFlow/utils')
import tensorflow as tf
from text_rnn import TextRNNClassifier
import dataproc

NCLASS = 2
NWORDS = 18765
SEQ_LEN = 100
LABEL_REPR = 'sparse'


def inp_fn(data):
    def _inp_fn(data, y_repr='dense'):
        inp_x = []
        inp_y = []
        for inst in data:
            flds = inst.split('\t')
            label = map(int, flds[0:1])
            feats = map(int, flds[1:])
            if y_repr == 'dense':
                inp_y.append(dataproc.sparse2dense(label, ndim=NCLASS))
            elif y_repr == 'sparse':
                inp_y.append(label)
            inp_x.append(dataproc.zero_padding(feats, SEQ_LEN))
        return np.array(inp_x), np.array(inp_y)
    return _inp_fn(data, LABEL_REPR)

train_file = './rt-polarity.shuf.train'
test_file = './rt-polarity.shuf.test'
freader = dataproc.BatchReader(train_file)
with open(test_file) as f:
    test_data = [x.rstrip('\n') for x in f.readlines()]
test_x, test_y = inp_fn(test_data)

mdl = TextRNNClassifier(
    seq_len=SEQ_LEN,
    emb_dim=256,
    nclass=NCLASS,
    vocab_size=NWORDS,
    reg_lambda=0.0,
    lr=1e-3,
    label_repr=LABEL_REPR,
    obj='softmax',
    nsample=1)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
metrics = ['loss', 'auc']
niter = 0
mdl_ckpt_dir = './model_ckpt/model.ckpt'
while niter < 500:
    niter += 1
    batch_data = freader.get_batch(128)
    if not batch_data:
        break
    train_x, train_y = inp_fn(batch_data)
    mdl.train_step(sess, train_x, train_y)
    train_eval = mdl.eval_step(sess, train_x, train_y, metrics)
    test_eval = mdl.eval_step(sess, test_x, test_y, metrics) \
        if niter % 20 == 0 else 'SKIP'
    print niter, 'train:', train_eval, 'test:', test_eval
save_path = mdl.saver.save(sess, mdl_ckpt_dir, global_step=mdl.global_step)
print "model saved:", save_path
sess.close()
