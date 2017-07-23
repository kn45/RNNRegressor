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

save_path = './model_export/model.ckpt-500'

# load model
mdl2 = TextRNNClassifier(
    seq_len=SEQ_LEN,
    emb_dim=256,
    nclass=NCLASS,
    vocab_size=NWORDS,
    reg_lambda=0.0,
    lr=1e-3,
    label_repr=LABEL_REPR,
    obj='softmax',
    nsample=1)

sess2 = tf.Session()
mdl2.saver.restore(sess2, save_path)
print sess2.run(mdl2.global_step)
