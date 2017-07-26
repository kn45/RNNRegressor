import numpy as np
import sys
sys.path.append('../../MLFlow/utils')
import tensorflow as tf
from text_cnn import TextCNNClassifier
import dataproc

NCLASS = 2
NWORDS = 18765
SEQ_LEN = 100
LABEL_REPR = 'sparse'
FILTER_SIZES = [3, 4, 5]
NFILTERS = 128

save_path = './model_ckpt/model.ckpt-500'

# load model
mdl2 = TextCNNClassifier(
    seq_len=SEQ_LEN,
    emb_dim=128,
    nclass=NCLASS,
    vocab_size=NWORDS,
    filter_sizes=FILTER_SIZES,
    nfilters=NFILTERS,
    reg_lambda=0.0,
    lr=1e-3,
    label_repr=LABEL_REPR)



sess2 = tf.Session()
mdl2.saver.restore(sess2, save_path)
print sess2.run(mdl2.global_step)
