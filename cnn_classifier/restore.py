import numpy as np
import sys
sys.path.append('../../MLFlow/utils')
import tensorflow as tf
from text_cnn import TextCNNClassifier
import dataproc

NCLASS = 2
NWORDS = 18765
SEQ_LEN = 100
FILTER_SIZES = [3, 4, 5]
NFILTERS = 128


def inp_fn_unilabel(data):
    inp_x = []
    inp_y = []
    for inst in data:
        flds = inst.split('\t')
        label = map(int, flds[0:1])
        feats = map(int, flds[1:])
        inp_y.append(label)
        inp_x.append(dataproc.zero_padding(feats, SEQ_LEN))
    return np.array(inp_x), np.array(inp_y)

save_path = './model_ckpt/'
test_file = './rt-polarity.shuf.test'
with open(test_file) as f:
    test_data = [x.rstrip('\n') for x in f.readlines()]
test_x, test_y = inp_fn_unilabel(test_data)

# load model
mdl2 = TextCNNClassifier(
    seq_len=SEQ_LEN,
    emb_size=128,
    nclass=NCLASS,
    vocab_size=NWORDS,
    filter_sizes=FILTER_SIZES,
    nfilters=NFILTERS,
    reg_lambda=0.0,
    lr=1e-3)

sess2 = tf.Session()
mdl2.saver.restore(sess2, tf.train.latest_checkpoint(save_path))
print sess2.run(mdl2.global_step)
with open('test_res', 'w') as f:
    preds = mdl2.predict_proba(sess2, test_x)
    for l, p in zip(test_y, preds):
        print >> f, '\t'.join(map(str, [l[0], p[1]]))
