#!/usr/bin/env python

import random
import numpy as np
import sys
import tensorflow as tf
import time


class MLPRegressor(object):
    """Multilayer Perceptron Regressor
    """
    def __init__(self, inp_dim=128, hid_dim=128, nclass=1,
                 nlayer=1, reg_lambda=0, lr=0.001):
        self.INP_DIM = inp_dim  # input layer dimension
        self.HID_DIM = hid_dim  # hidden layer dimension

        # prepare input and output placeholder
        self.inp_x = tf.placeholder(tf.float32, [None, self.INP_DIM], 'input_x')
        self.inp_y = tf.placeholder(tf.float32, [None, nclass], 'input_y')

        # layers
        w0 = tf.get_variable(
            'W0', shape=[inp_dim, hid_dim],
            initializer=tf.contrib.layers.xavier_initializer())
        b0 = tf.Variable(tf.constant(0.1, shape=[hid_dim]), name='b0')
        layer1 = tf.add(tf.matmul(self.inp_x, w0), b0)
        layer1 = tf.nn.relu(layer1)

        # output layer
        wo = tf.get_variable(
            'Wo', shape=[hid_dim, nclass],
            initializer=tf.contrib.layers.xavier_initializer())
        bo = tf.Variable(tf.constant(0.1, shape=[nclass]), name='bo')
        self.preds = tf.add(tf.matmul(layer1, wo), bo, name='prediction')

        # calculate loss
        self.loss = tf.sqrt(
            tf.reduce_mean(tf.square(tf.sub(self.inp_y, self.preds))))  # rmse
        # L2 loss for all params
        reg_loss = reg_lambda * (tf.nn.l2_loss(w0) + tf.nn.l2_loss(b0) +
                                 tf.nn.l2_loss(wo) + tf.nn.l2_loss(bo))
        self.total_loss = self.loss + reg_loss

        # bp optimization
        self.opt = tf.train.AdamOptimizer(
            learning_rate=lr).minimize(self.total_loss)
        # self.opt = tf.train.AdadeltaOptimizer().minimize(self.total_loss)

        # saver and loader
        self.saver = tf.train.Saver()

        # training steps
        self.global_steps = 0

    def pred(self, sess, input_x):
        pred_dict = {self.inp_x: input_x}
        return sess.run(self.preds, feed_dict=pred_dict)

    def evals(self, sess, dev_x, dev_y):
        evals_dict = {self.inp_x: dev_x, self.inp_y: dev_y}
        return self.loss.eval(feed_dict=evals_dict, session=sess)

    def train_step(self, sess, inp_batch_x, inp_batch_y, evals=None):
        input_dict = {self.inp_x: inp_batch_x, self.inp_y: inp_batch_y}
        sess.run(self.opt, feed_dict=input_dict)
        self.global_steps += 1

        # evaluation
        # evals: [data_set name, eval period, x, y]
        if evals:
            evlstr = ''
            for evl in evals:
                if (self.global_steps - 1) % evl[1] == 0:
                    loss = self.evals(sess, evl[2], evl[3])
                    evlstr += evl[0] + ': ' + str(loss) + '\t'
                else:
                    evlstr += evl[0] + ': --------\t'
            print evlstr.rstrip('\t')
            sys.stdout.flush()

    def get_batch(self, data, batch_size=1):
        inp_x, inp_y = [], []
        for _ in xrange(batch_size):
            raw_y, raw_x = random.choice(data)
            inp_y.append(np.array(raw_y, dtype=np.float32))
            inp_x.append(np.array(raw_x, dtype=np.float32))
        return np.array(inp_x), np.array(inp_y)

    def _sparse2plain(self, data_inst, ndim=None):
        x_inst = np.zeros((ndim), dtype=np.float32)
        if len(data_inst) > 1 and data_inst[1]:  # if there are non-zero features
            for idx, value in data_inst[1:]:
                x_inst[idx - 1] = value
        return np.array(x_inst, dtype=np.float32), \
            np.array([data_inst[0]], dtype=np.float32)

    def get_batch_from_sparse(self, data, batch_size=1, ndim=None):
        # data are [label, [idx, value], [idx, value] ... ] fields
        inp_x, inp_y = [], []
        for _ in xrange(batch_size):
            raw_yxs = random.choice(data)
            inst_x, inst_y = self._sparse2plain(raw_yxs, ndim=ndim)
            inp_x.append(inst_x)
            inp_y.append(inst_y)
        return np.array(inp_x), np.array(inp_y)


def test_sparse():
    mdl_mlp = MLPRegressor(
        inp_dim=10000,
        hid_dim=256,
        reg_lambda=0.01)

    ndim = 10000
    inp_data = [[0.2, [3, 1.1], [999, 0.2], [9999, -0.7]],
                [0.7, [37, 1.1], [399, 0.2], [5999, -0.7]],
                [0.9, [1, 1.1], [299, 0.2], [10000, -0.7]]]

    data_inp_x, data_inp_y = mdl_mlp.get_batch_from_sparse(inp_data, 5, ndim)
    evals = [['train-rmse', 1.0, data_inp_x, data_inp_y]]

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for _ in range(5):
        mdl_mlp.train_step(sess, data_inp_x, data_inp_y, evals)
        data_inp_x, data_inp_y = mdl_mlp.get_batch_from_sparse(
            inp_data, batch_size=5, ndim=ndim)
    sess.close()


def test_dense():
    mdl_mlp = MLPRegressor(
        inp_dim=10000,
        hid_dim=256,
        reg_lambda=0.01)

    ndata = 64
    ndim = 10000
    w = (np.random.rand(ndim) * 10).astype(np.float32).reshape((ndim, 1))
    print 'real value of W(probably with noise):'
    print w
    b = np.array([0.5] * ndata).reshape((ndata, 1))
    x_data = np.random.rand(ndata, ndim)
    y_data = np.dot(x_data, w) + b
    # noise = (np.random.rand(ndata).astype(np.float32) * 0.02).reshape((ndata, 1))
    # y_data = np.dot(x_data, w) + b + noise
    inp_data = np.array([np.array([np.array(y), np.array(x)]) for x, y in
                         zip(x_data.tolist(), y_data.tolist())])

    data_inp_x, data_inp_y = mdl_mlp.get_batch(inp_data, 64)
    evals = [['train-rmse', 1.0, data_inp_x, data_inp_y]]
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for _ in range(20):
        mdl_mlp.train_step(sess, data_inp_x, data_inp_y, evals)
        data_inp_x, data_inp_y = mdl_mlp.get_batch(inp_data, 64)
    sess.close()

if __name__ == '__main__':
    # test_dense()
    test_sparse()