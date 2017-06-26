#!/usr/bin/env python


import cPickle
import fasttext
import numpy as np
import os
import sys
from mlfutil import CharEncoder, draw_progress


data_file = sys.argv[1]
cencoder = CharEncoder()


def title_encoding(title):
    chars = []
    for char in title.decode('utf8'):
        idx = cencoder.cat2idx(char)
        idx = cencoder.cat2idx('UNK') if idx < 0 else idx
        chars.append(idx)
    return chars


def build_dict():
    cencoder.build_dict('data_all/data_all.tsv')
    cencoder.save_dict('chardict')


def init():
    cencoder.load_dict('chardict')


def main():
    outfile = sys.argv[2]
    build_dict()
    init()
    with open(data_file) as f:
        data = [l.rstrip('\r\n').split('\t') for l in f.readlines()]
    fo = open(outfile, 'w')
    data_size = len(data)
    for nr, rec in enumerate(data):
        title = rec[1]
        title_feats = title_encoding(title)
        print >> fo, \
            '\t'.join(map(str, rec[0:1] + title_feats))
        draw_progress(nr, data_size-1)
    fo.close()

if __name__ == '__main__':
    main()
