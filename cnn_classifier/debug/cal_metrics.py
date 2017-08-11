#!/usr/bin/env python

import sys
sys.path.append('../../../MLFlow/utils/')
import dataproc

data_t = []
data_p = []
with open(sys.argv[1]) as f:
    for line in f:
        t, p = line.rstrip('\n').split('\t')
        data_t.append(float(t))
        data_p.append(float(p))

print dataproc.auc(data_t, data_p)
