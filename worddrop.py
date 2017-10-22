#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 09:06:52 2017

@author: ray
"""
import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import warnings
import sys
import time
warnings.filterwarnings("ignore")  

def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

def make_idx_data_cv(revs, word_idx_map, cv, max_l=51, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)   
        sent.append(rev["y"])
        if rev["split"]==cv:            
            test.append(sent)        
        else:  
            train.append(sent)   
    train = np.array(train,dtype="int")
    test = np.array(test,dtype="int")
    return [train, test]

     
if __name__ == '__main__':
    print "loading data...",
    x = cPickle.load(open("mr.p","rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print "data loaded!"
    datasets = make_idx_data_cv(revs, word_idx_map, 0, max_l=56,k=300, filter_h=5)
    
####################
## only 3 drop
#==============================================================================
#     n_padding=4
#     a=datasets[0]
#     word_drop_rate=0.5
#     a2 = a[0:3,:] #dai cao zuo de shu zu
#     for i in xrange(3):
#         print i
#         n_nonzero=(a2[np.nonzero(a2[i].reshape(1,65))]).shape[0]-1 # nonzero
#         drop_pos = [np.random.randint(0, n_nonzero) for __ in range(int(n_nonzero*word_drop_rate))]        
#         a2[i,drop_pos[0]+n_padding]=0 # 4 shi qian mian bu ling de ge shu
#         
#==============================================================================

## drop the whole dataset
    n_padding=4
    a=datasets[0]
    word_drop_rate=0.5
    #a2 = a[0:3,:] #dai cao zuo de shu zu
    for i in xrange(a.shape[0]):
        n_nonzero=(a[np.nonzero(a[i].reshape(1,65))]).shape[0]-1 # nonzero
        drop_pos = [np.random.randint(0, n_nonzero) for __ in range(int(n_nonzero*word_drop_rate))]
        if len(drop_pos) == 0:
            print str(i) + "this sentence is not dropped out"
            continue
        for drop_po in drop_pos:
            a[i,drop_po+n_padding]=0 # 4 shi qian mian bu ling de ge shu
     