'''
Build a bidirectional LSTM Classifier Trained on SST2 dataset

Code is based on
https://github.com/zhegan27/sentence_classification
'''

import time
import cPickle
import numpy as np
import theano
import theano.tensor as tensor

from model.lstm_classifier import init_params, init_tparams
from model.lstm_classifier import build_model

from model.optimizers import Adam
from model.utils import get_minibatches_idx
from model.utils import zipp, unzip, numpy_floatX
from nltk.tokenize import word_tokenize


def load_adv():
    filename="sst2_0.4_two_examples.txt"
    lines = []
    x = []
    with open(filename, 'r') as f:
        lines.extend(f.readlines())
    for i, line in enumerate(lines):
        x.append(line.rstrip())
        
    adver_y=np.array([100])
    adver_x =[]
    for i in xrange(len(x)):
        if i%2 == 0: # remove original keep adver
            continue
        X=x[i].split( )
        sent = []
        if X[-1]== 'unimaginative':
            adver_y=np.row_stack([adver_y,np.array([1])])
        else:
            adver_y=np.row_stack([adver_y,np.array([0])])
        
        for j in X:
    #        print j
            if j =="unimaginative":
                continue        
    #        print wordtoix[j]
            sent.append(wordtoix[j])
        adver_x.append(sent)
    adver_y=adver_y[1:]
    adver_y=adver_y.reshape([len(x)/2,])
    Adver_x=[]
    if len(adver_x) < len(test[0]):
         n_tile=len(test[0]) / len(adver_x) + 1
         for i in xrange(n_tile):
             for j in xrange(len(adver_x)):
                 Adver_x.append(adver_x[j]) 
         Adver_y=np.tile(adver_y,(n_tile,))   
    Adver_x=Adver_x[:len(test[0])] 
    Adver_y=Adver_y[:len(test[0])]  
    Adver=(Adver_x,Adver_y)
    return Adver



""" used to calculate the prediction error. """

def pred_error(f_pred, prepare_data, data, iterator):
    
    """ compute the prediction error. 
    """
    valid_err = 0
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  np.array(data[1])[valid_index],
                                  maxlen=None)
        preds = f_pred(x, mask)
        targets = np.array(data[1])[valid_index]
        valid_err += (preds == targets).sum()
    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

    return valid_err

""" used to preprocess the dataset. """
    
def prepare_data(seqs, labels, maxlen=None):
    
    # seqs: a list of sentences
    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((maxlen, n_samples)).astype('int32')
    x_mask = np.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.
        
    labels = np.array(labels).astype('int32')
    
    return x, x_mask, labels


    
def create_valid(train_set,valid_portion=0.10):
    
    # split training set into validation set
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.random.permutation(n_samples)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)

    return train, valid
    

if __name__ == '__main__':
    
    print ('loading data...')
    x = cPickle.load(open("./data/sst2.p","rb"))
    train, test, W, ixtoword, wordtoix= x[0], x[1], x[2], x[3], x[4]
    del x
#     test = load_adv() # test on adversarial test set
    n_words = W.shape[0] # test on original test set
    
    results = []
    # run the cnn classifier ten times
    i=0
    train0, valid = create_valid(train, valid_portion=0.10)
# =============================================================================
    n_x=300;n_h=200; dropout_val=0.5; patience=10; max_epochs=20; lrate=0.0002; 
    batch_size=50; valid_batch_size=50; dispFreq=10; validFreq=100;
    saveFreq=200; saveto = 'trec_lstm_result.npz'
    
    """ train, valid, test : datasets
        W : the word embedding initialization
        n_words : vocabulary size
        n_x : word embedding dimension
        n_h : LSTM/GRU number of hidden units 
        dropout_val: dropput probability
        patience : Number of epoch to wait before early stop if no progress
        max_epochs : The maximum number of epoch to run
        lrate : learning rate
        batch_size : batch size during training
        valid_batch_size : The batch size used for validation/test set
        dispFreq : Display to stdout the training progress every N updates
        validFreq : Compute the validation error after this number of update.
        saveFreq: save the result after this number of update.
        saveto: where to save the result.
    """

    options = {}
    options['n_words'] = n_words
    options['n_x'] = n_x
    options['n_h'] = n_h
    options['patience'] = patience
    options['max_epochs'] = max_epochs
    options['lrate'] = lrate
    options['batch_size'] = batch_size
    options['valid_batch_size'] = valid_batch_size
    options['dispFreq'] = dispFreq
    options['validFreq'] = validFreq
    
    print ('Model options {}'.format(options))
    
    print ('{} train examples'.format(len(train[0])))
    print ('{} valid examples'.format(len(valid[0])))
    print ('{} test examples'.format(len(test[0])))

    print ('Building model...')
    
    n_y = np.max(train[1]) + 1
    options['n_y'] = n_y
    
    params = init_params(options,W)
    
    print ('Loading model...')
    with open('data/sst2_model.pickle', 'rb') as f:
        tmp = cPickle.load(f)
    
    for keys in params:
        params[str(keys)]=tmp[str(keys)]
    
    del tmp
    
    
    
    tparams = init_tparams(params)

    (use_noise, x, mask, y, f_pred_prob, f_pred, cost) = build_model(tparams,options)
    
    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = Adam(tparams, cost, [x, mask, y], lr)

    
    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)
    
    use_noise.set_value(0.)
    
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted)
    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
    test_err = pred_error(f_pred, prepare_data, test, kf_test)
    
    print ('Train {} Valid {} Test {}'.format(train_err, valid_err, test_err))
    
 
# =============================================================================

    print ("train_err %.2f, valid_err %.2f, test_err %.2f" % (train_err, valid_err, test_err))
    results.append(test_err)
    print ("accuarcy: %.2f" % (1-results[0]))