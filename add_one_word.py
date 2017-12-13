"""
author: Anyi Rao
chooes running options -nonstatic/-static -word2vec/-rand
    
Much of the code is modified from
https://github.com/yoonkim/CNN_sentence
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
import scipy
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from itertools import izip
warnings.filterwarnings("ignore")

#different non-linearities
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)
    
def invert_dict_fast(d):
    return dict(izip(d.itervalues( ), d.iterkeys( )))

def Tag(fir_din1_index,fir_din2_index,m,sent_n):
    sent_w1 = '';sent_w2= ''
    for j in xrange (sent_n.size):
        ind = sent_n[0,j]
        if ind ==0:
            continue
        sent_w1_temp=word_idx_map_invert[ind]
        sent_w1= sent_w1+' '+sent_w1_temp

    for j in xrange (sent_n.size):
        ind = sent_n[0,j]
        if ind ==0:
            continue
        if j == fir_din1_index[m]:
            sent_w2_temp=word_idx_map_invert[fir_din2_index[m]]
            sent_w2= sent_w2+' '+sent_w2_temp
            continue
        sent_w2_temp=word_idx_map_invert[ind]
        sent_w2= sent_w2+' '+sent_w2_temp
    words1 = word_tokenize(sent_w1)
    tag_set1=nltk.pos_tag(words1,tagset="universal")
    words2 = word_tokenize(sent_w2)
    tag_set2=nltk.pos_tag(words2,tagset="universal")
    tag1=tag_set1[(fir_din1_index[m]-4)[0]][1] ## 4 is the paddding number
    tag2=tag_set2[(fir_din1_index[m]-4)[0]][1]
    return tag1,tag2,tag_set1,tag_set2

def vec2sent(sent_n):
    sent_w =' '
    for j in xrange (sent_n.size):
        ind = sent_n[0,j]
        if ind ==0:
            continue
        sent_w_temp=word_idx_map_invert[ind]
        sent_w= sent_w+' '+sent_w_temp  
    return sent_w

def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')
        
def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param      
    return updates 

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)
    
def safe_update(dict_to, dict_from):
    """
    re-make update dictionary for safe updating
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to
    
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
  
   
if __name__=="__main__":
    print "loading data..."
    n_option = 500 #set number of option words
    sim_threshold=0.2 #set similarity threshold
    THEME="sst2"
    sim_string=str(sim_threshold)
    x = cPickle.load(open(THEME+".p","rb"))
    revs, W, W2, word_idx_map, vocab ,max_l = x[0], x[1], x[2], x[3], x[4], x[5]
    print "data loaded!"
    mode= sys.argv[1]
    word_vectors = sys.argv[2]    
    if mode=="-nonstatic":
        print "model architecture: CNN-non-static"
        non_static=True
    elif mode=="-static":
        print "model architecture: CNN-static"
        non_static=False
    execfile("conv_net_classes.py")    
    if word_vectors=="-rand":
        print "using: random vectors"
        U = W2
    elif word_vectors=="-word2vec":
        print "using: word2vec vectors"
        U = W   
    i=0
    datasets = make_idx_data_cv(revs, word_idx_map, i, max_l,k=300, filter_h=5)
    stopWords = set(stopwords.words('english'))
    stopWords.add('n\'t')
    wordtags = nltk.ConditionalFreqDist((w.lower(), t)\
    for w, t in nltk.corpus.brown.tagged_words(tagset="universal"))
    word_idx_map_invert=invert_dict_fast(word_idx_map)
    pair=dict()
    
    img_w=300
    activations=[Iden]                
    non_static=True  
    lr_decay=0.95
    filter_hs=[3,4,5]
    conv_non_linear="relu"
    hidden_units=[100,2]
    shuffle_batch=True
    n_epochs=25 #here is changed
    sqr_norm_lim=9
    batch_size=50
    dropout_rate=[0.5]

    """
    Train a simple conv net
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec)
    filter_hs = filter window sizes    
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """    
    rng = np.random.RandomState(3435)
    img_h = len(datasets[0][0])-1  
    filter_w = img_w    
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
    parameters = [("image shape",img_h,img_w),("filter shape",filter_shapes), ("hidden_units",hidden_units),
                  ("dropout", dropout_rate), ("batch_size",batch_size),("non_static", non_static),
                    ("learn_decay",lr_decay), ("conv_non_linear", conv_non_linear), ("non_static", non_static)
                    ,("sqr_norm_lim",sqr_norm_lim),("shuffle_batch",shuffle_batch)]
    print parameters    
    
    #define model architecture
    index = T.lscalar()
    m=T.lscalar()
    x = T.matrix('x')   
    y = T.ivector('y')
    add_x = T.matrix()
    add_y = T.matrix()
    Words = theano.shared(value = U, name = "Words")
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))], allow_input_downcast=True)
    layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,x.shape[1],Words.shape[1]))                                  
    conv_layers = []
    layer1_inputs = []
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,image_shape=(batch_size, 1, img_h, img_w),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
    layer1_input = T.concatenate(layer1_inputs,1)
    hidden_units[0] = feature_maps*len(filter_hs)    
    classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)
    
    #define parameters of the model and update functions using adadelta
    params = classifier.params     
    for conv_layer in conv_layers:
        params += conv_layer.params
    if non_static:
        #if word vectors are allowed to change, add them as model parameters
        params += [Words]
    cost = classifier.negative_log_likelihood(y) 
    dropout_cost = classifier.dropout_negative_log_likelihood(y)           
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)
    
    #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate 
    #extra data (at random)
    np.random.seed(3435)
    if datasets[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[0].shape[0] % batch_size
        train_set = np.random.permutation(datasets[0])   
        extra_data = train_set[:extra_data_num]
        new_data=np.append(datasets[0],extra_data,axis=0)
    else:
        new_data = datasets[0]
    new_data = np.random.permutation(new_data)
    n_batches = new_data.shape[0]/batch_size
    n_train_batches = int(np.round(n_batches*0.9))
    #divide train set into train/val sets 
    test_set_x = datasets[1][:,:img_h] 
    test_set_y = np.asarray(datasets[1][:,-1],"int32")
    train_set = new_data[:n_train_batches*batch_size,:]
    original_train_set = train_set
    
    val_set = new_data[n_train_batches*batch_size:,:]     
    
    train_set_x = theano.shared(np.asarray(train_set[:,:img_h],
                                               dtype=theano.config.floatX),
                                 borrow=True)
    train_set_y_before_cast = theano.shared(np.asarray(train_set[:,-1],
                                               dtype=theano.config.floatX),
                                 borrow=True)
    train_set_y = T.cast(train_set_y_before_cast, 'int32')
    
#    train_set_x, train_set_y = shared_dataset((train_set[:,:img_h],train_set[:,-1]))    
    val_set_x, val_set_y = shared_dataset((val_set[:,:img_h],val_set[:,-1]))
    n_val_batches = n_batches - n_train_batches
    val_model = theano.function([index], classifier.errors(y),
         givens={
            x: val_set_x[index * batch_size: (index + 1) * batch_size],
             y: val_set_y[index * batch_size: (index + 1) * batch_size]},
                                allow_input_downcast=True)            
    #compile theano functions to get train/val/test errors
    test_model = theano.function([index], classifier.errors(y),
             givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                 y: train_set_y[index * batch_size: (index + 1) * batch_size]},
                                 allow_input_downcast=True)               
    train_model = theano.function([index], cost, updates=grad_updates,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
              y: train_set_y[index*batch_size:(index+1)*batch_size]},
                                  allow_input_downcast = True)     
    test_pred_layers = []
    test_size = test_set_x.shape[0]
    test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))
    for conv_layer in conv_layers:
        test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))
    test_layer1_input = T.concatenate(test_pred_layers, 1)
    test_y_pred = classifier.predict(test_layer1_input)
    test_error = T.mean(T.neq(test_y_pred, y))
    test_model_all = theano.function([x,y], test_error, allow_input_downcast = True)       
    test_predict = theano.function([x], test_y_pred, allow_input_downcast=True)
    #test_probs = theano.function([x], test_y_pred_p_reduce, allow_input_downcast=True)
    
    #gradient-based update
    dinput=T.grad(dropout_cost,layer0_input)
    din_onehot=dinput.dot(W.transpose())
    all_din1_indextemp=T.max(din_onehot,axis=3)
    all_din1_index=T.argsort(all_din1_indextemp,axis=2)
    Fall_din1_index=theano.function([index], all_din1_index,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
              y: train_set_y[index*batch_size:(index+1)*batch_size]},
                                  allow_input_downcast = True)
    
    all_din2_index=T.argsort(din_onehot,axis=3)
    Fall_din2_index=theano.function([index], all_din2_index,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
              y: train_set_y[index*batch_size:(index+1)*batch_size]},
                                  allow_input_downcast = True)
  
    train_set_x_update=(train_set_x, T.concatenate([train_set_x,add_x[:,:img_h]],axis=0))
    Fadd_x= theano.function([add_x],updates=[train_set_x_update],allow_input_downcast=True)   
    train_set_y_update=(train_set_y_before_cast, T.concatenate([train_set_y_before_cast,add_y[:,-1]],axis=0))
    Fadd_y= theano.function([add_y],updates=[train_set_y_update],allow_input_downcast=True)

    Train_set_add = []
    U1=U
    #U1=Words.eval() # uncomment this line to finetune wordembeddings
    #start training over mini-batches
    print '... training similarity threshold: '+sim_string+' n_option: '+ str(n_option)
    epoch = 0
    best_val_perf = 0
    val_perf = 0
    test_perf = 0       
    cost_epoch = 0   
    while (epoch < n_epochs):
        count = 0
        start_time = time.time()
        epoch = epoch + 1
        if epoch == 1:
            pass
        else:
            train_set = original_train_set
            train_set = np.row_stack([original_train_set,Train_set_add])
            n_train_batches = train_set.shape[0]/50    
            
            train_set_x = theano.shared(np.asarray(train_set[:,:img_h],
                                               dtype=theano.config.floatX),
                                 borrow=True)
            train_set_y_before_cast = theano.shared(np.asarray(train_set[:,-1],
                                                   dtype=theano.config.floatX),
                                     borrow=True)
            train_set_y = T.cast(train_set_y_before_cast, 'int32')
            Fadd_x(Train_set_add)
            Fadd_y(Train_set_add)        
        train_set_xout= train_set[:,:img_h]        
        Train_set_add = np.zeros(img_h+1).reshape(1,img_h+1)
        
        if shuffle_batch:
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
        else:
            for minibatch_index in xrange(n_train_batches):
                cost_epoch = train_model(minibatch_index)  
                set_zero(zero_vec)
        train_losses = [test_model(i) for i in xrange(n_train_batches)]
        train_perf = 1 - np.mean(train_losses)
        val_losses = [val_model(i) for i in xrange(n_val_batches)]
        val_perf = 1- np.mean(val_losses) 

        print "...replacing"
        for i in xrange (n_train_batches):
            index =i
            all_din1_index=Fall_din1_index(i)
            all_din2_index=Fall_din2_index(i)
            fir_din1_index=all_din1_index[:,:,-1]
            for m in xrange (batch_size):
                w_2be_rep=int(train_set_xout[index*batch_size:(index+1)*batch_size][m:m+1,fir_din1_index[m]])              
                if w_2be_rep == 0:
                    continue 
                v1=U1[w_2be_rep,].flatten()                
                w1=word_idx_map_invert[w_2be_rep]
                if w1 in stopWords:
                    continue
                fir_din2_indextemp=np.empty((0,U.shape[0]), int)
                for i_batch in xrange(batch_size):
                    tempbatch=all_din2_index[i_batch,0,fir_din1_index[i_batch],:]
                    fir_din2_indextemp=np.row_stack((fir_din2_indextemp,tempbatch))
                for i_vocab in xrange(n_option):
                    if i_vocab == n_option-1:
                        count = count + 1                      
                    fir_din2_index=fir_din2_indextemp[:,-(i_vocab+1)]
                    v2=U1[fir_din2_index[m],].flatten()
                    w2=word_idx_map_invert[fir_din2_index[m]]
                    if pair.get(w1+w2) == -1:
                        continue
                    set_prefix="un"
                    w_prefix_1=set_prefix+w1
                    if w_prefix_1 == w2:
                        continue
                    w_prefix_2=set_prefix+w2
                    if w1 == w_prefix_2:
                        continue
                    sim=scipy.spatial.distance.cosine(v1, v2)
                    if sim > sim_threshold:
                        continue
                    sent_n1=train_set_xout[index*batch_size:(index+1)*batch_size][m:m+1]
                    tag1,tag2,tag_set1,tag_set2=Tag(fir_din1_index,fir_din2_index,m,sent_n1)
                    if tag1 != tag2:
                        pair.update({w1+w2:-1})
                        continue                    
                    train_set_add=train_set[index*batch_size:(index+1)*batch_size][m:m+1]
                    train_set_add[:,fir_din1_index[m]]=fir_din2_index[m]
                    Train_set_add = np.row_stack([Train_set_add,train_set_add])
                    if w1 != 0:
                        break                               
        print('epoch: %i, training time: %.2f secs, train perf: %.2f %%, val perf: %.2f %%' % (epoch, time.time()-start_time, train_perf * 100., val_perf*100.))
        print 'count number of not changing' + str(epoch) + ' ' +str(count)
        if val_perf >= best_val_perf:
            best_val_perf = val_perf
            test_loss = test_model_all(test_set_x,test_set_y)        
            test_perf = 1- test_loss
            print '...save the model'
            with open('save/'+THEME+'_model.pickle', 'wb') as file:
                model = params
                cPickle.dump(model, file)
    print "perf: " + str(test_perf)
