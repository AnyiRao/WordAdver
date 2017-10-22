"""
copy
For Word Replacement
Modified by Anyi Rao
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf
Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
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
from nltk.corpus import wordnet as wn

warnings.filterwarnings("ignore")   
THEANO_FLAGS="floatX=float32"

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
       
def loaddata():
    with open('save/Dinput_1.pickle', 'rb') as file:
            tempx=cPickle.load(file)
            Dinput_1=tempx #.astype(int)
#            print(Dinput_1[0][:10])
    
    with open('save/Dinput_2.pickle', 'rb') as file:
            Dinput_2=cPickle.load(file)
    
    with open('save/Train_set_xtest.pickle', 'rb') as file:
            tempx=cPickle.load(file)
            Train_set_x=tempx #.astype(int)
#            print(Train_set_x[0][:10])
    
    with open('save/Train_set_ytest.pickle', 'rb') as file:
            Train_set_y=cPickle.load(file)
#            print(Train_set_y[:10])
    
    ### replacement
    batch_size = 50
    for i in xrange (len(Dinput_1)):
        for j in xrange (batch_size):
            (Train_set_x[i])[j,Dinput_1[i][j]]=Dinput_2[i][j]
           
    #### 
    train_set_x=[]
    train_set_y=[]
    for i in xrange (len(Dinput_1)):
        if i == 0:
            train_set_x=Train_set_x[i]
            continue
        train_set_x=np.row_stack((train_set_x,Train_set_x[i]))
        
    for i in xrange (len(Dinput_1)):
        if i == 0:
            train_set_y=Train_set_y[i]
            continue
        train_set_y=np.row_stack((train_set_y,Train_set_y[i]))
    train_set_y=train_set_y.flatten()
    
    return train_set_x,train_set_y


def create_batches(datasets,batch_size):
        np.random.seed(3435)
        if datasets[0].shape[0] % batch_size > 0:
            extra_data_num = batch_size - datasets[0].shape[0] % batch_size
            train_set = np.random.permutation(datasets[0])   
            extra_data = train_set[:extra_data_num]
            new_data=np.append(datasets[0],extra_data,axis=0)
        else:
            new_data = datasets[0]
        new_data = np.random.permutation(new_data)
        return new_data


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
    
    """
    two question
    1. add hot vector into parameters. Use look up table index
    2. get argmax(grad(loss,hot vector))
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []# gradient of parameters
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
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim)) #transfer col_norms to between 0-3
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
    print "loading data...",
    x = cPickle.load(open("mr.p","rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
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
   
    results = []
    r = range(0,10)    
    i =0        
    datasets = make_idx_data_cv(revs, word_idx_map, i, max_l=52,k=300, filter_h=5)# max_l changed from 56
# =============================================================================
# 
# =============================================================================

    img_w=300
    activations=[Iden]                
    non_static=True  
    lr_decay=0.95
    filter_hs=[3,4,5]
    conv_non_linear="relu"
    hidden_units=[100,2]
    shuffle_batch=True
    n_epochs=1 #here is changed
    sqr_norm_lim=9
    batch_size=50
    dropout_rate=[0.5]
    
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
    x = T.matrix('x')   
    y = T.ivector('y')
    Words = theano.shared(value = U, name = "Words")
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))], allow_input_downcast=True)
    layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,x.shape[1],Words.shape[1]))# (50,1,64,300)
    conv_layers = []
    layer1_inputs = []
    for i in xrange(len(filter_hs)):#use 3 different filter and pooling size
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,image_shape=(batch_size, 1, img_h, img_w),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
    layer1_input = T.concatenate(layer1_inputs,1)
    #####
#    layer_printed=theano.printing.Print('layer')(layer1_input)
#    f0=theano.function([x],layer_printed)
    #####
    
    
    hidden_units[0] = feature_maps*len(filter_hs)    
    classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)
    
    #define parameters of the model and update functions using adadelta
    params = classifier.params  #[W, b, W_conv, b_conv, W_conv, b_conv, W_conv, b_conv, Words]   
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
    
    new_data = create_batches(datasets,batch_size)
    
    n_batches = new_data.shape[0]/batch_size
    n_train_batches = int(np.round(n_batches*0.9))
    #divide train set into train/val sets 
    test_set_x = datasets[1][:,:img_h] 
    test_set_y = np.asarray(datasets[1][:,-1],"int32")
    train_set = new_data[:n_train_batches*batch_size,:]
    val_set = new_data[n_train_batches*batch_size:,:]     
    train_set_x, train_set_y = shared_dataset((train_set[:,:img_h],train_set[:,-1]))
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
    
    ######
    dinput=T.grad(dropout_cost,layer0_input)
    dinput_1hot=dinput.dot(W.transpose())
    
    dinput1temp=T.max(dinput_1hot,axis=3)
    dinput1=T.argmax(dinput1temp,axis=2)
    dinput2temp=T.max(dinput_1hot,axis=2)
    dinput2=T.argmax(dinput2temp,axis=2)

    dinput_1printed = theano.printing.Print('dinput1')(dinput1)
    dinput_2printed = theano.printing.Print('dinput2')(dinput2)
    
    f1_printed=theano.function([index], dinput_1printed,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
              y: train_set_y[index*batch_size:(index+1)*batch_size]},
                                  allow_input_downcast = True)
    f2_printed=theano.function([index], dinput_2printed,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
              y: train_set_y[index*batch_size:(index+1)*batch_size]},
                                  allow_input_downcast = True)
    
    f1=theano.function([index], dinput1,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
              y: train_set_y[index*batch_size:(index+1)*batch_size]},
                                  allow_input_downcast = True)
    f2=theano.function([index], dinput2,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
              y: train_set_y[index*batch_size:(index+1)*batch_size]},
                                  allow_input_downcast = True)
    
    # sort
    din1temp=T.sort(dinput_1hot,axis=3)
    din1temp_printed = theano.printing.Print('din1')(din1temp)
    din_sort=theano.function([index], din1temp_printed,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
              y: train_set_y[index*batch_size:(index+1)*batch_size]},
                                  allow_input_downcast = True)
    
    din1_indextemp=T.argsort(dinput_1hot,axis=3)
    din1_indextemp_printed = theano.printing.Print('din1_index')(din1_indextemp)
    din_index_sort=theano.function([index], din1_indextemp_printed,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
              y: train_set_y[index*batch_size:(index+1)*batch_size]},
                                  allow_input_downcast = True)
    ######
    
    #start training over mini-batches
    print '... training'
    epoch = 0
    best_val_perf = 0
    val_perf = 0
    test_perf = 0       
    cost_epoch = 0    
    while (epoch < n_epochs):
        start_time = time.time()
        epoch = epoch + 1
        if shuffle_batch:
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
        else:
            for minibatch_index in xrange(n_train_batches):
                cost_epoch = train_model(minibatch_index)  
                set_zero(zero_vec)
        train_losses = [test_model(i) for i in xrange(n_train_batches)]
        print 'train '+ str(epoch) +' '+str(time.time()-start_time) +'secs'
        
#        din_value=din_sort(0)
#        din_value_reshape= din.reshape(50,60,18766)  
#        din_value_raw=din_reshape[0,:,:]
#        
#        din_index=din_index_sort(0)
#        din_index_reshape=din_index.reshape(50,60,18766)
#        din_index_raw=din_index_reshape[0,:,:]
        Din_value=[]
        Din_index=[]
        for i in xrange (n_train_batches):
            din_value=din_sort(i)
            din_value_reshape= din.reshape(50,60,18766)  
            din_index=din_index_sort(i)
            din_index_reshape=din_index.reshape(50,60,18766)
            for j in xrange (batch_size): 
                T.set_subtensor((train_set_x[index*batch_size:(index+1)*batch_size])[j:j+1,dinput_1[j]],dinput_2[j:j+1])
            Din_value.append(din_value)
            Din_index.append(din_index)
            
            
        print 'epoch '+ str(epoch)+ ' replacing time: ' + str(time.time()-start_time) +'secs'
  
         
        
#        train_set_xout=train_set[:,:img_h]
#        
#        n_inbatch=3 # which one in a batch
#        m=n_inbatch
#        wtemp=train_set_xout[m,d1_printed[m]]
#        w_temp=word_idx_map.keys()[word_idx_map.values().index(wtemp)]   
#        
#        s=wn.synsets(w_temp)
#        print s
#        s=s[2]
#        print s
#        w_rep = s.lemmas()[0].name()
#        w_rep_index=word_idx_map.get(w_rep)
#        print s.lemmas()[0].name()    # s.name() #, s.definition() #, s.examples()    
#        print w_rep_index
#            
##        for s in wn.synsets(w_temp):
##            print s
##            w_rep = s.lemmas()[0].name()
##            w_rep_index=word_idx_map.get(w_rep)
##            print s.lemmas()[0].name()    # s.name() #, s.definition() #, s.examples()    
##            print w_rep_index
#        
#        
#        index=0
#        j=n_inbatch
#        index_2be_rep=d2_printed[n_inbatch:n_inbatch+1]
#        value_2be_rep=0
#        if w_rep_index != None:
#            for p in xrange (18766):
#                if din_index_raw[d1_printed[n_inbatch],-(p+1)] == w_rep_index:
#                    print ("find out synnonym...."+str(p))
#                    index_2be_rep=w_rep_index
#                    value_2be_rep=din_raw[d1_printed[n_inbatch],-(p+1)]
#        else:
#            print "here is none, this sentence not changed"            
#        
#        (train_set_xout[index*batch_size:(index+1)*batch_size])[n_inbatch:n_inbatch+1,d1_printed[n_inbatch]]=index_2be_rep
#        
#        
        
   ###############        
        train_perf = 1 - np.mean(train_losses)
        val_losses = [val_model(i) for i in xrange(n_val_batches)]
        val_perf = 1- np.mean(val_losses)                        
        print('epoch: %i, whole training time: %.2f secs, train perf: %.2f %%, val perf: %.2f %%' % (epoch, time.time()-start_time, train_perf * 100., val_perf*100.))
        if val_perf >= best_val_perf:
            best_val_perf = val_perf
            test_loss = test_model_all(test_set_x,test_set_y)        
            test_perf = 1- test_loss
    
    
#    train_set_x = datasets[0][:,:img_h] 
#    train_set_y = np.asarray(datasets[0][:,-1],"int32")
#    train_set_x_adver = train_set_x[1000:2046]
#    train_set_y_adver = train_set_y[1000:2046]
    
    set_x_adver,set_y_adver =loaddata()    
    train_set_x_adver = set_x_adver[000:1046]
    train_set_y_adver = set_y_adver[000:1046]
              
    adver_loss = test_model_all(train_set_x_adver,train_set_y_adver)
    adver_perf = 1- adver_loss   
    print "adver_accuracy:" + str(adver_perf) 
     
    

# =============================================================================
#     
# =============================================================================
    perf = test_perf
    print "cv: " + str(i) + ", perf: " + str(perf)
    results.append(perf)  
    print str(np.mean(results))

    
#    with open('save/I.pickle', 'wb') as file:
#        model = I
#        cPickle.dump(model, file)
##        print(model[0][:10])
#    with open('save/M.pickle', 'wb') as file:
#        model = M
#        cPickle.dump(model, file)
##        print(model[0][:10])

    
    
    