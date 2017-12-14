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
from my_func import file_cover,file_save_string,file_save_int,file_save_nparray,file_save_nparray2d
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
    tag1=tag_set1[(fir_din1_index[m]-4)[0]][1] ## paddding number
    tag2=tag_set2[(fir_din1_index[m]-4)[0]][1]
    return tag1,tag2,tag_set1,tag_set2

def load_turker(): # load examples after Mechanical Turkers annotating.
    filename="examples_turker.txt"
    lines = []
    x = []
    with open(filename, 'r') as f:
        lines.extend(f.readlines())
    for i, line in enumerate(lines):
        x.append(line.rstrip())

    adver_x= np.zeros((len(x)/2,61))
    adver_y=np.array([100])
    for i in xrange(len(x)):
        if i%2 == 1: # 0 means removing original keep adver
            continue
        X=x[i].split( )
        if X[-1]== 'pos':
            adver_y=np.row_stack([adver_y,np.array([1])])
        else:
            adver_y=np.row_stack([adver_y,np.array([0])])

        for j,p in zip(X, xrange(len(X))):
            #            print j,p
            if j =="pos" or j=="neg":
                continue
            #            print word_idx_map[j]
            adver_x[i/2,p+4]=(word_idx_map[j])
adver_y=adver_y[1:]
    adver_y=adver_y.reshape([len(x)/2,])
    return adver_x,adver_y

def loaddata():
    I=np.loadtxt('save/'+SAVETHEME+'I.txt').astype('int32')
    M=np.loadtxt('save/'+SAVETHEME+'M.txt').astype('int32')
    Din1_index=np.loadtxt('save/'+SAVETHEME+'Fir_din1_index.txt').astype('int32')
    Din2_index=np.loadtxt('save/'+SAVETHEME+'Fir_din2_index.txt').astype('int32')
    Train_set_x=np.loadtxt('save/'+SAVETHEME+'Train_set_x.txt').astype('int32')
    Train_set_y=np.loadtxt('save/'+SAVETHEME+'Train_set_y.txt').astype('int32')
    
    train_set_x_ori=np.copy(np.reshape(Train_set_x,(len(Train_set_x)/img_h,img_h)))
    train_set_x=np.copy(np.reshape(Train_set_x,(len(Train_set_x)/img_h,img_h)))
    
    Sec_I=np.loadtxt('save/'+SAVETHEME+'secI.txt').astype('int32')
    Sec_M=np.loadtxt('save/'+SAVETHEME+'secM.txt').astype('int32')
    Sec_din1_index=np.loadtxt('save/'+SAVETHEME+'Sec_din1_index.txt').astype('int32')
    Sec_din2_index=np.loadtxt('save/'+SAVETHEME+'Sec_din2_index.txt').astype('int32')
    
    ### replacement
    
    for order in  xrange(len(I)):
        din1_index=Din1_index[order*50+M[order]]
        din2_index=Din2_index[order*50+M[order]]    
        train_set_x[order:order+1,din1_index]=din2_index
    
    Fir_IM=I*50+M
    Sec_IM=Sec_I*50+Sec_M
    for sec_order in xrange(len(Sec_I)):
        print sec_order
        sec_pos=np.where(Fir_IM==Sec_IM[sec_order])[0]
        print sec_pos
        sec_din1_index=Sec_din1_index[sec_order*50+Sec_M[sec_order]]
        sec_din2_index=Sec_din2_index[sec_order*50+Sec_M[sec_order]]
        train_set_x[sec_pos,sec_din1_index]=sec_din2_index
            
    return train_set_x_ori,train_set_x,Train_set_y

def save(i,m):  
    file_save_int(i,'save/'+SAVETHEME+"I.txt")  
    file_save_int(m,'save/'+SAVETHEME+"M.txt") 
    file_save_nparray(fir_din1_index,'save/'+SAVETHEME+"Fir_din1_index.txt") 
    file_save_nparray(fir_din2_index,'save/'+SAVETHEME+"Fir_din2_index.txt") 
    train_set_xsingle=Foutput_train_set_xsingle(i,m) 
    train_set_ysingle=Foutput_train_set_ysingle(i,m)
    file_save_nparray2d(train_set_xsingle,'save/'+SAVETHEME+"Train_set_x.txt") 
    file_save_nparray(train_set_ysingle,'save/'+SAVETHEME+"Train_set_y.txt") 
    return 

def save2(i,m):  
    file_save_int(i,'save/'+SAVETHEME+"secI.txt")  
    file_save_int(m,'save/'+SAVETHEME+"secM.txt") 
    file_save_nparray(sec_din1_index,'save/'+SAVETHEME+"Sec_din1_index.txt") 
    file_save_nparray(sec_din2_index,'save/'+SAVETHEME+"Sec_din2_index.txt") 
#    train_set_xsingle=Foutput_train_set_xsingle(i,m) 
#    train_set_ysingle=Foutput_train_set_ysingle(i,m)
#    file_save_nparray2d(train_set_xsingle,'save/'+SAVETHEME+"Train_set_x.txt") 
#    file_save_nparray(train_set_ysingle,'save/'+SAVETHEME+"Train_set_y.txt") 
    return 

def cover():
    file_cover('save/'+SAVETHEME+"_examples.txt")
    file_cover('save/'+SAVETHEME+"I.txt")
    file_cover('save/'+SAVETHEME+"M.txt") 
    file_cover('save/'+SAVETHEME+"Fir_din1_index.txt") 
    file_cover('save/'+SAVETHEME+"Fir_din2_index.txt") 
    file_cover('save/'+SAVETHEME+"Train_set_x.txt") 
    file_cover('save/'+SAVETHEME+"Train_set_y.txt") 
    file_cover('save/'+SAVETHEME+"secI.txt")
    file_cover('save/'+SAVETHEME+"secM.txt") 
    file_cover('save/'+SAVETHEME+"Sec_din1_index.txt") 
    file_cover('save/'+SAVETHEME+"Sec_din2_index.txt") 

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

def vec2sent(sent_n):
    sent_w =' '
    for j in xrange (sent_n.size):
        ind = sent_n[0,j]
        if ind ==0:
            continue
        sent_w_temp=word_idx_map_invert[ind]
        sent_w= sent_w+' '+sent_w_temp  
    return sent_w 
def vec2sent_onedim(sent_n):
    sent_w =' '
    for j in xrange (sent_n.size):
        ind = sent_n[j]
        if ind ==0:
            continue
        sent_w_temp=word_idx_map_invert[ind]
        sent_w= sent_w+' '+sent_w_temp  
    return sent_w 
      
if __name__=="__main__":
    print "loading data..."
    n_option = 500
    sim_threshold=0.4
    sim_string=str(sim_threshold)
    THEME="sst2" # for dataset
    MODELTHEME=THEME #+"two_same" # for model 
    SAVETHEME=THEME+ "_"+sim_string+"_two" # for adver test set
    CHANGE_FLAG=1
    
    x = cPickle.load(open(THEME+".p","rb"))
    revs, W, W2, word_idx_map, vocab, max_l = x[0], x[1], x[2], x[3], x[4],x[5]
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
    pair = dict()
        
    img_w=300
    activations=[Iden]                
    non_static=True  
    lr_decay=0.95
    filter_hs=[3,4,5]
    conv_non_linear="relu"
    hidden_units=[100,2]
    shuffle_batch=True
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
    m=T.lscalar()
    x = T.matrix('x')   
    y = T.ivector('y')
    din1 = T.ivector()
    din2 = T.scalar()
    Words = theano.shared(value = U, name = "Words")
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
    
    #load model parameters
    with open('save/'+MODELTHEME+'_model.pickle', 'rb') as f:
        tmp = cPickle.load(f)
    
    for i in range(len(classifier.params)):
        classifier.params[i].set_value(tmp[i].get_value())
    
    del tmp
    #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate 
    #extra data (at random)
    np.random.seed(3435)
    if datasets[1].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[1].shape[0] % batch_size
        test_set = np.random.permutation(datasets[1])   
        extra_data = test_set[:extra_data_num]
        new_data=np.append(datasets[1],extra_data,axis=0)
    else:
        new_data = datasets[1]
    new_data = np.random.permutation(new_data)
    n_batches = new_data.shape[0]/batch_size
    #divide train set into train/val sets
    
    n_test_length=datasets[1].shape[0]
    test_set =new_data
    test_set_x, test_set_y = shared_dataset((test_set[:,:img_h],test_set[:,-1]))
    test_pred_layers = []
    test_size = datasets[1].shape[0]            # modified linetest_set_x.shape[0]
    test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))
    for conv_layer in conv_layers:
        test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))
    test_layer1_input = T.concatenate(test_pred_layers, 1)
    test_y_pred = classifier.predict(test_layer1_input)
    test_y_pred_conf = classifier.predict_p(test_layer1_input)
    test_error = T.mean(T.neq(test_y_pred, y))
    test_model_all = theano.function([x,y], test_error, allow_input_downcast = True)   
    #test_probs = theano.function([x], test_y_pred_p_reduce, allow_input_downcast=True)
    test_predict = theano.function([x], test_y_pred, allow_input_downcast=True)
    test_predict_conf = theano.function([x], test_y_pred_conf, allow_input_downcast=True)
    #gradient-based update
    dinput=T.grad(dropout_cost,layer0_input)
    din_onehot=dinput.dot(W.transpose())
    all_din1_indextemp=T.max(din_onehot,axis=3)
    all_din1_index=T.argsort(all_din1_indextemp,axis=2)
    Fall_din1_index=theano.function([index], all_din1_index,
          givens={
            x: test_set_x[index*batch_size:(index+1)*batch_size],
              y: test_set_y[index*batch_size:(index+1)*batch_size]},
                                  allow_input_downcast = True)
    
    all_din2_index=T.argsort(din_onehot,axis=3)
    Fall_din2_index=theano.function([index], all_din2_index,
          givens={
            x: test_set_x[index*batch_size:(index+1)*batch_size],
              y: test_set_y[index*batch_size:(index+1)*batch_size]},
                                  allow_input_downcast = True)

    test_set_x_update = (test_set_x, T.set_subtensor(test_set_x[index*50+m:index*50+m+1,din1], din2))
    Frep = theano.function([index,m,din1,din2], updates=[test_set_x_update],allow_input_downcast=True)        
    
    train_set_xbatch=test_set_x[index*batch_size:(index+1)*batch_size]
    train_set_ybatch=test_set_y[index*batch_size:(index+1)*batch_size]   
    train_set_xsingle=train_set_xbatch[m:m+1]
    train_set_ysingle=train_set_ybatch[m:m+1]   
    Foutput_train_set_x=theano.function([index],train_set_xbatch)
    Foutput_train_set_y=theano.function([index],train_set_ybatch)   
    Foutput_train_set_xsingle=theano.function([index,m],train_set_xsingle)
    Foutput_train_set_ysingle=theano.function([index,m],train_set_ysingle)
    #start training over mini-batches
    print '... generating '+sim_string+' n_option: '+ str(n_option)
    epoch = 0
    best_val_perf = 0
    val_perf = 0
    test_perf = 0       
    cost_epoch = 0    
    
    if CHANGE_FLAG: # change here
        cover()
        start_time = time.time()      
        train_set_xout= test_set[:,:img_h]
        pair=dict()
        Train_set_x_adversingle=[];Train_set_y_adversingle=[];Fir_din1_index=[];Fir_din2_index=[]
        I=[];M=[];Sim=[];Sec_Train_set_x_adversingle=[];Sec_Train_set_y_adversingle=[]
        Sec_din1_index=[];Sec_din2_index=[];Sec_I=[];Sec_M=[];Sec_Sim=[]
        for i in xrange (n_batches):
            print str(i)+ ' ' + str(time.time()-start_time)+ 'sec'
            index =i
            all_din1_index=Fall_din1_index(i)
            all_din2_index=Fall_din2_index(i)
            fir_din1_index=all_din1_index[:,:,-1]
            for m in xrange (batch_size):
                w_2be_rep=train_set_xout[index*batch_size:(index+1)*batch_size][m:m+1,fir_din1_index[m]]
                if w_2be_rep == 0:
                    continue 
                
                v1=U[w_2be_rep,].flatten()                
                w1=word_idx_map_invert[w_2be_rep[0][0]]
                if w1 in stopWords:
                    continue
                
                fir_din2_indextemp=np.empty((0,U.shape[0]), int)
                for i_batch in xrange(batch_size):
                    tempbatch=all_din2_index[i_batch,0,fir_din1_index[i_batch],:]
                    fir_din2_indextemp=np.row_stack((fir_din2_indextemp,tempbatch))
                    
                for i_vocab in xrange(n_option):                      
                    fir_din2_index=fir_din2_indextemp[:,-(i_vocab+1)]
                    v2=U[fir_din2_index[m],].flatten()
                    w2=word_idx_map_invert[fir_din2_index[m]]
                    
                    if w1 == w2:
                        continue
                    if pair.get(w1+w2) == -1:
                            continue                    
                    prefix1="un"
                    w_prefix_1=prefix1+w1
                    if w_prefix_1 == w2:
                        continue
                    w_prefix_2=prefix1+w2
                    if w1 == w_prefix_2:
                        continue
                    affix1="s"
                    w_prefix_1=w1+affix1
                    if w_prefix_1 == w2:
                        continue
                    w_prefix_2=w2+affix1
                    if w1 == w_prefix_2:
                        continue
                    affix1="ing"
                    w_prefix_1=w1+affix1
                    if w_prefix_1 == w2:
                        continue
                    w_prefix_2=w2+affix1
                    if w1 == w_prefix_2:
                        continue
                    affix1="ed"
                    w_prefix_1=w1+affix1
                    if w_prefix_1 == w2:
                        continue
                    w_prefix_2=w2+affix1
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
                    
                    print w1,w2
                    test_set_original= np.copy(test_set[index*batch_size:(index+1)*batch_size][m:m+1,])
                    print "i_vocab"+str(i_vocab)
                    print vec2sent(test_set_original)
                    file_save_string(vec2sent(test_set_original),'save/'+SAVETHEME+'_examples_raw.txt')
                    file_save_string('\n','save/'+SAVETHEME+'_examples_raw.txt')
                    test_set[index*batch_size:(index+1)*batch_size][m:m+1,fir_din1_index[m]]=fir_din2_index[m]
                    test_set_changed = np.copy(test_set[index*batch_size:(index+1)*batch_size][m:m+1,])
                    print vec2sent(test_set_changed)
                    file_save_string(vec2sent(test_set_changed),'save/'+SAVETHEME+'_examples_raw.txt')
                    file_save_string('\n','save/'+SAVETHEME+'_examples_raw.txt')
                    if np.array(sent_n1.nonzero()).size > 10:
                        sec_din1_index=all_din1_index[:,:,-2]
                        sec_din2_indextemp=np.empty((0,U.shape[0]), int)
                        for i_batch in xrange(batch_size):
                                tempbatch=all_din2_index[i_batch,0,sec_din1_index[i_batch],:]
                                sec_din2_indextemp=np.row_stack((sec_din2_indextemp,tempbatch))
                        sec_w_2be_rep=train_set_xout[index*batch_size:(index+1)*batch_size][m:m+1,sec_din1_index[m]]
                        if sec_w_2be_rep == 0:
                            break                                 
                        sec_v1=U[sec_w_2be_rep,].flatten()                
                        sec_w1=word_idx_map_invert[sec_w_2be_rep[0][0]]
                        if sec_w1 in stopWords:
                            break
                            
                        for i_sec_vocab in xrange(n_option): 
                            sec_din2_index=sec_din2_indextemp[:,-(i_sec_vocab+1)]
                            sec_v2=U[sec_din2_index[m],].flatten()
                            sec_w2=word_idx_map_invert[sec_din2_index[m]]
                            if pair.get(sec_w1+sec_w2) == -1:
                                continue
                            if sec_w1==sec_w2:
                                continue
                            
                            sec_prefix1="un"
                            w_prefix_1=sec_prefix1+sec_w1
                            if w_prefix_1 == sec_w2:
                                continue
                            w_prefix_2=sec_prefix1+sec_w2
                            if sec_w1 == w_prefix_2:
                                continue
                            
                            sec_affix1="s"
                            sec_w_prefix_1=sec_w1+sec_affix1
                            if sec_w_prefix_1 == sec_w2:
                                continue
                            sec_w_prefix_2=sec_w2+sec_affix1
                            if sec_w1 == sec_w_prefix_2:
                                continue          
                            sec_affix1="ing"
                            sec_w_prefix_1=sec_w1+sec_affix1
                            if sec_w_prefix_1 == sec_w2:
                                continue
                            sec_w_prefix_2=sec_w2+sec_affix1
                            if sec_w1 == sec_w_prefix_2:
                                continue   
                            sec_affix1="ed"
                            sec_w_prefix_1=sec_w1+sec_affix1
                            if sec_w_prefix_1 == sec_w2:
                                continue
                            sec_w_prefix_2=sec_w2+sec_affix1
                            if sec_w1 == sec_w_prefix_2:
                                continue   
                            
                            sec_tag1=tag_set1[(sec_din1_index[m]-4)[0]][1]
                            sec_tag2=tag_set2[(sec_din1_index[m]-4)[0]][1]
                            if sec_tag1 != sec_tag2:
                                pair.update({sec_w1+sec_w2:-1})
                                continue
                            sec_sim=scipy.spatial.distance.cosine(sec_v1, sec_v2)
                            if sec_sim > sim_threshold:
                                continue                        
                            test_set[index*batch_size:(index+1)*batch_size][m:m+1,sec_din1_index[m]]=sec_din2_index[m]
                            test_set_changed = np.copy(test_set[index*batch_size:(index+1)*batch_size][m:m+1,])
                            print vec2sent(test_set_changed)
                            file_save_string(vec2sent(test_set_changed),'save/'+SAVETHEME+'_examples_raw.txt')
                            file_save_string('\n','save/'+SAVETHEME+'_examples_raw.txt')
                            if sec_w1 != 0:
                                save2(i,m)
                                break      
                    
                    if w1 != 0:                        
                        save(i,m)
                        break

    print "...predict..."
    set_x_ori,set_x,set_y=loaddata()
    if len(set_x) < datasets[1].shape[0]:
         n_tile=datasets[1].shape[0] / len(set_x)
         ori_x=np.tile(set_x_ori,(n_tile+1,1))
         adver_x=np.tile(set_x,(n_tile+1,1))
         adver_y=np.tile(set_y,(n_tile+1,))    
    else:     
        if len(set_x) % datasets[1].shape[0] > 0:
             extra_data_num = datasets[1].shape[0]- len(set_x) %datasets[1].shape[0]
             extra_data_x_ori=set_x_ori[:extra_data_num]
             extra_data_x = set_x[:extra_data_num]
             extra_data_y = set_y[:extra_data_num]
             ori_x=np.append(set_x_ori,extra_data_x,axis=0)
             adver_x=np.append(set_x,extra_data_x,axis=0)
             adver_y=np.append(set_y,extra_data_y,axis=0)
        else:
             ori_x = set_x_ori
             adver_x=set_x
             adver_y=set_y
    ori_x=ori_x[:datasets[1].shape[0],]
    adver_x=adver_x[:datasets[1].shape[0],]
    adver_y=adver_y[:datasets[1].shape[0],]    
    
    for i in xrange (len(adver_x)/ datasets[1].shape[0]):
         adver_set_x_temp=adver_x[datasets[1].shape[0]*i:datasets[1].shape[0]*(i+1)]
         adver_y_predict_temp=test_predict(adver_set_x_temp)         
         if i ==0:
             adver_y_predict=adver_y_predict_temp
         else:
             adver_y_predict=np.append(adver_y_predict,adver_y_predict_temp,axis=0)
         adver_y_predict_conf_temp=test_predict_conf(adver_set_x_temp)         
         if i ==0:
             adver_y_predict_conf=adver_y_predict_conf_temp
         else:
             adver_y_predict_conf=np.append(adver_y_predict_conf,adver_y_predict_conf_temp,axis=0)
    
    test_loss = test_model_all(adver_x,adver_y)        
    test_perf = 1- test_loss  
    print test_perf

    for i in xrange (len(ori_x)/ datasets[1].shape[0]):
         ori_x_temp=ori_x[datasets[1].shape[0]*i:datasets[1].shape[0]*(i+1)]
         ori_x_y_predict_temp=test_predict(ori_x_temp)         
         if i ==0:
             ori_x_y_predict=ori_x_y_predict_temp
         else:
             ori_x_y_predict=np.append(ori_x_y_predict,ori_x_y_predict_temp,axis=0)
         ori_x_y_predict_conf_temp=test_predict_conf(ori_x_temp)         
         if i ==0:
             ori_x_y_predict_conf=ori_x_y_predict_conf_temp
         else:
             ori_x_y_predict_conf=np.append(ori_x_y_predict_conf,ori_x_y_predict_conf_temp,axis=0)
         
    for i in xrange(len(set_x)):
        if adver_y[i] != adver_y_predict[i]: # and adver_y[i] != adver_y_predict_ori[i]: adver_y[i] == adver_y_predict_adv[i] and adver_y[i] != adver_y_predict_ori[i]:
            file_save_string(str(adver_y[i])+' '+str(ori_x_y_predict_conf[i])+' '+vec2sent_onedim(ori_x[i]),'save/'+SAVETHEME+'example_conf.txt')
            file_save_string('\n','save/'+SAVETHEME+'example_conf.txt')
            file_save_string(str(adver_y_predict[i])+' '+str(adver_y_predict_conf[i])+' '+vec2sent_onedim(adver_x[i]),'save/'+SAVETHEME+'example_conf.txt')
            file_save_string('\n','save/'+SAVETHEME+'example_conf.txt')
            print str(adver_y[i])+' '+str(ori_x_y_predict_conf[i])+' '+vec2sent_onedim(ori_x[i])
            print str(adver_y_predict[i])+' '+str(adver_y_predict_conf[i])+' '+vec2sent_onedim(adver_x[i])
    
