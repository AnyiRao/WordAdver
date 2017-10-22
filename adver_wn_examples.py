#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 10:30:53 2017

@author: anyi
"""
import cPickle
import numpy as np
from nltk.corpus import wordnet as wn

# =============================================================================
# print '...save'
# 
# with open('save/Dinput_1.pickle', 'wb') as file:
#     model = Dinput_1
#     cPickle.dump(model, file)
#     print(model[0][:10])
#     
# with open('save/Dinput_2.pickle', 'wb') as file:
#     model = Dinput_2
#     cPickle.dump(model, file)
#     print(model[0][:10])
# 
# with open('save/Train_set_xtest.pickle', 'wb') as file:
#     model = Train_set_xtest
#     cPickle.dump(model, file)
#     print(model[0][:10])
# 
# with open('save/Train_set_ytest.pickle', 'wb') as file:
#     model = Train_set_ytest
#     cPickle.dump(model, file)
#     print(model[0][:10])
#     
# with open('save/Train_set_temp.pickle', 'wb') as file:
#     model = Train_set_temp
#     cPickle.dump(model, file)
#     print(model[0][:10])
# =============================================================================
# =============================================================================
# with open('save/Dinput_1_wn.pickle', 'rb') as file:
#         tempx=cPickle.load(file)
#         Dinput_1=tempx #.astype(int)
#         print(Dinput_1[0][:10])
# 
# with open('save/Dinput_2_wn.pickle', 'rb') as file:
#         Dinput_2=cPickle.load(file)
# 
# with open('save/Train_set_xtest.pickle', 'rb') as file:
#         tempx=cPickle.load(file)
#         Train_set_x=tempx #.astype(int)
#         print(Train_set_x[0][:10])
# 
# with open('save/Train_set_ytest.pickle', 'rb') as file:
#         Train_set_y=cPickle.load(file)
#         print(Train_set_y)
# 
# ### replacement
# batch_size = 50
# for i in xrange (len(Dinput_1)):
#     for j in xrange (batch_size):
#         (Train_set_x[i])[j,Dinput_1[i][j]]=Dinput_2[i][j]
#        
# #### 
# train_set_x=[]
# train_set_y=[]
# for i in xrange (len(Dinput_1)):
#     if i == 0:
#         train_set_x=Train_set_x[i]
#         continue
#     train_set_x=np.row_stack((train_set_x,Train_set_x[i]))
#     
# for i in xrange (len(Dinput_1)):
#     if i == 0:
#         train_set_y=Train_set_y[i]
#         continue
#     train_set_y=np.row_stack((train_set_y,Train_set_y[i]))
# train_set_y=train_set_y.flatten()
# 
# =============================================================================

with open('save/I.pickle', 'rb') as file:
         tempx=cPickle.load(file)
         I=tempx #.astype(int)
#         print(Dinput_1[0][:10])
 
with open('save/M.pickle', 'rb') as file:
         M=cPickle.load(file)

with open('save/Dinput_1_wn.pickle', 'rb') as file:
         tempx=cPickle.load(file)
         Dinput_1=tempx #.astype(int)
#         print(Dinput_1[0][:10])
 
with open('save/Dinput_2_wn.pickle', 'rb') as file:
         Dinput_2=cPickle.load(file)

 
with open('save/adver_Train_set_x_adversingle_wn.pickle', 'rb') as file:
     tempx=cPickle.load(file)
     Train_set_x=tempx #.astype(int)
#     print(Train_set_x[0][:10])
 
with open('save/adver_Train_set_y_adversingle_wn.pickle', 'rb') as file:
     Train_set_y=cPickle.load(file)
#     print(Train_set_y)


train_set_x=[]
train_set_y=[]
for i in xrange (len(Train_set_x)):
    if i == 0:
         train_set_x=Train_set_x[i]
         continue
    train_set_x=np.row_stack((train_set_x,Train_set_x[i]))
 
for i in xrange (len(Train_set_y)):
    if i == 0:
         train_set_y=Train_set_y[i]
         continue
    train_set_y=np.row_stack((train_set_y,Train_set_y[i]))

train_set_y=train_set_y.flatten()
### replacement
for order in  xrange(len(Train_set_x)):
    dinput1=Dinput_1[I[order][0]][M[order]]
    dinput2=Dinput_2[I[order][0]][M[order]]
    wtemp=Train_set_x[order][0,dinput1]
    w_rep_index = 0
    if wtemp ==0:
#        print "senten: "+str(order) + " wtemp is zero"
        continue
    w_temp=word_idx_map.keys()[word_idx_map.values().index(wtemp)] 
#                print str(m)+ '......' + w_temp
    for s in wn.synsets(w_temp):
        if s.lemmas()[0].name() != w_temp:
            w_rep = s.lemmas()[0].name()
            w_rep_index=word_idx_map.get(w_rep)
#                        print s.lemmas()[0].name()
#                        print w_rep_index
            if w_rep_index != None:
                
                print "this sentence changed ......"+str(order)\
                    +"position: "+str(dinput1)
                train_set_x[order:order+1,dinput1]=w_rep_index
                break
            else:
                break
    if wn.synsets(w_temp) == []:
#        print "this sentence  No synonm"+str(order)
        continue
    if w_rep_index == None or w_rep_index == 0:
#        print "none"
        continue
    print "this sentence save"+str(order)
