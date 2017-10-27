#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 10:30:53 2017

@author: anyi
"""
import cPickle
import numpy as np
from nltk.corpus import wordnet as wn

with open('save/I.pickle', 'rb') as file:
         tempx=cPickle.load(file)
         I=tempx 
with open('save/M.pickle', 'rb') as file:
         M=cPickle.load(file)
with open('save/Din1_index.pickle', 'rb') as file:
         tempx=cPickle.load(file)
         Din1_index=tempx 
with open('save/Din2_index.pickle', 'rb') as file:
         Din2_index=cPickle.loadx(file)
with open('save/anto_Train_set_x.pickle', 'rb') as file:
     tempx=cPickle.load(file)
     Train_set_x=tempx 
with open('save/anto_Train_set_y.pickle', 'rb') as file:
     Train_set_y=cPickle.load(file)

with open('save/Sec_I.pickle', 'rb') as file:
         tempx=cPickle.load(file)
         Sec_I=tempx 
with open('save/Sec_M.pickle', 'rb') as file:
         Sec_M=cPickle.load(file)
with open('save/Sec_Din1_index.pickle', 'rb') as file:
         tempx=cPickle.load(file)
         Sec_din1_index=tempx
with open('save/Sec_Din1_index.pickle', 'rb') as file:
         Sec_din2_index=cPickle.load(file)
with open('save/Sec_anto_Train_set_x.pickle', 'rb') as file:
     tempx=cPickle.load(file)
     Sec_Train_set_x=tempx #.astype(int)
#     print(Train_set_x[0][:10])
with open('save/Sec_anto_Train_set_y.pickle', 'rb') as file:
     Sec_Train_set_y=cPickle.load(file)
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
if len(I) == len(Sec_I):
    for order in  xrange(len(I)):
        din1_index=Din1_index[I[order][0]][M[order]]
        din2_index=Din2_index[I[order][0]][M[order]]    
        train_set_x[order:order+1,din1_index]=din2_index
else:      
    for order in  xrange(len(I)):
        din1_index=Din1_index[I[order][0]][M[order]]
        din2_index=Din2_index[I[order][0]][M[order]]    
        train_set_x[order:order+1,din1_index]=din2_index
    ### replacement second
    IM=I*50+M
    Sec_IM=Sec_I*50+Sec_M
    sec_pos=np.where(IM==Sec_IM)[0]
    for sec_order in xrange(len(Sec_I)):
        sec_din1_index=Sec_din1_index[Sec_I[sec_order][0]][Sec_M[sec_order]]
        sec_din2_index=Sec_din2_index[Sec_I[sec_order][0]][Sec_M[sec_order]]  
        train_set_x[sec_pos[sec_order],din1_index]=din2_index
    
# =============================================================================
# with open('save/I.pickle', 'rb') as file:
#          tempx=cPickle.load(file)
#          I=tempx #.astype(int)
# #         print(Din1_index[0][:10])
#  
# with open('save/M.pickle', 'rb') as file:
#          M=cPickle.load(file)
# 
# with open('save/Din1_index_wn.pickle', 'rb') as file:
#          tempx=cPickle.load(file)
#          Din1_index=tempx #.astype(int)
# #         print(Din1_index[0][:10])
#  
# with open('save/Din2_index_wn.pickle', 'rb') as file:
#          Din2_index=cPickle.load(file)
# 
#  
# with open('save/adver_Train_set_x_adversingle_wn.pickle', 'rb') as file:
#      tempx=cPickle.load(file)
#      Train_set_x=tempx #.astype(int)
# #     print(Train_set_x[0][:10])
#  
# with open('save/adver_Train_set_y_adversingle_wn.pickle', 'rb') as file:
#      Train_set_y=cPickle.load(file)
# #     print(Train_set_y)
# 
# 
# train_set_x=[]
# train_set_y=[]
# for i in xrange (len(Train_set_x)):
#     if i == 0:
#          train_set_x=Train_set_x[i]
#          continue
#     train_set_x=np.row_stack((train_set_x,Train_set_x[i]))
#  
# for i in xrange (len(Train_set_y)):
#     if i == 0:
#          train_set_y=Train_set_y[i]
#          continue
#     train_set_y=np.row_stack((train_set_y,Train_set_y[i]))
# 
# train_set_y=train_set_y.flatten()
# ### replacement
# for order in  xrange(len(Train_set_x)):
#     din1_index=Din1_index[I[order][0]][M[order]]
#     din2_index=Din2_index[I[order][0]][M[order]]
#     wtemp=Train_set_x[order][0,din1_index]
#     w_rep_index = 0
#     if wtemp ==0:
# #        print "senten: "+str(order) + " wtemp is zero"
#         continue
#     w_temp=word_idx_map.keys()[word_idx_map.values().index(wtemp)] 
# #                print str(m)+ '......' + w_temp
#     for s in wn.synsets(w_temp):
#         if s.lemmas()[0].name() != w_temp:
#             w_rep = s.lemmas()[0].name()
#             w_rep_index=word_idx_map.get(w_rep)
# #                        print s.lemmas()[0].name()
# #                        print w_rep_index
#             if w_rep_index != None:
#                 
#                 print "this sentence changed ......"+str(order)\
#                     +"position: "+str(din1_index)
#                 train_set_x[order:order+1,din1_index]=w_rep_index
#                 break
#             else:
#                 break
#     if wn.synsets(w_temp) == []:
# #        print "this sentence  No synonm"+str(order)
#         continue
#     if w_rep_index == None or w_rep_index == 0:
# #        print "none"
#         continue
#     print "this sentence save"+str(order)
# 
# =============================================================================
