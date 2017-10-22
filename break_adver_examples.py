
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 10:30:53 2017

@author: anyi
"""
import cPickle
import numpy as np
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
with open('save/Dinput_1.pickle', 'rb') as file:
        tempx=cPickle.load(file)
        Dinput_1=tempx #.astype(int)
        print(Dinput_1[0][:10])

with open('save/Dinput_2.pickle', 'rb') as file:
        Dinput_2=cPickle.load(file)

with open('save/Train_set_xtest.pickle', 'rb') as file:
        tempx=cPickle.load(file)
        Train_set_x=tempx #.astype(int)
        print(Train_set_x[0][:10])

with open('save/Train_set_ytest.pickle', 'rb') as file:
        Train_set_y=cPickle.load(file)
        print(Train_set_y)

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