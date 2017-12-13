#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:33:58 2017

@author: anyi
"""

import os
import numpy as np

def file_cover(file_name):
    if os.path.isfile(file_name):
        f=open(file_name,"w")
        f.truncate()
        f.close()
    return

def file_save_string(string,file_name):
    with open(file_name, 'a') as file:
        file.write(string)

def file_save_int(integer,file_name):
    with open (file_name,"a") as file:
        write_integer = '%d\n'%(integer)  
        file.write(write_integer)

def file_save_nparray(nparray,file_name):
    with open (file_name,"a") as file:
        for i in xrange (len(nparray)):
            write_nparray = '%d\n'%(nparray[i])  
            file.write(write_nparray)
def file_save_nparray2d(nparray,file_name):
    with open (file_name,"a") as file:
        for i in xrange (nparray.size):
            write_nparray = '%d\n'%(nparray[0,i])  
            file.write(write_nparray)
        
if __name__=="__main__":   
    file_name='test.txt'
    file_cover(file_name)
    file_save_string('hello',file_name)
    for i in xrange (10):
        file_save_int(i,file_name)
    a =np.loadtxt(file_name)    
    b= np.ones(5)
    c= np.zeros(5)
    file_save_nparray(b,file_name)
    file_save_nparray(c,file_name)
