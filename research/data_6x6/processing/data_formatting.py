# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 12:14:12 2024

@author: Krishna
"""
import os
import pandas as pd
import numpy as np

data=[]
data.append(np.array(pd.read_csv('epvm_eq_0.csv',header=None)))
data.append(np.array(pd.read_csv('epvm_deq_0.csv',header=None)))
data.append(np.array(pd.read_csv('epvm_deq_1.csv',header=None)))
data.append(np.array(pd.read_csv('epvm_deq_2.csv',header=None)))
data.append(np.array(pd.read_csv('epvm_deq_3.csv',header=None)))



SV=[0.0,0.01,0.02,0.03,0.04]

for i in range(len(data)):
    if i==0:
        print('in i==0 loop')
        transposed_data=np.concatenate((np.array(data[i]),np.repeat(SV[i],len(data[i])).reshape(-1,1)),axis=1)
        transposed_data=np.vstack((transposed_data,np.zeros(data[i].shape[-1]+1)+np.nan))
    elif i==len(data)-1:
        print('in last loop')
        transposed_data=np.vstack((transposed_data,np.concatenate((np.array(data[i]),np.repeat(SV[i],len(data[i])).reshape(-1,1)),axis=1)))
        
    else:
        print('other loops')
        transposed_data=np.vstack((transposed_data,np.concatenate((np.array(data[i]),np.repeat(SV[i],len(data[i])).reshape(-1,1)),axis=1)))
        transposed_data=np.vstack((transposed_data,np.zeros(data[i].shape[-1]+1)+np.nan))
        
for i in range(len(data[0])):
    if i==0:
        temp=np.stack([data[j][i,:] for j in range(len(SV))])
        bingo_format=np.concatenate((temp,np.asarray(SV).reshape(-1,1)),axis=1)
        bingo_format=np.vstack((bingo_format,np.zeros(7)+np.nan))
    elif i==len(data[0])-1:
        temp=np.stack([data[j][i,:] for j in range(len(SV))])
        bingo_format=np.vstack((bingo_format,np.concatenate((temp,np.asarray(SV).reshape(-1,1)),axis=1)))
    else:
        temp=np.stack([data[j][i,:] for j in range(len(SV))])
        bingo_format=np.vstack((bingo_format,np.concatenate((temp,np.asarray(SV).reshape(-1,1)),axis=1)))
        bingo_format=np.vstack((bingo_format,np.zeros(7)+np.nan))
        
np.savetxt('bingo_format.txt',bingo_format)
np.savetxt('transpoed_data.txt',transposed_data)