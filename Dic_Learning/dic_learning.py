# -*- coding: utf-8 -*-
import numpy as np
from sklearn import linear_model
data = np.loadtxt('data/communitycrime/crimecommunity.csv', delimiter=',')
random_index = np.loadtxt('data/communitycrime/crimecommunity_index.csv', delimiter=',')
sample = data[:,1:-2]
label = data[:,-1]
sensitive = data[:,0]
[n,p] = sample.shape
[row,column] = random_index.shape
for i in range(row):
    for j in range(column):
        random_index[i,j] = random_index[i,j]-1
#dic update        
def dict_update(y,d,x,n_components):
    for i in range (n_components):
        index = np.nonzero(x[i,:])[0]
        if len(index) ==0:
            continue;
        d[:,i]=0
        r = (y-np.dot(d,x))[:,index]
        u,s,v = np.linalg.svd(r,full_matrices=False)
        d[:,i]=u[:,0]
        for j,k in enumerate(index):
            x[i,k] = s[0]*v[0,j]
    return d,x
final_dic =[]
num = 3 
threshold=0.01
for loop in range (num):
    start = 0 #hyperparameter
    end = int(1/num*n) #hyperparameter
    sample_train = sample[random_index[start:end,loop].astype('int64'),:]
    label_train = label[random_index[start:end,loop].astype('int64')]
    sensitive_train = sensitive[random_index[start:end,loop].astype('int64')]
    [n1,p]=sample_train.shape
    u,s,v = np.linalg.svd(np.transpose(sample_train))  
    n_comp=p
    dictionary = u[:,:n_comp]
    max_iter = 1000
    tolerance = 1e-6
    temp_dic = np.zeros((p,n_comp))
    y = np.transpose(sample_train) 
    e=np.zeros((max_iter,1))
    for i in range(max_iter):
        x = linear_model.orthogonal_mp(dictionary,y)  
        e[i]= np.linalg.norm(y-np.dot(dictionary,x))
        if e[i]<tolerance:
            break
        dict_update(y,dictionary,x,n_comp)
    dic_acc=dictionary
    for i in range (n_comp):
        cov_sum = 0
        for j in range(n1):
            pre = np.dot(sample_train[j,:],dic_acc[:,i])
            cov_sum = cov_sum+(sample_train[j,0]-np.mean(sample_train[:,0]))*pre
        random_cov = cov_sum / n1
        if(abs(random_cov)<=threshold):
            temp_dic[:,i] = dic_acc[:,i]
    idx = np.argwhere(np.all(temp_dic[..., :] == 0, axis=0))
    final_dic.append(np.delete(temp_dic, idx, axis=1))
    print("Complete the iteration",loop)



