#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from tqdm import tqdm
from scipy import stats


# In[3]:


# Reading and arranging data
# All the mfcc features for each utterance is compiled in a folder called MFCC
train_speakers = ['ac',  'bh',  'cg',  'dg',  'eg',  'hg',  'il',  'jn',  'kh',  'la',
'ag',  'bi',  'cl',  'ea',  'ei',  'hp',  'jc',  'jp',  'kk',  'ld',
'ai',  'br', 'cm',  'ec',  'ek',  'ig',  'ji',  'kc',  'kn',  'ls',
'an',  'ca',  'dc',  'ee',  'es',  'ih',  'jj',  'kf',  'kt'
]
test_speakers =['mk',  'mm',  'ms',  'mw',  'nc',  'ng',  'nh',  'pe',  'pk',  'pm',  'pp',  'ra'] 
digits = ['1','4','6','9','o']
train_mfcc = {} #dictionary to be addressed with labels
test_mfcc = {}

for speaker in train_speakers:
    for i in digits:
        s = speaker+'_'+i+'.wav.mfcc' #generating file name
        
        with open('MFCC/'+s) as f:
            lines = f.readlines() #reading the lines of the mfcc files
            
        size = np.asarray(list(map(int,lines[0].split()))) #converting the space separated info to an array
        dim = size[0]  #dimension of feature vectors = 38
        length = size[1] #number of feature vectors in the utterance
        mfcc_coeff = np.zeros((length,dim)) #coefficients info row-> feature vector number,column->coefficients
        
        for i in range(length):
            a = np.asarray(list(map(float,lines[i+1].split())))
            mfcc_coeff[i] = a
            
        train_mfcc[s] = mfcc_coeff
        
for speaker in test_speakers:
    for i in digits:
        s = speaker+'_'+i+'.wav.mfcc' #generating file name
        
        with open('MFCC/'+s) as f:
            lines = f.readlines() #reading the lines of the mfcc files
            
        size = np.asarray(list(map(int,lines[0].split()))) #converting the space separated info to an array
        dim = size[0]  #dimension of feature vectors = 38
        length = size[1] #number of feature vectors in the utterance
        mfcc_coeff = np.zeros((length,dim)) #coefficients info row-> feature vector number,column->coefficients
        
        for i in range(length):
            a = np.asarray(list(map(float,lines[i+1].split())))
            mfcc_coeff[i] = a
            
        test_mfcc[s] = mfcc_coeff
        


# In[5]:


#Function for dynamic time warping

def DTW(sample,test):  # function returns DTW cost matrix and the warped path, vertical movement allowed since test could be smaller than sample
    n = len(sample)
    m = len(test)
    d = len(sample[0])
    if(d != len(test[0])):
        return np.inf,np.inf
    else:
        phi = np.zeros((n+1,m+1))  #dtw cost matrix
        epsilon = np.zeros((n+1,m+1))  #dtw path matrix

        #initialisation
        for i in range(n+1):
            for j in range(m+1):
                if(i*j == 0):
                    if((i == 0)and(j == 0)):
                        phi[i,j] = 0
                    else:
                        phi[i,j] = np.inf
                #recursion
                else:
                    prev_min = np.min([phi[i-1,j],phi[i,j-1],phi[i-1,j-1]])
                    phi[i,j] = np.linalg.norm(sample[i-1,:] - test[j-1,:]) + prev_min #euclidean distance between the feature vectors + previous min val


        #backtracking to find optimal warped path
        for i in range(n+1):
            for j in range(m+1):
                i = n-i
                j = m-j

                if((phi[i-1,j] <= phi[i,j-1])and(phi[i-1,j] <= phi[i-1,j-1])):
                    epsilon[i-1,j] = 1
                elif((phi[i,j-1] <= phi[i-1,j] )and(phi[i,j-1] <= phi[i-1,j-1] )):
                    epsilon[i,j-1] = 1
                else:
                    epsilon[i-1,j-1] = 1


        epsilon[0,0] = 1
        epsilon[1,1] = 1 #start together
        epsilon[n,m] = 1 #end together

        return phi,epsilon
            
def predictor(score,k=len(train_speakers)): # K-NN based prediction given score matrix 
    n_digits,n_speaker = score.shape
    score_array = [] # score array with all score values
    pred_array = np.zeros(k)
    for i in range(n_digits):
        for j in range(n_speaker):
            score_array.append(score[i,j])
    score_array = np.asarray(score_array) # converting to numpy array
    indices = [b[0] for b in sorted(enumerate(score_array),key=lambda i:i[1])] #indices when sorted in ascending order
    
    for i in range(k):
        index = indices[i]
        pred_array[i] = int(index/n_speaker)
        
    prediction = stats.mode(pred_array) #majority prediction
    
    return prediction
        
    
        

    
    


# In[2]:


# K-Means Algorithm

#K-Means helper functions
def dist(x,centroid):   #computes distance between two vectors
    distance = np.square(x-centroid).sum()
    return distance


def closest_centroid(x,centroids):  #computes the index of the closest centroid
    distance = []
    for i in range(len(centroids)):
        distance.append(dist(x,centroids[i]))
        
    closest_centroid_index = distance.index(min(distance))
    return closest_centroid_index 

def tot_error(data,centroids,assigned_centroids): # returns total error incurred
    error = 0
    
    for i,x in enumerate(data): #i is index, x is value at that index
        centroid = centroids[int(assigned_centroids[i])]
        error += dist(x,centroid)
        
    error /= len(data)
    return error
        
    
def KMeans(data,n_clusters,niter=50,tolerance = 0.0001):  #niter taken to be 50 as it converges before that
    cluster_centroids = np.zeros((n_clusters,data.shape[1]))
    assigned_centroids= np.zeros(data.shape[0])
    r                 = np.zeros((data.shape[0],n_clusters))
    
    #initialisation
    # assigning the cluster_centroids to random data points
    indices = np.random.randint(data.shape[0],size = n_clusters)
    
    for i,index in enumerate(indices):
        cluster_centroids[i] = data[index]
        
    error = np.zeros(niter)
    #Assignment and Update 
    for n in range(niter):
        
        #Assignment 
        for i,x in enumerate(data):
            ind = closest_centroid(x,cluster_centroids)
            assigned_centroids[i] = ind #storing the assigned centroid
            r[i,ind] = 1 #responsibility r[n,k] = 1

        #Update
        for i in range(n_clusters):
            R = 0  #total responsibility R
            for j,x in enumerate(data):
                cluster_centroids[i] += r[j,i]*x  # Sigma(r[n,k]x[n])
                R                    += r[j,i]
            cluster_centroids[i] /= R

        error[n] = tot_error(data,cluster_centroids,assigned_centroids)
        if((error[n]-error[n-1])<tolerance):
            break
    return cluster_centroids,assigned_centroids,error   
        


# In[120]:


# DTW based digit recognition
pred = {}
error = 0
for speaker in tqdm(test_speakers):
    for i in digits:
        s_test = speaker+'_'+i+'.wav.mfcc'
        score = np.zeros((len(digits),len(train_speakers)))#score matrix with dynamic time warping scores 
        
        for k in range(len(train_speakers)):
            speaker2 = train_speakers[k]
            for j in range(len(digits)):
                    digit = digits[j]
                    s_train = speaker2+'_'+digit+'.wav.mfcc'
                    #print(train_mfcc[s_train].shape)
                    phi = DTW(train_mfcc[s_train],test_mfcc[s_test])[0]
                    score[j,k] = phi[-1,-1]
        
        #making prediction 
        predicted_index = predictor(score,20)[0]
        prediction = digits[int(predicted_index)]
        pred[s_test] = prediction
        if(prediction != i):
            error += 1

print(pred)
print(error)
        
                
            


# In[ ]:


# Generating the K-Means codebook and the observation sequences





