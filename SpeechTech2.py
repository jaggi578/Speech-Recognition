#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
from tqdm import tqdm
from scipy import stats
import os
from sklearn.model_selection import train_test_split



# In[19]:


#organizing all the data
# Since this is a speaker identification/verification project,all the speakers can be clustered be they in train or test in the database
#MFCC5 is the directory containing all the Mel Filterbank Cepstral Coefficients of all the data

#initializing list
male_speakers =[]
female_speakers=[] 

#initialising dictionaries for record keeping purpose
train_male={}
test_male={}
train_female={}
test_female={}

#initialising the dictionaries storing the mfcc coefficients
train_male_mfcc = {}
test_male_mfcc = {}
train_female_mfcc = {}
test_female_mfcc = {}
 
directories = os.listdir('MFCC5')
for dir in directories:
    inner_dir = os.listdir('MFCC5/'+dir)
    in_dir = dir
    for dir in inner_dir:
        folder = dir
        inner_inner_dir = os.listdir('MFCC5/'+in_dir+'/'+folder)
        if(folder == 'man'):
            for file in inner_inner_dir:
                mfcc_dir = os.listdir('MFCC5/'+in_dir+'/'+folder+'/'+file)
                file = file[:-4] #truncating mfcc
                male_speakers.append(file)
                train_male[file],test_male[file] = train_test_split(mfcc_dir,test_size = 0.2,random_state=42) #randomstate= int for reproducable output, 20% test,80% train
                train_male_mfcc[file] = []
                test_male_mfcc[file] = []
    
                #reusing code snippet from assignment 4
                for utterance in train_male[file]:
                    with open('MFCC5/'+in_dir+'/'+folder+'/'+file+'mfcc/'+ utterance) as f: #open the file containing mfcc coefficients
                        lines = f.readlines() #reading the lines of the mfcc files

                    size = np.asarray(list(map(int,lines[0].split()))) #converting the space separated info to an array
                    dim = size[0]  #dimension of feature vectors = 38
                    length = size[1] #number of feature vectors in the utterance
                    mfcc_coeff = np.zeros((length,dim)) #coefficients info row-> feature vector number,column->coefficients

                    for i in range(length):
                        a = np.asarray(list(map(float,lines[i+1].split())))
                        mfcc_coeff[i] = a

                    train_male_mfcc[file].append(mfcc_coeff)
                
                for utterance in test_male[file]:
                    with open('MFCC5/'+in_dir+'/'+folder+'/'+file+'mfcc/'+ utterance) as f: #open the file containing mfcc coefficients
                        lines = f.readlines() #reading the lines of the mfcc files

                    size = np.asarray(list(map(int,lines[0].split()))) #converting the space separated info to an array
                    dim = size[0]  #dimension of feature vectors = 38
                    length = size[1] #number of feature vectors in the utterance
                    mfcc_coeff = np.zeros((length,dim)) #coefficients info row-> feature vector number,column->coefficients

                    for i in range(length):
                        a = np.asarray(list(map(float,lines[i+1].split())))
                        mfcc_coeff[i] = a

                    test_male_mfcc[file].append(mfcc_coeff)
                    

        else:
            for file in inner_inner_dir:
                mfcc_dir = os.listdir('MFCC5/'+in_dir+'/'+folder+'/'+file)
                file = file[:-4] #truncating mfcc
                female_speakers.append(file)
                train_female[file],test_female[file] = train_test_split(mfcc_dir,test_size = 0.2,random_state=42) #randomstate= int for reproducable output, 20% test,80% train
                train_female_mfcc[file] = []
                test_female_mfcc[file] = []
    
                #reusing code snippet from assignment 4
                for utterance in train_female[file]:
                    with open('MFCC5/'+in_dir+'/'+folder+'/'+file+'mfcc/'+ utterance) as f: #open the file containing mfcc coefficients
                        lines = f.readlines() #reading the lines of the mfcc files

                    size = np.asarray(list(map(int,lines[0].split()))) #converting the space separated info to an array
                    dim = size[0]  #dimension of feature vectors = 38
                    length = size[1] #number of feature vectors in the utterance
                    mfcc_coeff = np.zeros((length,dim)) #coefficients info row-> feature vector number,column->coefficients

                    for i in range(length):
                        a = np.asarray(list(map(float,lines[i+1].split())))
                        mfcc_coeff[i] = a

                    train_female_mfcc[file].append(mfcc_coeff)
                
                for utterance in test_female[file]:
                    with open('MFCC5/'+in_dir+'/'+folder+'/'+file+'mfcc/'+ utterance) as f: #open the file containing mfcc coefficients
                        lines = f.readlines() #reading the lines of the mfcc files

                    size = np.asarray(list(map(int,lines[0].split()))) #converting the space separated info to an array
                    dim = size[0]  #dimension of feature vectors = 38
                    length = size[1] #number of feature vectors in the utterance
                    mfcc_coeff = np.zeros((length,dim)) #coefficients info row-> feature vector number,column->coefficients

                    for i in range(length):
                        a = np.asarray(list(map(float,lines[i+1].split())))
                        mfcc_coeff[i] = a

                    test_female_mfcc[file].append(mfcc_coeff)

'''
# In[114]:


print(len(train_male_mfcc['ar'][5][35]))


# In[49]:


data = train_male_mfcc['ar']
n_utter = len(data)
data_matrix = []
    
#create a data matrix accumulating all the feature vectors across utterances
for i in range(n_utter):
    n_fv = len(data[i]) # number of feature vectors in that utterance
    for j in range(n_fv):
        data_matrix.append(data[i][j])
print(np.asarray(data_matrix).shape[1])


# In[14]:


with open('MFCC5/testmfcc/man/ar/28a.wav.mfcc') as f: #open the file containing mfcc coefficients
    lines = f.readlines()
print(lines)

'''
# In[22]:


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
    return cluster_centroids,assigned_centroids,error[-1]


# In[109]:


# Function to return GMM params given speaker data
def GMM_pdf(x,mean,covariance): 
    D = len(x) #dimension
    sigma_det = np.linalg.det(covariance)
    Norm_factor = 1/(np.sqrt(((2*np.pi)**D)*sigma_det))
    sigma_inverse = np.linalg.inv(covariance)
    exp_factor =-0.5*(np.dot((x-mean).T,np.dot(sigma_inverse,(x-mean))))
    pdf = Norm_factor*(np.power(np.e,exp_factor))
    return pdf
                       
                       
    
    
    
def GMM(data,n_clusters = 10,n_iter=10,tolerance = 1e-8): #data is a list of 2d arrays where each 2d array are the feature vectors for an utterance
    #find no of utterances in data
    n_utter = len(data)
    data_matrix = []
    
    #create a data matrix accumulating all the feature vectors across utterances
    for i in range(n_utter):
        n_fv = len(data[i]) # number of feature vectors in that utterance
        for j in range(n_fv):
            data_matrix.append(data[i][j])
            
    data_matrix = np.asarray(data_matrix) #make it into a numpy array for processing
    
    #initialization
    weights = np.zeros(n_clusters)
    N_count = np.zeros(n_clusters)
    means = np.zeros((n_clusters,data_matrix.shape[1])) #data_matrix.shape[1] = size of the feature vector =38
    covariances = np.zeros((n_clusters,data_matrix.shape[1],data_matrix.shape[1]))
    gamma = np.zeros((len(data_matrix),n_clusters)) #gamma[n][k]
    gaussian_pdf = np.zeros(n_clusters) #temporary variable used to compute gamma
    
    #initialization for the means
    means,assigned_centroids,error = KMeans(data_matrix,n_clusters)
    
    #initializing variance and weights
    for i in range(len(assigned_centroids)): #iterate across assigned_centroids to update weights and covariances
        k = int(assigned_centroids[i])  #centroid assigned to datapoint i
        #covariances[k]+=(data_matrix[i]-means[k]).dot((data_matrix[i]-means[k]).T) # (x[n]-mu[k])(x[n]-mu[k])^T
        for d in range(data_matrix.shape[1]):
            covariances[k][d][d] += np.power((data_matrix[i][d] - means[k][d]),2) #diagonal covariance matrix, elliptical gaussian
        N_count[k] += 1
    
    for i in range(n_clusters):
        covariances[i] /= N_count[i]
        weights[i]      = N_count[i]/(len(data_matrix))  #Nk/N
    
    #Begin E-M algorithm
    
    for i in range(n_iter):
        temp_N_count = np.zeros(n_clusters)
        temp_means = np.zeros((n_clusters,data_matrix.shape[1])) #data_matrix.shape[1] = size of the feature vector =38
        temp_covariances = np.zeros((n_clusters,data_matrix.shape[1],data_matrix.shape[1]))
        
        
        #E step algorithm
        for n in range(len(data_matrix)):
            weighted_sum = 0
            for k in range(n_clusters):
                corrected_cov = covariances[k]+tolerance*(np.identity(data_matrix.shape[1])) #to make matrix non singular
                #gaussian_pdf[k] = stats.multivariate_normal.pdf(data_matrix[n], mean = means[k] , cov = corrected_cov)
                gaussian_pdf[k] = GMM_pdf(data_matrix[n], mean = means[k] , covariance = corrected_cov)
                weighted_sum += weights[k]*gaussian_pdf[k]
            
            for k in range(n_clusters):
                gamma[n][k] = (weights[k]*gaussian_pdf[k])/(weighted_sum)
                temp_N_count[k] += gamma[n][k]
        
        #M step algorithm
        #means
        for n in range(len(data_matrix)):
            for k in range(n_clusters):
                temp_means[k] += gamma[n][k]*data_matrix[n]
                
        #updating means
        for k in range(n_clusters):
            means[k]       = temp_means[k]/temp_N_count[k]
        
        #covariances
        for n in range(len(data_matrix)):
            for k in range(n_clusters):
                #temp_covariances[k] += gamma[n][k]*((data_matrix[n]-means[k]).dot((data_matrix[n]-means[k]).T)) # gamma[n][k](x[n]-mu[k])(x[n]-mu[k])^T
                for d in range(data_matrix.shape[1]):
                    temp_covariances[k][d][d] += gamma[n][k]*np.power((data_matrix[n][d] - means[k][d]),2) #diagonal covariance matrix, elliptical gaussian
        #updating covariances and weights
        for k in range(n_clusters):
            covariances[k] = temp_covariances[k]/temp_N_count[k]
            weights[k]     = temp_N_count[k]/(len(data_matrix))
        
    
    return weights,means,covariances
    
    
    


# In[249]:


# Function to make predictions based on different metrics
# average posterior probability across feature vectors
# average log posterior probability
# majority vote across feature vectors
# conventional posterior probability

def predictor_GMM(data,orig_class,gmm_params): # return prediction and also prediction score of the original class to plot RoC and DET curves
    
    pred_label = [] #list of size 4
    error = np.zeros(4) # 1 if error 0 otherwise
    orig_index = 0 #index of orig_class
    
    avg_post = 0 #avg posterior prob
    avg_logpost = 0 # avg log posterior prob (geometric mean)
    conv_post = 0 #conventional posterior prob
    
    N_vec = len(data) # no of feature vectors in the test utterance
    N_speakers = len(gmm_params.keys()) # no of speakers
    
    #temporary arrays to keep scores
    post_scores = np.zeros((N_speakers,N_vec)) # each row corresponds to a speaker, each column correspond to feature vector
    speaker_scores = np.zeros((N_speakers,3)) # since we have three different metrics to compute scores
    original_class_score = np.zeros(3)
    
    #computing posterior scores
    for i in range(N_speakers):
        for j in range(N_vec):
            weights,means,covariances = gmm_params[list(gmm_params.keys())[i]]
            for k in range(len(weights)):
                pdf = GMM_pdf(data[j],means[k],covariances[k])
                post_scores[i][j] += weights[k]*pdf
                
        # having computed the posterior scores compute speaker scores
        
        #speaker_scores[i][0] = post_scores[i].mean() #average posterior score
        #speaker_scores[i][2] = post_scores[i].prod() #conventional posterior score
        speaker_scores[i][1] = np.power(speaker_scores[i][2],1/N_vec) #average log posterior score
        
        #original scores
        if(list(gmm_params.keys())[i] == orig_class):
            orig_index = i
            original_class_score[0] = np.around(speaker_scores[i][0],5) #rounded off to 5 decimal places
            original_class_score[1] = np.around(speaker_scores[i][1],5) 
            original_class_score[2] = np.around(speaker_scores[i][2],5)
        
    #make predictions
    for i in range(3):
        pred_index = np.where(speaker_scores[:,i] == speaker_scores[:,i].max())[0]
        print(speaker_scores[:,i])
        if(len(pred_index) > 1):
            pred_index = int(pred_index[0])
        #else:
            #pred_index = int(pred_index)
        pred_label.append(list(gmm_params.keys())[pred_index])
        if(pred_index != orig_index):
            error[i] = 1
            
        
    
    #majority vote prediction
    pred_vec = np.zeros(N_vec)
    maj_pred_ind = 0
    
    for i in range(N_vec):
        pred_vec[i] = np.where(post_scores[:,i] == post_scores[:,i].max())[0]
        if(len(pred_vec[i]) > 1):
            pred_vec[i] = int(pred_vec[i][0])
        #else:
            #pred_vec[i] = int(pred_vec[i])
        
    maj_pred_ind = int(stats.mode(pred_vec)[0])
    if(maj_pred_ind != orig_index):
        error[3] = 1
    pred_label.append(list(gmm_params.keys())[maj_pred_ind])
    
    # return 0 if correct pred 1 if wrong
    
    
    
    
    return pred_label,original_class_score,error
    
    
    
    

        
        
    
    
    
    
    


# In[220]:


# Training step


#initialising dictionaries
gmm_params_male = {}
gmm_params_female = {}


for speaker in tqdm(male_speakers):
    gmm_params_male[speaker] = GMM(data = train_male_mfcc[speaker])
    
for speaker in tqdm(female_speakers):
    gmm_params_female[speaker] = GMM(data = train_female_mfcc[speaker])
    


# In[227]:


gmm_params_male1 = gmm_params_male
gmm_params_female1 = gmm_params_female


# In[250]:


# Testing step
predictions_male = [[] for i in range(4)] #as four different metrics are being employed for prediction
predictions_female = [[] for i in range(4)] 
orig_male = []
orig_female = []
orig_class_scores_male = [[] for i in range(3)] 
orig_class_scores_female = [[] for i in range(3)]
error_male = [[] for i in range(4)]
error_female= [[] for i in range(4)]

temp_error = np.zeros(4)
temp_pred_label = []
temp_orig_score = np.zeros(3)

for speaker in tqdm(male_speakers):
    N_utter = len(test_male_mfcc[speaker])
    for i in range(N_utter):
        orig_male.append(speaker)
        temp_pred_label,temp_orig_score,temp_error = predictor_GMM(data = test_male_mfcc[speaker][i],orig_class = speaker,gmm_params = gmm_params_male)
        predictions_male[0].append(temp_pred_label[0])
        predictions_male[1].append(temp_pred_label[1])
        predictions_male[2].append(temp_pred_label[2])
        predictions_male[3].append(temp_pred_label[3])
        orig_class_scores_male[0].append(temp_orig_score[0])
        orig_class_scores_male[1].append(temp_orig_score[1])
        orig_class_scores_male[2].append(temp_orig_score[2])
        error_male[0].append(temp_error[0])
        error_male[1].append(temp_error[1])
        error_male[2].append(temp_error[2])
        error_male[3].append(temp_error[3])
        
        
    
for speaker in tqdm(female_speakers):
    N_utter = len(test_female_mfcc[speaker])
    for i in range(N_utter):
        orig_male.append(speaker)
        temp_pred_label,temp_orig_score,temp_error = predictor_GMM(data = test_female_mfcc[speaker][i],orig_class = speaker,gmm_params = gmm_params_female)
        predictions_female[0].append(temp_pred_label[0])
        predictions_female[1].append(temp_pred_label[1])
        predictions_female[2].append(temp_pred_label[2])
        predictions_female[3].append(temp_pred_label[3])
        orig_class_scores_female[0].append(temp_orig_score[0])
        orig_class_scores_female[1].append(temp_orig_score[1])
        orig_class_scores_female[2].append(temp_orig_score[2])
        error_female[0].append(temp_error[0])
        error_female[1].append(temp_error[1])
        error_female[2].append(temp_error[2])
        error_female[3].append(temp_error[3])


# In[ ]:


# UBM GMM

# creating the UBM GMM framework with 1024 Gaussians

UBM_male_data = []
UBM_female_data = []

for speaker in male_speakers:
    






