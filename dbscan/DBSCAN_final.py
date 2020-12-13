#!/usr/bin/env python
# coding: utf-8

# In[7]:


import scipy as scipy
from scipy import sparse
from scipy.spatial import cKDTree as KDTree
#from sklearn import cluster
import matplotlib.pyplot as plt
from astropy.table import Table, Column
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
from seaborn import pairplot
from tqdm import tqdm
from scipy.spatial import distance
from sklearn.neighbors import KDTree
#from sklearn.neighbors import DistanceMetric
#from sklearn.cluster import DBSCAN
#from numpy.linalg import norm

#generate datasets
def CreateDataset():
    tab = Table.read('DR16Q_v4.fits')
    tab.colnames
    psfmag = np.array(tab['PSFMAG'])
    u = psfmag[:,0]
    g = psfmag[:,1]
    r = psfmag[:,2]
    i = psfmag[:,3]
    z = psfmag[:,4]

#  SDSS mag SN
    psfmag_error = tab['PSFMAGERR']
    psfmag_SN = psfmag/psfmag_error
# SDSS mag Signal to Noise -> Flux*(1/Flux_variance)
    u_SN = psfmag_SN[:,0]
    g_SN = psfmag_SN[:,1]
    r_SN = psfmag_SN[:,2]
    i_SN = psfmag_SN[:,3]
    z_SN = psfmag_SN[:,4]

# WISE magnitudes
    W1 = tab['W1_MAG']
    W2 = tab['W2_MAG']

# Wise MAG SN 
    W1_error = tab['W1_MAG_ERR']
    W2_error = tab['W2_MAG_ERR']
# WISE MAG SN
    W1_SN = W1/W1_error
    W2_SN = W2/W2_error

#  Other features 
    redshift = tab['Z']
    zwarning = tab['ZWARNING']
    e=3
    mask = (u_SN>e) & (g_SN>e) & (r_SN>e) & (i_SN>e) & (z_SN>e) & (zwarning==0)  & (W1_SN>e) & (W2_SN>e) &  (redshift<=2.7) & (redshift>=2)

    # Filtering DR16Q
    tab.remove_rows(mask==0)
    #tab.write('reduced_dr16q.fits')
    # Definiging colors -> difference between magnitude in two filters
    r = r[mask]
    i = i[mask]
    z = z[mask]
    W1 = W1[mask]
    W2 = W2[mask]
    # colors
    rz = r-z
    rW1 = r-W1
    iW1 = i-W1
    zW1 = z-W1
    rW2 = r-W2
    iW2 = i-W2
    zW2 = z-W2

    data = np.array(list(zip(rz, rW1, iW1, zW1, rW2, iW2, zW2)))
    data_scaled = StandardScaler().fit_transform(data)
    df = pd.DataFrame(data=data, columns=['rz_0', 'rW1_0', 'iW1_0', 'zW1_0', 'rW2_0', 'iW2_0', 'zW2_0'])
    df_scaled = pd.DataFrame(data=data_scaled, columns=['rz_sc', 'rW1_sc', 'iW1_sc', 'zW1_sc', 'rW2_sc', 'iW2_sc', 'zW2_sc'])
    return df_scaled

def DBSCAN(dataset, eps,MinPts):
#    Dataset is a mxn matrix, m is number of item and n is the dimension of data
    m,n=dataset.shape
    print(m)
    print(n)
    Visited=np.zeros(m,'int') #stores whether a point is visited or not
    Type=np.zeros(m) #it stores the type of a point 
    ClusterNumber=np.zeros(m) #stores which point belongs to which cluster
    ClusterIndex=1 #here each cluster is represented by some indexes starting from 1
    #PointNeighbors=[] #stores neighbouring point of a given point
    cluster=0
    X=dataset
    tree = KDTree(X, leaf_size=2)
    for i in tqdm(range(m)):
        PointNeighbors=[] #stores neighbouring point of a given point
        PointNeigbors2=[]
        #X=dataset
        #if (cluster==1):
            #break
        if Visited[i]==0:
            Visited[i]=1
            for j in range(m):
                x=distance.euclidean(dataset[i:i+1],dataset[j:j+1])
                if x<=eps:
                    PointNeighbors.append(j)
            #ind=tree.query_radius(X[i:i+1], eps, return_distance=False, count_only=False, sort_results=False)
            #for j in ind[0]:
                #PointNeighbors.append(j)
            if len(PointNeighbors)==1:
                Type[i]=-1
                #print("Outlier")
            elif (len(PointNeighbors)<MinPts) & (len(PointNeighbors)>1):
                Type[i]=1
                #print("Border Point")
            else:
               
                ClusterNumber[i]=ClusterIndex
                #print(type(PointNeighbors))
                #print(PointNeighbors)
                #PointNeighbors2=set_to_list(PointNeighbors)
                PointNeighbors2=PointNeighbors
                #print(PointNeighbors)
                print(len(PointNeighbors2))
                print("n")
                #print(PointNeighbors2)
                print(PointNeighbors2[0])
                CheckNeighbourhood(dataset,X,cluster,tree,m,PointNeighbors2,MinPts,eps,Visited,ClusterNumber,ClusterIndex)
                ClusterIndex=ClusterIndex+1
               
    return ClusterNumber 





def CheckNeighbourhood(dataset,X,cluster,tree,m,PointNeighbors,MinPts,eps,Visited,ClusterNumber,ClusterIndex):
    Neighbors=[]
    Neighbors2=[]
    cluster=cluster+1
    count=0
    for i in PointNeighbors:
            count=count+1
            print(count)
            #print(i)
            if Visited[i]==0:
                Visited[i]=1
                ind2=tree.query_radius(X[i:i+1], eps, return_distance=False, count_only=False, sort_results=False)
                if(len(list(ind2[0]))>=MinPts):
                    Neighbors3=set(list(ind2[0]))-set(PointNeighbors)
# Neighbors merge with PointNeighbors
                    for j in Neighbors3:
                        PointNeighbors.append(j)
            if ClusterNumber[i]==0:
                ClusterNumber[i]=ClusterIndex
    return


Data=CreateDataset()
fig = plt.figure()
 
Epsilon=1.5
MinumumPoints=10
print(Data[1:5])
#test=Data[0:5000]
test= Data
result =DBSCAN(test,Epsilon,MinumumPoints)
print(result)
data_with_clust = pd.concat([test,pd.DataFrame({'cluster':result})], axis = 1)
#print(data_with_clust.head())
outlier=data_with_clust[data_with_clust['cluster']==0]
print(outlier.head())
from sklearn.decomposition import PCA
import seaborn as sns
pca=PCA(n_components=2)
principal_comp=pca.fit_transform(test)
pca_df = pd.DataFrame(data = principal_comp, columns =['pca1','pca2'])
pca_df.head()
pca_df = pd.concat([pca_df,pd.DataFrame({'cluster':result})], axis = 1)
print(pca_df.head())
plt.figure(figsize=(10,10))
filt=(pca_df['cluster']<1)
#print(pca_df[filt].head())
#ax = sns.scatterplot(x="pca1", y="pca2", hue = "cluster", data = pca_df[filt])
ax = sns.scatterplot(x="pca1", y="pca2", hue = "cluster", data = pca_df)
plt.show()
print(len(outlier))


# In[ ]:





# In[ ]:





# In[8]:


print(test.head())


# In[3]:


print(Data.head())


# In[ ]:




