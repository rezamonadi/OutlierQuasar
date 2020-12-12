#!/usr/bin/env python
# coding: utf-8

# In[5]:


#import numpy as numpy
import scipy as scipy
from sklearn import cluster
import matplotlib.pyplot as plt
from astropy.table import Table, Column
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
from seaborn import pairplot
#from sklearn.cluster import DBSCAN

#converts a numpy array to a list
def set_to_list(numpy_array):
	List = []
	for item in numpy_array:
		List.append(item.tolist())
	return List

#generate datasets
def GenerateData():
    tab = Table.read('DR16Q_v4.fits')
    tab.colnames
    psfflux = np.array(tab['PSFFLUX'])
    Fu = psfflux[:,0]
    Fg = psfflux[:,1]
    Fr = psfflux[:,2]
    Fi = psfflux[:,3]
    Fz = psfflux[:,4]

#  SDSS flux inverse variance
    psfflux_ivar = tab['PSFFLUX_IVAR']
    psfflux_SN = psfflux*psfflux_ivar


# SDSS FLux Signal tp Noise -> Flux*(1/Flux_variance)
    Fu_SN = psfflux_SN[:,0]
    Fu_SN = psfflux_SN[:,0]
    Fg_SN = psfflux_SN[:,1]
    Fr_SN = psfflux_SN[:,2]
    Fi_SN = psfflux_SN[:,3]
    Fz_SN = psfflux_SN[:,4]

# WISE fluxes
    W1_Flux = tab['W1_FLUX']
    W2_Flux = tab['W2_FLUX']
 
 # Wise Flux inverse variance 
    W1_Flux_ivar = tab['W1_FLUX_IVAR']
    W2_Flux_ivar = tab['W2_FLUX_IVAR']

    W1_SN, W2_SN = W1_Flux*W1_Flux_ivar, W2_Flux*W1_Flux_ivar

#  Other features 
    redshift = tab['Z']
    zwarning = tab['ZWARNING']
    e=2
    mask = (Fu_SN>e) & (Fg_SN>e) & (Fr_SN>e) & (Fi_SN>e) &(Fz_SN>e) & (redshift>0) & (zwarning==0)   & (W1_SN>e) & (W2_SN>e) 
    print(sum(mask))
    print(min(redshift[mask]), max(redshift[mask]), np.median(redshift[mask]))
#  SDSS-SDSS flux ratios
    Fug = np.log(Fu[mask]/Fg[mask])
    Fur = np.log(Fu[mask]/Fr[mask])
    Fui = np.log(Fu[mask]/Fi[mask])
    Fuz = np.log(Fu[mask]/Fz[mask])
    Fgr = np.log(Fg[mask]/Fr[mask])
    Fgi = np.log(Fg[mask]/Fi[mask])
    Fgz = np.log(Fg[mask]/Fz[mask])
    Fri = np.log(Fr[mask]/Fi[mask])
    Frz = np.log(Fr[mask]/Fz[mask])
    Fiz = np.log(Fi[mask]/Fz[mask])

# WISE-WISE flux ratios
    FW1W2 = np.log(W1_Flux[mask]/W2_Flux[mask])

# SDSS-WISE flux ratios
    FuW1 = np.log(Fu[mask]/W1_Flux[mask])
    FuW2 = np.log(Fu[mask]/W2_Flux[mask])
    FgW1 = np.log(Fg[mask]/W1_Flux[mask])
    FgW2 = np.log(Fg[mask]/W2_Flux[mask])
    FrW1 = np.log(Fr[mask]/W1_Flux[mask])
    FrW2 = np.log(Fr[mask]/W2_Flux[mask])
    FiW1 = np.log(Fi[mask]/W1_Flux[mask])
    FiW2 = np.log(Fi[mask]/W2_Flux[mask])
    FzW1 = np.log(Fz[mask]/W1_Flux[mask])
    FzW2 = np.log(Fz[mask]/W2_Flux[mask])
    data = np.array(list(zip(FuW1,
                        FgW1,  FuW2,FgW2)))
    data_scaled = StandardScaler().fit_transform(data)
    #df = pd.DataFrame(data=data_scaled, columns=['Fug', 'Fur', 'Fui', 'Fuz','Fgr','Fgi','Fgz','Fri','Frz','Fiz', 'FW1W2','FuW1',
                        #'FgW1', 'FrW1', 'FiW1', 'FzW1', 'FuW2', 'FgW2', 'FrW2', 'FiW2', 'FzW2'])
    # df = pd.DataFrame(data=data_scaled, columns=['FuW1','FgW1', 'FuW2', 'FgW2',])
    return data_scaled
    

def DBSCAN(dataset, eps,MinPts,DistanceMethod = 'euclidean'):
#    Dataset is a mxn matrix, m is number of item and n is the dimension of data
    m,n=dataset.shape
    print(m)
    print(n)
    Visited=np.zeros(m,'int') #stores whether a point is visited or not
    Type=np.zeros(m) #it stores the type of a point 
#   -1 noise, outlier
#    0 border
#    1 core
    #ClustersList=[] #set of clusters
    #Cluster=[]
    PointClusterNumber=np.zeros(m) #stores which point belongs to which cluster
    PointClusterNumberIndex=1 #here each cluster is represented by some indexes starting from 1
    PointNeighbors=[] #stores neighbouring point of a given point
    DistanceMatrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(dataset, DistanceMethod))
    for i in range(m):
        if Visited[i]==0:
            Visited[i]=1
            PointNeighbors=np.where(DistanceMatrix[i]<eps)
            #PointNeighbors=set_to_list(PointNeighbors)
            print("Length of Point neighbors are")
            print(len(PointNeighbors[0]))
            print((PointNeighbors[0]))
            if len(PointNeighbors[0])<MinPts:
                Type[i]=-1
            else:
                #for k in range(len(Cluster)):
                    #Cluster.pop()
                #Cluster.append(i)
                PointClusterNumber[i]=PointClusterNumberIndex
                PointNeighbors2=set_to_list(PointNeighbors[0])
                print(PointNeighbors)
                print(len(PointNeighbors2))
                print("n")
                print(PointNeighbors2)
                #print(PointNeighbors2[0])
                expandCluster(dataset[i],PointNeighbors2,Cluster,MinPts,eps,Visited,DistanceMatrix,PointClusterNumber,PointClusterNumberIndex)
                #Cluster.append(PointNeighbors[:])
                #ClustersList.append(Cluster[:])
                PointClusterNumberIndex=PointClusterNumberIndex+1
                #print(PointClusterNumberIndex)
    return PointClusterNumber 





def expandCluster(PointToExpand, PointNeighbors,Cluster,MinPts,eps,Visited,DistanceMatrix,PointClusterNumber,PointClusterNumberIndex  ):
    Neighbors=[]
    Neighbors2=[]
    #for i in PointNeighbors:
    for i in range(len(PointNeighbors)):
            #print(Visited[i])
            #print((Visited[PointNeighbors[i]]))
            if Visited[PointNeighbors[i]]==0:
                Visited[PointNeighbors[i]]=1
                Neighbors=np.where(DistanceMatrix[i]<eps)[0]
                print(Neighbors)
                Neighbors2=set_to_list(Neighbors)
                #print(Neighbors2[1])
# Neighbors merge with PointNeighbors
            for j in Neighbors:
                try:
                     PointNeighbors.index(j)
                except ValueError:
                     PointNeighbors.append(j)
            if PointClusterNumber[i]==0:
                #Cluster.append(i)
                PointClusterNumber[i]=PointClusterNumberIndex
    return


Data=GenerateData()
 
#Adding some noise with uniform distribution 
#X between [-3,17],
#Y between [-3,17]
#noise=scipy.rand(50,2)*20 -3
 
#Noisy_Data=numpy.concatenate((Data,noise))
#size=20
 
 
#fig = plt.figure()
#ax1=fig.add_subplot(2,1,1) #row, column, figure number
#ax2 = fig.add_subplot(212)
 
#ax1.scatter(Data[:,0],Data[:,1], alpha =  0.5 )
#ax1.scatter(noise[:,0],noise[:,1],color='red' ,alpha =  0.5)
#ax2.scatter(noise[:,0],noise[:,1],color='red' ,alpha =  0.5)
 
 
Epsilon=2.7
MinumumPoints=4
result =DBSCAN(Data[0:1000,:],Epsilon,MinumumPoints)
 

print(result)


 
#for i in xrange(len(result)):
#	ax2.scatter(Noisy_Data[i][0],Noisy_Data[i][1],color='yellow' ,alpha =  0.5)
      
#plt.show()
from sklearn.decomposition import PCA
import seaborn as sns
pca=PCA(n_components=2)
principal_comp=pca.fit_transform(Data)
pca_df = pd.DataFrame(data2 = principal_comp, columns =['pca1','pca2'])
pca_df.head()
pca_df = pd.concat([pca_df,pd.DataFrame({'cluster':result})], axis = 1)
print(pca_df.head())
plt.figure(figsize=(10,10))
filt=(pca_df['cluster']<1)
print(pca_df[filt].head())
ax = sns.scatterplot(x="pca1", y="pca2", hue = "cluster", data = pca_df[filt])
#ax = sns.scatterplot(x="pca1", y="pca2", hue = "cluster", data = pca_df)
plt.show()


# In[ ]:


print(m)


# In[ ]:




