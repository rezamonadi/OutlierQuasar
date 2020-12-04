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
#from sklearn.neighbors import DistanceMetric
#from sklearn.cluster import DBSCAN
#from numpy.linalg import norm

#converts a numpy array to a list
def set_to_list(numpy_array):
	List = []
	for item in numpy_array:
		List.append(item.tolist())
	return List

#generate datasets
def CreateDataset():
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
    data = np.array(list(zip(Fui, Fuz, FuW1, FuW2, Fgz, FgW1, FgW2, FrW1, FrW2, FiW2)))  
    data_scaled = StandardScaler().fit_transform(data)
    df =pd.DataFrame(data=data_scaled, columns=['Fui', 'Fuz', 'FuW1', 'FuW2', 'Fgz', 'FgW1', 'FgW2', 'FrW1', 'FrW2', 'FiW2'])
    return df

def DBSCAN(dataset, eps,MinPts,DistanceMethod = 'euclidean'):
#    Dataset is a mxn matrix, m is number of item and n is the dimension of data
    m,n=dataset.shape
    print(m)
    print(n)
    Visited=np.zeros(m,'int') #stores whether a point is visited or not
    Type=np.zeros(m) #it stores the type of a point 
#   -1 noise, outlier
#    0 border
#    1 corea = np.random.random((50, 4))
# make tree
    A = KDTree(dataset)
# list all pairs within 0.05 of each other in 2-norm
# format: (i, j, v) - i, j are indices, v is distance
    print('Distance matrix...')
    D = A.sparse_distance_matrix(A, eps, p=2.0, output_type='ndarray')
    #DistanceMatrix=np.zeros((m,m))
    PointClusterNumber=np.zeros(m) #stores which point belongs to which cluster
    PointClusterNumberIndex=1 #here each cluster is represented by some indexes starting from 1
    #PointNeighbors=[] #stores neighbouring point of a given point
    #DistanceMatrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(dataset, DistanceMethod))
    #dist=DistanceMetric.get_metric('euclidean')
    #DistanceMatrix=dist.pairwise(dataset)
    #print('distance matrix')
    for i in tqdm(range(m)):
        PointNeighbors=[] #stores neighbouring point of a given point
        PointNeigbors2=[]
        if Visited[i]==0:
            Visited[i]=1
            #PointNeighbors=np.where(DistanceMatrix[i]<eps)
            #PointNeighbors=set_to_list(PointNeighbors)
            #print("Length of Point neighbors are")
            #print(len(PointNeighbors[0]))
            #print((PointNeighbors[0]))
            #for j in range(m):
                #if norm(dataset[i:i+1]-dataset[j:j+1])<=eps:
            #PointNeighbors=np.where(DistanceMatrix[i]<=eps)
                    #PointNeighbors.append(j)
            DU=D[D['i']==i]
            PointNeighbors=DU['j']
            if len(PointNeighbors)<MinPts:
                Type[i]=-1
            else:
               
                PointClusterNumber[i]=PointClusterNumberIndex
                PointNeighbors2=set_to_list(PointNeighbors)
                print(PointNeighbors)
                print(len(PointNeighbors2))
                print("n")
                #print(PointNeighbors2)
                print(PointNeighbors2[0])
                AddNeighbours(dataset,m,D,PointNeighbors2,MinPts,eps,Visited,PointClusterNumber,PointClusterNumberIndex)
                PointClusterNumberIndex=PointClusterNumberIndex+1
                #print(PointClusterNumberIndex)
    return PointClusterNumber 





def AddNeighbours(dataset,m,D,PointNeighbors,MinPts,eps,Visited,PointClusterNumber,PointClusterNumberIndex):
    Neighbors=[]
    Neighbors2=[]
    for i in PointNeighbors:
    #for i in range(len(PointNeighbors)):
            #print(Visited[i])
            #print((Visited[PointNeighbors[i]]))
            #if Visited[PointNeighbors[i]]==0:
            if Visited[i]==0:
                #Visited[PointNeighbors[i]]=1
                Visited[i]=1
                DU=D[D['i']==i]
                Neighbors=DU['j']
                #for j in range(m):
                    #if norm(dataset[i:i+1]-dataset[j:j+1])<=eps:
            #PointNeighbors=np.where(DistanceMatrix[i]<=eps)
                       #Neighbors.append(j)
                #Neighbors=np.where(DistanceMatrix[i]<eps)[0]
                #print(Neighbors)
                Neighbors2=set_to_list(Neighbors)
                #print(Neighbors2[1])
# Neighbors merge with PointNeighbors
            for j in Neighbors:
                try:
                     PointNeighbors.index(j)
                except ValueError:
                     PointNeighbors.append(j)
            if PointClusterNumber[i]==0:
                PointClusterNumber[i]=PointClusterNumberIndex
    return


Data=CreateDataset()
 
 
fig = plt.figure()
 
Epsilon=0.6
MinumumPoints=6
print(Data[1:5])
#test=Data[1:100000]
test= Data
result =DBSCAN(test,Epsilon,MinumumPoints)
      
#plt.show()
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
print(pca_df[filt].head())
#ax = sns.scatterplot(x="pca1", y="pca2", hue = "cluster", data = pca_df[filt])
ax = sns.scatterplot(x="pca1", y="pca2", hue = "cluster", data = pca_df)
plt.show()
