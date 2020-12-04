from astropy.table import Table, Column
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from seaborn import pairplot
from sklearn.cluster import DBSCAN
from numpy.linalg import norm
from scipy.spatial import distance
# Assuming you have located the table fits file in the same directory 
tab = Table.read('data/DR16Q_v4.fits')
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
data = np.array(list(zip(Fui,Fuz,FuW1,FuW2,Fgz,FgW1,FgW2,FrW1,FrW2,FiW2)))
data_scaled = StandardScaler().fit_transform(data)
df = pd.DataFrame(data=data_scaled, columns=['Fui', 'Fuz', 'FuW1', 'FuW2', 'Fgz', 'FgW1', 'FgW2', 'FrW1', 'FrW2', 'FiW2'])
#df = pd.DataFrame(data=data_scaled, columns=[ 'FW1W2','FuW1','FgW1', 'FrW1', 'FiW1', 'FzW1', 'FuW2', 'FgW2', 'FrW2', 'FiW2', 'FzW2'])
print(df.head())
print("Hello World")
#sns_plot = pairplot(df, corner=True, vars=['Fug','Fur','Fui','Fgr','Fgi'])
#sns_plot.savefig('sdss_scaled_plot.png', dpi=1800)
#ID = tab['SDSS_NAME']
#ID = ID[mask]
#data = np.array(list(zip(Fug, Fur, Fui, Fuz,Fgr,Fgi,Fgz,Fri,Frz,Fiz, FW1W2,FuW1,FgW1, FrW1, FiW1, FzW1, FuW2, FgW2, FrW2, FiW2, FzW2)))
#np.savetxt('data.dat',data)
#np.savetxt('names.txt', ID, fmt='%s')
print("Hello World")
pointId=1
points=[]
large=0
#for j in range(len(df)):
     #count=0
     #large=large+1
     #for i in range(len(df)):
        #Euclidian distance using L2 Norm
            #x=distance.euclidean(df[i:i+1],df[j:j+1])
                #points.append(i)
            #print(x)
            #count=count+1
            #print(large)
            #print(count)
            #print(j,i)
db=DBSCAN(eps=1.2,min_samples=11)
test=df[0:110000]
model=db.fit(test)
label=model.labels_
print(label[1:10])
from sklearn.decomposition import PCA
import seaborn as sns
data_with_clust = pd.concat([test,pd.DataFrame({'cluster':label})], axis = 1)
#print(data_with_clust.head())
outlier=data_with_clust[data_with_clust['cluster']!=0]
#print(outlier.head())
pca=PCA(n_components=2)
principal_comp=pca.fit_transform(test)
pca_df = pd.DataFrame(data = principal_comp, columns =['pca1','pca2'])
pca_df.head()
pca_df = pd.concat([pca_df,pd.DataFrame({'cluster':label})], axis = 1)
print(pca_df.head())
plt.figure(figsize=(10,10))
filt=(pca_df['cluster']<1)
#print(pca_df[filt].head())
#ax = sns.scatterplot(x="pca1", y="pca2", hue = "cluster", data = pca_df[filt])
ax = sns.scatterplot(x="pca1", y="pca2", hue = "cluster", data = pca_df)
plt.show()
print(len(outlier))