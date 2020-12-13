# OutlierQuasar

## ```data/..```
### Downloading Quasar Catalog
- For Downloading Quasar Catalog, go to the `data` directory and run
 this command in your linux terminal to get ```DR16Q_v4.fits```:
> ```bash dl-dr6q.sh```
### Building the main data-set 
- In the same directory run ```preprocessing.ipynb``` to get the scaled and raw
data set. After running this script you should shave ```data_0.csv``` (raw features created from catalog) and ```data_scaled.csv``` (scaled features to be used for further analysis) and a reduced version of the original catalog (```reduced_dr16q.fits```)
- ```preprocessing.ipynb``` also makes pair-plot, correlation matrix plot, and box plot. 
### Downloading Spectra
- This script also makes 
- Run ```build_list.ipynb``` to get the directories need for downloading quasar spectra form SDSS data base.
- Run: 
>```bash dl-spectra.sh``` 

in terminal which takes a while and needs ~40GB of storage.

### Dimensionality reduction 
- Run ```tSNE_dim_reduc.m``` in MATLAB to get mapped data-set from 7D to 2D with tSNE algorithm stored at ```Y.mat```. 

## ```dbscan/..```

## ```AgglomerativeClustering/..```
- Run ```agg.m```  in MATLAB 

## ```IsolationForest/..```
- Run ```iForest-dist-plot.m``` to get the distribution of iForest scores in each cluster labe;ed by ```dbscan```. 

## ```MedSpec/..```
- Running ```MedSpec.ipynb``` in this directory gives median spectra for each cluster
labeled by ```DBSCAN```. 
