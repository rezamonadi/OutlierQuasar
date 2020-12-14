# OutlierQuasar

Here we explain each directory's functionality. 
## ```data/..```
#### Downloading Quasar Catalog
- For Downloading Quasar Catalog, go to the `data` directory and run
 this command in your linux terminal to get ```DR16Q_v4.fits```:
> ```bash dl-dr6q.sh```
#### Building the main data-set 
- In the same directory run ```preprocessing.ipynb``` to get the scaled and raw
data set. After running this script you should shave ```data_0.csv``` (raw features created from catalog) and ```data_scaled.csv``` (scaled features to be used for further analysis) and a reduced version of the original catalog (```reduced_dr16q.fits```)
- ```preprocessing.ipynb``` also makes pair-plot, correlation matrix plot, and box plot. 
#### Downloading Spectra
- This script also makes 
- Run ```build_list.ipynb``` to get the directories need for downloading quasar spectra form SDSS data base.
- Run: 
>```bash dl-spectra.sh``` 

in terminal which takes a while and needs ~40GB of storage.

#### Dimensionality reduction 
- Run ```tSNE_dim_reduc.m``` in MATLAB to get mapped data-set from 7D to 2D with tSNE algorithm stored at ```Y.mat```. 

## ```dbscan/..```


#### ```AgglomerativeClustering/..```
- Run ```agg.m```  in MATLAB. It gives the denrograms and a plot showing clusters found by Agglomerative clustering on the tSNE mapped dataset.

## ```IsolationForest/..```
- Run ```iForest-dist-plot.m``` to get the distribution of iForest scores in each 
cluster labeled by ```dbscan```
- Scores would be stored in ```Outliers.txt```. 
- Run ```iForest-dist-plot.m``` in MATLAB to get Isolation Forest score distribution 
for different clusters found by DBSCAN. 

## ```MedSpec/..```
- Running ```MedSpec.ipynb``` in this directory gives median spectra for each cluster
labeled by ```DBSCAN```. 
- ```stacker.py``` is a module that ```MedSpec.ipynb``` uses for stacking quasars spectra for calculating median spectrum.
-```line_db.py``` is a module that ```MedSpec.ipynb``` utilizes to plot vertical dashed lines
in the position specific emission lines in the spectrum. 

## ```PropertyInvestigation/..```
- Running ```PhysicalProperties.ipynb``` gives the color distribution plots for each cluster.

