T = readtable('data.dat');%read data file
K = table2array(T);%put into array
numberOfDimensions = 3;%set num of dimensions to 3
[COEFF, SCORE, LATENT, TSQUARED, EXPLAINED] = pca(K); % Perform PCA analysis
reducedDimension = COEFF(:,1:numberOfDimensions);% Use reduced dimentions
S = linkage(reducedDimension,'average','chebychev');%get average distances
W = cluster(S,'maxclust',3);%Put into 3 clusters
cutoff = median([S(end-2,3) S(end-1,3)]);%Make a cutoff value
dendrogram(S,'ColorThreshold',cutoff)%display as dendrogram% clf
clc
clear
T = readtable('../data/data_scaled.csv');%read data file
K = table2array(T);%put igscatter(Y(:,1), Y(:,2),idx)nto array

load('../data/Y.mat');% loading tSNE maped data
S = linkage(Y_20,'single', 'euclidean', 'savememory','off');%get linkage 
cl=12; % number of clusters
W = cluster(S,'maxclust',cl);%Put into 12 clusters
cutoff = median([S(end-2,3) S(end-1,3)]);%Make a cutoff value
dendrogram(S,'ColorThreshold',cutoff)%display as dendrogram
cmap = colormap(jet(cl)); % making color map
% % plotting different clusters with different colors
for i=1:cl
    mask = (W==i);
    y_plot = Y_20(mask,:);
    
    scatter(y_plot(:,1), y_plot(:,2), 2, cmap(i,:), 'MarkerEdgeAlpha', 0.5)
    hold on
end

