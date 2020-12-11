clf
clc
clear
T = readtable('../data/data_scaled.csv');%read data file
K = table2array(T);%put igscatter(Y(:,1), Y(:,2),idx)nto array
Y = tsne(K,'Algorithm','barneshut','Distance','euclidean',...
    'NumDimensions',2,'Theta', 0.1, 'Verbose',2, 'Perplexity',20);
load('Y-Theta-0.1.mat');
load('S-single-tSNE_p30.mat');
gscatter(Y(:,1),Y(:,2))
numberOfDimensions = 3;%set num of dimensions to 3
[COEFF, SCORE, LATENT, TSQUARED, EXPLAINED] = pca(K); % Perform PCA analysis
reducedDimension = COEFF(:,1:numberOfDimensions);% Use reduced dimentions
load('Y-Theta-0.1.mat');

S = linkage(Y,'ward', 'euclidean', 'savememory','on');%get average distances
cl=5;
W = cluster(S,'maxclust',cl);%Put into 3 clusters
cutoff = median([S(end-5,3) S(end-4,3)]);%Make a cutoff value
histogram(W(W~=2))
dendrogram(S,'ColorThreshold',cutoff)%display as dendrogram
cmap = colormap(parula(cl));

for i=1:cl
    mask = (W==i);
    y_plot = Y(mask,:);
    
    scatter(y_plot(:,1), y_plot(:,2), 2, 'MarkerEdgeAlpha', 0.5)
    hold on
end
hold off 


idx = dbscan(Y,1.2, 20);
gscatter(Y(:,1), Y(:,2),idx)