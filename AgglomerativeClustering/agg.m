clc
T = readtable('../data/data_scaled.csv');%read data file
K = table2array(T);%put into array
Y = tsne(K,'Algorithm','barneshut','Distance','euclidean', 'NumDimensions',2,...
                                                      'Theta', 0.1, 'Verbose',2);

% numberOfDimensions = 3;%set num of dimensions to 3
% [COEFF, SCORE, LATENT, TSQUARED, EXPLAINED] = pca(K); % Perform PCA analysis
% reducedDimension = COEFF(:,1:numberOfDimensions);% Use reduced dimentions
% S = linkage(K,'average','chebychev');%get average distances
% W = cluster(S,'maxclust',3);%Put into 3 clusters
% cutoff = median([S(end-2,3) S(end-1,3)]);%Make a cutoff value
% dendrogram(S,'ColorThreshold',cutoff)%display as dendrogram
