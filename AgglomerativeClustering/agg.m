clc
clear
T = readtable('../data/data_scaled.csv');%read data file
K = table2array(T);%put into array
% Y = tsne(K,'Algorithm','barneshut','Distance','euclidean', 'NumDimensions',2,...
%                                                       'Theta', 0.1, 'Verbose',2);

% numberOfDimensions = 3;%set num of dimensions to 3
% [COEFF, SCORE, LATENT, TSQUARED, EXPLAINED] = pca(K); % Perform PCA analysis
% reducedDimension = COEFF(:,1:numberOfDimensions);% Use reduced dimentions
% load('Y-Theta-0.1.mat');

S = linkage(K,'ward', 'euclidean', 'savememory','on');%get average distances
W = cluster(S,'maxclust',20);%Put into 3 clusters
cutoff = median([S(end-5,3) S(end-4,3)]);%Make a cutoff value
histogram(W)
dendrogram(S,'ColorThreshold',cutoff)%display as dendrogram
% % c=['r','b','g','m','y'];
% for i=1:5
%     mask = (W==i);
%     gscatter(y(mask,1), y(mask,2), 'Color', c(i))
%     hold on
% end
% hold off