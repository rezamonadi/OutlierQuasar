clf
% clc
% clear
T = readtable('../data/data_scaled.csv');%read data file
K = table2array(T);%put into array
Y = tsne(K,:),'Algorithm','barneshut','Distance','euclidean',...
    'NumDimensions',2,'Theta', 0.1, 'Verbose',2, 'Perplexity',20);
gscatter(Y(:,1),Y(:,2))
% numberOfDimensions = 3;%set num of dimensions to 3
% [COEFF, SCORE, LATENT, TSQUARED, EXPLAINED] = pca(K); % Perform PCA analysis
% reducedDimension = COEFF(:,1:numberOfDimensions);% Use reduced dimentions
% load('Y-Theta-0.1.mat');
% 
% S = linkage(Y,'ward', 'euclidean', 'savememory','on');%get average distances
% cl=1;
% W = cluster(S,'maxclust',cl);%Put into 3 clusters
% cutoff = median([S(end-5,3) S(end-4,3)]);%Make a cutoff value
% % histogram(W)
% % dendrogram(S,'ColorThreshold',cutoff)%display as dendrogram
% cmap = colormap(parula(cl));
% for i=1:cl
%     mask = (W==i);
%     y_plot = Y(mask,:);
%     scatter(y_plot(:,1), y_plot(:,2), 'MarkerEdgeAlpha', 0.1)
%     hold on
% end
% hold off 
