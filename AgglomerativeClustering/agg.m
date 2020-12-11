clc
clear
T = readtable('../data/data_scaled.csv');%read data file
K = table2array(T);%put into array
Y_10 = tsne(K,'Algorithm','barneshut','Distance','euclidean', 'NumDimensions',2,...
    'Verbose',2, 'Theta',0.1, 'Perplexity',10);
Y_20 = tsne(K,'Algorithm','barneshut','Distance','euclidean', 'NumDimensions',2,...
    'Verbose',2, 'Theta',0.1, 'Perplexity',20);
Y_30 = tsne(K,'Algorithm','barneshut','Distance','euclidean', 'NumDimensions',2,...
    'Verbose',2, 'Theta',0.1, 'Perplexity',30);
Y_5 = tsne(K,'Algorithm','barneshut','Distance','euclidean', 'NumDimensions',2,...
    'Verbose',2, 'Theta',0.1, 'Perplexity',5);
%                                                       

% numberOfDimensions = 3;%set num of dimensions to 3
% [COEFF, SCORE, LATENT, TSQUARED, EXPLAINED] = pca(K); % Perform PCA analysis
% reducedDimension = COEFF(:,1:numberOfDimensions);% Use reduced dimentions
% load('Y-Theta-0.1.mat');

% Single Link
S = linkage(K,'single', 'euclidean');%get average distances
save('S-single-FullData.mat', 'S')

S = linkage(Y_5,'single', 'euclidean');%get average distances
save('S-single-tSNE_p5.mat', 'S')

S = linkage(Y_10,'single', 'euclidean');%get average distances
save('S-single-tSNE_p10.mat', 'S');
S = linkage(Y_20,'single', 'euclidean');%get average distances
save('S-single-tSNE_p20.mat', 'S');
S = linkage(Y_30,'single', 'euclidean');%get average distances
save('S-single-tSNE_p30.mat', 'S');

% Complete linkage
S = linkage(K,'complete', 'euclidean');%get average distances
save('S-complete-FullData.mat', 'S')

S = linkage(Y_5,'complete', 'euclidean');%get average distances
save('S-complete-tSNE_p5.mat', 'S')

S = linkage(Y_10,'complete', 'euclidean');%get average distances
save('S-complete-tSNE_p10.mat', 'S')
S = linkage(Y_20,'complete', 'euclidean');%get average distances
save('S-complete-tSNE_p20.mat', 'S')
S = linkage(Y_30,'complete', 'euclidean');%get average distances
save('S-complete-tSNE_p30.mat', 'S')


% S = linkage(Y,'ward', 'euclidean');%get average distances
% save('S-ward-tSNE.mat', 'S')
% 
% S = linkage(K,'ward', 'euclidean');%get average distances
% save('S-ward-FullData.mat', 'S')
%W = cluster(S,'maxclust',20);%Put into 3 clusters
%cutoff = median([S(end-5,3) S(end-4,3)]);%Make a cutoff value
%histogram(W)
%dendrogram(S,'ColorThreshold',cutoff)%display as dendrogram
% % c=['r','b','g','m','y'];
% for i=1:5
%     mask = (W==i);
%     gscatter(y(mask,1), y(mask,2), 'Color', c(i))
%     hold on
% end
% hold off
