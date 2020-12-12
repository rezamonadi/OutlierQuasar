% clf
% clc
% % clear
% % % T = readtable('../data/data_scaled.csv');%read data file
% % % K = table2array(T);%put igscatter(Y(:,1), Y(:,2),idx)nto array
% % % Y = tsne(K,'Algorithm','barneshut','Distance','euclidean',...
% % %     'NumDimensions',2,'Theta', 0.1, 'Verbose',2, 'Perplexity',20);
% load('../data/Y.mat');
% % gscatter(Y_20(:,1),Y_20(:,2))
% % 
% % 
% % S = linkage(Y_20,'single', 'euclidean', 'savememory','off');%get average distances
% load('S-complete-tSNE_p20.mat');
% % 
% % 
cl=9;
% W = cluster(S,'maxclust',cl);%Put into 3 clusters
% cutoff = median([S(end-5,3) S(end-4,3)]);%Make a cutoff value
% % % % histogram(W(W~=2))
% dendrogram(S,'ColorThreshold',cutoff)%display as dendrogram
cmap = colormap(jet(cl));
% % 
% for i=1:cl
%     mask = (W==i);
%     y_plot = Y_20(mask,:);
%     
%     scatter(y_plot(:,1), y_plot(:,2), 2, cmap(i,:), 'MarkerEdgeAlpha', 0.5)
%     hold on
% end


% load('dbscan-idx-eps-1.5-mpt-10.mat');
% color = ['k', 'g', 'b', 'm', 'C4', 'C5', 'C6', 'C7', 'black', 'blue', 'C8', 'gray']

% idx = dbscan(Y_20,1.2, 10);
load('dbscan-idx-eps-1.5-mpt-10.mat');
iForest=load('../IsolationForest/iForestScores.dat');
fig = figure();
clf();
hold on
for i=[1,6,7,8,9]
    
    mask=(idx==i);
    histogram(iForest(mask), 'Normalization', 'probability');
    
end
fidpdf = sprintf('iForest-dist-%d-%d.pdf',1,9);

% legend('cl-1', 'cl-2', 'cl-3', 'cl-4', 'cl-5') 
legend('cl-1', 'cl-6', 'cl-7', 'cl-8', 'cl-9')

exportgraphics(fig, fidpdf,'ContentType','vector')



%     gscatter(Y_20(:,1), Y_20(:,2),idx)
% save('dbscan-idx-eps-1.5-mpt-10','idx');


