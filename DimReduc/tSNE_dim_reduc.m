T = readtable('../data/data_scaled.csv');%read data file
K = table2array(T);%put igscatter(Y(:,1), Y(:,2),idx)nto array
Y_20 = tsne(K,'Algorithm','barneshut','Distance','euclidean',...
    'NumDimensions',2,'Theta', 0.1, 'Verbose',2, 'Perplexity',20);
Y_25 = tsne(K,'Algorithm','barneshut','Distance','euclidean',...
    'NumDimensions',2,'Theta', 0.1, 'Verbose',2, 'Perplexity',25);
Y_30 = tsne(K,'Algorithm','barneshut','Distance','euclidean',...
    'NumDimensions',2,'Theta', 0.1, 'Verbose',2, 'Perplexity',30);

save('Y_20.mat',Y_20);
save('Y_25.mat',Y_25);
save('Y_30.mat',Y_30);