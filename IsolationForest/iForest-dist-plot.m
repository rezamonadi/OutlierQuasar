load('../dbscan/dbscan-idx-eps-1.5-mpt-10.mat');

iForest=load('Outliers.txt');
fig = figure();
clf();
hold on
% for i=[1,6,7,8,9]
for i=1:5
    
    mask=(idx==i);
    histogram(iForest(mask), 'Normalization', 'probability');
    
end