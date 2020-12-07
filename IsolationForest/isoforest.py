from astropy.io import ascii
import matplotlib
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation
import matplotlib.pyplot as plt   
import pandas as pd
import numpy as np
from dask.dataframe import from_pandas
import dask.bag as db
import dask_ml.datasets
import dask_ml.cluster
import dask.array as da
from dask.distributed import Client
import joblib
import random
from pandas.plotting import scatter_matrix
import seaborn as sns


#Selecting a random column for sub-sampling
def select_feaure(data):
    return random.choice(data.columns)

def select_value(data,feature):
    min_split = data[feature].min()     #Min value of attribute 
    max_split = data[feature].max()     #Max value of attribute
    
    return (max_split-min_split)*np.random.random()+min_split       #Randomly selected split point from sequence
    

def split(data,split_column, split_value):
    data_above = data[data[split_column] > split_value]
    data_below = data[data[split_column] <= split_value]
   
    return data_above, data_below

def classify_data(data):
    
    label_column = data.values[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
    
    return classification

def isolationTree(data, curHeight=0,heightLimit=50):
    if(curHeight>=heightLimit or len(data) <=1):
        classification = classify_data(data)
        
        return classification

    else:
        curHeight += 1

        split_attribute = select_feaure(data)       #Select feature
        split_value = select_value(data,split_attribute)   # Select Value to split on
        data_above, data_below = split(data,split_attribute,split_value)

        # Instantiate Subtree
        question = "{} <= {}".format(split_attribute,split_value)
        sub_tree = {question: []}

        # Recursive Logic
        below_answer = isolationTree(data_below,curHeight,heightLimit=heightLimit)
        above_answer = isolationTree(data_above,curHeight,heightLimit=heightLimit)

        if(below_answer == above_answer):
            sub_tree = below_answer

        else:
            sub_tree[question].append(below_answer)
            sub_tree[question].append(above_answer)

        return sub_tree

def isolationForest(data,numTrees,heightLimit, samplingSize):
    forest=[]

    for i in range(numTrees):
        if(samplingSize <=1):
            data = data.sample(frac=samplingSize)

        else:
            data = data.sample(samplingSize)

        tree = isolationTree(data,heightLimit=numTrees)
        forest.append(tree)
        
    return forest


def pathLength(data, tree,path=0):
    path = path + 1
    question = list(tree.keys())[0]
    feature_name, comparison_operator,value =  question.split()

    if data[feature_name].values <= float(value):
        answer = tree[question][0]
    else:
        answer = tree[question][1]

    if(not isinstance(answer,dict)):
        return path
    
    else:
        residual_tree = answer
        return pathLength(data,residual_tree,path=path)

    return path
    
def evaluate_instance(instance,forest):
    paths = []
    for tree in forest:
        paths.append(pathLength(instance,tree))
        
    return paths
    
def c_factor(n) :
    """
    Average path length of unsuccesful search in a binary search     tree given n points
    
    Parameters
    ----------
    n : int
        Number of data points for the BST.
    Returns
    -------
    float
        Average path length of unsuccesful search in a BST
        
    """
    return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))
def anomaly_score(data_point,forest,n):
    '''
    Anomaly Score
    
    Returns
    -------
    ~0.5 -- sample does not have any distinct anomaly
    ~0 -- Normal Instance
    ~1 -- An anomaly
    '''
    # Mean depth for an instance
    E = np.mean(evaluate_instance(data_point,forest))
    
    c = c_factor(n)
    
    return 2**-(E/c)




if __name__ == "__main__":
    data = ascii.read("C:/Users/arman/Workspace/OutlierQuasar/data.dat",guess=False)
    print("here")
    df = data.to_pandas()
    sns.set_theme(style="ticks")

    # sns_plot = sns.pairplot(df.sample(1000),kind="scatter",diag_kind="kde")
    # fig = sns_plot.fig
    # fig.savefig("output.png")
    #scatter_matrix(df.sample(1000), alpha=0.2, figsize=(40, 40), diagonal='kde',cmap="#415BCB")
    

    iForest = isolationForest(df,20, 150,512)
    # outlier = evaluate_instance(X.head(1), iForest)
    # normal = evaluate_instance(X.sample(1), iForest)
    an= []
    for i in range(df.shape[0]):
        an.append(anomaly_score(df.iloc[[i]],iForest,512))
  

    # print(an)
    # Y=[[] for i in range(len(df))]
    file1 = open("Outliers.txt","a")
    outlier_index = []
    for i in range (df.shape[0]):
        if an[i] > 0.65:
            outlier_index.append(i)
            file1.write(str(i)+","+str(an[i])+"\n")
            # x_val = df.iloc[i,0]
            # y_val = df.iloc[i,1]
            # Y[i] =[x_val,y_val]
    file1.close()

    
    # Y = pd.DataFrame(Y, columns=["feat1","feat2"])
    # Y = Y.dropna(axis=0)
    # ax = df.plot.scatter(x='feat1',y='feat2',color='DarkBlue',label = 'Outliers')
    # Y.plot.scatter(x='feat1',y='feat2',color='DarkGreen', label='Group 2',ax=ax)
    # plt.figure(figsize=(7,7))
    # plt.plot(x,y,'bo')
# Evaluate one instance
