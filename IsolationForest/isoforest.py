from astropy.io import ascii
import matplotlib
import matplotlib.pyplot as plt   
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import pandas as pd
import numpy as np
import random
from pandas.plotting import scatter_matrix
import time

"""
Arman Irani
University of California, Riverside
CS 235 
Isolation Forest implementation for Outlier Detection in Quasars

References Used:

    Isolation Forest Logic & Pseudocode:
        {https://ieeexplore.ieee.org/document/4781136}

    Decision Tree implementation using Python Dictionary data structure and Path Length evaluation:
        {https://github.com/karangautam/Decision-Tree-From-Scratch}
    
    Isolation Forest C_score
        {https://github.com/mgckind/iso_forest}

    Extended Isolation Forest (not implemented but architecture was used as reference)
        {https://github.com/sahandha/eif}
"""



def split_data(data):
    """
    This function first selects a random column or "attribute" for sub-sampling purposes. 
    The binary trees will therefore will all have randomly generated partitions. 

    Then the function accesses the maximum and minimum value from the 'randomly' selected feature.
    Then to get our randomly selected value from within the attributes value range, 
    calculate the difference betweent the max & min and multiply by  randomly selected float 
    between 0 -> 1. Add by the min_split_value in edge case of 'np.random.random() == 0. 

    Returns random column and random split value
    """
    rand_attribute = random.choice(data.columns)
    min_split_value = data[rand_attribute].min()                          # Min value of attribute 
    max_split_value = data[rand_attribute].max()                          # Max value of attribute
    
    max_min_diff = max_split_value-min_split_value                                 # Min/Max Range
    return rand_attribute, max_min_diff*np.random.random() + min_split_value       # Randomly selected split point from sequence of min/max

def partition_data(data, column_part, split_value):
    """
    Partitions our data based on randomly selected value we derived in 'select_value()' in section 'above' and 'below' our split value.


    Parameters
    ----------
    data : Pandas Dataframe
        dataset
    
    column_part: float
        column chosen to partition on.
    
    split_value: float
        random value to partition data on.

    Returns
    --------
    data_greater: Pandas Dataframe
        data that is grater than split value

    data_less: Pandas Dataframe
        data that is less than split value
        

    """
    data_right = data[data[column_part] > split_value]          # Selecting dataframe that is greater than the randomly selected split value
    data_left = data[data[column_part] <= split_value]          # Selecting dataframe that is less or equal to than the randomly selected split value
   
    return data_right, data_left

def make_decision_tree(data, currHeight=0,heightLimit=20):
    """
    This function recursively constructs the decision tree. 
    The function will return when the decision tree reaches an external node (branch with no elements left) or
    when the height limit specified is reached.

    Follows assumption of dataframe data structure.

     Parameters
    ----------
    data : Pandas Dataframe
        Dataset

    currHeight: int
        Variable to keep track of current depth of decision tree
    
    heightLimit: int
        Variable that sets the maximum depth of the tree
        
    Returns
    --------
    sub_tree: dict
        Dictionary representation of decision tree containing attribute and branched on <= and nodes.
    
    """
    if(currHeight==heightLimit or len(data) <=1):                                                     # If we have reached the maximum height limit, or we have isolated the last element, return external node.
        label_column = data.values[:, -1]                                                             # Takes all rows of last column in data
        sorted_unique_values, unique_classes_count = np.unique(label_column, return_counts=True)      # Returns the counts and elements of the sorted unique elements in the retrieved 'label_column' 
        max_index = unique_classes_count.argmax()                                                     # Returns index of sorted unique value that occurs the most 
        return sorted_unique_values[max_index]                                                        # Returns unique value that occurs the most often
        
    else:
        split_attribute, split_value = split_data(data)                                 # Select random feature & Select random value to split on between the max and min values of 'split_attribute' in data                                   
        data_right, data_left = partition_data(data,split_attribute,split_value)        # Filter data with partition

        # Instantiate Subtree
        node = "{} <= {}".format(split_attribute,split_value)                           # Set up our node construction to add to our faux decision tree (dict)
        sub_tree = {node: []}                                                           # dict construction

        # Recursive Logic
        currHeight += 1                                                                 # Variable to keep track of current height in tree
        left_sub_tree = make_decision_tree(data_left,currHeight,heightLimit=heightLimit)
        right_sub_tree = make_decision_tree(data_right,currHeight,heightLimit=heightLimit)

        if(left_sub_tree == right_sub_tree):
            sub_tree = left_sub_tree
        
        # Adding nodes to decision tree
        else:
            sub_tree[node].append(left_sub_tree)  
            sub_tree[node].append(right_sub_tree)

        """
                    split_attribute
                        /   \ 
                 (<= split_value < )
                    /            \ 
                ______           _______ 
               |_left_|         |_right_|
            
        """
        return sub_tree

def isolationForest(data,numTrees,heightLimit, samplingSize):
    """
    This function constructs the 'forest' or ensemble of decision trees. 
    Select a subsample (size specified by subsampling size) of the data to pass to the tree
    While in the range of numTrees specified in the program, construct a decision tree for each sample of data.

    Follows assumption of dataframe data structure.

     Parameters
    ----------
    data : Pandas Dataframe
        Dataset

    numTrees: int
        Number of trees to construct
    
    heightLimit: int
        Variable that sets the maximum depth of the tree

    samplingSize: int
        Sample size of data to take
        
    Returns
    --------
    forest_ensemble: list
        Decision trees(dict)
    
    """

    forest_ensemble=list()
    for i in range(numTrees):
        data = data.sample(samplingSize)                                # Random selection of attributes
        decision = make_decision_tree(data,heightLimit=heightLimit)     # Tree creation
        forest_ensemble.append(decision)                                # Adding tree to ensemble
        
    return forest_ensemble                                              # Return an ensemble of trees



def get_path_length(data, decision_tree,path=0):
    """
    This function calculates the path length for . 
    Select a subsample (size specified by subsampling size) of the data to pass to the tree
    While in the range of numTrees specified in the program, construct a decision tree for each sample of data.

    Follows assumption of dataframe data structure.

     Parameters
    ----------
    data : Pandas Dataframe
        Dataset

    decision_tree: dict
        Decision tree implementation for the value
    
    path: int
        Tree depth counter
        
    Returns
    --------
    path: int
        Depth traversed in decision tree to reach value
    
    """
    node = list(decision_tree.keys())[0]                                       # Root Node
    feature, comparison_op, split_value =  node.split()                        
    if data[feature].values <= float(split_value):                             # Choose branch to descend 
        attr = decision_tree[node][0]
    else:
        attr = decision_tree[node][1]

    path += 1                                                                  # Incrementing path value each recursive visit    
    if(not isinstance(attr,dict)):                                             # Equivalent to if T is an external node (leaf)
        return path                                                            # Base case of recursion
    
    else:
        residual_tree = attr
        return get_path_length(data,residual_tree,path=path)                 # Recursive logic to continue down path

    return path                                                                # Returns path length
    
def evaluate_instance(instance,forest):
    """
    This function evaluates each instance's path length. 
    
    Follows assumption of dataframe data structure.

     Parameters
    ----------
    instance : Pandas Dataframe

    forest: List of Dicts
        Ensemble of decision trees
        
    Returns
    --------
    paths: list
        path lengths for instance
    
    """
    paths = list()
    for decision_tree in forest:
        paths.append(get_path_length(instance,decision_tree))            # Evaluation is based off of path length. Need this to calculate anomaly score
        
    return paths
    
def c_score(n) :
    """
    If the number of data points is equal to 2 then return 1.0
    If the number is < 2.0 return 0
    Else return the average path length of unsuccesful search in a binary search tree given n points
    
    Parameters
    ----------
    n : int
        Number of data points for the decision tree.
    Returns
    -------
    float
        Average path length of unsuccesful search in a decision tree
        
    """
    if(n==2):
        return 1.0
    if(n>2):
        return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.0)/(n*1.0))
    else:
        return 0.0

def an_score(data_point,forest,n):
    """
    Anomaly Score calculation

    Parameters
    ----------
    data_point: float

    forest: List of Dicts
        Ensemble of decision trees
    
    n: int
        Number of data points in the decision tree
    
    Returns
    -------
    float
        Anomaly score between 0.0 and 1.0
    """

    
    return 2**-((np.mean(evaluate_instance(data_point,forest)))/c_score(n))

def path_length(data_point,forest,n):
    """
        Returns average path length for each instance

        For plotting purposes only.
    """
     
    return np.mean(evaluate_instance(data_point,forest))

import pandas as pd
if __name__ == "__main__":
    df = pd.read_csv("../data/data_scaled.csv")                # Using astropy.io to read in large data file
    
    start_time = time.time()    
    iForest = isolationForest(df,20, 150,256)                                                       # Creating the forest

    print("Time Taken: ", time.time() - start_time)
    
    start_time = time.time()
    an= list()                                                                                      # List to hold anomaly scores for every instance
    for i in range(df.shape[0]):
        an.append(an_score(df.iloc[[i]],iForest,256))                                          
    print("Time Taken: ", time.time() - start_time)


    ########################### Path Length v. Anamoly Score Plotting Logic ###########################
    start_time = time.time()
    path = list()
    for i in range(df.shape[0]):
        path.append(path_length(df.iloc[[i]],iForest,256))
    print("Time Taken: ", time.time() - start_time)


    plt.hexbin(path,an)
    plt.colorbar(label='Count in bin')
    plt.xlabel('Path Length')
    plt.ylabel('Anomaly Score')
    plt.show()
    ########################### Path Length v. Anamoly Score Plotting Logic ###########################

    # Output Number of outliers found for debugging purposes
    count = 0
    for i in range(df.shape[0]):
        if an[i] > 0.70:
            count += 1
    print(count)


    ########################### Writing Outliers to File ###########################
    file1 = open("Outliers.txt","a")
    outlier_index = []
    for i in range (df.shape[0]):
        if an[i] > 0.70:
            outlier_index.append(i)
            file1.write(str(i)+","+str(an[i])+"\n")
    file1.close()
    ########################### Writing Outliers to File ###########################
