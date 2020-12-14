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

    Decision Tree construction using Python Dictionary data structure and Path Length evaluation:
        {https://github.com/karangautam/Decision-Tree-From-Scratch}
    


"""



def select_value(data,feature):
    """
    First access the maximum and minimum value from the 'randomly' selected feature.
    Then to get our randomly selected value from within the attributes value range, 
    calculate the difference betweent the max & min and multiply by  randomly selected float 
    between 0 -> 1. Add by the min_split_value in edge case of 'np.random.random() == 0. 

    Parameters
    ----------
    data : Pandas Dataframe
        dataset

    feature: float
        Feature returned from get_random_attribute
    Returns
    --------
    float
        random value between [Feature.Min, Feature.Max]
    """
    min_split_value = data[feature].min()                          # Min value of attribute 
    max_split_value = data[feature].max()                          # Max value of attribute
    
    max_min_diff = max_split_value-min_split_value                 # Min/Max Range
    return max_min_diff*np.random.random() + min_split_value       # Randomly selected split point from sequence of min/max

    
def get_random_attribute(data):
    """
    This function selects a random column or "attribute" for sub-sampling purposes. 
    The binary trees will therefore will all have randomly generated partitions. 

    Follows assumption of dataframe data structure architecture.

     Parameters
    ----------
    data : Pandas Dataframe
        dataset
        
    Returns
    --------
    float:
        random column for columns in dataset
    
    """
    return random.choice(data.columns)

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
    data_greater = data[data[column_part] > split_value]        # Selecting dataframe that is greater than the randomly selected split value
    data_less = data[data[column_part] <= split_value]          # Selecting dataframe that is less or equal to than the randomly selected split value
   
    return data_greater, data_less

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
        ex_node_value = sorted_unique_values[max_index]                                               # Returns unique value that occurs the most often
        
        return ex_node_value

    else:
        currHeight += 1                                                                 # Variable to keep track of current height in tree
        split_attribute = get_random_attribute(data)                                    # Select random feature
        split_value = select_value(data,split_attribute)                                # Select random value to split on between the max and min values of 'split_attribute' in data
        data_above, data_below = partition_data(data,split_attribute,split_value)       # Filter data with partition

        # Instantiate Subtree
        node = "{} <= {}".format(split_attribute,split_value)                           # Set up our node construction to add to our faux decision tree (dict)
        sub_tree = {node: []}                                                           # dict construction


        # Recursive Logic
        left = make_decision_tree(data_below,currHeight,heightLimit=heightLimit)
        right = make_decision_tree(data_above,currHeight,heightLimit=heightLimit)

        # End of split
        if(left == right):
            sub_tree = left
        
        # Adding nodes to decision tree
        else:
            sub_tree[node].append(left)  
            sub_tree[node].append(right)

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
        data = data.sample(samplingSize)                             # Random selection of attributes
        decision = make_decision_tree(data,heightLimit=numTrees)     # Tree creation
        forest_ensemble.append(decision)                             # Adding tree to ensemble
        
    return forest_ensemble                                           # Return an ensemble of trees



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
    path += 1                                                                  # Incrementing path value each recursive visit
    node = list(decision_tree.keys())[0]                                       # Root Node
    feature, comparison_op, split_value =  node.split()                        
    if data[feature].values <= float(split_value):                             # Choose branch to descend 
        attr = decision_tree[node][0]
    else:
        attr = decision_tree[node][1]

    
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
    Average path length of unsuccesful search in a binary search tree given n points
    
    Parameters
    ----------
    n : int
        Number of data points for the decision tree.
    Returns
    -------
    float
        Average path length of unsuccesful search in a decision tree
        
    """
    return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))

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

    # Mean depth for an instance
    E = np.mean(evaluate_instance(data_point,forest))
    
    c = c_score(n)
    
    return 2**-(E/c)

def path_length(data_point,forest,n):
    """
        Returns average path length for each instance

        For plotting purposes only.
    """
    # Mean depth for an instance
    E = np.mean(evaluate_instance(data_point,forest))
     
    return E


if __name__ == "__main__":
    data = ascii.read("C:/Users/arman/Workspace/OutlierQuasar/data.dat",guess=False)                # Using astropy.io to read in large data file
    df = data.to_pandas()                                                                           # Converting to Pandas to use for this implementation
    
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
