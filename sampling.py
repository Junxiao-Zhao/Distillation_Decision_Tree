import numpy as np
import pandas as pd
from itertools import product

def sample_new(ratio:np.ndarray, Z_matrix:np.ndarray):
    '''
        Sampling new covariates according to ratio.

        Args:
            ratio (ndarray):
                An array contains the proportions.
            
            Z_matrix (ndarray):
                A n*q matrix of new covariates.

        Returns:
            samples (ndarray):
                A n*q matrix of new covariates after PCA.
    '''
    
    #calculate how many values should be selected in each column
    num_rows = Z_matrix.shape[0]
    ratio *= num_rows
    ratio = np.append(ratio, np.ones([Z_matrix.shape[1]-len(ratio)]))

    samples = []

    #random select
    for i, each in enumerate(ratio):
        each = int(each)
        selected_values = list(np.random.choice(Z_matrix[:,i], each))  #random sample for each attribute
        selected_values *= int(np.ceil(num_rows / each)) #mutiple and cut the samples to the certain length
        samples.append(selected_values[:num_rows])

    samples = np.array(samples).T

    return samples

def sample(each_num:int, attributes:dict):
    '''
    Sampling data to the specified size.

    Args:
        each_num (int):
            The number of times sampling each attribute.
        
        attributes: (dict):
            The name and range of all the attributes:
                Continuous: {attr1: {"min": x, "max": y}, attr2: ...};
                Categorical: {attr1: [c1, c2, ...], attr2: ...}.
    
    Returns:
        samples (DataFrame):
            The random sample without labels.
    '''

    samples = []
    each_num = int(each_num)
    
    for attr, values in attributes.items():
        #categorical
        if isinstance(values, list):
            samples.append(np.random.choice(values, each_num))
        #continuous
        else:
            samples.append(np.random.uniform(values['min'], values['max'], each_num))
    
    #use cartesian product to generate the samples
    #samples = pd.DataFrame(product(*samples), columns=attrs)
    samples = np.array(samples).T
    samples = pd.DataFrame(samples, columns=list(attributes.keys()))

    return samples
    
def predict_label(samples:pd.DataFrame, teacher_predict, *args):
    '''
    Predict the labels for the sampled data.

    Args:
        samples (DataFrame):
            The random sample without labels.
        
        teacher:
            The trained model for prediction.
        
        *args:
            The parameters for the teacher_predict method.

    Returns:
        samples (DataFrame):
            The random sample with labels.
    '''

    #predict the samples using the teacher model
    labels = teacher_predict(samples, *args)
    #insert the labels into the samples
    samples.insert(samples.shape[1], "label", labels)

    return samples