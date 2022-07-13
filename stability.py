import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import t

def first_stability(attribute_count:dict, num:int):
    '''
        Find the most frequent attribute and calculate its first level stability.

        Args:
            attribute_count (dict):
                The record of how many times each attribute/value is selected. 
                For example: {attr1: 5, attr2: 3, ...}
            
            num (int):
                The number of Monte Carlo simulations.
        
        Returns:
            best_attr:
                The most frequent attribute been selected
            n_k (float):
                The first level stability round to two decimal (the proportion).
    '''

    key = np.argmax(list(attribute_count.values()))
    best_attr = list(attribute_count.keys())[key] #the attribute which is most frequently selected
    n_k = round(attribute_count[best_attr]/num, 2)     #the number of times the attribute is selected
    
    return best_attr, n_k

def second_stability(Xs, categorical:bool, confidence:float=None):
    '''
        Find the most frequent value/the range of the values and its second level stability.

        Args:
            Xs (list or Series): 
                The record of the values been selected in each iteration.
            
            categorical (bool):
                Whether the values in Xs are categorical or continuous.
            
            confidence (float):
                For calculating the confidence interval. Only work when it's continuous attribute. 

        Returns:
            If it's categorical, the return values are the same as that of first_stability;
            If it's continuous, it returns a dictionary contains mean, standard deviation, variance, confidence variance, and confidence interval.
    '''

    #if it's a categorical variable
    if categorical:
        #similar to the first level stability
        count = Counter(Xs)
        return first_stability(count, len(Xs))
    
    #if it's a continuous variable
    else:
        if confidence is None:
            raise ValueError("confidence is needed for calculating the confidence interval of a continuous attribute")

        var = np.var(Xs)
        std = np.std(Xs)
        mean = np.mean(Xs)
        cv = std/mean       #confidence variance
        dof = len(Xs)-1     #Degree of Freedom

        t_crit = np.abs(t.ppf((1-confidence)/2,dof))
        interval = mean-std*t_crit/np.sqrt(len(Xs)), mean+std*t_crit/np.sqrt(len(Xs)) #confidence interval

        return {"mean": mean, "std": std, "var": var, "cv": cv, "c_interval": interval}