import numpy as np
import pandas as pd

def split_domain(value_domain, split_value):
    '''
        Split the attribute's domain.

        Args:
            value_domain (list|dict):
                Continuous: {"min": x, "max": y};
                Categorical: [c1, c2, ...].
            
            split_value:
                The split value.
        
        Returns (tuple):
            Continuous: ({"min": min, "max": split_value}, {"min": split_value, "max": max});
            Categorical: ([split_value], a list after removing the split value).
    '''

    if isinstance(value_domain, list):  #categorical
        return list(split_value), value_domain.remove(split_value)
    else:   #continuous
        left = right = value_domain.copy()
        #generate a random value in the confidence interval as the split value
        split_value = np.random.uniform(split_value[0], split_value[1], 1)[0]
        left["max"] = split_value
        right["min"] = split_value
        return left, right

def true_support_filter(df:pd.DataFrame, attr_info:dict):
    '''
        Filter the sample after PCA by the true support of attributes.

        Args:
            df (DataFrame):
                A DataFrame contains attributes and labels.
            
            attr_info (dict):
                The name and range of all the attributes:
                    Continuous: {attr1: {"min": x, "max": y}, attr2: ...};
                    Categorical: {attr1: [c1, c2, ...], attr2: ...}.
        
        Returns:
            df (DataFrame):
                A DataFrame contains attributes and labels after filtration.
    '''
    
    for attr, values_domain in attr_info.items():
        df = filter_sample(df, attr, values_domain)

    return df

def filter_sample(df:pd.DataFrame, attr, values_domain):
    '''
        Filter the sample according to a specified attribute's domain.

        Args:
            df (DataFrame):
                A DataFrame contains attributes and labels.
            
            attr:
                An attribute.

            values_domain (list|dict):
                Continuous: {"min": x, "max": y};
                Categorical: [c1, c2, ...].
        
        Returns:
            df (DataFrame):
                A DataFrame contains attributes and labels after filtration.
    '''

    #continuous
    if isinstance(values_domain, dict):
        df = df[df[attr]>=values_domain["min"]]
        df = df[df[attr]<=values_domain["max"]]
    #categorical
    else:
        df = df[df[attr].apply(lambda x: x in values_domain)]
    
    return df

def binary_search(na: np.ndarray, criteria: float):
    '''
        Binary Search the index i when the sum of na[:i] greater or equal to the criteria.

        Args:
            na (ndarray):
                A array of proportions sum to 1.
            
            criteria (float):
                A number small or equal to 1.
        
        Returns:
            split (int):
                The index i which makes the sum of na[:i] greater or equal to the criteria.
    '''

    criteria = np.round(criteria, 2)
    length = len(na)
    start = 0
    end = length - 1
    split = end // 2

    while True:
        cur_sum = np.sum(na[:split])

        if cur_sum < criteria:
            if np.round(cur_sum + na[split], 2) >= criteria:
                break
            else:
                start = split + 1
        
        else:
            end = split
        
        split = (end + start) // 2
    
    return split + 1

if __name__ == "__main__":
    test = np.array([0.1,0.2,0.1,0.3,0.1,0.2])
    print(binary_search(test, 1))