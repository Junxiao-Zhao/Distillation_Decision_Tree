import numpy as np
import pandas as pd

def split_df(split_value, df:pd.DataFrame):
    '''
        Split the data according to the split value.

        Args:
            split_value:
                The split value.

            df (DataFrame):
                The DataFrame for spliting.
        
        Returns:
            A tuple with a DataFrame contains attribute values smaller or equal to split_value at position 0 and another half at position 1.
    '''
    
    df_l = df[df.iloc[:,0]<=split_value]
    df_r = df[df.iloc[:,0]>split_value]

    return df_l, df_r

def MSE(df_each:pd.DataFrame, df:pd.DataFrame) -> float:
    '''
        Calculate the mean square error according to the split category.

        Args:
            df_each (DataFrame):
                A DataFrame contains one category of the attribute in df.
            
            df (DataFrame):
                A DataFrame contains one column of a specified attribute and one column of labels.
        
        Returns:
            mse (float):
                The mean square error of spliting according to the specified category.
    '''

    each_mean = np.mean(df_each['label'])
    df_rest = df[df.iloc[:,0]!=df_each.iloc[0,0]]
    rest_mean = np.mean(df_rest['label'])

    mse = (np.sum(np.square(df_each['label']-each_mean)) + np.sum(np.square(df_rest['label']-rest_mean))) / df.shape[0]

    return mse

def mean_square_error(df:pd.DataFrame) -> pd.Series:
    '''
        Calculate all the mean square error of choosing each category of the attribute as the split category.

        Args:
            df (DataFrame):
                A DataFrame contains one column of a specified attribute and one column of labels.
        
        Returns:
            mse (Series):
                A Series contains all the mean square error of choosing each category of the attribute as the split category.
    '''

    mse = df.groupby(df.columns[0]).apply(MSE, df=df)

    return mse

def SSE(split_value:float, df:pd.DataFrame) -> float:
    '''
        Calculate the sum square error according to the split value.

        Args:
            split_value (float):
                The split value.
            
            df (DataFrame):
                A DataFrame contains one column of a specified attribute and one column of labels.
        
        Returns:
            sse (float):
                The sum square error.
    '''

    #split the data according to the split value
    df_l, df_r = split_df(split_value, df)

    #calculate each labels mean
    mean_l = np.mean(df_l["label"])
    mean_r = np.mean(df_r["label"])

    #calculate the sum square error
    sse = np.sum(np.square(df_l['label']-mean_l)) + np.sum(np.square(df_r['label']-mean_r))

    return sse

def sum_square_error(df:pd.DataFrame):
    '''
        Calculate all the sum square errors of choosing each value of the attribute as the split value.

        Args:
            df (DataFrame):
                A DataFrame contains one column of a specified attribute and one column of labels.
        
        Returns:
            sses (Series):
                A Series contains all the sum square errors of choosing each value of the attribute as the split value.
    '''

    #the sum square errors of choosing each value of the attribute as the split value
    sses = df.iloc[:,0].apply(SSE, df=df)

    return sses

def entropy(df:pd.DataFrame) -> list:
    '''
        Calculate the entropy for the given attribute.

        Args:
            df (DataFrame):
                A DataFrame contains one column of a specified attribute and one column of labels.
        
        Returns:
            entropy (list):
                A list with the entropy at position 0 and the number of rows in df at position 1.
    '''

    proportion = np.array(df.groupby(df.columns[1]).apply("count").iloc[:,0])/df.shape[0]  #the proportion of each label category
    entr = -np.sum(proportion*np.log2(proportion)) #calculate the entropy

    return [entr, df.shape[0]]

def entropy_continuous(split_value, df:pd.DataFrame):
    '''
        Calculate the entropy for each split value.

        Args:
            split_value:
                The split value.
            
            df (DataFrame):
                A DataFrame contains one column of a specified attribute and one column of labels.
        
        Returns:
            entr (Series):
                A Series contains all the entropy of spliting by each split value.
    '''

    #split the data according to the split value
    df_l, df_r = split_df(split_value, df)

    #calculate the sum of entropy * proportion
    entr_l, entr_r = entropy(df_l), entropy(df_r)
    entr = entr_l[0] * entr_l[1] + entr_r[0] * entr_r[1]
    entr /= df.shape[0]

    return entr

def gain_ratio_continuous(df:pd.DataFrame) -> pd.Series:
    '''
        Calculate the inverse of Gain ratio for choosing each value of the given continuous attribute.

        Args:
            df (DataFrame):
                A DataFrame contains one column of a specified attribute and one column of labels.
        
        Returns:
            gain (Series):
                A Series contains the inverse of Gain ratios of choosing each value as the split value for the attribute.
    '''

    #the entropy of this attribute
    node_entropy = entropy(df)[0]

    #calculate the entropy of selecting each value as the split value
    all_entr = df.iloc[:,0].apply(entropy_continuous, df=df)

    #calculate the Gain ratio
    gain = 1 / (node_entropy - all_entr)

    return gain

def entropy_categorical(split_cat, df:pd.DataFrame):
    '''
        Calculate the entropy for the categories besides the selected one.

        Args:
            split_cat:
                The split category.

            df (DataFrame):
                A DataFrame contains one column of a specified attribute and one column of labels.
        
        Returns:
            entropy_rest (list):
                A list with the entropy at position 0 and the number of rows in df at position 1.

    '''

    df_rest = df[df.iloc[:,0]!=split_cat]
    entropy_rest = entropy(df_rest)
    
    return entropy_rest

def gain_ratio_categorical(df:pd.DataFrame) -> pd.Series:
    '''
        Calculate the inverse of Gain ratio for choosing each category of the given categorical attribute.

        Args:
            df (DataFrame):
                A DataFrame contains one column of a specified attribute and one column of labels.
        
        Returns:
            gain (Series):
                A Series contains the inverse of Gain ratios of choosing each category as the split category for the attribute.
    '''

    #the entropy of this attribute
    node_entropy = entropy(df)[0]
    
    #calculate the entropy of each attribute category
    each_entropy = df.groupby(df.columns[0]).apply(entropy)
    each_entropy = pd.DataFrame(list(each_entropy), index=each_entropy.index, columns=["entropy", "count"])
    
    #calculate the entropy of the rest categories
    not_entropy = pd.Series(each_entropy.index).apply(entropy_categorical, df=df)
    not_entropy = pd.DataFrame(list(not_entropy), index=each_entropy.index, columns=["not_entropy", "not_count"])

    #concate each_entropy with not_entropy
    each_entropy = pd.concat([each_entropy, not_entropy], axis=1)

    #calculate the Gain ratio
    gain = each_entropy['entropy'] * each_entropy['count'] + each_entropy['not_entropy'] * each_entropy['not_count']
    gain /= df.shape[0]
    gain = 1 / (node_entropy - gain)

    return gain

def each_gini(df:pd.DataFrame):
    '''
        Calculate the Gini index for each category of the given attribute.

        Args:
            df (DataFrame):
                A DataFrame contains one column of a specified attribute and one column of labels.
        
        Returns:
            gini_each (list):
                A list with Gini index at position 0 and the number of rows in df at position 1.
    '''

    proportion = np.array(df.groupby(df.columns[1]).apply("count").iloc[:,0])/df.shape[0]  #the proportion of each label category
    gini_each = 1 - np.sum(np.square(proportion))
    return [gini_each, df.shape[0]]

def rest_gini_categorical(split_cat, df:pd.DataFrame):
    '''
        Calculate the Gini index for the categories besides the selected one.

        Args:
            split_cat:
                The split category.

            df (DataFrame):
                A DataFrame contains one column of a specified attribute and one column of labels.
        
        Returns:
            entropy_rest (list):
                A list with the Gini index at position 0 and the number of rows in df at position 1.
    '''

    df_rest = df[df.iloc[:,0]!=split_cat]
    gini_rest = each_gini(df_rest)

    return gini_rest

def gini_categorical(df:pd.DataFrame):
    '''
        Calculate the overall Gini index of the given categorical attribute.

        Args:
            df (DataFrame):
                A DataFrame contains one column of a specified attribute and one column of labels.
        
        Returns:
            gini_index (Series)
                A Series contains the Gini index of choosing each category as the split category for the attribute.
    '''

    #calculate the Gini index of each attribute category
    gini_each = df.groupby(df.columns[0]).apply(each_gini)
    gini_each = pd.DataFrame(list(gini_each), index=gini_each.index, columns=["each gini", "count"])

    #calculate the entropy of the rest categories
    not_gini = pd.Series(gini_each.index).apply(rest_gini_categorical, df=df)
    not_gini = pd.DataFrame(list(not_gini), index=gini_each.index, columns=["not_gini", "not_count"])

    #concat gini_each with not_gini
    gini_each = pd.concat([gini_each, not_gini], axis=1)
    
    #calculate the Gini index
    gini_index = gini_each['each gini'] * gini_each['count'] + gini_each['not_gini'] * gini_each['not_count']
    gini_index /= df.shape[0]

    return gini_index

def gini_each_continuous(split_value, df:pd.DataFrame):
    '''
        Calculate the Gini index for each split value.

        Args:
            split_value:
                The split value.
            
            df (DataFrame):
                A DataFrame contains one column of a specified attribute and one column of labels.
        
        Returns:
            entr (Series):
                A Series contains all the Gini index of spliting by each split value.
    '''

    #split the data according to the split value
    df_l, df_r = split_df(split_value, df)

    #calculate the sum of Gini index * proportion
    gini_l, gini_r = each_gini(df_l), each_gini(df_r)
    gini = gini_l[0] * gini_l[1] + gini_r[0] * gini_r[1]
    gini /= df.shape[0]

    return gini

def gini_continuous(df:pd.DataFrame):
    '''
        Calculate the overall Gini index of the given continuous attribute.

        Args:
            df (DataFrame):
                A DataFrame contains one column of a specified attribute and one column of labels.
        
        Returns:
            gini_index (Series)
                A Series contains the Gini index of choosing each value as the split value for the attribute.
    '''

    #calculate the Gini index of selecting each value as the split value
    all_gini = df.iloc[:,0].apply(gini_each_continuous, df=df)

    return all_gini

if __name__ == "__main__":
    #watermelon
    df1 = pd.DataFrame([["green", "yes"],
                        ["black", "yes"],
                        ["black", "yes"],
                        ["green", "yes"],
                        ["white", "yes"],
                        ["green", "yes"],
                        ["black", "yes"],
                        ["black", "yes"],
                        ["black", "no"],
                        ["green", "no"],
                        ["white", "no"],
                        ["white", "no"],
                        ["green", "no"],
                        ["white", "no"],
                        ["black", "no"],
                        ["white", "no"],
                        ["green", "no"],], columns=['root color', 'label'])
    
    #defaults
    df2 = pd.DataFrame([["yes", "no"],
                        ["yes", "no"],
                        ["yes", "no"],
                        ["no", "no"],
                        ["no", "no"],
                        ["no", "no"],
                        ["no", "no"],
                        ["no", "yes"],
                        ["no", "yes"],
                        ["no", "yes"],], columns=["has house", "label"])
    
    df3 = pd.DataFrame([["single", "not"],
                        ["married", "not"],
                        ["single", "not"],
                        ["married", "not"],
                        ["divorce", "yes"],
                        ["married", "not"],
                        ["divorce", "not"],
                        ["single", "yes"],
                        ["married", "not"],
                        ["single", "yes"],], columns=["marital status", "label"])

    df4 = pd.DataFrame([[1, 5.56],
                        [2, 5.7],
                        [3, 5.91],
                        [4, 6.4],
                        [5, 6.8],
                        [6, 7.05],
                        [7, 8.9],
                        [8, 8.7],
                        [9, 9],
                        [10, 9.05]], columns=['x', 'label'])
    
    df5 = pd.DataFrame([[5.56, 1],
                        [5.7, 1],
                        [5.91,1],
                        [6.4, 2],
                        [6.8, 1],
                        [7.05, 2],
                        [8.9, 3],
                        [8.7, 3],
                        [9, 2],
                        [9.05, 3]], columns=['x', 'label'])
    
    df6 = pd.concat([df3['marital status'], df4['label']], axis=1)
    
    a = gain_ratio_continuous(df5)
    print(a)
    #print(a[np.argmin(a)])
    #print(sum_square_error(df4))
    #print(gini_continuous(df5))
    #print(gini_categorical(df3))
    """ print(df6)
    print(mean_square_error(df6)) """