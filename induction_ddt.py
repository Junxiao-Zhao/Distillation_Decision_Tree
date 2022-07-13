import numpy as np
import pandas as pd
import lightgbm as lgb
import sampling as sp
import stability as sb
import split_criteria as sc
import pickle
from sklearn.model_selection import train_test_split
from utils import *

class Node():
    '''
        The Node of Distillation Decision Tree.

        Args:
            node_id (int):
                The ID of the node.
            
            split_attr:
                The split attribute. Default None.
            
            split_value:
                The split value. Default None.
            
            attr_info (dict):
                The name and range of all the attributes:
                    Continuous: {attr1: {"min": x, "max": y}, attr2: ...};
                    Categorical: {attr1: [c1, c2, ...], attr2: ...}.
            
            min_sample_each (int):
                The sample size of each attribute. Default 2.
            
            left:
                The left child. Default None.
            
            right:
                The right child. Default None.

            model:
                The LightGBM model. Default None.
    '''

    def __init__(self, node_id, split_attr=None, split_value=None, attr_info=None, min_sample_each=2, left=None, right=None, model=None):
        self.id = node_id
        self.attr = split_attr
        self.value = split_value
        self.attr_info = attr_info
        self.sample_size = min_sample_each
        self.left = left
        self.right = right
        self.model = model

class DDT():
    '''
        Construct a Distillation Decision Tree Model.

        Args:
            teacher_predict (function): 
                The teacher (trained complex) model's predict method.

            attr_info (dict):
                The name and range of all the attributes:
                    Continuous: {attr1: {"min": x, "max": y}, attr2: ...};
                    Categorical: {attr1: [c1, c2, ...], attr2: ...}.

            label_info (dict/list):
                The name and range of the labels:
                    Continuous: {"min": x, "max": y};
                    Categorical: [c1, c2, ...].
            
            data (ndarray):
                The data values without label, which is used to train the teacher model.
'''

    def __init__(self, teacher_predict, attr_info:dict, label_info, data:np.ndarray):
        self.teacher_predict = teacher_predict
        self.attr_info = attr_info
        self.label_info = label_info

        #calculate the mean and std for normalize the pseudo data
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)

        self.W = np.linalg.svd(np.matmul(data.T, data))[0]   #The singular vector of the real data which trains the teacher model
        self.student = None #The student model after distillation
    
    def stable_sample_size(self, min_sample_each=2, max_sample_size:int=None, stability:tuple=(0.5,0.7), **kwargs):
        '''
            Find the stable sample size for each split node.

            Args:
                min_sample_each (int):
                    The minimum sample size for sampling each attribute. Default 2.
                    
                max_sample_size (int):
                    The maximum sample size. Default None. If specified, stable_sample_size function will stop when the sample size reaches the limit.
                    
                stability (tuple/list):
                    The first and second level stability. If specified, the stable_sample_size function will stop when the both stability are satisfied.

                **kwargs:
                    attr_info (dict):
                        The name and range of all the attributes:
                            Continuous: {attr1: {"min": x, "max": y}, attr2: ...};
                            Categorical: {attr1: [c1, c2, ...], attr2: ...}.

                    split_criteria (str):
                        The spliting criteria used for classification. Accept "Gain ratio" or "Gini index".
                    
                    num_of_iter (int): 
                        The number of times sampling the data for each size. Default 100.
                    
                    pca (bool):
                        Whether doing PCA on the samples. Default False.
                    
                    criteria (float):
                        The precentage of variance explained by the first t new covariates. Default 0.9. Only work when pca is True.
            
            Returns:
                cur_size (int):
                    The number of times sampling each attribute to form a stable/maximum size sample.

                best_attr:
                    The most frequently been selected attribute.
                
                best_value:
                    Categorical: The most frequently been selected category;
                    Continuous (tuple): The confidence interval.
        '''

        print("Finding the stable sample size...")

        #the initial number of times sampling each attribute
        cur_size = min_sample_each
        #initialize two level stability to 0
        stability1, stability2 = 0, 0

        #stop condition depends on the max_sample_size
        if max_sample_size is None:
            not_stop = lambda x: True
        else:
            not_stop = lambda x: x <= max_sample_size / len(self.attr_info)

        #stop when reaching the specified max sample size or when both level stability are met
        while not_stop(cur_size) and (stability1 < stability[0] or stability2 < stability[1]):
            
            #calculate the first and second level stability of the sample of current size after sampling num_of_iter times
            best_attr, best_value, stability1, stability2 = self.sample_n_times(cur_size=cur_size, second_level_stability=stability[1], **kwargs)

            #next round each attribute sample size
            cur_size *= 2
        
        #the stable or maximun sample size for all the attributes
        cur_size //= 2

        return cur_size, best_attr, best_value
    
    def sample_n_times(self, num_of_iter:int, **kwargs):
        '''
            Calculate the first and second level stability of the sample of specified size after sampling num_of_iter times.

            Args:
                num_of_iter (int): 
                    The number of times sampling the data. Default 100.
                
                **kwargs:
                    attr_info (dict):
                        The name and range of all the attributes:
                            Continuous: {attr1: {"min": x, "max": y}, attr2: ...};
                            Categorical: {attr1: [c1, c2, ...], attr2: ...}.
                    
                    cur_size (int):
                        The number of times sampling each attribute.
                    
                    split_criteria (str):
                        The spliting criteria used for classification. Accept "Gain ratio" or "Gini index".

                    second_level_stability (float):
                        For calculating the confidence interval. Only work when it's continuous attribute.
                    
                    pca (bool):
                        Whether doing PCA on the samples. Default False.

                    criteria (float):
                        The precentage of variance explained by the first t new covariates. Default 0.9. Only work when pca is True.
            
            Returns:
                overall_best_attr:
                    The most frequently been selected attribute.
                
                overall_best_value:
                    Categorical: The most frequently been selected category;
                    Continuous (tuple): The confidence interval.
                
                stability1 (float):
                    The first level stability.
                
                stability2 (float):
                    The second level stability.
        '''

        count_attributes = {}   #count the selected attributes
        count_values = []   #count the selected values/categories

        #sample n times
        for i in range(num_of_iter):
            #sample the data
            results = self.each_sample(**kwargs)

            count_values.append(results[0])
            count_attributes.setdefault(results[1], 0)
            count_attributes[results[1]] += 1
        
        count_values = pd.DataFrame(count_values, columns=list(self.attr_info.keys()))

        #calculate the best split attribute and its first level stability
        overall_best_attr, stability1 = sb.first_stability(count_attributes, num_of_iter)
        #the best split value of the attribute and its second level stability
        if kwargs.setdefault('split_criteria', None) is not None:  #categorical
            overall_best_value, stability2 = sb.second_stability(count_values[overall_best_attr], True)
        else:   #continuous
            overall_best_value = sb.second_stability(count_values[overall_best_attr], False, kwargs['second_level_stability'])['c_interval']
            stability2 = 1
        
        return overall_best_attr, overall_best_value, stability1, stability2
            
    def each_sample(self, attr_info:dict=None, cur_size:int=None, split_criteria:str=None, pca=False, criteria=0.9, samples=None, **kwargs):
        '''
            Sampling and split the data.

            Args:
                attr_info (dict):
                    The name and range of all the attributes:
                        Continuous: {attr1: {"min": x, "max": y}, attr2: ...};
                        Categorical: {attr1: [c1, c2, ...], attr2: ...}.

                cur_size (int):
                    The number of times sampling each attribute.

                split_criteria (str):
                    The spliting criteria used for classification. Accept "Gain ratio" or "Gini index".
                
                pca (bool):
                    Whether doing PCA on the samples. Default False.

                criteria (float):
                    The precentage of variance explained by the first t new covariates. Default 0.9. Only work when pca is True.
                
                samples (DataFrame):
                    A sample with labels. Default None. If specified, all previous args except split_criteria will not work.
            
                **kwargs:
                    No use here.

            Returns:
                count_values (list):
                    A list contains the split value/category for all the attributes.
                
                best_attr:
                    The best split attribute.

                best_value:
                    The best split value.
        '''

        #generate samples
        if samples is None:
            samples = sp.sample(cur_size, attr_info)    #sample the data without label
            
            #reduce the sample size if indicated
            if pca:
                samples = self.marginal_pca(samples, attr_info, criteria)
            
            #predict the labels using the teacher model
            samples = sp.predict_label(samples, self.teacher_predict, self.mean, self.std)

        count_values = []
        count_score = []

        for attr, values_domain in self.attr_info.items():

            if isinstance(values_domain, dict):
                #if both attributes and label are continuous (case 1)
                if isinstance(self.label_info, dict):
                    results = sc.sum_square_error(samples.loc[:,[attr, 'label']])

                #if the attributes are continuous and the label is categorical (case 2)
                else:
                    if split_criteria == "Gain ratio":
                        results = sc.gain_ratio_continuous(samples[attr, 'label'])
                    
                    elif split_criteria == "Gini index":
                        results = sc.gini_continuous(samples[attr, 'label'])

                    else:
                        raise ValueError('split_criteria only supports "Gain ratio" or "Gini index"')
            
            else:
                #if the attributes are categorical and the label is continuous (case 3)
                if isinstance(self.label_info, dict):
                    results = sc.mean_square_error(samples[attr, 'label'])

                #if both attributes and label are categorical (case 4)
                else:
                    if split_criteria == "Gain ratio":
                        results = sc.gain_ratio_categorical(samples[attr, 'label'])
                    
                    elif split_criteria == "Gini index":
                        results = sc.gini_categorical(samples[attr, 'label'])

                    else:
                        raise ValueError('split_criteria only supports "Gain ratio" or "Gini index"')

            split_value = results.index[np.argmin(results)]
            score = np.min(results)
            
            count_values.append(split_value) #record the split values
            count_score.append(score)   #record the score
        
        #find the best split attribute with the smallest score in this round
        index = np.argmin(count_score)
        best_attr = list(self.attr_info.keys())[index]
        best_value = count_values[index]

        return count_values, best_attr, best_value

    def marginal_pca(self, samples:pd.DataFrame, attr_info:dict, criteria:float=0.9):
        '''
            To reduce the sample size.

            Args:
                samples (DataFrame):
                    A sample without labels.
                
                attr_info (dict):
                    The name and range of all the attributes:
                        Continuous: {attr1: {"min": x, "max": y}, attr2: ...};
                        Categorical: {attr1: [c1, c2, ...], attr2: ...}.

                criteria (float):
                    The precentage of variance explained by the first t new covariates. Default 0.9.
            
            Returns:
                filter_sample (DataFrame):
                    Samples filtered by the true support of attributes.

        '''

        if criteria > 1 or criteria < 0:
            raise ValueError("criteria should be between 0 and 1")
        
        samples_value = samples.values #q covariates, n*q matrix
    
        Z_matrix = np.matmul(samples_value, self.W)    #transfer the sample covariates to new covariates z, n*q matrix
        eigen_values = np.linalg.svd(np.matmul(Z_matrix.T, Z_matrix))[1]   #the eigen values lambda, which reflect the variance explained by covariates z
        proportion = eigen_values / np.sum(eigen_values)    #the proportion of variance

        #first t covariates that can explain the percentage of variance equal to or greater than the criteria
        fst_t = binary_search(proportion, criteria)
        sub_proportion = proportion[:fst_t] 
        
        #calculate the ratio
        ratio = sub_proportion / sub_proportion[0]

        #sampling new covariates according to ratio
        new_samples = sp.sample_new(ratio=ratio, Z_matrix=Z_matrix)
        new_samples = np.matmul(new_samples, self.W.T)
        new_samples = pd.DataFrame(new_samples, columns=samples.columns)

        #filter by the true support of attributes
        filter_samples = true_support_filter(new_samples, attr_info)

        return filter_samples

    def subtree(self, sample_size, attr_info:dict, train_test_split_para:dict, lgb_para:dict, fit_para:dict, **kwargs):
        '''
            Using LightGBM to generate the subtrees under certian depth.

            Args:
                sample_size (int):
                    The sample size for sampling each attribute.
                
                attr_info (dict):
                    The name and range of all the attributes:
                        Continuous: {attr1: {"min": x, "max": y}, attr2: ...};
                        Categorical: {attr1: [c1, c2, ...], attr2: ...}.
                
                train_test_split_para (dict):
                    The parameters passed to sklearn's train_test_split method.

                lgb_para (dict):
                    The parameters passed to LightGBM Classifer or Regressor.
                
                fit_para (dict):
                    The parameters passed to the fit method of LightGBM.
                
                **kwargs:
                    pca (bool):
                        Whether doing PCA on the samples. Default False.
                    
                    criteria (float):
                        The precentage of variance explained by the first t new covariates. Default 0.9. Only work when pca is True.
            
            Returns:
                model:
                    The LightGBM model.      
        '''

        samples = sp.sample(sample_size, attr_info)
        
        if kwargs['pca']:
            samples = self.marginal_pca(samples, attr_info, criteria=kwargs['criteria'])
        
        samples_label = sp.predict_label(samples.copy(), self.teacher_predict, self.mean, self.std)['label']
        
        x_train,x_test,y_train,y_test = train_test_split(samples, samples_label, **train_test_split_para)

        #categorical
        if isinstance(self.label_info, list):
            model = lgb.LGBMClassifier(**lgb_para)
        #continuous
        else:
            model = lgb.LGBMRegressor(**lgb_para)
        
        model.fit(x_train, y_train, eval_set=[(x_test,y_test), (x_train,y_train)], **fit_para)

        return model

    def distill(self, cur_node:Node, train_test_split_para:dict, lgb_para:dict, fit_para:dict, all_in_one=False, **kwargs):
        '''
            Generate child nodes.

            Args:
                cur_node (Node):
                    The parent node.

                train_test_split_para (dict):
                    The parameters passed to sklearn's train_test_split method.

                lgb_para (dict):
                    The parameters passed to LightGBM Classifer or Regressor.
                
                fit_para (dict):
                    The parameters passed to the fit method of LightGBM.
                
                all_in_one (Bool):
                    Whether use LightGBM to generate the subtrees. Default False.
                
                **kwargs:
                    split_criteria (str):
                        The spliting criteria used for classification. Accept "Gain ratio" or "Gini index".
                    
                    num_of_iter (int): 
                        The number of times sampling the data for each size. Default 100.
                    
                    min_sample_each (int):
                        The minimum sample size for sampling each attribute. Default 2.
                    
                    max_sample_size (int):
                        The maximum sample size. Default None. If specified, stable_sample_size function will stop when the sample size reaches the limit.
                    
                    stability (tuple/list):
                        The first and second level stability. If specified, the stable_sample_size function will stop when the both stability are satisfied.
                    
                    pca (bool):
                        Whether doing PCA on the samples. Default False.
                    
                    criteria (float):
                        The precentage of variance explained by the first t new covariates. Default 0.9. Only work when pca is True.
        '''

        print(f"Generating Node {cur_node.id}..." )

        cur_domain = cur_node.attr_info[cur_node.attr]

        left_domain, right_domain = split_domain(cur_domain, cur_node.value)    #split the domain according the split value
        #create new domain for left and right child
        left_attr_info, right_attr_info = cur_node.attr_info.copy(), cur_node.attr_info.copy()
        left_attr_info[cur_node.attr] = left_domain
        right_attr_info[cur_node.attr] = right_domain

        #before reaching the certain depth
        if not all_in_one:
            #use the current sample each attribute size as the min_sample_size
            left_kwargs = kwargs.copy()
            left_kwargs["min_sample_each"] = cur_node.sample_size
            right_kwargs = kwargs.copy()
            right_kwargs["min_sample_each"] = cur_node.sample_size

            #calculate the sample each size, split attributes and split values
            left_size, left_attr, left_value = self.stable_sample_size(attr_info=left_attr_info, **left_kwargs)
            right_size, right_attr, right_value = self.stable_sample_size(attr_info=right_attr_info, **right_kwargs)

            #create left and right child and add them into the queue
            cur_node.left = Node(2*cur_node.id+1, left_attr, left_value, left_attr_info, left_size)
            cur_node.right = Node(2*cur_node.id+2, right_attr, right_value, right_attr_info, right_size)
            
        
        #generate LightGBM subtrees
        else:
            left_model = self.subtree(cur_node.sample_size, left_attr_info, train_test_split_para, lgb_para, fit_para, **kwargs)
            cur_node.left = Node(2*cur_node.id+1, model=left_model)
            right_model = self.subtree(cur_node.sample_size, right_attr_info, train_test_split_para, lgb_para, fit_para, **kwargs)
            cur_node.right = Node(2*cur_node.id+2, model=right_model)

    def fit(self, stopping_criteria, **kwargs):
        '''
            Build the distillation decision tree.

            Args:
                stop_criteria (int):
                    The depth of DDT of which each node samples the pseudo data.

                **kwargs:
                    split_criteria (str):
                        The spliting criteria used for classification. Accept "Gain ratio" or "Gini index".
                    
                    num_of_iter (int): 
                        The number of times sampling the data for each size. Default 100.
                    
                    min_sample_each (int):
                        The minimum sample size for sampling each attribute. Default 2.
                    
                    max_sample_size (int):
                        The maximum sample size. Default None. If specified, stable_sample_size function will stop when the sample size reaches the limit.
                    
                    stability (tuple/list):
                        The first and second level stability. If specified, the stable_sample_size function will stop when the both stability are satisfied.
                    
                    pca (bool):
                        Whether doing PCA on the samples. Default False.
                    
                    criteria (float):
                        The precentage of variance explained by the first t new covariates. Default 0.9. Only work when pca is True.
                    
                    train_test_split_para (dict):
                        The parameters passed to sklearn's train_test_split method.
                    
                    lgb_para (dict):
                        The parameters passed to LightGBM Classifer or Regressor.
                    
                    fit_para (dict):
                        The parameters passed to the fit method of LightGBM.
                    
                
            Returns:
                root (Node):
                    The root node.
        '''

        #create the root node
        cur_size, best_attr, best_value = self.stable_sample_size(attr_info=self.attr_info, **kwargs) #, split_criteria=split_criteria, num_of_iter=num_of_iter, min_sample_each=min_sample_each, max_sample_size=max_sample_size, stability=stability, pca=pca, is_root=is_root, criteria=criteria)
        root = Node(0, best_attr, best_value, self.attr_info, cur_size)

        #Breadth-first sampling strategy based on BFS
        queue1 = [root]
        queue2 = []
        depth = 0

        while depth <= stopping_criteria:

            cur_node = queue1.pop(0)
            self.distill(cur_node=cur_node, **kwargs)
            queue2 += [cur_node.left, cur_node.right]

            #calculate the current depth
            if queue1 == []:
                queue1, queue2 = queue2, queue1
                depth += 1

        if queue1 == []:
            return root
        
        #generate subtree using LightGBM
        else:
            for each in queue1:
                self.distill(cur_node=each, all_in_one=True, **kwargs)
        
        self.student = root

        return root
    
    def prediction(self, node:Node, df:pd.DataFrame):
        '''
            Recursion Prediction.

            Args:
                node (Node):
                    The DDT Node.
                
                df (DataFrame):
                    A sample for prediction.
            
            Returns:
                df (DataFrame):
                    A sample with labels.
        '''

        #LightGBM model prediction
        if node.model is not None:
            labels = node.model.predict(df)
            df.insert(df.shape[1], "label", labels)

            return df

        else:
            left_domain, right_domain = split_domain(node.attr_info[node.attr], node.value)    #split the domain according the split value
            df_left = df_right = df.copy()
            #split df using the node's attr and its split value
            df_left = filter_sample(df_left, node.attr, left_domain)
            df_right = filter_sample(df_right, node.attr, right_domain)

            #data after prediction, with labels
            new_left = self.prediction(node.left, df_left)
            new_right = self.prediction(node.right, df_right)
            #concat left and right nodes' prediction together
            new_df = pd.concat([new_left, new_right])

            return new_df

    def predict(self, root_node:Node, data:pd.DataFrame):
        '''
            Return the predicted value for each sample.

            Args:
                root_node (Node):
                    The root of DDT.
                
                data (DataFrame):
                    The data without labels.
            
            Returns:
                label (Series):
                    A Series of labels.
        '''
        
        df = self.prediction(root_node, data)
        #sort the index
        df.sort_index(inplace=True)

        return df['label']
    
    def save(self, filepath:str): 
        '''
            Save the root node.

            Args:
                filepath (str):
                    The model path.
        '''

        pickle.dump(self.student, open(filepath, 'wb'))
        print(f"The model has been saved to {filepath}!")
    
    def load(self, filepath:str):
        '''
            Load the root node.

            Args:
                filepath (str):
                    The model path.

            Returns:
                root (Node):
                    The root node of DDT.
        '''

        root = pickle.load(open(filepath, "rb"))

        return root