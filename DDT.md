# Distillation Decision Tree

#### class DDT(teacher_predict, attr_info:dict, label_info, data:np.ndarray)

`__init__(teacher_predict, attr_info:dict, label_info, data:np.ndarray)`

Construct a Distillation Decision Tree Model.

###### Parameters:
- teacher_prdict - The teacher (trained complex) model's predict method.
- attr_info (dict) - The name and range of all the attributes: Continuous: {attr1: {"min": x, "max": y}, attr2: ...}; Categorical: {attr1: [c1, c2, ...], attr2: ...}.
- label_info (dict or list) - The name and range of the labels: Continuous: {"min": x, "max": y}; Categorical: [c1, c2, ...].
- data (ndarray) - The data values without label, which is used to train the teacher model.

|Methods||
|:----|:----|
|`__init__`|Construct a Distillation Decision Tree Model|
| `fit`|Build a Distillation Decision Tree Model|
|`predict`|Return the predicted value for each sample|
|`save`|Save the model|
|`load`|Load the model|
<br>

`fit(stopping_criteria, **kwargs)`

Build a Distillation Decision Tree Model.

###### Parameters:
- stop_criteria (int) - The depth of DDT of which each node samples the pseudo data.
- **kwargs - Other parameters of the model:
  - split_criteria (str) - The spliting criteria used for classification. Accept "Gain ratio" or "Gini index".
  - num_of_iter (int) - The number of times sampling the data for each size. Default 100.
  - min_sample_each (int) - The minimum sample size for sampling each attribute. Default 2.
  - max_sample_size (int) - The maximum sample size. Default None. If specified, stable_sample_size function will stop when the sample size reaches the limit.
  - stability (tuple or list) - The first and second level stability. If specified, the stable_sample_size function will stop when the both stability are satisfied.
  - pca (bool) - Whether doing PCA on the samples. Default False.
  - criteria (float) - The precentage of variance explained by the first t new covariates. Default 0.9. Only work when pca is True.
  - train_test_split_para (dict) - The parameters passed to sklearn's train_test_split method.
  - lgb_para (dict) - The parameters passed to LightGBM Classifer or Regressor.
  - fit_para (dict) - The parameters passed to the fit method of LightGBM.

###### Returns:
- root - The root node of DDT.
<br>

`predict(root_node:Node, data:pd.DataFrame)`

Return the predicted value for each sample.

###### Parameters:
- root_node (Node) - The root of DDT.
- data (DataFrame) - The data without labels.

###### Returns:
- label (Series) - A Series of predicted values.
<br>

`save(filepath:str)`

Save the root node.

###### Parameters:
- filepath (str) - The model path.
<br>

`load(filepath:str)`

Load the root node.

###### Parameters:
- filepath (str) - The model path.

###### Returns:
- root (Node) - The root node.
<br>