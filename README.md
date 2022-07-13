# Distillation Decision Tree
### It's the code reproduction of the essay: Lu, Xuetao, etc. Distillation Decision Tree. 9 Jun 2022. https://doi.org/10.48550/arXiv.2206.04661.

Distillation Decision Tree is a model applying Knowledge Distillation and Decision Tree to open the black boxes and simplify the complex machine learning models.

Advantages:
- Through generating pseudo data at each node, it breaks the dependency chains between the parent nodes and the child nodes
- Through generating the pseudo data many times then finding the most frequently selected attribute and split value, it solves the stability problem of the decision tree (The traditional decision trees are sensitive to the slight changes in the training set)
- It accepts both categorical and continuous attributes and labels
- Do not have the overfitting or underfitting problem
- Provide a way to reduce the dimensionality of the pseudo sample
- Switch to generate traditional decision subtrees after certain depth (or meeting other specified criteria) to save resources

Disadvantages/Problems of the current code:
- High Space and Time Complexity
- Finding the stable sample size for each node consumes great time and resources
- Currently cannot handle the data with both continuous and categorical attributes
- Filtering the true support after Dimensionality Reduction always generates empty DataFrame
