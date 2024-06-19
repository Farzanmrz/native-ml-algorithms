# Native Machine Learning Algorithms

This project contains native implementations of fundamental machine learning algorithms and demonstrates their application on various datasets. The goal is to provide a hands-on experience with core ML algorithms without the use of high-level libraries such as scikit-learn or TensorFlow.

## Structure
- `/algorithms`: Contains the implementation of each machine learning algorithm.
- `/evaluation`: Contains scripts for evaluating the algorithms on different datasets.
- `/data`: Includes datasets used for various analyses.
- `/notebooks`: Contains Jupyter notebooks for detailed analysis and reporting.


## Algorithms

Each algorithm is implemented in a modular way to showcase the core mathematical concepts and their application. The following algorithms are included:

### PCA (Principal Component Analysis)

**File**: `algorithms/pca.py`  
**Evaluation**: `evaluation/evaluate_pca.py`  
**Description**: PCA is used for dimensionality reduction by projecting data onto a lower-dimensional space while preserving as much variance as possible.

### Closed-form Linear Regression

**File**: `algorithms/linear_regression.py`  
**Evaluation**: `evaluation/evaluate_linear_regression.py`  
**Description**: Linear regression is used for predicting a continuous target variable based on one or more predictor variables using a closed-form solution.

### Logistic Regression

**File**: `algorithms/logistic_regression.py`  
**Evaluation**: `evaluation/evaluate_logistic_regression.py`  
**Description**: Logistic regression is used for binary classification tasks by estimating the probability that a given input belongs to a certain class.

### LDA (Linear Discriminant Analysis)

**File**: `algorithms/lda.py`  
**Evaluation**: `evaluation/evaluate_lda.py`  
**Description**: LDA is used for classification and dimensionality reduction by finding the linear combinations of features that best separate different classes.

### Decision Tree

**File**: `algorithms/decision_tree.py`  
**Evaluation**: `evaluation/evaluate_decision_tree.py`  
**Description**: Decision trees are used for classification and regression tasks by recursively splitting the data into subsets based on feature values.

### Naive Bayes

**File**: `algorithms/naive_bayes.py`  
**Evaluation**: `evaluation/evaluate_naive_bayes.py`  
**Description**: Naive Bayes is a probabilistic classifier based on Bayes' theorem with the assumption of feature independence.

## Future Work
- Implement more complex ML algorithms.
- Include more examples and tutorials on how to use the algorithms for classification, regression, and other tasks.
- Enhance the project with additional features and optimizations.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
* Farzan Mirza: [farzan.mirza@drexel.edu](mailto:farzan.mirza@drexel.edu) | [LinkedIn](https://www.linkedin.com/in/farzan-mirza13/)
