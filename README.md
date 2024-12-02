# Native Machine Learning Algorithms

This project contains native implementations of fundamental machine learning algorithms and demonstrates their application on various datasets. The goal is to provide a hands-on experience with core ML algorithms without the use of high-level libraries such as scikit-learn or TensorFlow.

## Structure
- `/models`: Contains the implementation of each machine learning algorithm.
- `/evaluation`: Contains scripts for evaluating the algorithms on different datasets.
- `/utilities`: Includes utility functions shared across different evaluation scripts.
- `/data`: Includes datasets used for various analyses.

## Algorithms

Each algorithm is implemented in a modular way to showcase the core mathematical concepts and their application. The algorithms are all stored in the `models.py` file at root. The following algorithms are included:

### PCA (Principal Component Analysis)

**Evaluation**: `evaluation/pca.py`  
**Description**: PCA is used for dimensionality reduction by projecting data onto a lower-dimensional space while preserving as much variance as possible.

### Closed-form Linear Regression

**Evaluation**: `evaluation/linreg.py`  
**Description**: Linear regression is used for predicting a continuous target variable based on one or more predictor variables using a closed-form solution.

### Logistic Regression

**Evaluation**: `evaluation/logreg.py`  
**Description**: Logistic regression is used for binary classification tasks by estimating the probability that a given input belongs to a certain class.

### LDA (Linear Discriminant Analysis)

**Evaluation**: `evaluation/lda.py`  
**Description**: LDA is used for classification and dimensionality reduction by finding the linear combinations of features that best separate different classes.

### Decision Tree

**File**: `models/dtl.py`  
**Evaluation**: `evaluation/dt.py`  
**Description**: Decision trees are used for classification and regression tasks by recursively splitting the data into subsets based on feature values.

## Future Work
- Implement more complex ML algorithms.
- Include more examples and tutorials on how to use the algorithms for classification, regression, and other tasks.
- Enhance the project with additional features and optimizations.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
* Farzan Mirza: [farzan.mirza@drexel.edu](mailto:farzan.mirza@drexel.edu) | [LinkedIn](https://www.linkedin.com/in/farzan-mirza13/)
