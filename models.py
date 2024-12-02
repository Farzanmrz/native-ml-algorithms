import numpy as np

def entropy( y ):
	"""
	Calculate the entropy of the target variable.

	Parameters:
	y (numpy.ndarray): The target variable vector.

	Returns:
	float: The entropy of the target variable.
	"""
	# Get the unique elements and their counts
	elements, counts = np.unique(y, return_counts = True)

	# Return entropy
	return -np.sum([ (counts[ i ] / np.sum(counts)) * (np.log2(counts[ i ] / np.sum(counts))) for i in range(len(elements)) ])

def chooseAttribute( x, y, features ):
	"""
	Choose the best attribute for splitting the data based on entropy.

	Parameters:
	x (pandas.DataFrame): The input feature matrix.
	y (pandas.Series): The target variable vector.
	features (list): The list of features to consider for splitting.

	Returns:
	int: The index of the best feature for splitting.
	"""
	# Initialize best
	ft_best = None

	# Initialize variable to store lowest weighted average entropy initially infinite
	w_entropy_best = np.inf

	# Loop through all the features
	for ft in features:
		# Get s the number of subsets created by the current feature will be 1 or 0
		s = x.iloc[ :, ft ].unique()

		# Initialize variable to store weighted average entropy for current feature
		w_entropy = 0

		# Loop through each subset i in s
		for i in s:
			# Get the subset y for the current feature where it matches subset i
			y_i = y[ x.iloc[ :, ft ] == i ]

			# Update weighted entropy accordingly
			w_entropy += (len(y_i) / len(y)) * entropy(y_i)

		# Check if this weighted average entropy is the lowest yet
		if w_entropy < w_entropy_best:
			# If it is then set this to best entropy
			w_entropy_best = w_entropy

			# Set the current feature to best feature
			ft_best = ft

	# Return the best feature
	return ft_best

def pca( data, n_components = 2 ):
	"""
	Perform Principal Component Analysis (PCA) on the dataset.

	Parameters:
	data (numpy.ndarray): The input data matrix where each row represents a sample.
	n_components (int): The number of principal components to retain.

	Returns:
	tuple: A tuple containing the projected data, the principal component vectors, and the transpose of V matrix from SVD.
	"""
	# Standardize the data matrix
	data_st = (data - np.mean(data, axis = 0)) / np.std(data, axis = 0)

	# Perform SVD
	u, s, vt = np.linalg.svd(data_st)

	# Select the first n principal components from the Vt matrix
	w = vt[ :n_components, : ]

	# Project the standardized data onto the principal components
	z = np.real(np.dot(data_st, w.T))

	return z, w, vt


def linreg( X, y ):
	"""
	Perform closed-form linear regression to compute the best-fit parameters
	using the normal equation with the Moore-Penrose pseudo-inverse for stability.

	Args:
		X (numpy.ndarray): A 2D array of shape (n_samples, n_features) representing
			the input feature matrix.
		y (numpy.ndarray): A 1D array of shape (n_samples,) representing the target
			variable vector.

	Returns:
		numpy.ndarray: A 1D array of shape (n_features,) containing the best-fit
		parameters (weights) for the linear regression model.
	"""

	# Return the weight vector computed using the normal equation
	return np.dot(

		np.linalg.pinv(

			np.dot(X.T, X) # Gram matrix

		), # Pseudo-inverse of the Gram matrix

		np.dot(X.T, y) # Projection of y onto X

	) # Dot the projection and pseudo-inverse for weight vector

def logreg( X, y, lr = 0.01, num_iter = 10000 ):
	"""
	Perform logistic regression using gradient descent.

	Parameters:
:
	X (numpy.ndarray): The input feature matrix.
	y (numpy.ndarray): The target variable vector.
	lr (float): The learning rate for gradient descent.
	num_iter (int): The number of iterations for gradient descent.

	Returns:
	numpy.ndarray: The parameters for logistic regression.
	"""
	# Add a column of ones to the input feature matrix for the intercept term
	X = np.c_[ np.ones((X.shape[ 0 ], 1)), X ]

	# Initialize the parameters to zeros
	theta = np.zeros(X.shape[ 1 ])

	# Loop over the number of iterations
	for i in range(num_iter):
		# Calculate the linear combination of inputs and parameters
		z = np.dot(X, theta)

		# Apply the sigmoid function to get the predictions
		h = 1 / (1 + np.exp(-z))

		# Calculate the gradient of the loss function
		gradient = np.dot(X.T, (h - y)) / y.size

		# Update the parameters using gradient descent
		theta -= lr * gradient

	return theta



def lda( X, y ):
	"""
	Perform Linear Discriminant Analysis (LDA) for dimensionality reduction.

	Parameters:
	X (numpy.ndarray): The input feature matrix.
	y (numpy.ndarray): The target variable vector.

	Returns:
	numpy.ndarray: The transformation matrix for LDA.
	"""
	# Calculate the mean vectors for each class
	mean_vectors = [ np.mean(X[ y == cl ], axis = 0) for cl in np.unique(y) ]

	# Initialize the within-class scatter matrix
	S_W = np.zeros((X.shape[ 1 ], X.shape[ 1 ]))

	# Loop over each class and mean vector
	for cl, mv in zip(np.unique(y), mean_vectors):
		# Calculate the class scatter matrix
		class_scatter = np.cov(X[ y == cl ].T)

		# Add to the within-class scatter matrix
		S_W += class_scatter

	# Calculate the overall mean of the data
	overall_mean = np.mean(X, axis = 0)

	# Initialize the between-class scatter matrix
	S_B = np.zeros((X.shape[ 1 ], X.shape[ 1 ]))

	# Loop over each class and mean vector
	for cl, mean_vec in enumerate(mean_vectors):
		# Calculate the number of samples in the class
		n = X[ y == cl, : ].shape[ 0 ]

		# Reshape the mean vectors for matrix operations
		mean_vec = mean_vec.reshape(X.shape[ 1 ], 1)

		# Reshape the overall mean for matrix operations
		overall_mean = overall_mean.reshape(X.shape[ 1 ], 1)

		# Add to the between-class scatter matrix
		S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

	# Calculate the eigenvalues and eigenvectors for the matrix (S_W^-1 S_B)
	eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

	# Create the transformation matrix using the top eigenvectors
	W = eig_vecs[:, np.argmax(eig_vals)]

	return W


def dtl( x, y, features = None, default = None ):
	"""
	Implement the Decision Tree Learning algorithm.

	Parameters:
	x (pandas.DataFrame): The input feature matrix.
	y (pandas.Series): The target variable vector.
	features (list, optional): The list of features to consider for splitting.
	default: The default value to return if the dataset is empty.

	Returns:
	dict: The trained decision tree.
	"""
	# Initialize features if not provided
	if features is None:
		features = list(range(x.shape[ 1 ]))

	# Set the default value to the mode of the target variable if not provided
	if default is None:
		default = y.mode()[ 0 ]

	# If x is empty return default
	if len(x) == 0:
		return default

	# Else if all y have same values return y1
	elif len(np.unique(y)) == 1:
		return y.iloc[ 0 ]

	# Else if features is empty then return probabilities for each class
	elif len(features) == 0:
		class_counts = y.value_counts(normalize = True).to_dict()
		return class_counts

	# If normal case
	else:
		# Get best attribute to split using chooseAttribute function
		best = chooseAttribute(x, y, features)

		# Define new internal node using best feature
		tree = { best: { } }

		# Remove the best feature from the feature set
		features = [ i for i in features if i != best ]

		# Elements of x, y with best == 1 or True
		x_t = x[ x.iloc[ :, best ] == 1 ]
		y_t = y[ x.iloc[ :, best ] == 1 ]

		# Elements of x, y with best == 0 or False
		x_f = x[ x.iloc[ :, best ] == 0 ]
		y_f = y[ x.iloc[ :, best ] == 0 ]

		# Set the default value to mode of Y if not empty else previous default
		default_t = y_t.mode()[ 0 ] if not y_t.empty else default

		# Create left child recursively and set to leftChild of tree
		leftChild = dtl(x_t, y_t, features, default_t)
		tree[ best ][ 1 ] = leftChild

		# Set the default value to mode of Y if not empty else previous default
		default_f = y_f.mode()[ 0 ] if not y_f.empty else default

		# Create right child recursively and set to rightChild of tree
		rightChild = dtl(x_f, y_f, features, default_f)
		tree[ best ][ 0 ] = rightChild

		return tree

def dt(xtrain, ytrain, xvalid):
	"""
	Build and predict with a decision tree.

	Parameters:
		xtrain (pd.DataFrame): Training feature matrix.
		ytrain (pd.Series): Training target variable.
		xvalid (pd.DataFrame): Validation feature matrix.

	Returns:
		list: Predictions for the validation dataset.
	"""
	features = list(range(xtrain.shape[1]))
	default = ytrain.mode()[0]
	root = dtl(xtrain, ytrain, features, default)
	predictions = []

	def predict(dt, row):
		if not isinstance(dt, dict):
			return dt
		if all(not isinstance(subtree, dict) for subtree in dt.values()):
			return max(dt.items(), key=lambda item: item[1])[0]
		for ft, subtree in dt.items():
			ft_val = row.iloc[ft]
			if ft_val in subtree:
				return predict(subtree[ft_val], row)
		return default

	for _, row in xvalid.iterrows():
		predictions.append(predict(root, row))

	return predictions


def nb( X, y ):
	"""
	Implement a Naive Bayes classifier.

	Parameters:
	X (numpy.ndarray): The input feature matrix.
	y (numpy.ndarray): The target variable vector.

	Returns:
	GaussianNB: The trained naive bayes classifier.
	"""
	from sklearn.naive_bayes import GaussianNB

	# Create the Gaussian Naive Bayes classifier
	gnb = GaussianNB()

	# Fit the classifier to the data
	gnb.fit(X, y)

	return gnb
