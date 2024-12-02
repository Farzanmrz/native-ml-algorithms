# Imports
import os
import cv2
import numpy as np
from PIL import Image
import math

def calc_rmse(y_true, y_pred):
    """
    Calculate the Root Mean Square Error (RMSE) between actual and predicted values.

    Args:
        y_true (numpy.ndarray): The actual target values.
        y_pred (numpy.ndarray): The predicted target values.

    Returns:
        float: The RMSE value, representing the average prediction error.
    """
    # Compute the RMSE between actual and predicted values
    return ((sum((y_true - y_pred) ** 2)) / len(y_true)) ** 0.5

def calc_smape(y_true, y_pred):
    """
    Calculate the Symmetric Mean Absolute Percentage Error (SMAPE)
    between actual and predicted values.

    Args:
        y_true (numpy.ndarray): The actual target values.
        y_pred (numpy.ndarray): The predicted target values.

    Returns:
        float: The SMAPE value, representing the average percentage error.
    """
    # Compute the SMAPE between actual and predicted values
    return sum(abs(y_true - y_pred) / (abs(y_true) + abs(y_pred))) / len(y_true)



def load_yale_faces( data_dir, img_size = (40, 40) ):
	files = [ file for file in os.listdir(data_dir) if file != ".DS_Store" ]
	data = np.array([ np.array(Image.open(os.path.join(data_dir, file)).resize(img_size)).ravel() for file in files ])
	return data

def cross_validation( df, S, num_seeds = 20 ):
	"""
	Perform cross-validation on the dataset to evaluate the model performance.

	Args:
		df (DataFrame): The dataset containing features and the target variable.
		S (int): The number of folds for cross-validation.
		num_seeds (int): The number of random seeds to use for averaging the results.

	Prints:
		The mean and standard deviation of the RMSE across all seeds.
	"""
	from models import linreg  # Import here to avoid circular import

	# List to store RMSE values for each seed
	rmse_vals = [ ]

	# Loop through the number of seeds
	for seed in range(num_seeds):

		# Set seed for reproducibility
		np.random.seed(seed)

		# Shuffle the dataset, reset the index, and add a dummy variable for intercept in one step
		currdf = df.sample(frac = 1).reset_index(drop = True).assign(dummy = 1.0)

		# Declare variable to hold total sum of SE values over different folds
		se_foldSum = 0

		# Loop through each fold
		for i in range(S):

			# Split into training and validation sets as per fold
			training_data = currdf.loc[ currdf.index[ currdf.index % S != i ].tolist() ]
			validation_data = currdf.loc[ currdf.index[ currdf.index % S == i ].tolist() ]

			# Set x and y separate
			y_train = training_data[ 'charges' ]
			y_val = validation_data[ 'charges' ]
			x_train = training_data.drop('charges', axis = 1)
			X_val = validation_data.drop('charges', axis = 1)

			# Get the squared error for current fold and add to sum
			se_foldSum += np.sum(np.square(y_val - np.dot(X_val, linreg(x_train, y_train))))

		# Append RMSE for current seed to all RMSE values
		rmse_vals.append(math.sqrt(((se_foldSum) / (currdf.shape[ 0 ]))))

	# Output the mean and standard deviation of the RMSE across all seeds
	print(f"Cross-Val S = {S} :-\tMean: {np.mean(rmse_vals)}\t Stdev: {np.std(rmse_vals)}")
