import os
import cv2
import numpy as np
from PIL import Image


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
		np.random.seed(seed)  # Set the seed for reproducibility

		# Shuffle the dataset
		currdf = df.sample(frac = 1)

		# Reset the index
		currdf.reset_index(drop = True, inplace = True)

		# Add a dummy variable for intercept
		currdf.insert(0, 'dummy', np.ones(len(currdf)))

		# Declare variable to hold total sum of SE values over different folds
		se_foldSum = 0

		# Loop through each fold
		for i in range(S):
			# Get rows for training and validation
			row_train = currdf.index[ currdf.index % S != i ].tolist()
			row_val = currdf.index[ currdf.index % S == i ].tolist()

			# Split into training and validation sets
			training_data = currdf.loc[ row_train ]
			validation_data = currdf.loc[ row_val ]

			# Set x and y separate
			y_train = training_data[ 'charges' ]
			y_val = validation_data[ 'charges' ]
			x_train = training_data.drop('charges', axis = 1)
			X_val = validation_data.drop('charges', axis = 1)

			# Calculate the weight vector using the linreg function
			w = linreg(x_train, y_train)

			# Make predictions on the validation set
			y_val_pred = np.dot(X_val, w)

			# Calculate squared error values
			se_val = (y_val - y_val_pred) ** 2

			# Sum all squared error values in vector
			se_sum = np.sum(se_val)

			# Add to total sum over folds
			se_foldSum += se_sum

		# Calculate RMSE for current seed
		currRmse = ((se_foldSum) / (currdf.shape[ 0 ])) ** 0.5

		# Append to all RMSE values
		rmse_vals.append(currRmse)

	# Output the mean and standard deviation of the RMSE across all seeds
	print(f"Mean RMSE for S = {S}: {np.mean(rmse_vals)}")
	print(f"Standard Deviation for S = {S}: {np.std(rmse_vals)}")
	print()
