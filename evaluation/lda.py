# Reset the imported modules
import importlib
import models
import utilities
importlib.reload(models)
importlib.reload(utilities)

# Other imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from models import lda
import warnings

# Suppress all FutureWarning messages
warnings.simplefilter(action="ignore", category=FutureWarning)


def predict_classes( z, mu0, mu1 ):
	"""
	Predict classes for LDA projected data based on class means.

	Args:
		z (pd.Series): Projected data (1D Pandas Series).
		mu0 (float): Mean of class 0 in the projected space.
		mu1 (float): Mean of class 1 in the projected space.

	Returns:
		np.ndarray: Predicted classes (0 or 1) for the input data.
	"""
	# Initialize an array for predictions
	predictions = np.zeros(len(z))

	# Iterate through each projected value using .iloc for Pandas indexing
	for i in range(len(z)):
		if abs(z.iloc[ i ] - mu1) > abs(z.iloc[ i ] - mu0):
			predictions[ i ] = 0
		else:
			predictions[ i ] = 1

	return predictions


def process_data(df):
	"""
	Preprocess the spambase dataset by standardizing features and separating the target.

	Args:
		df (pd.DataFrame): The input spambase dataset.

	Returns:
		tuple: Standardized features (x), target variable (y).
	"""
	# Separate target (y) and features (x)
	y = df.iloc[:, -1]
	x = df.iloc[:, :-1]

	# Standardize features using mean and standard deviation from training set
	mu = x.mean()
	sigma = x.std()
	x_standardized = (x - mu) / sigma

	return x_standardized, y


def evaluate_lda(y_actual, y_pred):
	"""
	Evaluate the LDA model using precision, recall, F-measure, and accuracy.

	Args:
		y_actual (pd.Series): Actual target values.
		y_pred (np.ndarray): Predicted target values.

	Returns:
		dict: Evaluation metrics including precision, recall, F-measure, and accuracy.
	"""
	# Calculate true positives, false positives, true negatives, and false negatives
	TP = sum((y_actual == 1) & (y_pred == 1))
	TN = sum((y_actual == 0) & (y_pred == 0))
	FP = sum((y_actual == 0) & (y_pred == 1))
	FN = sum((y_actual == 1) & (y_pred == 0))

	# Compute metrics
	accuracy = np.mean(y_pred == y_actual)
	precision = TP / (TP + FP) if (TP + FP) > 0 else 0
	recall = TP / (TP + FN) if (TP + FN) > 0 else 0
	f_measure = (
		2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
	)

	return {
		"Precision": precision,
		"Recall": recall,
		"F-Measure": f_measure,
		"Accuracy": accuracy,
	}


def main():
	"""
	Perform Linear Discriminant Analysis (LDA) on the spambase dataset and evaluate its performance.

	This function preprocesses the dataset, trains an LDA model, projects the data,
	predicts the classes, and evaluates its performance using metrics.
	"""
	# Set random seed for reproducibility
	np.random.seed(0)

	# Load and shuffle the dataset
	df = pd.read_csv("../data/spambase.data", header=None).sample(
		frac=1, random_state=0
	).reset_index(drop=True)

	# Process data to standardize features and separate target
	x, y = process_data(df)

	# Split into train and test sets
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(1 / 3))

	# Perform LDA to get the transformation matrix
	w = lda(x_train.values, y_train.values)

	# Project training and test data onto the LDA components
	z_train = np.dot(x_train, w)
	z_test = np.dot(x_test, w)

	# Calculate the means of the projected classes
	mu0, mu1 = np.mean(z_train[y_train == 0]), np.mean(z_train[y_train == 1])

	# Predict training and test classes
	y_train_pred = predict_classes(pd.Series(z_train), mu0, mu1)
	y_test_pred = predict_classes(pd.Series(z_test), mu0, mu1)

	# Evaluate the model on training and test sets
	train_metrics = evaluate_lda(y_train.values, y_train_pred)
	test_metrics = evaluate_lda(y_test.values, y_test_pred)

	# Print training metrics
	print("Training Metrics")
	print(f"Precision: {train_metrics['Precision'] * 100:.2f}%")
	print(f"Recall: {train_metrics['Recall'] * 100:.2f}%")
	print(f"F-Measure: {train_metrics['F-Measure'] * 100:.2f}%")
	print(f"Accuracy: {train_metrics['Accuracy'] * 100:.2f}% \n")

	# Print test metrics
	print("Testing Metrics")
	print(f"Precision: {test_metrics['Precision'] * 100:.2f}%")
	print(f"Recall: {test_metrics['Recall'] * 100:.2f}%")
	print(f"F-Measure: {test_metrics['F-Measure'] * 100:.2f}%")
	print(f"Accuracy: {test_metrics['Accuracy'] * 100:.2f}%")


if __name__ == "__main__":
	# Execute the main function
	main()
