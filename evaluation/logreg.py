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
from models import logreg
import warnings

# Suppress all FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)


def process_data(df):
    """
    Preprocess the spambase dataset by standardizing features and separating the target.

    Args:
        df (pd.DataFrame): The input spambase dataset.

    Returns:
        pd.DataFrame: Standardized features (x).
        pd.Series: Target variable (y).
    """
    # Separate target (y) and features (x)
    y = df.iloc[:, -1]
    x = df.iloc[:, :-1]

    # Standardize features using mean and standard deviation from training set
    mu = x.mean()
    sigma = x.std()
    x_standardized = (x - mu) / sigma

    return x_standardized, y


def evaluate_model(x_test, y_test, theta):
    """
    Evaluate logistic regression model using precision, recall, F-measure, and accuracy.

    Args:
        x_test (pd.DataFrame): Test feature set.
        y_test (pd.Series): Actual target values for the test set.
        theta (np.ndarray): Learned logistic regression parameters.

    Returns:
        dict: Evaluation metrics including precision, recall, F-measure, and accuracy.
    """
    # Add dummy column to x_test for the intercept term
    x_test = np.c_[np.ones((x_test.shape[0], 1)), x_test]

    # Predict probabilities
    y_test_pred_prob = 1 / (1 + np.exp(-np.dot(x_test, theta)))

    # Convert probabilities to binary predictions (0 or 1)
    y_test_pred = np.where(y_test_pred_prob >= 0.5, 1, 0)

    # Calculate true positives, false positives, true negatives, and false negatives
    TP = sum((y_test == 1) & (y_test_pred == 1))
    TN = sum((y_test == 0) & (y_test_pred == 0))
    FP = sum((y_test == 0) & (y_test_pred == 1))
    FN = sum((y_test == 1) & (y_test_pred == 0))

    # Compute metrics
    accuracy = np.mean(y_test_pred == y_test)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "Precision": precision,
        "Recall": recall,
        "F-Measure": f_measure,
        "Accuracy": accuracy
    }


def main():
    """
    Perform logistic regression on the spambase dataset and evaluate its performance.

    This function preprocesses the dataset, trains a logistic regression model,
    evaluates its performance using metrics (precision, recall, F-measure, accuracy),
    and visualizes training and validation losses over epochs.

    Prints:
        - Evaluation metrics including precision, recall, F-measure, and accuracy.
    """
    # Set random seed for reproducibility
    np.random.seed(0)

    # Load the dataset
    raw_df = pd.read_csv('../data/spambase.data', header=None)

    # Shuffle rows and reset the index
    df = raw_df.sample(frac=1, random_state=0).reset_index(drop=True)

    # Process data to standardize features and separate target
    x, y = process_data(df)

    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(1 / 3))

    # Train logistic regression model using gradient descent
    theta = logreg(x_train.values, y_train.values, lr=0.01, num_iter=5000)

    # Evaluate the model on the test set
    metrics = evaluate_model(x_test, y_test, theta)

    # Print evaluation metrics
    print(f"Precision: {metrics['Precision'] * 100:.2f}%")
    print(f"Recall: {metrics['Recall'] * 100:.2f}%")
    print(f"F-Measure: {metrics['F-Measure'] * 100:.2f}%")
    print(f"Accuracy: {metrics['Accuracy'] * 100:.2f}%")


if __name__ == "__main__":
    # Execute the main function
    main()
