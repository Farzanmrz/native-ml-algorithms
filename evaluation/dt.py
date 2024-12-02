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
import seaborn as sns
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import re
from sklearn.metrics import confusion_matrix as cm
from models import dt
import warnings

# Suppress all FutureWarning messages
warnings.simplefilter(action="ignore", category=FutureWarning)


def preprocess_yalefaces():
    """
    Load and preprocess the Yale Faces dataset.

    Returns:
        tuple: Training features (x_train), test features (x_test),
        training targets (y_train), and test targets (y_test).
    """
    # Load Yale Faces dataset
    files = os.listdir('../data/yalefaces/')

    # Keep only files starting with "subject"
    files = [file for file in files if file.startswith('subject')  and not file.endswith('.gif')]

    # Define variables to store x and y
    x = np.zeros((154, 1600))
    y = np.zeros((154,))

    # Define row index to iterate through matrix
    row = 0

    # Process each file
    for file in files:
        # Read image
        image = Image.open(f"../data/yalefaces/{file}")

        # Resize image to 40x40
        image = image.resize((40, 40))

        # Flatten the image to row vector and add to x
        x[row] = np.array(image).ravel()

        # Extract subject number from filename and add to y
        match = re.search(r'subject(\d+)', file)
        if match:
            y[row] = int(match.group(1))

        # Increment row
        row += 1

    # Combine x and y into a DataFrame
    df = pd.concat(
        [pd.DataFrame(x), pd.DataFrame(y, columns=['target'])], axis=1
    )

    # Initialize variables to store training and testing sets
    x_train = pd.DataFrame(dtype="float64")
    x_test = pd.DataFrame(dtype="float64")
    y_train = pd.Series(dtype="float64")
    y_test = pd.Series(dtype="float64")

    # Get the unique target values
    target_unique = df["target"].unique()

    # Split each class into training and testing sets
    for target in target_unique:
        # Get the subset of the DataFrame for the current target
        curr_df = df[df["target"] == target]

        # Shuffle and reset the index
        np.random.seed(0)
        curr_df = curr_df.sample(frac=1).reset_index(drop=True)

        # Allocate first 7 rows to training and the rest to testing
        curr_xtrain = curr_df.iloc[:7, :-1]
        curr_ytrain = curr_df.iloc[:7, -1]
        curr_xtest = curr_df.iloc[7:, :-1]
        curr_ytest = curr_df.iloc[7:, -1]

        # Append to the global training and testing sets
        x_train = pd.concat([x_train, curr_xtrain], ignore_index=True)
        y_train = pd.concat([y_train, curr_ytrain], ignore_index=True)
        x_test = pd.concat([x_test, curr_xtest], ignore_index=True)
        y_test = pd.concat([y_test, curr_ytest], ignore_index=True)

    # Shuffle training and testing DataFrames again for better randomness
    train_df = pd.concat([x_train, y_train], axis=1)
    test_df = pd.concat([x_test, y_test], axis=1)

    np.random.seed(0)
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    test_df = test_df.sample(frac=1).reset_index(drop=True)

    # Separate features and targets
    x_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    x_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]

    return x_train, x_test, y_train, y_test


def preprocess_ctg():
    """
    Load and preprocess the CTG dataset.

    Returns:
        tuple: Features (x), target (y).
    """
    # Load the CTG dataset
    df = pd.read_csv("../data/CTG.csv")

    # Drop the first row (NA) and the second last column (CLASS)
    df = df.drop(0, axis=0)
    df = df.drop("CLASS", axis=1)

    # Separate features and target
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return x, y


def evaluate_dt(y_actual, y_pred):
    """
    Evaluate the decision tree model using accuracy and confusion matrix.

    Args:
        y_actual (pd.Series): Actual target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        dict: Evaluation metrics and the confusion matrix.
    """
    accuracy = np.mean(y_actual == y_pred)
    conf_matrix = cm(y_actual, y_pred)

    return {"Accuracy": accuracy, "Confusion Matrix": conf_matrix}


def main():
    """
    Train decision trees on CTG and Yale Faces datasets and evaluate their performance.

    This function preprocesses both datasets, trains decision trees, predicts classes,
    and evaluates performance using accuracy and confusion matrices.
    """
    # Process and evaluate the CTG dataset
    x_ctg, y_ctg = preprocess_ctg()
    x_train_ctg, x_test_ctg, y_train_ctg, y_test_ctg = train_test_split(
        x_ctg, y_ctg, test_size=0.33, random_state=0
    )

    predictions_ctg = dt(x_train_ctg, y_train_ctg, x_test_ctg)
    results_ctg = evaluate_dt(y_test_ctg, predictions_ctg)

    print("CTG Dataset Results:")
    print(f"Accuracy: {results_ctg['Accuracy'] * 100:.2f}%")
    print()

    # Process and evaluate the Yale Faces dataset
    x_train_yale, x_test_yale, y_train_yale, y_test_yale = preprocess_yalefaces()
    predictions_yale = dt(x_train_yale, y_train_yale, x_test_yale)
    results_yale = evaluate_dt(y_test_yale, predictions_yale)

    print("Yale Faces Dataset Results:")
    print(f"Accuracy: {results_yale['Accuracy'] * 1000:.2f}%")


if __name__ == "__main__":
    main()
