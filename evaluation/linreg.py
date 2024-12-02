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
from utilities import calc_rmse, calc_smape, cross_validation
from models import linreg
from sklearn.model_selection import train_test_split
import seaborn as sns
import warnings

# Suppress all FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

def process_data( df ):
    """
    Preprocess the insurance dataset and return features (x) and target (y).

    Args:
        df (pd.DataFrame): The input insurance dataset.

    Returns:
        pd.DataFrame: Features (x) with a dummy variable.
        pd.Series: Target variable (y).
    """
    # Convert 'sex' and 'smoker' to binary values
    df[ [ 'sex', 'smoker' ] ] = df[ [ 'sex', 'smoker' ] ].replace({ 'male': 1, 'female': 0, 'yes': 1, 'no': 0 })

    # One-hot encode 'region' and convert to float
    df = pd.get_dummies(df, columns = [ 'region' ], prefix = '', prefix_sep = '').astype(float)

    # Shuffle rows and reset the index
    df = df.sample(frac = 1, random_state = 0).reset_index(drop = True)

    # Set target (y) and features (x), adding a dummy column to x in one step
    x = pd.concat(
        [
            pd.Series(1.0, index = df.index, name = 'dummy'),
            df.drop(columns = 'charges')
        ], axis = 1
    )
    y = df[ 'charges' ]

    # Return the processed df with x, y
    return df, x, y


def main():
    """
    Perform linear regression on the insurance dataset and evaluate its performance.

    This function preprocesses the dataset, trains a linear regression model,
    evaluates its performance using RMSE and SMAPE, visualizes results,
    and performs k-fold cross-validation.

    Prints:
        - RMSE and SMAPE metrics for training and validation sets.
        - Mean RMSE and standard deviation from cross-validation.
    """

    # Set random seed for reproducibility
    np.random.seed(0)

    # Load the dataset into df
    raw_df = pd.read_csv('../data/insurance.csv')

    # Get the feature matrix and target vector from processed df
    df, x, y = process_data(raw_df)

    # Split into train and test set
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = (1 / 3))

    # Calculate the weight vector using the linreg function
    w = linreg(x_train, y_train)

    # Make predictions on the training and validation sets
    y_train_pred = np.dot(x_train, w)
    y_val_pred = np.dot(x_val, w)

    # Calculate RMSE for training and validation sets
    rmse_train = calc_rmse(y_train, y_train_pred)
    rmse_val = calc_rmse(y_val, y_val_pred)

    # Calculate SMAPE for training and validation sets
    smape_train = calc_smape(y_train, y_train_pred)
    smape_val = calc_smape(y_val, y_val_pred)

    # Print the RMSE and SMAPE results
    print(f"Training RMSE: {rmse_train:.2f}, Validation RMSE: {rmse_val:.2f}")
    print(f"Training SMAPE: {smape_train:.2%}, Validation SMAPE: {smape_val:.2%}\n")

    # Create the scatter plot with a regression line and add labels/title
    sns.regplot(
        x = y_val, y = y_val_pred,
        scatter_kws = { "alpha": 0.7, "label": "Data points" },
        line_kws = { "color": "red", "label": "Line of best fit" },
        truncate = False, ci = None
    )
    plt.xlabel('Actual charges')
    plt.ylabel('Predicted charges')
    plt.title('Actual vs Predicted charges')
    plt.show()

    # Perform cross-validation with different S values
    cross_validation(df, S = 3)
    cross_validation(df, S = 223)
    cross_validation(df, S = 1338)


if __name__ == "__main__":
    # Execute the main function
    main()
