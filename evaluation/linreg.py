import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utilities import cross_validation
from models import linreg
from sklearn.model_selection import train_test_split

pd.set_option('future.no_silent_downcasting', True)
def main():
    """
    Main function to perform linear regression on the insurance dataset,
    evaluate the model using RMSE and SMAPE, and perform cross-validation.
    """
    # Set random seed for reproducibility
    np.random.seed(0)

    # Load the dataset
    df = pd.read_csv('../data/insurance.csv')

    # Convert sex, smoker to binary
    df[ 'sex' ] = df[ 'sex' ].replace({ 'male': 1, 'female': 0 });
    df[ 'smoker' ] = df[ 'smoker' ].replace({ 'yes': 1, 'no': 0 });

    # One-hot encode the region column and concatenate to original df
    df = pd.get_dummies(df, columns = [ 'region' ], prefix = '', prefix_sep = '')

    # Convert all columns to float for processing
    df = df.astype(float);

    # Set random seem
    np.random.seed(0)

    # Shuffle rows
    df = df.sample(frac = 1)

    # Reset the index
    df.reset_index(drop = True, inplace = True)

    # Seperate charges column as value to predict
    y = df[ 'charges' ];
    x = df.drop('charges', axis = 1)

    # Add dummy variable to x
    x.insert(0, 'dummy', np.ones(shape = (x.shape[ 0 ], 1)));

    # Split into train and test set
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = (1 / 3));

    # Calculate the weight vector using the linreg function
    w = linreg(x_train, y_train)

    # Make predictions on the training and validation sets
    y_train_pred = np.dot(x_train, w)
    y_val_pred = np.dot(x_val, w)

    # Calculate RMSE for training and validation sets
    rmse_train = (sum((y_train - y_train_pred)**2) / len(y_train))**0.5
    rmse_val = (sum((y_val - y_val_pred)**2) / len(y_val))**0.5

    # Calculate SMAPE for training and validation sets
    smape_train = sum((abs(y_train - y_train_pred)) / (abs(y_train) + abs(y_train_pred))) / len(y_train)
    smape_val = sum((abs(y_val - y_val_pred)) / (abs(y_val) + abs(y_val_pred))) / len(y_val)

    # Print the RMSE and SMAPE results
    print(f"Training RMSE: {rmse_train:.2f}, Validation RMSE: {rmse_val:.2f}")
    print(f"Training SMAPE: {smape_train:.2%}, Validation SMAPE: {smape_val:.2%}")
    print()

    # Plot the predicted vs actual charges
    plt.scatter(y_val, y_val_pred)
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
