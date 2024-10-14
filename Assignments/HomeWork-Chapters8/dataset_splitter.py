# dataset_splitter.py

from sklearn.model_selection import train_test_split

def train_val_test_split(X, y, random_state=42):
    data_size = len(X)
    
    # Determine the proportions for training, validation, and test sets based on data size
    if data_size <= 10000:
        train_size = 0.7
        val_size = 0.15
        test_size = 0.15
    elif data_size <= 100000:
        train_size = 0.9
        val_size = 0.05
        test_size = 0.05
    elif data_size <= 1000000:
        train_size = 0.9
        val_size = 0.05
        test_size = 0.05
    else:
        train_size = 0.98
        val_size = 0.01
        test_size = 0.01

    # Split data into training and temp sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1 - train_size), random_state=random_state)

    # Calculate the proportion of validation and test from the temp set
    val_test_ratio = val_size / (val_size + test_size)

    # Further split temp data into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1 - val_test_ratio), random_state=random_state)
    
    return X_train, X_val, X_test, y_train, y_val, y_test