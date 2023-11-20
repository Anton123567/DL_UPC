

import pandas as pd
import numpy as np
import re

def custom_parser(array_string):
    """
    Parse a string representation of an array with missing commas and convert it to a NumPy array.

    Args:
    array_string (str): The string representation of the array.

    Returns:
    numpy.ndarray: The resulting NumPy array.
    """
    # Remove brackets and split the string using spaces while filtering out empty strings
    elements = [float(x) for x in re.findall(r"\d+\.\d*", array_string)]
    return np.array(elements)

def convert_column_to_numpy(df, column_name):
    """
    Convert a column of custom stringified arrays into actual NumPy arrays.

    Args:
    df (pandas.DataFrame): The DataFrame containing the column.
    column_name (str): The name of the column to convert.

    Returns:
    pandas.DataFrame: The DataFrame with the converted column.
    """
    df[column_name] = df[column_name].apply(custom_parser)
    return df



# Now df['array_column'] contains actual NumPy arrays


train = pd.read_csv("./../features_train.csv")
train = convert_column_to_numpy(train, "embedding")

X_train = train["embedding"]
y_train = train["label"]

from sklearn import svm

clf = svm.LinearSVC()
clf.fit(X_train, y_train)