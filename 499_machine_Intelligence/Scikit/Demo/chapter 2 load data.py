import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

HOUSING_PATH = os.path.join("datasets", "housing")


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def split_train_test(data, test_ratio):  # picks indices at random each time you run it
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]


housing = load_housing_data()
'''
housing.info()  # The info method is usefult to get a quick description of the data
print(housing[
          "ocean_proximity"].value_counts())  # Shows what categories exist and how many districts belong to each category
print(housing.describe())  # This method shows a summary of the numerical attributes
housing.hist(bins=50, figsize=(20, 15))  # shows the number of instances (vertical axis) that have a given value range
plt.show()  # Plots a histogram for each numerical attribute'''

train_set, test_set = split_train_test(housing, 0.2)

print(len(train_set))
print(len(test_set))



'''
REFERENCES: 
https://en.wikipedia.org/wiki/Hash_function
https://pythonhealthcare.org/2018/04/18/74-using-numpy-to-generate-random-numbers-or-shuffle-arrays/
'''
