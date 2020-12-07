import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from zlib import crc32

#The imports below are new for 4/7 lecture
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix

HOUSING_PATH = os.path.join("datasets", "housing")

def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

'''
def split_train_test(data, test_ratio): #picks indices at random each time you run it
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]
'''

def do_the_cut(housing):
    housing['income_cat'] = pd.cut(housing["median_income"], bins=[0.,1.5,3.,4.5,6., np.inf], labels=[1,2,3,4,5])
    housing["income_cat"].hist()
    plt.show() #You have to delete the popup window to continue the program

'''def remove_income_cat(strat_train_set, strat_test_set):
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)'''

housing = load_housing_data()
#train_set, test_set = split_train_test(housing, 0.2)

#housing_with_id = housing.reset_index()  #adds an index column
#train_set, test_set = split_train_test_by_id(housing_with_id, 0.2,'index')
#print(len(train_set))
#print(len(test_set))

do_the_cut(housing)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
print(strat_train_set["income_cat"].value_counts() / len(strat_train_set))


#remove_income_cat(strat_train_set, strat_test_set)

housing = strat_train_set.copy()
print(len(housing))

#housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
#plt.show() #This is required in PyCharm - not shown in book

#housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100, label="population", figsize=(10,7),c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True)
#plt.legend()
#plt.show() #This is required in PyCharm - not shown in book

#corr_matrix = housing.corr()
#print(corr_matrix["median_house_value"].sort_values(ascending=False))

#attributes = ["median_house_value", "median_income", "total_rooms","housing_median_age"]
#scatter_matrix(housing[attributes],figsize=(12,8))
#plt.show()

#housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
#plt.show()


housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))



'''import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from zlib import crc32

HOUSING_PATH = os.path.join("datasets", "housing")

def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def split_train_test(data, test_ratio): #picks indices at random each time you run it
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

def do_the_cut(housing):
    housing['income_cat'] = pd.cut(housing["median_income"], bins=[0.,1.5,3.,4.5,6., np.inf], labels=[1,2,3,4,5])
    housing["income_cat"].hist()
    plt.show()

housing = load_housing_data()
train_set, test_set = split_train_test(housing, 0.2)


#housing_with_id = housing.reset_index()  #adds an index column
#train_set, test_set = split_train_test_by_id(housing_with_id, 0.2,'index')
print(len(train_set))
print(len(test_set))

do_the_cut(housing)
'''
