
'''
Vincent Perez, MS-IDBT Cohort 6 - 5847842892
ACAD 499 Machine Intelligence / AI for Design Applications, Spring 2020
vincenjp@usc.edu
Final Project Part 2
'''

#Importing Libraries
import numpy as np
from numpy import array
from numpy import argmax
import matplotlib.pyplot as plt #This for math?
from matplotlib.pyplot import figure
import csv # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd # importing pandas as pd
pd.set_option('display.max_rows',20000, 'display.max_columns',100)
import seaborn as sb #import Seaborn to for visualizing data
import plotly.express as px
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.linear_model import LinearRegression
from sklearn import metrics
#import warnings
#warnings.filterwarnings('ignore')


# Making data frame from the csv file
'''columns = "name pokedex_number generation is_sub_legendary is_legendary is_mythical type_number abilities_number " \
          "total_points hp attack defense sp_attack sp_defense speed catch_rate base_friendship base_experience " \
          "egg_type_number".split(" ") # Declare the columns names'''

pokeDf = pd.read_csv("Dataset/pokedex_(Update.04.20).csv")

print(pokeDf.shape)
print(pokeDf.describe())

#This is where we try and replace the typing with a number, decided against it because the algorithm could try and use
# the ascending numbers as value indicators

'''def type_numbering(string) :
    if string == 'Normal' :
        return 1
    elif string== 'Fire' :
        return 2
    elif string == 'Fighting' :
        return 3
    elif string == 'Water' :
        return 4
    elif string == 'Flying' :
        return 5
    elif string == 'Grass' :
        return 6
    elif string == 'Poison' :
        return 7
    elif string == 'Electric' :
        return 8
    elif string == 'Ground' :
        return 9
    elif string == 'Psychic' :
        return 10
    elif string == 'Rock' :
        return 11
    elif string == 'Ice' :
        return 12
    elif string == 'Bug' :
        return 13
    elif string == 'Dragon' :
        return 14
    elif string == 'Ghost' :
        return 15
    elif string == 'Dark' :
        return 16
    elif string == 'Steel' :
        return 17
    elif string == 'Fairy' :
        return 18
    else :
        return 0

pokeDf['Type 1'] = pokeDf['type_1'].apply(type_numbering)
pokeDf['Type 2'] = pokeDf['type_2'].apply(type_numbering)

poketypes = ['normal','fire','water', 'water', 'electric', 'grass', 'ice', 'fighting', 'poison', 'ground',
              'flying','psychic', 'bug', 'rock', 'ghost', 'dragon', 'dark', 'dark','steel', 'fairy']'''

#general fact finding
print(pokeDf.groupby(['generation','is_legendary']).count()['name'])
print(pokeDf['is_legendary'].value_counts())

#Varrying lists that are important

list_of_num_fields = ['Unnamed: 0','pokedex_number', 'generation', 'is_sub_legendary', 'is_legendary', 'is_mythical',
                      'type_number', 'hp', 'defense', 'attack','sp_attack','sp_defense','speed', 'catch_rate', 'base_friendship',
                      'base_experience', 'against_fairy', 'against_steel', 'against_dark', 'against_dragon', 'against_ghost',
                      'against_rock','against_bug', 'against_psychic', 'against_flying', 'against_ground', 'against_poison',
                      'against_fight','against_ice','against_ice', 'against_grass', 'against_electric', 'against_water',
                      'against_fire','against_normal', 'abilities_number', 'height_m', 'weight_kg', 'type_number', ]

list_of_object_fields = ['type_1', 'type_2', 'egg_type_2', 'egg_type_1','growth_rate', 'ability_hidden','ability_2', 'ability_1',]


#This is where we create a list of the fields that are in our data frame so that we can effectively pivot and analyze data
listOTypes =  ['Unnamed: 0']

list_of_object_field= ['name,', 'type_1', 'type_2',]

list_of_final_fields = ['pokedex_number', 'generation', 'is_sub_legendary', 'is_legendary', 'is_mythical', 'type_number'
                        'hp', 'defense', 'attack','sp_attack','sp_defense','speed', 'catch_rate', 'base_friendship',
                        'base_experience']

list_of_stats = ['hp', 'defense', 'attack','sp_attack','sp_defense','speed', 'catch_rate', 'base_friendship',
                 'base_experience']

list_to_drop = ['Unnamed: 0','name', 'german_name', 'japanese_name', 'species', 'type_1', 'type_2', 'ability_1', 'ability_2',
                'growth_rate', 'egg_type_1', 'egg_type_2', 'type_1', 'type_2', 'egg_type_2', 'egg_type_1','growth_rate',
                'ability_hidden' ]

list_of_in_game_random= ['height_m', 'weight_kg', 'ability_2', 'ability_1','abilities_number', 'ability_hidden'
                         'growth_rate', 'egg_type_2', 'egg_type_1', 'percentage_male', 'egg_cycles', ]

list_of_against= ['against_fairy', 'against_steel', 'against_dark', 'against_dragon', 'against_ghost','against_rock',
                'against_bug', 'against_psychic', 'against_flying', 'against_ground', 'against_poison', 'against_fight',
                'against_ice','against_ice', 'against_grass', 'against_electric', 'against_water', 'against_fire',
                'against_normal',]

pokemon_stats = pokeDf.drop(['against_fairy', 'against_steel', 'against_dark', 'against_dragon', 'against_ghost',
                             'against_rock', 'against_bug', 'against_psychic', 'against_flying', 'against_ground',
                             'against_poison', 'against_fight', 'against_ice','against_ice', 'against_grass',
                             'against_electric', 'against_water', 'against_fire', 'against_normal','height_m',
                             'weight_kg', 'ability_2', 'ability_1','abilities_number', 'ability_hidden', 'growth_rate',
                             'egg_type_2', 'egg_type_1', 'percentage_male', 'egg_cycles', 'Unnamed: 0','name', 'german_name',
                             'japanese_name', 'species', 'type_1', 'type_2', 'ability_1', 'ability_2', 'growth_rate',
                             'egg_type_2', 'egg_type_1','growth_rate', 'ability_hidden', 'type_1', 'type_2','pokedex_number',
                             'generation', 'is_sub_legendary', 'is_legendary', 'is_mythical', 'type_number'], axis=1)

#this is our new DF for the algorithm
pokePure = pokeDf.drop(list_to_drop, axis=1)

#This is where we fill in any missing values with a 0
pokePure.fillna(0, inplace=True)

#poke print (count, sum, mean, median, mode, std, min, max, abs, prod, cumsum, cumprod
#print(pokePure.mean())
#print("total_points mean:", pokePure['total_points'].mean())
#print("base_experience mean:", pokePure['base_experience'].mean())
print(pokePure.describe().astype(int))

#this just shows us the number of missing data in cells
#print("Number of missing cells: ", pokePure.isnull().sum().sum())

#Pokesearch
'''pokesearch = input('Enter Pokemon name to Find: ')
while pokesearch is True:
    for row in pokeDf:
    #if current rows 2nd value is equal to input, print that row
        if pokesearch == pokeDf['name']:
            print("yes")
        continue'''

#displot will show both a histogram of you data and the Kernel Density Estimation (KDE)

plt.figure(figsize=(10,5))
sb.distplot(pokePure.base_experience, kde=False)

#boxplot for one column of data
plt.figure(figsize=(8,5))
sb.boxplot(pokePure.total_points)

plt.figure(figsize=(15,15))
sb.boxplot(data=pokemon_stats)

#General Graph
pokeDf.plot(x='total_points', y='base_experience', style='o')
plt.title('Total Points vs Base Experience')
plt.xlabel('total_points')
plt.ylabel('base_experience')
plt.show()

#Average Stats
plt.figure(figsize=(15,10))
plt.tight_layout()
sb.distplot(pokeDf['total_points'])

#Using the Pearson Correlation since we want to calculate the pearson coefficient of correlation.
#pokecorr = pokeDf.corr(method='pearson')
#print(pokecorr)

#Correlation matrix
pokeCorrMatrix = pokemon_stats.corr()
plt.figure(figsize=(16, 16))
sb.heatmap(pokeCorrMatrix, annot=True)
plt.show()

#Regression model with x & Y
#this is our linear regression model
# define the target variable (dependent variable) as y
'''
x = pokePure.base_experience
y = pokePure.total_points
'''

X = pokePure['base_experience'].values.reshape(-1,1)
y = pokePure['total_points'].values.reshape(-1,1)

# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(pokePure, y, test_size=0.2)
print('This is our training set:', X_train.shape, y_train.shape)
print('This is our test set:',X_test.shape, y_test.shape)

# fit a model
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)

#To retrieve the intercept:
print('Intercept:', lm.intercept_)
#For retrieving the slope:
print('Slope:', lm.coef_)

print('These are our predictions:',predictions[:5])

# The line / model
plt.scatter(y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')

#Print Score, The left plot has a perfect positive linear relationship between x and y, so r = 1
print('Score:', model.score(X_test, y_test))

#Predictions
y_pred = lm.predict(X_test)
pokePure = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(pokePure[:10])

#algorithm performance evals
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

'''
#Linear Regression Model 2 (Simple)
linearRegressor = LinearRegression()
linearRegressor.fit(X_train, y_train)
yPrediction = linearRegressor.predict(X_test)

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, linearRegressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
'''



#References

'''
Starter: Complete Pokemon Dataset 35d8be4d-7
    https://www.kaggle.com/kerneler/starter-complete-pokemon-dataset-35d8be4d-7
Data Cleaning with Python and Pandas: Detecting Missing Values
    https://towardsdatascience.com/data-cleaning-with-python-and-pandas-detecting-missing-values-3e9c6ebcf78b

Topic 1. Exploratory Data Analysis with Pandas
    https://www.kaggle.com/kashnitsky/topic-1-exploratory-data-analysis-with-pandas

PYTHON DATA ANALYSIS WITH PANDAS AND MATPLOTLIB
    https://ourcodingclub.github.io/tutorials/pandas-python-intro/

How to build your own AlphaZero AI using Python and Keras
    https://medium.com/applied-data-science/how-to-build-your-own-alphazero-ai-using-python-and-keras-7f664945c188

Pandas Tutorial: Analyzing Video Game Data with Python and Pandas
    https://www.dataquest.io/blog/pandas-python-tutorial/

Python | Pandas dataframe.corr()
    https://www.geeksforgeeks.org/python-pandas-dataframe-corr/

Starter: Complete Pokemon Dataset 35d8be4d-7
    https://www.kaggle.com/kerneler/starter-complete-pokemon-dataset-35d8be4d-7

Pearson Coefficient of Correlation with Python
https://levelup.gitconnected.com/pearson-coefficient-of-correlation-using-pandas-ca68ce678c04

Data Cleaning with Python and Pandas: Detecting Missing Values
    https://towardsdatascience.com/data-cleaning-with-python-and-pandas-detecting-missing-values-3e9c6ebcf78b

Python | Pandas dataframe.drop_duplicates()
    https://www.geeksforgeeks.org/python-pandas-dataframe-drop_duplicates/

The leading front-end for ML & data science models in Python, R, and Julia.
    https://plotly.com/python/

Data Science with Python: Intro to Loading, Subsetting, and Filtering Data with pandas
    https://towardsdatascience.com/data-science-with-python-intro-to-loading-and-subsetting-data-with-pandas-9f26895ddd7f

Beginner’s Guide to Machine Learning with Python
    https://towardsdatascience.com/beginners-guide-to-machine-learning-with-python-b9ff35bc9c51

Python Basics for Data Science
    https://towardsdatascience.com/python-basics-for-data-science-6a6c987f2755

sklearn.preprocessing.StandardScaler
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

Categorical encoding using Label-Encoding and One-Hot-Encoder
    https://towardsdatascience.com/categorical-encoding-using-label-encoding-and-one-hot-encoder-911ef77fb5bd

Python Basics for Data Science
    https://towardsdatascience.com/python-basics-for-data-science-6a6c987f2755

NumPy, SciPy, and Pandas: Correlation With Python
    https://realpython.com/numpy-scipy-pandas-correlation-python/#spearman-correlation-coefficient

Mapping Categorical Data in pandas
    https://benalexkeen.com/mapping-categorical-data-in-pandas/
'''
