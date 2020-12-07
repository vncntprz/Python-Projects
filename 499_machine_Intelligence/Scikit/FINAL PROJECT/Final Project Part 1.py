
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
pd.options.display.max_columns = 500
import seaborn as sb #import Seaborn to for visualizing data
import plotly.express as px
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Making data frame from the csv file
'''columns = "name pokedex_number generation is_sub_legendary is_legendary is_mythical type_number abilities_number " \
          "total_points hp attack defense sp_attack sp_defense speed catch_rate base_friendship base_experience " \
          "egg_type_number".split(" ") # Declare the columns names'''

pokeDf = pd.read_csv("Dataset/pokedex_(Update.04.20).csv")

#transform types into numbers
pokeDf['type_1'] = pokeDf['type_1'].replace({1: 'normal', 2: 'fire', 3: 'water', 4: 'water', 5: 'elelctric', 6: 'grass',
                                             7: 'ice', 8: 'fighting', 9: 'poison', 10:'ground', 11: 'flying', 12: 'pyschic',
                                             13: 'bug', 14: 'rock', 16: "ghost", 17: 'dragon', 18: 'dark', 19: 'dark',
                                             20: 'steel', 21: 'fairy'})
pokeDf['type_2'] = pokeDf['type_2'].replace({1: 'normal', 2: 'fire', 3: 'water', 4: 'water', 5: 'elelctric', 6: 'grass',
                                             7: 'ice', 8: 'fighting', 9: 'poison', 10:'ground', 11: 'flying', 12: 'pyschic',
                                             13: 'bug', 14: 'rock', 16: "ghost", 17: 'dragon', 18: 'dark', 19: 'dark',
                                             20: 'steel', 21: 'fairy'})

#this just prints the head for all columns
'''
print(pokeDf.head())
pokeDf.info()
'''

#this was a test to see what replacing values with the median looks like
'''
median = pokeDf['percentage_male'].median()
pokeDf['percentage_male'].fillna(median, inplace=True)
'''

#Dropping ALL duplicte values
'''
pokeDf.drop_duplicates(subset ="name", keep=False, inplace=True)
'''

#This just displays data info from the .csv

pokePure = pokeDf.drop(['name','german_name', 'japanese_name', 'species', 'ability_1', 'ability_2', 'ability_hidden',
                        'height_m', 'weight_kg', 'growth_rate', 'egg_type_2', 'egg_type_1', 'Unnamed: 0', 'percentage_male',
                        'egg_cycles', ], axis=1)

#Where we drop all against features
'''
pokePureNoAgaist = pokeDf.drop(['against_fairy', 'against_steel', 'against_dark', 'against_dragon', 'against_ghost',
                        'against_rock', 'against_bug', 'against_psychic', 'against_flying', 'against_ground', 'against_poison',
                        'against_fight', 'against_ice','against_ice', 'against_grass', 'against_electric', 'against_water',
                        'against_fire', 'against_normal', 'german_name', 'japanese_name', 'species', 'ability_2', 'ability_1',
                        'height_m', 'weight_kg', 'growth_rate', 'egg_type_2', 'egg_type_1', 'Unnamed: 0', 'percentage_male',
                        'egg_cycles',], axis=1)
'''

#This is where we now see the new data with the adjustments
'''
print(pokePure.info())
print(pokePure.head(2))
'''

#PokeCorrelations with less features
'''
pokeCorrPearson = pokePure.corr('pearson')
print(pokeCorrPearson)

pokeCorrKendall = pokePure.corr('kendall')
print(pokeCorrKendall)

pokeCorrspearman = pokePure.corr('spearman')
print(pokeCorrspearman)

pokeCorrMatrix = pokePure.corr()
print(pokeCorrMatrix)'''

#Covariance Matrix
'''pokeCovMatrix = np.cov(pokePure,bias=True)
sb.heatmap(pokeCovMatrix, annot=True, fmt='g')
plt.show()'''

#Pokesearch
'''
pokesearch = input('Enter Pokemon name to Find: ')
for row in pokePure:
    #if current rows 2nd value is equal to input, print that row
    if pokesearch == pokePure['name']:
        print("yes")
'''

# Replace missing values with a number

pokePure.fillna(0, inplace=True)


#this just shows us the number of missing data in cells
'''
print("Number of missing cells: ", pokePure.isnull().sum().sum())
'''

#Using the Pearson Correlation since we want to calculate the pearson coefficient of correlation.
'''pokecorr = pokeDf.corr(method='pearson')
print(pokecorr)
'''

#For Printing
'''
plt.figure(figsize=(16, 16))
sb.heatmap(pokeCorrMatrix, annot=True)

#Charts
pokepowerfig = px.histogram(pokePure, x="name", y="total_points", color="generation", histfunc="sum")
pokegenfig = px.pie(pokePure, values='pokedex_number', names='generation')

pokepowerfig.plt.show()
pokegenfig.plt.show()'''

#We are looking here for correlations using the corr function
'''
corr_matrix = pokeDf.corr()
'''

#this is just testing plots
'''
print(corr_matrix["generation"].sort_values(ascending=False))
pokeDf.plot(kind="scatter", x="generation", y="hp", alpha=0.1)
pokeDf.plot(kind="scatter", x="generation", y="total_points", alpha=0.1)
pokeDf.plot(kind="scatter", x="generation", y="attack", alpha=0.1)
'''

# define the target variable (dependent variable) as y
y = pokePure.total_points

# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(pokePure, y, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# fit a model
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)

print(predictions[0:5])

# The line / model
plt.scatter(y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')

#Print Score
print('Score:', model.score(X_test, y_test))

