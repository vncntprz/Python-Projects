from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        # Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()

# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()


    # Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()

nRowsRead = None # specify 'None' if want to read whole file
# pokedex_(Update.04.20).csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('/Users/vincent/PycharmProjects/499_machine_Intelligence/Scikit/FINAL PROJECT/Dataset/pokedex_(Update.04.20).csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = '/Users/vincent/PycharmProjects/499_machine_Intelligence/Scikit/FINAL PROJECT/Dataset/pokedex_(Update.04.20).csv'
nRow, nCol = df1.shape
print('There are {nRow} rows and {nCol} columns')

df1.info()

plotPerColumnDistribution(df1, 10, 5)

plotScatterMatrix(df1, 20, 10)

plotCorrelationMatrix(df1, 10)


'''
Starter: Complete Pokemon Dataset 35d8be4d-7
    https://www.kaggle.com/kerneler/starter-complete-pokemon-dataset-35d8be4d-7
'''

#References
'''
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

One-Hot Encoding Categorical Data using SKLEARN
    https://www.kaggle.com/sun4gh/one-hot-encoding-categorical-data-using-sklearn
    
Introduction to Machine Learning with Python's Scikit-learn
    https://www.codementor.io/@garethdwyer/introduction-to-machine-learning-with-python-s-scikit-learn-czha398p1
'''



