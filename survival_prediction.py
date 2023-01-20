import numpy as numpy
import pandas as pandas
import matplotlib.pyplot as plot
import seaborn as seaborn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading the data into Pandas and printing
titanic_data = pandas.read_csv('./data/train.csv')
# printing the first 5 rows of the dataframe
titanic_data.head()
# number of rows and Columns
titanic_data.shape
# getting some informations about the data
titanic_data.info()
# check the number of missing values in each column
titanic_data.isnull().sum()

# Handling the Missing values
# drop the cabin column bcs too many missing values
titanic_data = titanic_data.drop(columns='Cabin', axis=1)

# replacing the missing values in "Age" column with mean value
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)

# finding the mode value of "Embarked" column
titanic_data['Embarked'].mode()[0]

# replacing the missing values in "Embarked" column with mode value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

# check the number of missing values in each column
titanic_data.isnull().sum()

# Data Analysis

# getting some statistical measures about the data
titanic_data.describe()

# finding the number of people survived and not survived
titanic_data['Survived'].value_counts()

# Data visualisation

seaborn.set()


# making a count plot for "Survived" column
print(seaborn.countplot(data=titanic_data, x='Survived'))

titanic_data['Sex'].value_counts()

# making a count plot for "Sex" column
seaborn.countplot(data=titanic_data, x='Sex')

# number of survivors Gender wise
seaborn.countplot(data=titanic_data, x='Sex', hue='Survived', )

# making a count plot for "Pclass" column
seaborn.countplot(x='Pclass', data=titanic_data)
seaborn.countplot(x='Pclass', hue='Survived', data=titanic_data)


# Encoding the Categorical Columns

print(titanic_data['Sex'].value_counts())

# converting categorical Columns
titanic_data.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)

# Separating features and targets
X = titanic_data.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
Y = titanic_data['Survived']

# print(X)
# print(Y)

# Splitting the data into training and test sets

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=3)

print(X.shape, X_train.shape, X_test.shape)

# Model Training - Logistical Regression

model = LogisticRegression(solver='lbfgs', max_iter=1000)
# training the Logistic Regression model with training data
model.fit(X_train, Y_train)

# Module Eval - Accuracy score

# accuracy on training data
X_train_prediction = model.predict(X_train)

print(X_train_prediction)

training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)

test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)
