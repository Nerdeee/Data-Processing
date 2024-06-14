import numpy as np
import pandas as pd

dataset = pd.read_csv('employee_data.csv')
print(dataset.describe())

# create independent and dependent variables
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:-1].values

# count the number of missing values in each column
print(dataset.isnull().sum())

# drop missing value records
dataset.dropna(inplace=True)

# replace the missing values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

# data encooding : handle/encode categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# one hot encoding
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder="passthrough")
x=np.array(ct.fit_transform(x))
print(x)

# label encoding
from sklearn.preprocessing import LabelEncoder, StandardScaler
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state=1)
print(x_train)
print(y_train)
print(x_test)
print(y_test)

# feature scaling - standardization and normalization
scaler = StandardScaler()
x_train[:,4:] = scaler.fit_transform(x_train[:, 4:])
x_test[:,4:] = scaler.fit_transform(x_test[:, 4:])
print(x_train)
print(x_test)