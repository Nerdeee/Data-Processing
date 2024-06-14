import numpy as np
import pandas as pd

'''dataset = pd.read_csv('employee_data.csv')
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

from imblearn.over_sampling import SMOTE
from collections import Counter

smote = SMOTE(random_state=27)
smote_X_train, smote_Y_train = smote.fit_resample(x_train, y_train)

print('Before SMOTE: ', Counter(y_train))
print('After SMOTE: ', Counter(smote_Y_train))'''

from sklearn.datasets import load_diabetes
data = load_diabetes()

df = pd.DataFrame(data.data, columns=data.feature_names)

df['test'] = range(1, len(df)+1)
print(df)

# print(df.describe)    // shows stats of the dataset for each label such as the min, max, different quartiles, etc.
# print(df.corr())      // shows the correlation between labels / categories.
# print(df.nunique())   // shows the number of unique datapoints per label.
# print(df['label_name'].unique())  // shows the number of unique datapoints for the label specified.
# print(df.info())      // shows general information about dataframe such as the number of columns, their names, data type, non-null count, etc
# .strip()              // string method that removes any spaces before or after non-space character string

# Note: Feature selection is the process of choosing features from the original dataset whereas feature extraction is using selected features and creating new features 
# using them, either through finding correlations betweem the selected features or some other method.
