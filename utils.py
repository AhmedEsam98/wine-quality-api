# Import Libraries
import numpy as np
import pandas as pd
import os
import missingno
import joblib
from datasist.structdata import detect_outliers
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,OneHotEncoder, PolynomialFeatures, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector

# Reading Dataset
File_Path = os.path.join(os.getcwd(),'winequality-red.csv')
df = pd.read_csv(File_Path)


df.columns = df.columns.str.replace(' ', '_')



X =df.drop(columns=['quality'], axis=1)
y = df['quality']

# Feature Selection
# Feature Selection is a techinque of finding out the features that contribute the most to our model i.e. the best predictors.
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.20, shuffle=True, random_state=45, stratify=y)

num_cols = df.select_dtypes(include='number').columns.to_list()
categ_cols = df.select_dtypes(include='object').columns.tolist()
num_cols1 = df.select_dtypes(include='number').columns.tolist()[:-1]


# Pipeline
# Create separate pipelines for numeric and categorical columns
# For Numeric
num_pipeline = Pipeline(steps=[
    ('selector',DataFrameSelector(num_cols)),
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# For Categorical
categ_pipeline = Pipeline(steps=[
    ('selector',DataFrameSelector()),
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('OHE', OneHotEncoder(sparse_output=False))
])


## all pipline
all_pipline = FeatureUnion(transformer_list=[
                ('numerical', num_pipeline),
                ('categorical', categ_pipeline)

])

_ = all_pipline.fit(X_train)

def process_new(X_new):
    df_new = pd.DataFrame([X_new])
    df_new.columns = X_train.columns

    # Adjust the Datatypes
    df_new['fixed_acidity'] = df_new['fixed_acidity'].astype('float')
    df_new['volatile_acidity'] = df_new['volatile_acidity'].astype('float')
    df_new['citric_acid'] = df_new['citric_acid'].astype('float')
    df_new['residual_sugar'] = df_new['residual_sugar'].astype('float')
    df_new['chlorides'] = df_new['chlorides'].astype('float')
    df_new['free_sulfur_dioxide'] = df_new['free_sulfur_dioxide'].astype('float')
    df_new['total_sulfur_dioxide'] = df_new['total_sulfur_dioxide'].astype('float')
    df_new['density'] = df_new['density'].astype('float')
    df_new['pH'] = df_new['pH'].astype('float')
    df_new['sulphates'] = df_new['sulphates'].astype('float')
    df_new['alcohol'] = df_new['alcohol'].astype('float')

    X_processed = all_pipline.transform(df_new)
    
    return X_processed

