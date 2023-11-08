import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.base import BaseEstimator,TransformerMixin

class Add_Features(BaseEstimator,TransformerMixin):
    def __init__(self,add_rooms=True):
        self.add_rooms=add_rooms
        self.ln=0 # longitude index
        self.lt=1
        self.pop=5
        self.hs=6
        self.rm=3 # rooms index
        self.bd=4
        
    def fit(self,X,y=None):
        return self
    
#     def transform(self,X,y=None):
#         # here we are using indexes because this pipeline will be called after Imputer and imputer returns an array not a dataframe
#         ln_lat=X[:,self.ln]+X[:,self.lt]
#         hs_pop=X[:,self.hs]/X[:,self.pop]
#         if self.add_rooms:
#             bd_rm=X[:,self.bd]/X[:,self.rm]
#             return np.hstack((X, ln_lat.reshape(-1, 1), hs_pop.reshape(-1, 1), bd_rm.reshape(-1, 1)))
#         return np.hstack((X, ln_lat.reshape(-1, 1), hs_pop.reshape(-1, 1)))
    def transform(self, X, y=None):
        ln_lat = X[:, self.ln] + X[:, self.lt]
        hs_pop = X[:, self.hs] / X[:, self.pop]
        if self.add_rooms:
            bd_rm  = X[:, self.bd] / X[:, self.rm]
            return np.c_[X, ln_lat, hs_pop, bd_rm]

        return np.c_[X, ln_lat, hs_pop]

def rmse(y,y_hat):
    return np.sqrt(mean_squared_error(y,y_hat))

def model_report(models,X_train,X_test,y_train,y_test):
    for name,model in models:
        model.fit(X_train,y_train)
        y_train_hat=model.predict(X_train)
        y_test_hat=model.predict(X_test)
        print("-"*80)
        print(f"Report For {name}".center(80))
        print("-"*80)
        print(f"Training RMSE Error : {rmse(y_train,y_train_hat):.2f}")
        print(f"Test RMSE Error : {rmse(y_test,y_test_hat):.2f}")
        print()
        print(f"Training Accuracy : {r2_score(y_train,y_train_hat):.2f}")
        print(f"Test Accuracy : {r2_score(y_test,y_test_hat):.2f}")
        print("\n\n")

def load_data():
    data_root_dir="C:\\Users\\KIIT\\Desktop\\AI_ML"
    data_path=os.path.join(data_root_dir,"datasets/housing/housing.csv")
    housing=pd.read_csv(data_path)
    X=housing.drop(['median_house_value'],axis=1).copy()
    y=housing['median_house_value']
    return X,y
        
if __name__=="__main__":
    
    X,y=load_data()
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=123)
    
    num_features=['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',]
    cat_features=['ocean_proximity']


    num_pipeline=Pipeline([
        ('imputer',SimpleImputer(strategy='median')),
        ('add features',Add_Features()),
        ('standard scaler',StandardScaler()),
    ])
    #drop_features=['longitude','latitude']
    #pass_through=['median_income']
    final_pipeline=ColumnTransformer([
        # (name,Transform,Column_List)
        ("numerical pipeline",num_pipeline,num_features),
        ("categorical pipeline",OneHotEncoder(),cat_features),
        #("remove features",'drop',drop_features),
        #("pass through",'passthrough',pass_through),
    ])


    X_train_tr=final_pipeline.fit_transform(X_train)
#     cat_col=list(final_pipeline.named_transformers_['categorical pipeline'].categories_[0])
#     columns=num_features+cat_col#+pass_through
#     X_train_tr=pd.DataFrame(X_train_tr,columns=columns)

    X_test_tr=final_pipeline.transform(X_test)
#     X_test_tr=pd.DataFrame(X_test_tr,columns=columns)
    
    
    models=[
        ("Linear Regression",LinearRegression()),
        ("SGD Regressor",SGDRegressor(max_iter=10000)),
        ("Decision Tree",DecisionTreeRegressor(max_depth=7)),
        ("Support Vector Machines",SVR(kernel='linear')),
        ("Random Forest",RandomForestRegressor(max_depth=9)),
        ("K-Nearest Neighbors",KNeighborsRegressor(n_neighbors=51)),
    ]
    model_report(models,X_train_tr,X_test_tr,y_train,y_test)
