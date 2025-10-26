import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
import os


def load_data(path:str)->pd.DataFrame:
    data=pd.read_csv(path)
    return data

def data_processing(data:pd.DataFrame):
    data_train=data.iloc[:,:-2]
    data_test=data.iloc[:,-1]
    xtrain,xtest,ytrain,ytest=train_test_split(data_train,data_test,test_size=0.2,random_state=42)
    transformer=ColumnTransformer(transformers=[
    ('tf1',OneHotEncoder(drop='first',sparse_output=False),['Gender','Department']),
    ('tf2',OrdinalEncoder(categories=[['Bachelor','Master','PhD']]),['Education_Level']),
    ('tf3',MinMaxScaler(),['Age','Experience_Years']),
    ('tf4',OrdinalEncoder(categories=[['Intern','Analyst','Executive','Engineer','Manager']]),['Job_Title'])],remainder='passthrough')

    xtrain_tf=transformer.fit_transform(xtrain)
    xtest_tf=transformer.transform(xtest)

    output_path='/Users/rocky/Desktop/MLops/Employee Salary Prediction/salary_prediction/data/raw'
    os.makedirs(output_path,exist_ok=True)
    np.savetxt(os.path.join(output_path,'xtest.txt'),xtest_tf,delimiter=',')
    ytest.to_csv(os.path.join(output_path,'ytest.csv'))

    np.savetxt(os.path.join(output_path,'xtrain.txt'),xtrain_tf,delimiter=',')
    ytrain.to_csv(os.path.join(output_path,'ytrain.csv'))


if __name__=='__main__':
    path='/Users/rocky/Desktop/MLops/Employee Salary Prediction/salary_prediction/data/processed/cleaned_data.csv'
    data=load_data(path)
    data_processing(data=data)



