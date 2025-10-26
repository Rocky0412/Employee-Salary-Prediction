import numpy as np
import pandas as pd
import os


def load_data(path:str)->pd.DataFrame:
    data=pd.read_csv(path)
    return data


def data_cleaning(data:pd.DataFrame)->pd.DataFrame:
    cleaned_data=data.drop(['Employee_ID','Name'],axis=1)
    return cleaned_data


if __name__ == '__main__':
    path='/Users/rocky/Desktop/MLops/Employee Salary Prediction/salary_prediction/data/external/Employers_data.csv'
    df=load_data(path)
    cleaned_data=data_cleaning(df)
    output_path='/Users/rocky/Desktop/MLops/Employee Salary Prediction/salary_prediction/data/processed'
    os.makedirs(output_path,exist_ok=True)
    output_path = os.path.join(output_path, 'cleaned_data.csv')
    cleaned_data.to_csv(output_path)