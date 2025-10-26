from sklearn.linear_model import LinearRegression
import pickle
import numpy as np
import pandas as pd
import os


# Load data
xtrain = np.genfromtxt(
    '/Users/rocky/Desktop/MLops/Employee Salary Prediction/salary_prediction/data/raw/xtrain.txt',
    delimiter=','
)
ytrain = pd.read_csv(
    '/Users/rocky/Desktop/MLops/Employee Salary Prediction/salary_prediction/data/raw/ytrain.csv'
).to_numpy()

# Train model
model = LinearRegression()
model.fit(xtrain, ytrain)

# Save model
save_folder = '/Users/rocky/Desktop/MLops/Employee Salary Prediction/salary_prediction/models'
os.makedirs(save_folder, exist_ok=True)

save_file = 'linear_model.pkl'
save_path = os.path.join(save_folder, save_file)

with open(save_path, 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved at {save_path}")
 

