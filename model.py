import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv('electricity_bill_dataset.csv')

# Features
X = data[['Units', 'AC_Hours', 'Temperature']]

# Target
y = data['Bill']

# Train Model
model = LinearRegression()
model.fit(X, y)

# Save Model
pickle.dump(model, open('model.pkl', 'wb'))

print('✅ Model trained successfully!')