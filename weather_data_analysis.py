#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


# Load the dataset
weather_data = pd.read_csv('weather.csv')


# In[3]:


# Data Exploration
print("Data Information:")
print(weather_data.info())

print("Few Rows:")
print(weather_data.head())

print("Summary of Data")
print(weather_data.describe())


# In[4]:


# Data Visualization: Pair plots to visualize relationships between numeric variables
sns.pairplot(weather_data[['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine']])
plt.title('Pair Plot of Weather Variables')
plt.show()


# In[5]:


# Data Analysis: Calculate statistics or relationships for specific columns
rainfall_mean = weather_data['Rainfall'].mean()
print(f"\nMean Rainfall: {rainfall_mean} mm")


# In[6]:


# Data Visualization 2
plt.figure(figsize=(10, 6))
sns.scatterplot(x='MaxTemp', y='Rainfall', data=weather_data,color='orange')
plt.title('Scatter Plot: Max Temperature vs Rainfall')
plt.xlabel('Max Temperature (°C)')
plt.ylabel('Rainfall (mm)')
plt.show()


# In[7]:


plt.figure(figsize=(10, 6))
sns.violinplot(x='RainToday', y='Rainfall', data=weather_data)
plt.title('Violin Plot: Distribution of Rainfall across the season')
plt.xlabel('Rain Today')
plt.ylabel('Rainfall (mm)')
plt.show()


# In[8]:


plt.figure(figsize=(10, 6))
sns.histplot(weather_data['Rainfall'], bins=50, kde=False, color='blue', alpha=0.7)
plt.title('Distribution of Rainfall across the season')
plt.xlabel('Rainfall (mm)')
plt.ylabel('Frequency of Rainfall')
plt.tight_layout()
plt.show()


# In[9]:


# Linear Regression for Rainfall Prediction
X = weather_data[['MinTemp', 'MaxTemp',]]
y = weather_data['Rainfall']


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[11]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[12]:


# Make predictions
y_pred = model.predict(X_test)


# In[13]:


# Model Evaluation
mse = mean_squared_error(y_test, y_pred)

print(f"\nMean Squared Error: {mse}")


# In[14]:


# Visualize Predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Rainfall (mm)")
plt.ylabel("Predicted Rainfall (mm)")
plt.title("Actual vs Predicted Rainfall")
plt.show()


# In[15]:


# Conclusions and Insights
highest_rainfall_month = weather_data['Rainfall'].idxmax()
lowest_rainfall_month = weather_data['Rainfall'].idxmin()

print("\nHighest Rainfall Month:")
print(highest_rainfall_month)
print("\nLowest Rainfall Month:")
print(lowest_rainfall_month)

