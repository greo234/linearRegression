# -*- coding: utf-8 -*-
"""
A linear regression  project to predict rainfall and productivity




Created on Tue Jan  3 08:15:41 2023

@author: Omoregbe Olotu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Reading CSV file from file path in the computer using pandas
input_file = r"C:\Users\matah\Downloads\inputdata1.csv"
df = pd.read_csv(input_file)
print(df)

# Converting dataset to float type using numpy

rainfall = np.array(df['Rainfall'])
productivity = np.array(df['Productivity'])

#plotting scatter plot
plt.figure(figsize=(10,8), dpi=140)
plt.scatter(rainfall,productivity)
plt.title("Rainfall vs Productivity", fontsize = 15)
a,b = np.polyfit(rainfall, productivity,1)
plt.plot(rainfall,a*rainfall+b)
plt.xlabel('Rainfall (mm)')
plt.ylabel('Productivity')
plt.show()

#reshaping dataframe for regression analysis
x = rainfall.reshape(28,1)
y = productivity

# Creating a Linear Regression Model and fitting the model
model = LinearRegression().fit(x,y)

# Coefficient of determination, R^2
r_sq = model.score(x,y)
print(f"r_sq: {r_sq}")

#intercept
i = model.intercept_
print(f"intercept: {i}")

#slope
s = model.coef_
print(f" slope: {s}")

# rainfall predictions
p = model.predict(x)
print(f"prediction: {p}")

#plotting 
plt.figure(figsize=(10,8), dpi=140)
plt.scatter(x,y)
plt.plot(x,p,label ="Line of Best Fit")
plt.title("Rainfall vs Productivity", fontsize = 15)
plt.ylabel("Productivity", fontsize = 10)
plt.xlabel("Rainfall", fontsize = 10)
plt.legend(bbox_to_anchor = (1.02,1))
plt.show()

# Evaluating the productivity coefficient of the field 
# if the amount of precipitations is X mm.
X = 300
Xp = model.predict([[X]])

#plotting
plt.figure(figsize=(10,8), dpi=140)
plt.scatter(x,y)
plt.plot(x, p, label ="Line of best Fit")
plt.scatter(X, Xp, label ="Prediction")
plt.title("Rainfall vs Productivity", fontsize = 15)
plt.xlabel("Rainfall", fontsize=10)
plt.ylabel("Productivity", fontsize =10)
plt.annotate(Xp, (X, Xp), fontsize=10)
plt.legend(bbox_to_anchor = (1.02,1))
plt.show()

