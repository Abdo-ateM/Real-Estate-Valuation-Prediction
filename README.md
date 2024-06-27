---

# Real Estate Valuation Prediction

This project aims to build a linear regression model to predict the house price per unit area based on various features such as transaction date, house age, distance to the nearest MRT station, number of convenience stores, latitude, and longitude. The dataset used in this project is the "Real Estate Valuation Data Set."

## Dataset

The dataset contains the following columns:
- **No**: Serial number
- **X1 transaction date**: The transaction date (e.g., 2012.917 means the 917th day after 2012)
- **X2 house age**: The age of the house (in years)
- **X3 distance to the nearest MRT station**: The distance to the nearest MRT station (in meters)
- **X4 number of convenience stores**: The number of convenience stores within walking distance
- **X5 latitude**: The latitude coordinate of the property
- **X6 longitude**: The longitude coordinate of the property
- **Y house price of unit area**: The house price per unit area (in 10,000 New Taiwan Dollars)

## Project Steps

1. **Load the Dataset**: Load the dataset from an Excel file and inspect its structure.
2. **Exploratory Data Analysis (EDA)**: Visualize the data relationships and compute the correlation matrix to understand the dependencies between variables.
3. **Feature Selection**: Select the relevant features (independent variables) and the target variable (dependent variable) from the dataset.
4. **Data Splitting**: Split the dataset into training and testing sets using `train_test_split` from `sklearn.model_selection`.
5. **Model Training**: Train a linear regression model using `LinearRegression` from `sklearn.linear_model`.
6. **Model Prediction**: Make predictions on the testing set.
7. **Model Evaluation**: Evaluate the model's performance using metrics such as Mean Squared Error (MSE) and R-squared (R²) from `sklearn.metrics`.
8. **Accuracy Calculation**: Calculate the accuracy within a defined tolerance.
9. **Confusion Matrix**: Create a confusion matrix by binning the predicted and actual values.
10. **Frequency Distribution**: Plot the frequency distribution of the predicted and actual values.

## Code Overview

### Load the Dataset

```python
import pandas as pd

# Load the dataset from the Excel file
file_path = 'Real estate valuation data set.xlsx'
df = pd.read_excel(file_path)

# Display the first few rows of the dataset to understand its structure
print(df.head())
```

### Exploratory Data Analysis (EDA)

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Visualize the data
sns.pairplot(df)
plt.show()

# Correlation matrix
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()
```

### Feature Selection

```python
# Select features and target variable
X = df[['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']]
y = df['Y house price of unit area']
```

### Data Splitting

```python
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Model Training

```python
from sklearn.linear_model import LinearRegression

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
```

### Model Prediction

```python
# Make predictions
y_pred = model.predict(X_test)
```

### Model Evaluation

```python
from sklearn.metrics import mean_squared_error, r2_score

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
```

### Accuracy Calculation

```python
import numpy as np

# Define tolerance for accuracy calculation
tolerance = 5  # Within 5 units of the actual value

# Calculate accuracy
accurate_predictions = (np.abs(y_test - y_pred) <= tolerance).astype(int)
accuracy = accurate_predictions.sum() / len(accurate_predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')
```

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix

# Create bins for confusion matrix
bins = np.linspace(y.min(), y.max(), 10)  # 10 bins
y_test_binned = pd.cut(y_test, bins=bins, labels=False)
y_pred_binned = pd.cut(y_pred, bins=bins, labels=False)

# Create confusion matrix
conf_matrix = confusion_matrix(y_test_binned, y_pred_binned)

print('Confusion Matrix:')
print(conf_matrix)

# Plot the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title('Confusion Matrix')
plt.show()
```

### Frequency Distribution

```python
# Frequency distribution of predicted values
plt.hist(y_pred, bins=20, alpha=0.7, label='Predicted')
plt.hist(y_test, bins=20, alpha=0.7, label='Actual')
plt.legend(loc='upper right')
plt.xlabel('House Prices per Unit Area')
plt.ylabel('Frequency')
plt.title('Frequency Distribution of Predicted and Actual Values')
plt.show()
```

## Conclusion

This project demonstrates how to build and evaluate a linear regression model using Python and the `sklearn` library. The model is used to predict house prices per unit area based on various features. The evaluation metrics include Mean Squared Error (MSE), R-squared (R²), accuracy, confusion matrix, and frequency distribution of the predicted values.

Feel free to clone the repository, try out the code, and make improvements. Contributions are welcome!

---

You can copy and paste this description into your README file on GitHub. It provides a comprehensive overview of the project, including the steps involved and the code used. If you have any further questions or need additional modifications, feel free to ask!
