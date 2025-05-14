
# Titanic Data Analysis - Simple ML Project

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data (in real case, use pd.read_csv('titanic.csv'))
data = pd.DataFrame({
    'Pclass': [3, 1, 3, 1, 3, 3, 2, 2],
    'Sex': ['male', 'female', 'female', 'female', 'male', 'male', 'female', 'male'],
    'Age': [22, 38, 26, 35, 28, np.nan, 30, 40],
    'SibSp': [1, 1, 0, 1, 0, 0, 0, 1],
    'Fare': [7.25, 71.28, 7.92, 53.10, 8.05, 8.46, 13.00, 15.50],
    'Survived': [0, 1, 1, 1, 0, 0, 1, 0]
})

# Data preprocessing
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Age'].fillna(data['Age'].mean(), inplace=True)

# Features and target
X = data.drop('Survived', axis=1)
y = data['Survived']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Plot
plt.figure(figsize=(6,4))
sns.barplot(x=['Accuracy'], y=[accuracy])
plt.ylim(0, 1)
plt.title('Model Accuracy')
plt.tight_layout()
plt.show()
