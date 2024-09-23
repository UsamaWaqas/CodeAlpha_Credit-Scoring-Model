# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Step 1: Load the dataset
data = pd.read_csv('credit_data.csv')

# Step 2: Handle missing values
data = data.dropna()

# Step 3: Encode categorical variables
label_encoder = LabelEncoder()
data['Employment Status'] = label_encoder.fit_transform(data['Employment Status'])
data['Credit History'] = label_encoder.fit_transform(data['Credit History'])

# Ensure all columns are numeric
data = data.apply(pd.to_numeric, errors='coerce')

# Step 4: Split the data into features and target variable
X = data.drop('Creditworthiness', axis=1)
y = data['Creditworthiness']

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 8: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Step 10: Hyperparameter Tuning (Optional)
param_grid = {
    'n_estimators': [100, 200],  # Reduced to 2 options to limit execution time
    'max_depth': [10, 20],       # Reduced to 2 options to limit execution time
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f'Best Parameters: {best_params}')

# Train the model with the best parameters
best_model = grid_search.best_estimator_

# Evaluate the best model
best_y_pred = best_model.predict(X_test)
best_accuracy = accuracy_score(y_test, best_y_pred)
best_conf_matrix = confusion_matrix(y_test, best_y_pred)
best_class_report = classification_report(y_test, best_y_pred)

print(f'Best Accuracy: {best_accuracy}')
print('Best Confusion Matrix:')
print(best_conf_matrix)
print('Best Classification Report:')
print(best_class_report)
