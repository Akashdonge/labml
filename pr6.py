import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Step 1: Create dataset
data = {
    'Age': ['Youth', 'Youth', 'Middle', 'Senior', 'Senior', 'Senior', 'Middle', 'Youth', 'Youth', 'Senior', 'Youth', 'Middle', 'Middle', 'Senior'],
    'Income': ['High', 'High', 'High', 'Medium', 'Low', 'Low', 'Low', 'Medium', 'Low', 'Medium', 'Medium', 'Medium', 'High', 'Medium'],
    'Buys_Computer': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}
df = pd.DataFrame(data)

# Step 2: Preprocessing - Label Encoding
le_age = LabelEncoder()
le_income = LabelEncoder()
le_target = LabelEncoder()

df['Age'] = le_age.fit_transform(df['Age'])
df['Income'] = le_income.fit_transform(df['Income'])
df['Buys_Computer'] = le_target.fit_transform(df['Buys_Computer'])

# Optional: Feature Scaling (Min-Max Scaling)
scaler = MinMaxScaler()
df[['Age', 'Income']] = scaler.fit_transform(df[['Age', 'Income']])

# Step 3: Split data
X = df[['Age', 'Income']]
y = df['Buys_Computer']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Step 4: Train model
model = GaussianNB()
model.fit(X_train, y_train)

# Step 5: Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 6: Predict on new data
new_data = pd.DataFrame({'Age': ['Youth', 'Senior'], 'Income': ['High', 'Low']})
new_data['Age'] = le_age.transform(new_data['Age'])
new_data['Income'] = le_income.transform(new_data['Income'])

# Apply same scaling to new data
new_data[['Age', 'Income']] = scaler.transform(new_data[['Age', 'Income']])

# Predict
new_pred = model.predict(new_data)

# Step 7: Show predictions
print("\nPredictions:")
for i in range(len(new_data)):
    age = le_age.inverse_transform([int(scaler.inverse_transform([new_data.iloc[i]])[0][0])])[0]
    income = le_income.inverse_transform([int(scaler.inverse_transform([new_data.iloc[i]])[0][1])])[0]
    result = le_target.inverse_transform([new_pred[i]])[0]
    print(f"Age: {age}, Income: {income} => Predicted: {result}")
