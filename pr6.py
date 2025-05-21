import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Create dataset
data = {
    'Age': ['Youth', 'Youth', 'Middle', 'Senior', 'Senior', 'Senior', 'Middle', 'Youth', 'Youth', 'Senior', 'Youth', 'Middle', 'Middle', 'Senior'],
    'Income': ['High', 'High', 'High', 'Medium', 'Low', 'Low', 'Low', 'Medium', 'Low', 'Medium', 'Medium', 'Medium', 'High', 'Medium'],
    'Buys_Computer': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}
df = pd.DataFrame(data)

# Encode categorical values
le_age = LabelEncoder()
le_income = LabelEncoder()
le_target = LabelEncoder()

df['Age'] = le_age.fit_transform(df['Age'])
df['Income'] = le_income.fit_transform(df['Income'])
df['Buys_Computer'] = le_target.fit_transform(df['Buys_Computer'])

# Split data
X = df[['Age', 'Income']]
y = df['Buys_Computer']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train model
model = GaussianNB()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict new data
new_data = pd.DataFrame({'Age': ['Youth', 'Senior'], 'Income': ['High', 'Low']})
new_data['Age'] = le_age.transform(new_data['Age'])
new_data['Income'] = le_income.transform(new_data['Income'])

new_pred = model.predict(new_data)

# Show predictions
print("\nPredictions:")
for i in range(len(new_data)):
    age = le_age.inverse_transform([new_data.loc[i, 'Age']])[0]
    income = le_income.inverse_transform([new_data.loc[i, 'Income']])[0]
    result = le_target.inverse_transform([new_pred[i]])[0]
    print(f"Age: {age}, Income: {income} => Predicted: {result}")
