import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Step 1: Dataset with 4 features
df = pd.DataFrame({
    'Age': ['Youth', 'Youth', 'Middle', 'Senior', 'Senior', 'Senior', 'Middle', 'Youth', 'Youth', 'Senior', 'Youth', 'Middle', 'Middle', 'Senior'],
    'Income': ['High', 'High', 'High', 'Medium', 'Low', 'Low', 'Low', 'Medium', 'Low', 'Medium', 'Medium', 'Medium', 'High', 'Medium'],
    'Student': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No'],
    'Credit_Rating': ['Fair', 'Excellent', 'Fair', 'Fair', 'Fair', 'Excellent', 'Excellent', 'Fair', 'Fair', 'Fair', 'Excellent', 'Excellent', 'Fair', 'Excellent'],
    'Buys': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
})

# Step 2: Encode categorical values
le_age = LabelEncoder()
le_income = LabelEncoder()
le_student = LabelEncoder()
le_credit = LabelEncoder()
le_buys = LabelEncoder()

df['Age'] = le_age.fit_transform(df['Age'])
df['Income'] = le_income.fit_transform(df['Income'])
df['Student'] = le_student.fit_transform(df['Student'])
df['Credit_Rating'] = le_credit.fit_transform(df['Credit_Rating'])
df['Buys'] = le_buys.fit_transform(df['Buys'])

# Step 3: Train/test split
X = df[['Age', 'Income', 'Student', 'Credit_Rating']]
y = df['Buys']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Step 4: Train model
model = GaussianNB()
model.fit(X_train, y_train)

# Step 5: Accuracy
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# Step 6: Predict new data
new = pd.DataFrame({
    'Age': ['Youth', 'Senior'],
    'Income': ['High', 'Low'],
    'Student': ['Yes', 'No'],
    'Credit_Rating': ['Fair', 'Excellent']
})
new['Age'] = le_age.transform(new['Age'])
new['Income'] = le_income.transform(new['Income'])
new['Student'] = le_student.transform(new['Student'])
new['Credit_Rating'] = le_credit.transform(new['Credit_Rating'])
pred = model.predict(new)

# Step 7: Output predictions
# Step 7: Custom Prediction Output (All 4 features)
print("\nPredictions:")
for i in range(len(new)):
    age = le_age.inverse_transform([new.loc[i, 'Age']])[0]
    income = le_income.inverse_transform([new.loc[i, 'Income']])[0]
    student = le_student.inverse_transform([new.loc[i, 'Student']])[0]
    credit = le_credit.inverse_transform([new.loc[i, 'Credit_Rating']])[0]
    result = le_buys.inverse_transform([pred[i]])[0]
    
    print(f"Age: {age}, Income: {income}, Student: {student}, Credit Rating: {credit} => Predicted: {result}")


# Step 8: Plot 1 - Age vs Income
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(X['Age'], X['Income'], c=y, label='Original')
plt.scatter(new['Age'], new['Income'], c=pred, label='Predicted')
plt.xlabel("Age (Encoded)")
plt.ylabel("Income (Encoded)")
plt.title("Naive Bayes: Age vs Income")
plt.legend()
plt.grid()

# Step 9: Plot 2 - Student vs Credit_Rating
plt.subplot(1, 2, 2)
plt.scatter(X['Student'], X['Credit_Rating'], c=y, label='Original')
plt.scatter(new['Student'], new['Credit_Rating'], c=pred, label='Predicted')
plt.xlabel("Student (Encoded)")
plt.ylabel("Credit Rating (Encoded)")
plt.title("Naive Bayes: Student vs Credit Rating")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
