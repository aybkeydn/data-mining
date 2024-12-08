import pandas as pd
data = pd.read_csv("heart_v2.csv")
print(data.head())
print(data.columns)
print(data.shape)
print(data.info())

X = data.drop("heart disease", axis = 1)
Y = data['heart disease']

#Importing the scikit learn module

from  sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.7, random_state = 42)

X_train.shape #(189, 4)

X_test.shape # (81, 4)

Y_train.shape #(189,)

Y_test.shape


print(data.isnull().sum())


import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x=Y)
plt.title("Heart Disease Class Distribution")
plt.show()


correlation = data.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)


accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")


conf_matrix = confusion_matrix(Y_test, Y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix for Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, Y_train)


rf_Y_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(Y_test, rf_Y_pred)
rf_precision = precision_score(Y_test, rf_Y_pred)
rf_recall = recall_score(Y_test, rf_Y_pred)
rf_f1 = f1_score(Y_test, rf_Y_pred)

print("Random Forest Metrics:")
print(f"Accuracy: {rf_accuracy}")
print(f"Precision: {rf_precision}")
print(f"Recall: {rf_recall}")
print(f"F1 Score: {rf_f1}")


results = pd.DataFrame({
    "Model": ["Decision Tree", "Random Forest"],
    "Accuracy": [accuracy, rf_accuracy],
    "Precision": [precision, rf_precision],
    "Recall": [recall, rf_recall],
    "F1 Score": [f1, rf_f1]
})

print(results)
