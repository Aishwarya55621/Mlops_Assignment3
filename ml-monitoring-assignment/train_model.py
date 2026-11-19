import pandas as pd
import pickle

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = iris.target

# Save reference data for later drift monitoring
X.to_csv("iris.csv", index=False)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# Evaluate model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Model accuracy: {accuracy:.2%}")


# Save trained model
with open("model/iris_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully.")
print("Reference data saved to iris.csv.")