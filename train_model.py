# train_model.py

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Step 1: Load iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

# Step 2: Save data to CSV
df.to_csv("data/iris.csv", index=False)
print("✅ Saved iris.csv in /data folder")

# Step 3: Split dataset
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)
print("✅ Model trained successfully")

# Step 5: Save model
with open("app/model/iris_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved to app/model/iris_model.pkl")
