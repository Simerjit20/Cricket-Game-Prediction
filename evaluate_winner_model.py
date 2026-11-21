
import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.dummy import DummyClassifier

print("Loading improved winner model...")

model = joblib.load("models/winner_model.pkl")

X_test = joblib.load("models/winner_X_test.pkl")
y_test = joblib.load("models/winner_y_test.pkl")
y_train = joblib.load("models/winner_y_train.pkl")

print("Making predictions...")

y_pred = model.predict(X_test)

# Model accuracy
model_accuracy = accuracy_score(y_test, y_pred)

# Majority-class baseline calculated from training data
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(np.zeros((len(y_train), 1)), y_train)

baseline_pred = baseline.predict(
    np.zeros((len(y_test), 1))
)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_pred
)

print("\n================================")
print("IMPROVED MATCH-WINNER EVALUATION")
print("================================")

print(f"Model Accuracy: {model_accuracy * 100:.2f}%")
print(f"Majority Baseline: {baseline_accuracy * 100:.2f}%")

print("\nClassification Report:\n")

print(classification_report(
    y_test,
    y_pred,
    labels=["Team 1", "Team 2"],
    zero_division=0
))

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Team 1", "Team 2"]
)

print("Confusion Matrix:")
print(cm)

# Save confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Team 1", "Team 2"],
    yticklabels=["Team 1", "Team 2"]
)

plt.title("Improved Match-Winner Confusion Matrix")
plt.xlabel("Predicted Winner")
plt.ylabel("Actual Winner")
plt.tight_layout()

plt.savefig("models/winner_confusion_matrix.png")
plt.close()

print("\nConfusion matrix saved!")
print("Evaluation completed!")
