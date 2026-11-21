
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

print("Loading trained model...")

model = joblib.load("models/next_ball_model.pkl")
encoder = joblib.load("models/label_encoder.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

X_test = joblib.load("models/X_test.pkl")
y_test = joblib.load("models/y_test.pkl")
y_train = joblib.load("models/y_train.pkl")

# Model predictions
predictions = model.predict(X_test)

model_accuracy = accuracy_score(y_test, predictions)

# Correct majority-class baseline
baseline_input_train = np.zeros((len(y_train), 1))
baseline_input_test = np.zeros((len(y_test), 1))

baseline_model = __import__(
    "sklearn.dummy", fromlist=["DummyClassifier"]
).DummyClassifier(strategy="most_frequent")

baseline_model.fit(baseline_input_train, y_train)
baseline_predictions = baseline_model.predict(baseline_input_test)

baseline_accuracy = accuracy_score(
    y_test, baseline_predictions
)

print("\nMODEL ACCURACY:", round(model_accuracy * 100, 2), "%")
print("MAJORITY BASELINE ACCURACY:", round(baseline_accuracy * 100, 2), "%")

print("\nCLASSIFICATION REPORT:")
print(classification_report(
    y_test,
    predictions,
    labels=np.arange(len(encoder.classes_)),
    target_names=encoder.classes_,
    zero_division=0
))

# Confusion matrix
cm = confusion_matrix(
    y_test,
    predictions,
    labels=np.arange(len(encoder.classes_))
)

plt.figure(figsize=(12, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)

plt.title("Cricket Next-Ball Prediction Confusion Matrix")
plt.xlabel("Predicted Outcome")
plt.ylabel("Actual Outcome")
plt.tight_layout()

plt.savefig("models/confusion_matrix.png")
plt.show()

# Most informative phrases
feature_names = vectorizer.get_feature_names_out()
log_probs = model.feature_log_prob_

print("\nMOST INFORMATIVE PHRASES BY OUTCOME:")

for class_index, class_name in enumerate(encoder.classes_):

    other_classes = np.delete(
        log_probs, class_index, axis=0
    )

    scores = log_probs[class_index] - other_classes.mean(axis=0)

    top_indices = np.argsort(scores)[-10:][::-1]

    print("\n", class_name.upper())

    for index in top_indices:
        print(feature_names[index])

print("\nEvaluation completed!")
