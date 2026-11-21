
import pandas as pd
import os
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import LabelEncoder

from feature_extraction import extract_features

print("Loading next-ball dataset...")

data = pd.read_csv("dataset/ball_data.csv")

data = data.dropna(
    subset=["Previous_Commentary", "PlayType_description", "Match_ID"]
)

X = data["Previous_Commentary"].astype(str)
y = data["PlayType_description"].astype(str)
groups = data["Match_ID"].astype(str)

# Encode outcome labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split by match to prevent the same match appearing in both sets
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_index, test_index = next(
    splitter.split(X, y_encoded, groups)
)

X_train_text = X.iloc[train_index]
X_test_text = X.iloc[test_index]

y_train = y_encoded[train_index]
y_test = y_encoded[test_index]

print("Extracting text features...")

X_train, X_test, vectorizer = extract_features(
    X_train_text,
    X_test_text
)

print("Training Naive Bayes model...")

model = ComplementNB(alpha=1.0)
model.fit(X_train, y_train)

# Save model and supporting files
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/next_ball_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
joblib.dump(encoder, "models/label_encoder.pkl")
joblib.dump(X_test, "models/X_test.pkl")
joblib.dump(y_test, "models/y_test.pkl")
joblib.dump(y_train, "models/y_train.pkl")

print("\nMODEL TRAINING COMPLETED!")
print("Training records:", len(train_index))
print("Testing records:", len(test_index))
print("Vocabulary size:", len(vectorizer.get_feature_names_out()))
print("Outcome classes:", list(encoder.classes_))
print("Model saved in the models folder.")
