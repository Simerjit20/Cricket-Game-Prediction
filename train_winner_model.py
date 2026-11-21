
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

print("Loading final match-winner dataset...")

data = pd.read_csv("dataset/match_winner_data.csv")

numeric_features = [
    "First_Batting_Side",
    "First_Innings_Runs",
    "First_Innings_Wickets",
    "First_Innings_Balls",
    "First_Innings_Run_Rate"
]

categorical_features = ["Match_Format"]

required_columns = (
    ["Match_ID", "Match_Text", "Winner_Side"]
    + numeric_features
    + categorical_features
)

data = data.dropna(subset=required_columns)
data = data.drop_duplicates(subset=["Match_ID"])

X = data[
    ["Match_Text"] + numeric_features + categorical_features
]

y = data["Winner_Side"]

print("Total matches:", len(data))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "text",
            TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=3000,
                min_df=2,
                stop_words="english"
            ),
            "Match_Text"
        ),
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "format",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    )
])

print("Training final winner model...")

model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/winner_model.pkl")
joblib.dump(X_test, "models/winner_X_test.pkl")
joblib.dump(y_test, "models/winner_y_test.pkl")
joblib.dump(y_train, "models/winner_y_train.pkl")

print("\nFINAL WINNER MODEL TRAINED!")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Features: Commentary + cricket statistics + match format")
print("Model saved successfully!")
