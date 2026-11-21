
import pandas as pd
import glob
import os

print("Preparing three-ball context dataset...")

files = glob.glob(
    "dataset/COMMENTARY_INTL_MATCH/**/*.csv",
    recursive=True
)

print("Commentary files found:", len(files))

all_data = []

for i, file in enumerate(files):
    try:
        data = pd.read_csv(
            file,
            usecols=lambda c: c in [
                "Commentary",
                "PlayType_description",
                "Innings"
            ]
        )

        required = [
            "Commentary",
            "PlayType_description",
            "Innings"
        ]

        if not all(col in data.columns for col in required):
            continue

        data = data.dropna(
            subset=["Commentary", "PlayType_description", "Innings"]
        ).copy()

        data["Commentary"] = data["Commentary"].astype(str).str.strip()
        data["PlayType_description"] = (
            data["PlayType_description"].astype(str).str.strip().str.lower()
        )

        # Get commentary from the previous three deliveries
        grouped = data.groupby("Innings")["Commentary"]

        data["Previous_1"] = grouped.shift(1)
        data["Previous_2"] = grouped.shift(2)
        data["Previous_3"] = grouped.shift(3)

        # Combine past deliveries in chronological order
        data["Previous_Commentary"] = (
            data[["Previous_3", "Previous_2", "Previous_1"]]
            .fillna("")
            .agg(" ".join, axis=1)
            .str.strip()
        )

        data["Match_ID"] = os.path.basename(file).split("_")[0]

        data = data[
            data["Previous_Commentary"] != ""
        ]

        all_data.append(
            data[[
                "Match_ID",
                "Previous_Commentary",
                "PlayType_description"
            ]]
        )

    except Exception as e:
        print("Skipped:", os.path.basename(file), e)

    if (i + 1) % 100 == 0:
        print("Processed:", i + 1, "files")

combined = pd.concat(all_data, ignore_index=True)

combined.to_csv("dataset/ball_data.csv", index=False)

print("\nThree-ball context dataset created!")
print("Total records:", len(combined))
print("\nOutcome distribution:")
print(combined["PlayType_description"].value_counts())
