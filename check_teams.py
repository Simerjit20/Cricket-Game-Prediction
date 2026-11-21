
import pandas as pd
import glob

files = glob.glob(
    "dataset/COMMENTARY_INTL_MATCH/**/*.csv",
    recursive=True
)

team_names = set()

for file in files:
    try:
        data = pd.read_csv(
            file,
            usecols=["Batting_Team_name"]
        )

        names = data["Batting_Team_name"].dropna()
        team_names.update(names.astype(str).unique())

    except Exception:
        continue

print("Unique batting team names:", len(team_names))

for team in sorted(team_names):
    print(team)
