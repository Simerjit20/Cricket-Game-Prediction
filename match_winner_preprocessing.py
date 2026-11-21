
import pandas as pd
import numpy as np
import glob
import os

print("Preparing final match-winner dataset...")

matches = pd.read_csv("dataset/INTERNATIONAL_MATCH.csv")

# Match dataset names -> commentary dataset names
TEAM_NAMES = {
    "Mum Indians": "Mumbai Indians",
    "KKR": "Kolkata Knight Riders",
    "RCB": "Royal Challengers Bangalore",
    "Super Kings": "Chennai Super Kings",
    "Sixers": "Sydney Sixers",
    "Sunrisers": "Sunrisers Hyderabad",
    "Scorchers": "Perth Scorchers",
    "Strikers": "Adelaide Strikers",
    "Hurricanes": "Hobart Hurricanes",
    "Heat": "Brisbane Heat",
    "Thunder": "Sydney Thunder",
    "Stars": "Melbourne Stars",
    "Renegades": "Melbourne Renegades",
    "Guj Lions": "Gujarat Lions",
    "Daredevils": "Delhi Capitals",
    "Supergiants": "Rising Pune Supergiant",
    "Supergiant": "Rising Pune Supergiant",
    "Kings XI": "Kings XI Punjab",
    "Royals": "Rajasthan Royals",
    "United": "Islamabad United",
    "Zalmi": "Peshawar Zalmi",
    "Qalandars": "Lahore Qalandars",
    "Gladiators": "Quetta Gladiators",
    "Sultans": "Multan Sultans"
}

# Names like "Kings" are ambiguous and are
# intentionally not mapped.


def normalize_name(name):
    name = str(name).strip()
    return TEAM_NAMES.get(name, name).casefold()


def normalize_id(value):
    try:
        return str(int(float(value)))
    except (ValueError, TypeError):
        return str(value).strip()


def last_numeric(df, column):
    if column not in df.columns:
        return np.nan

    values = pd.to_numeric(
        df[column], errors="coerce"
    ).dropna()

    if values.empty:
        return np.nan

    return float(values.iloc[-1])


def common_team(df, column):
    if column not in df.columns:
        return ""

    values = df[column].dropna().astype(str).str.strip()
    values = values[values != ""]

    if values.empty:
        return ""

    return values.mode().iloc[0]


matches["Match_ID"] = matches["Match_number"].apply(
    normalize_id
)

lookup = (
    matches.drop_duplicates("Match_ID")
    .set_index("Match_ID")
)

files = glob.glob(
    "dataset/COMMENTARY_INTL_MATCH/**/*.csv",
    recursive=True
)

records = []
skipped = 0
seen = set()

print("Commentary files:", len(files))

for index, file in enumerate(files):
    try:
        match_id = os.path.basename(file).split("_")[0]

        if match_id in seen:
            continue

        if match_id not in lookup.index:
            skipped += 1
            continue

        info = lookup.loc[match_id]

        team1 = str(info["Team_1"]).strip()
        team2 = str(info["Team_2"]).strip()
        winner = str(info["Winner_team"]).strip()

        if winner not in [team1, team2]:
            skipped += 1
            continue

        df = pd.read_csv(file)

        required = [
            "Innings",
            "Commentary",
            "Batting_Team_name"
        ]

        if not all(c in df.columns for c in required):
            skipped += 1
            continue

        first = df[
            df["Innings"].astype(str).str.strip()
            == "1st innings"
        ].copy()

        if first.empty:
            skipped += 1
            continue

        commentary = (
            first["Commentary"]
            .dropna()
            .astype(str)
            .str.strip()
        )
        commentary = commentary[commentary != ""]

        if commentary.empty:
            skipped += 1
            continue

        batting = common_team(
            first, "Batting_Team_name"
        )
        bowling = common_team(
            first, "Bowling_Team_name"
        )

        if normalize_name(batting) == normalize_name(team1):
            batting_side = 1
        elif normalize_name(batting) == normalize_name(team2):
            batting_side = 2
        else:
            skipped += 1
            continue

        # Numeric cricket statistics
        runs = last_numeric(first, "Innings_runs")
        wickets = last_numeric(first, "Innings_wickets")
        balls = last_numeric(first, "Innings_balls")
        rate = last_numeric(first, "Innings_run_rate")

        score = str(first.iloc[-1].get("Score", ""))

        if pd.isna(runs) and "/" in score:
            try:
                runs = float(score.split("/")[0])
            except ValueError:
                pass

        if pd.isna(wickets) and "/" in score:
            try:
                wickets = float(score.split("/")[1])
            except ValueError:
                pass

        if pd.isna(balls):
            over = last_numeric(first, "Over_number")
            ball = last_numeric(first, "Over_ball")

            if not pd.isna(over):
                balls = max(
                    0,
                    (over - 1) * 6 +
                    (0 if pd.isna(ball) else ball)
                )

        runs = 0 if pd.isna(runs) else runs
        wickets = 0 if pd.isna(wickets) else wickets
        balls = 0 if pd.isna(balls) else balls

        if pd.isna(rate) or rate == 0:
            rate = runs / (balls / 6) if balls > 0 else 0

        match_format = str(info["Match_Type"]).strip()

        match_text = (
            f"Team one {team1} "
            f"Team two {team2} "
            f"Batting first {batting} "
            f"Bowling first {bowling} "
            f"Batting side {batting_side} "
            f"Match format {match_format} "
            f"Ground {info['Ground']} "
            f"Commentary {' '.join(commentary.tolist())}"
        )

        records.append({
            "Match_ID": match_id,
            "Match_Text": match_text,
            "Team_1": team1,
            "Team_2": team2,
            "Winner_team": winner,
            "Winner_Side": (
                "Team 1" if winner == team1 else "Team 2"
            ),
            "First_Batting_Team": batting,
            "First_Bowling_Team": bowling,
            "First_Batting_Side": batting_side,
            "Match_Format": match_format,
            "First_Innings_Runs": runs,
            "First_Innings_Wickets": wickets,
            "First_Innings_Balls": balls,
            "First_Innings_Run_Rate": rate
        })

        seen.add(match_id)

    except Exception as e:
        skipped += 1
        print("Error:", os.path.basename(file), e)

    if (index + 1) % 200 == 0:
        print("Processed:", index + 1)

if not records:
    raise ValueError("No valid matches found.")

result = pd.DataFrame(records)

result.to_csv(
    "dataset/match_winner_data.csv",
    index=False
)

print("\nDATASET READY!")
print("Valid matches:", len(result))
print("Skipped:", skipped)

print("\nWinner distribution:")
print(result["Winner_Side"].value_counts())

print("\nBatting side distribution:")
print(result["First_Batting_Side"].value_counts())

print("\nMatch formats:")
print(result["Match_Format"].value_counts())
