from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "results.csv"
MODELS_DIR = ROOT_DIR / "models"

MODEL_PATH = MODELS_DIR / "second_sight_model.joblib"
TEAM_DATA_PATH = MODELS_DIR / "team_data.joblib"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.joblib"


def load_data():
    """Load the historical international football results dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find dataset at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_columns = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "neutral",
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])

    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"])

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(bool).astype(int)

    return df.sort_values("date").reset_index(drop=True)


def classify_result(row):
    """Classify a match result from the home team's perspective."""
    if row["home_score"] > row["away_score"]:
        return "Team A win"
    if row["home_score"] < row["away_score"]:
        return "Team B win"
    return "Draw"


def build_team_data(df):
    """Create team-level historical summaries used by the app and model."""
    team_rows = []

    for _, row in df.iterrows():
        team_rows.append({
            "team": row["home_team"],
            "goals_for": row["home_score"],
            "goals_against": row["away_score"],
            "won": int(row["home_score"] > row["away_score"]),
            "date": row["date"],
        })
        team_rows.append({
            "team": row["away_team"],
            "goals_for": row["away_score"],
            "goals_against": row["home_score"],
            "won": int(row["away_score"] > row["home_score"]),
            "date": row["date"],
        })

    long_df = pd.DataFrame(team_rows)
    team_data = {}

    for team, group in long_df.groupby("team"):
        group = group.sort_values("date")
        recent = group.tail(10)

        team_data[team] = {
            "matches_played": int(len(group)),
            "win_rate": float(group["won"].mean()),
            "avg_goals_for": float(group["goals_for"].mean()),
            "avg_goals_against": float(group["goals_against"].mean()),
            "recent_form": float(recent["won"].mean()),
        }

    return team_data


def major_tournament_flag(tournament_name):
    """Mark tournaments that roughly represent higher-pressure international settings."""
    major_terms = [
        "FIFA World Cup",
        "World Cup",
        "UEFA Euro",
        "Copa América",
        "Copa America",
        "African Cup of Nations",
        "AFC Asian Cup",
        "CONCACAF Gold Cup",
        "Oceania Nations Cup",
        "Confederations Cup",
        "Nations League",
    ]

    tournament_text = str(tournament_name)
    return int(any(term.lower() in tournament_text.lower() for term in major_terms))


def build_training_rows(df, team_data):
    """Build model-ready match rows from historical team summaries."""
    rows = []

    for _, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]

        if home_team not in team_data or away_team not in team_data:
            continue

        home = team_data[home_team]
        away = team_data[away_team]

        rows.append({
            "home_win_rate": home["win_rate"],
            "away_win_rate": away["win_rate"],
            "win_rate_difference": home["win_rate"] - away["win_rate"],
            "home_avg_goals_for": home["avg_goals_for"],
            "away_avg_goals_for": away["avg_goals_for"],
            "avg_goals_for_difference": home["avg_goals_for"] - away["avg_goals_for"],
            "home_avg_goals_against": home["avg_goals_against"],
            "away_avg_goals_against": away["avg_goals_against"],
            "avg_goals_against_difference": home["avg_goals_against"] - away["avg_goals_against"],
            "home_recent_form": home["recent_form"],
            "away_recent_form": away["recent_form"],
            "recent_form_difference": home["recent_form"] - away["recent_form"],
            "neutral": int(row["neutral"]),
            "major_tournament": major_tournament_flag(row["tournament"]),
            "result": classify_result(row),
        })

    return pd.DataFrame(rows)


def train_model(training_df):
    """Train a compact RandomForest model that stays small enough for GitHub."""
    feature_columns = [column for column in training_df.columns if column != "result"]

    X = training_df[feature_columns]
    y = training_df["result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    return model, feature_columns, accuracy


def save_assets(model, team_data, feature_columns):
    """Save model assets with compression so the repository remains GitHub-friendly."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH, compress=3)
    joblib.dump(team_data, TEAM_DATA_PATH, compress=3)
    joblib.dump(feature_columns, FEATURE_COLUMNS_PATH, compress=3)


def main():
    """Train and save Second Sight FC model assets."""
    print("Loading match data...")
    df = load_data()

    print("Building team summaries...")
    team_data = build_team_data(df)

    print("Building training rows...")
    training_df = build_training_rows(df, team_data)

    print("Training compact model...")
    model, feature_columns, accuracy = train_model(training_df)

    print("Saving assets...")
    save_assets(model, team_data, feature_columns)

    print()
    print("Training complete.")
    print(f"Rows used: {len(training_df):,}")
    print(f"Teams summarized: {len(team_data):,}")
    print(f"Validation accuracy: {accuracy:.3f}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved team data: {TEAM_DATA_PATH}")
    print(f"Saved feature columns: {FEATURE_COLUMNS_PATH}")


if __name__ == "__main__":
    main()