from pathlib import Path
import joblib
import pandas as pd
import streamlit as st


APP_VERSION = "1.0.0"

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"

MODEL_PATH = MODELS_DIR / "second_sight_model.joblib"
TEAM_DATA_PATH = MODELS_DIR / "team_data.joblib"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.joblib"


TEAM_CODES = {
    "Argentina": "ARG",
    "Australia": "AUS",
    "Austria": "AUT",
    "Belgium": "BEL",
    "Brazil": "BRA",
    "Cameroon": "CMR",
    "Canada": "CAN",
    "Chile": "CHI",
    "China PR": "CHN",
    "Colombia": "COL",
    "Costa Rica": "CRC",
    "Croatia": "CRO",
    "Czech Republic": "CZE",
    "Denmark": "DEN",
    "Ecuador": "ECU",
    "Egypt": "EGY",
    "England": "ENG",
    "France": "FRA",
    "Germany": "GER",
    "Ghana": "GHA",
    "Greece": "GRE",
    "Hungary": "HUN",
    "Iceland": "ISL",
    "Iran": "IRN",
    "Italy": "ITA",
    "Ivory Coast": "CIV",
    "Japan": "JPN",
    "Mexico": "MEX",
    "Morocco": "MAR",
    "Netherlands": "NED",
    "New Zealand": "NZL",
    "Nigeria": "NGA",
    "North Korea": "PRK",
    "Northern Ireland": "NIR",
    "Norway": "NOR",
    "Panama": "PAN",
    "Paraguay": "PAR",
    "Peru": "PER",
    "Poland": "POL",
    "Portugal": "POR",
    "Qatar": "QAT",
    "Republic of Ireland": "IRL",
    "Romania": "ROU",
    "Russia": "RUS",
    "Saudi Arabia": "KSA",
    "Scotland": "SCO",
    "Senegal": "SEN",
    "Serbia": "SRB",
    "Slovakia": "SVK",
    "Slovenia": "SVN",
    "South Africa": "RSA",
    "South Korea": "KOR",
    "Spain": "ESP",
    "Sweden": "SWE",
    "Switzerland": "SUI",
    "Tunisia": "TUN",
    "Turkey": "TUR",
    "Ukraine": "UKR",
    "United Arab Emirates": "UAE",
    "United States": "USA",
    "Uruguay": "URU",
    "Wales": "WAL",
}


st.set_page_config(
    page_title="Second Sight FC",
    page_icon="⚽",
    layout="wide"
)


CUSTOM_CSS = """
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(26, 115, 232, 0.18), transparent 32%),
            radial-gradient(circle at top right, rgba(0, 214, 201, 0.10), transparent 28%),
            linear-gradient(135deg, #07111f 0%, #0d1726 45%, #111827 100%);
        color: #f8fafc;
    }

    [data-testid="stSidebar"] {
        background: rgba(3, 7, 18, 0.82);
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }

    .hero {
        padding: 2rem;
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 28px;
        background:
            linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(30, 41, 59, 0.72)),
            radial-gradient(circle at 85% 15%, rgba(34, 211, 238, 0.18), transparent 35%);
        box-shadow: 0 24px 80px rgba(0, 0, 0, 0.28);
        margin-bottom: 1.5rem;
    }

    .eyebrow {
        color: #67e8f9;
        font-size: 0.82rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .hero-title {
        color: #ffffff;
        font-size: 3.2rem;
        line-height: 1;
        font-weight: 850;
        margin-bottom: 0.75rem;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1.15rem;
        max-width: 820px;
        line-height: 1.6;
    }

    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-top: 1.25rem;
    }

    .pill {
        border: 1px solid rgba(103, 232, 249, 0.26);
        background: rgba(8, 47, 73, 0.42);
        color: #cffafe;
        padding: 0.42rem 0.72rem;
        border-radius: 999px;
        font-size: 0.86rem;
        font-weight: 650;
    }

    .glass-card {
        padding: 1.25rem;
        border-radius: 22px;
        background: rgba(15, 23, 42, 0.74);
        border: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow: 0 18px 56px rgba(0, 0, 0, 0.20);
        min-height: 100%;
    }

    .section-title {
        color: #f8fafc;
        font-size: 1.3rem;
        font-weight: 800;
        margin-bottom: 0.7rem;
    }

    .small-muted {
        color: #94a3b8;
        font-size: 0.92rem;
        line-height: 1.5;
    }

    .result-badge {
        display: inline-block;
        padding: 0.55rem 0.85rem;
        border-radius: 999px;
        background: rgba(34, 211, 238, 0.16);
        border: 1px solid rgba(34, 211, 238, 0.34);
        color: #a5f3fc;
        font-weight: 800;
        margin-bottom: 0.85rem;
    }

    .big-number {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 850;
        line-height: 1.05;
    }

    .confidence {
        color: #e0f2fe;
        font-size: 1.05rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }

    .explain-item {
        padding: 0.9rem 1rem;
        margin: 0.65rem 0;
        border-radius: 16px;
        background: rgba(30, 41, 59, 0.70);
        border-left: 4px solid rgba(34, 211, 238, 0.75);
        color: #e2e8f0;
        line-height: 1.55;
    }

    .trust-box {
        padding: 1rem;
        border-radius: 18px;
        background: rgba(2, 6, 23, 0.52);
        border: 1px solid rgba(251, 191, 36, 0.28);
        color: #fde68a;
        line-height: 1.55;
    }

    h1, h2, h3 {
        color: #f8fafc !important;
    }

    p, li, label, span {
        color: inherit;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
    }

    .stMetric {
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.14);
        padding: 1rem;
        border-radius: 18px;
    }

    .footer-note {
        color: #94a3b8;
        text-align: center;
        padding-top: 2rem;
        font-size: 0.9rem;
    }
</style>
"""


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    """Load the trained model, team data, and feature column list from disk."""
    model = joblib.load(MODEL_PATH)
    team_data = joblib.load(TEAM_DATA_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    return model, team_data, feature_columns


def team_label(team_name):
    """Return a display label with a scoreboard-style team code when one is available."""
    code = TEAM_CODES.get(team_name)
    if code:
        return f"{code} · {team_name}"
    return team_name


def get_display_teams(team_data):
    """Return teams that have scoreboard codes and are present in the trained dataset."""
    return sorted(team for team in team_data.keys() if team in TEAM_CODES)


def build_match_features(team_a, team_b, team_data, neutral=True, major_tournament=True):
    """Build the one-row feature table used by the model for a selected matchup."""
    a = team_data[team_a]
    b = team_data[team_b]

    features = {
        "home_win_rate": a["win_rate"],
        "away_win_rate": b["win_rate"],
        "win_rate_difference": a["win_rate"] - b["win_rate"],
        "home_avg_goals_for": a["avg_goals_for"],
        "away_avg_goals_for": b["avg_goals_for"],
        "avg_goals_for_difference": a["avg_goals_for"] - b["avg_goals_for"],
        "home_avg_goals_against": a["avg_goals_against"],
        "away_avg_goals_against": b["avg_goals_against"],
        "avg_goals_against_difference": a["avg_goals_against"] - b["avg_goals_against"],
        "home_recent_form": a["recent_form"],
        "away_recent_form": b["recent_form"],
        "recent_form_difference": a["recent_form"] - b["recent_form"],
        "neutral": int(neutral),
        "major_tournament": int(major_tournament),
    }

    return pd.DataFrame([features])


def confidence_label(probabilities):
    """Convert model probability spread into a human-readable uncertainty label."""
    sorted_probs = sorted(probabilities, reverse=True)
    gap = sorted_probs[0] - sorted_probs[1]

    if gap < 0.10:
        return "Toss-up", gap, "The model sees this as close. Treat the result as a conversation starter, not a verdict."
    if gap < 0.20:
        return "Moderate confidence", gap, "One outcome has an edge, but the matchup still carries meaningful uncertainty."
    return "Higher confidence", gap, "The model sees a clearer historical signal, though football still refuses to behave neatly."


def format_percent(value):
    """Format a decimal value as a percentage string."""
    return f"{value * 100:.1f}%"


def explain_match(team_a, team_b, row, new_fan_mode):
    """Turn model features into plain-language explanations for fans."""
    notes = []

    win_rate_diff = row["win_rate_difference"]
    goals_diff = row["avg_goals_for_difference"]
    defense_diff = row["avg_goals_against_difference"]
    form_diff = row["recent_form_difference"]

    team_a_display = team_label(team_a)
    team_b_display = team_label(team_b)

    if new_fan_mode:
        if win_rate_diff > 0.03:
            notes.append(f"{team_a_display} has the stronger historical win rate. That means they have won a larger share of past international matches.")
        elif win_rate_diff < -0.03:
            notes.append(f"{team_b_display} has the stronger historical win rate. That gives them a stronger long-term signal.")
        else:
            notes.append("The teams have similar historical win rates, so the model does not see a huge long-term gap.")

        if goals_diff > 0.10:
            notes.append(f"{team_a_display} has scored more goals per match historically. This can matter because teams with stronger scoring records may create more ways to change a match.")
        elif goals_diff < -0.10:
            notes.append(f"{team_b_display} has scored more goals per match historically. This can matter because teams with stronger scoring records may create more ways to change a match.")

        if defense_diff < -0.10:
            notes.append(f"{team_a_display} has allowed fewer goals per match. This matters because a stronger defensive record can keep close matches within reach.")
        elif defense_diff > 0.10:
            notes.append(f"{team_b_display} has allowed fewer goals per match. This matters because a stronger defensive record can keep close matches within reach.")

        if form_diff > 0.10:
            notes.append(f"{team_a_display} has better recent form. Recent form means how often a team has been winning lately.")
        elif form_diff < -0.10:
            notes.append(f"{team_b_display} has better recent form. Recent form means how often a team has been winning lately.")
        else:
            notes.append("Recent form is close, so the model treats the matchup as more uncertain.")
    else:
        if win_rate_diff > 0.03:
            notes.append(f"{team_a_display} leads on long-term win rate by {format_percent(abs(win_rate_diff))}.")
        elif win_rate_diff < -0.03:
            notes.append(f"{team_b_display} leads on long-term win rate by {format_percent(abs(win_rate_diff))}.")
        else:
            notes.append("Long-term win rate is nearly even.")

        if goals_diff > 0.10:
            notes.append(f"{team_a_display} has the stronger attacking profile by {goals_diff:.2f} goals per match.")
        elif goals_diff < -0.10:
            notes.append(f"{team_b_display} has the stronger attacking profile by {abs(goals_diff):.2f} goals per match.")

        if defense_diff < -0.10:
            notes.append(f"{team_a_display} has the stronger defensive profile, conceding {abs(defense_diff):.2f} fewer goals per match.")
        elif defense_diff > 0.10:
            notes.append(f"{team_b_display} has the stronger defensive profile, conceding {abs(defense_diff):.2f} fewer goals per match.")

        if form_diff > 0.10:
            notes.append(f"{team_a_display} has a recent-form edge of {format_percent(abs(form_diff))}.")
        elif form_diff < -0.10:
            notes.append(f"{team_b_display} has a recent-form edge of {format_percent(abs(form_diff))}.")
        else:
            notes.append("Recent form is close, increasing uncertainty.")

    return notes


def comparison_table(team_a, team_b, team_data):
    """Create a side-by-side table of team-level historical signals."""
    a = team_data[team_a]
    b = team_data[team_b]

    return pd.DataFrame({
        "Signal": [
            "Matches played",
            "Historical win rate",
            "Average goals scored",
            "Average goals allowed",
            "Recent form"
        ],
        team_label(team_a): [
            int(a["matches_played"]),
            format_percent(a["win_rate"]),
            f"{a['avg_goals_for']:.2f}",
            f"{a['avg_goals_against']:.2f}",
            format_percent(a["recent_form"])
        ],
        team_label(team_b): [
            int(b["matches_played"]),
            format_percent(b["win_rate"]),
            f"{b['avg_goals_for']:.2f}",
            f"{b['avg_goals_against']:.2f}",
            format_percent(b["recent_form"])
        ],
    })


try:
    with st.spinner("Loading Second Sight FC..."):
        model, team_data, feature_columns = load_assets()
except FileNotFoundError as error:
    st.error("Second Sight FC could not find one or more model files.")
    st.info("Run `python src/train_model.py` from the project root, then restart the app.")
    st.caption(f"Missing file details: {error}")
    st.stop()
except Exception as error:
    st.error("Second Sight FC could not load the model assets.")
    st.info("Try regenerating the model files with `python src/train_model.py`, then restart the app.")
    st.caption(f"Technical details: {error}")
    st.stop()

teams = get_display_teams(team_data)

if len(teams) < 2:
    st.error("Second Sight FC needs at least two coded teams to display a matchup.")
    st.info("Check the TEAM_CODES list in app.py or regenerate the model assets.")
    st.stop()

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Explainable AI for World Cup understanding</div>
        <div class="hero-title">Second Sight FC</div>
        <div class="hero-subtitle">
            A pre-match AI companion that explains how historical signals inform fan understanding of a matchup,
            where uncertainty lives, and why fans should understand the model instead of blindly trusting it.
        </div>
        <div class="pill-row">
            <span class="pill">Pre-match insight</span>
            <span class="pill">Trust-first AI</span>
            <span class="pill">New fan friendly</span>
            <span class="pill">Historical match signals</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("Match setup")
    st.caption("Choose two national teams and adjust the match context.")

    team_a = st.selectbox(
        "Team A",
        teams,
        index=teams.index("Brazil") if "Brazil" in teams else 0,
        format_func=team_label
    )
    team_b = st.selectbox(
        "Team B",
        teams,
        index=teams.index("Argentina") if "Argentina" in teams else 1,
        format_func=team_label
    )

    st.divider()

    neutral = st.checkbox("Neutral venue", value=True)
    major_tournament = st.checkbox("Major tournament context", value=True)
    new_fan_mode = st.checkbox("Explain for new fans", value=True)

    st.divider()
    st.caption("This prototype uses historical international results from the lab dataset.")
    st.caption(f"Showing {len(teams)} coded national teams.")
    st.caption(f"Version {APP_VERSION}")

if team_a == team_b:
    st.info("Choose two different teams to generate a matchup reading.")
    st.stop()

features = build_match_features(
    team_a,
    team_b,
    team_data,
    neutral=neutral,
    major_tournament=major_tournament
)

features = features[feature_columns]

prediction = model.predict(features)[0]
probabilities = model.predict_proba(features)[0]
classes = model.classes_

probability_df = pd.DataFrame({
    "Outcome": classes,
    "Probability": probabilities
}).sort_values("Probability", ascending=False)

label, gap, confidence_note = confidence_label(probabilities)

left, right = st.columns([1.05, 1])

with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Pattern-Based Reading</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="result-badge">{team_label(team_a)} vs {team_label(team_b)}</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-muted">Historical patterns indicate</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="big-number">{prediction}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="confidence">{label}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="small-muted">Top outcome gap: {gap * 100:.1f} percentage points</div>', unsafe_allow_html=True)
    st.markdown(f'<br><div class="small-muted">{confidence_note}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Outcome Probabilities</div>', unsafe_allow_html=True)
    st.dataframe(
        probability_df.assign(Probability=probability_df["Probability"].map(format_percent)),
        use_container_width=True,
        hide_index=True
    )
    st.markdown('<div class="small-muted">The probabilities are model signals, not certainties.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">What historical signals shaped this reading</div>', unsafe_allow_html=True)

explanations = explain_match(
    team_a,
    team_b,
    features.iloc[0],
    new_fan_mode
)

for note in explanations:
    st.markdown(f'<div class="explain-item">{note}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

table_left, trust_right = st.columns([1.1, 0.9])

with table_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Team Signal Comparison</div>', unsafe_allow_html=True)
    st.dataframe(
        comparison_table(team_a, team_b, team_data),
        use_container_width=True,
        hide_index=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

with trust_right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Trust Check</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="trust-box">
            This AI reads historical patterns, not the live truth of a match.
            It does not know injuries, lineups, weather, tactical plans,
            red cards, substitutions, travel fatigue, or pressure inside the stadium.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="small-muted">
            Use Second Sight FC as an explanation and learning tool, not as a betting tool.
            The model may reflect historical bias because it learns from past results.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

with st.expander("How to use Second Sight FC"):
    st.write(
        "Use this tool as a pre-match conversation guide. Start by choosing two national teams, "
        "then read the pattern-based reading, the confidence label, and the signal comparison table."
    )
    st.write(
        "The goal is not to blindly accept the model output. The goal is to compare the AI's reading "
        "with your own intuition, learn which historical signals matter, and notice where uncertainty remains."
    )
    st.write(
        "A strong use case is discussing a matchup before kickoff: What does history indicate? "
        "Which team has stronger long-term signals? Is the model confident, or is the match too close to call?"
    )

with st.expander("Learn the signals"):
    st.markdown(
        """
        **Historical win rate**  
        How often a team has won past international matches. This gives the model a long-term strength signal.

        **Average goals scored**  
        How many goals a team typically scores per match. This helps describe attacking output.

        **Average goals allowed**  
        How many goals a team typically concedes per match. This helps describe defensive record.

        **Recent form**  
        How often a team has won in its most recent matches. This gives the model a short-term signal.

        **Neutral venue**  
        A match played without a home-field advantage. World Cup matches are often neutral or semi-neutral contexts.

        **Major tournament context**  
        A signal that the match belongs to a higher-pressure competition environment.
        """
    )

with st.expander("Model limitations and data notes"):
    st.write(
        "Second Sight FC uses historical international match results. "
        "It compares broad signals such as win rate, goals scored, goals allowed, recent form, "
        "neutral venues, and major tournament context."
    )
    st.write(
        "This prototype does not use live match feeds, player tracking, injury reports, tactical formations, "
        "or official World Cup squad data."
    )
    st.write(
        "The dropdown is intentionally limited to coded national teams for a cleaner World Cup-style experience, "
        "even though the source dataset contains many more historical and non-FIFA representative teams."
    )
    st.write(
        "The goal is not to replace analysts, coaches, referees, or fans. "
        "The goal is to make the model's reasoning easier to understand."
    )

st.markdown(
    """
    <div class="footer-note">
        Built for the IBM SkillsBuild AI Builders June Challenge.
        Focus: explainability, trust, and fan understanding before the match.
    </div>
    """,
    unsafe_allow_html=True
)