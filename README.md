# Second Sight FC

**Understand the match behind the number.**

Second Sight FC is an explainable pre-match AI companion built for the IBM SkillsBuild AI Builders June Challenge. It helps fans understand how historical football signals can inform a matchup, where the model is uncertain, and why AI output should be interpreted carefully instead of blindly trusted.

This project is focused on **explainability, trust, and fan understanding**. It is not a pure match prediction engine.

Demo video: **https://youtu.be/hPihf5D6y5U**

---

## Challenge Alignment

The June Challenge asks participants to build an AI-powered solution that helps humans understand soccer before, during, or after a match. Second Sight FC focuses on the **before the match** experience.

Instead of simply asking, "Who will win?", Second Sight FC asks:

- What historical signals inform this matchup?
- Which team has stronger long-term patterns?
- Is the model confident, or is this a toss-up?
- What does the AI not know?
- How can a newer fan understand the reasoning?

The goal is to make AI more transparent and useful for everyday fans.

---

## What Second Sight FC Does

Second Sight FC allows users to choose two national teams and view an explainable AI reading based on historical international match data.

The app includes:

- Team A vs Team B selector
- Historical pattern reading
- Outcome probability table
- Confidence label: Toss-up, Moderate confidence, or Higher confidence
- Plain-English explanation of the signals behind the result
- New Fan Mode for simpler explanations
- Team signal comparison table
- Trust Check section explaining model limits
- Fan education sections
- Sleek Streamlit interface

---

## What Second Sight FC Is Not

Second Sight FC is **not**:

- A betting tool
- A live match tracker
- A replacement for coaches, analysts, players, or referees
- A VAR decision system
- A FIFA laws assistant
- A tactical reconstruction platform
- A pure score prediction engine

The model uses historical patterns to support fan understanding. It does not know live match conditions, injuries, lineups, formations, weather, red cards, substitutions, travel fatigue, or emotional pressure inside the stadium.

---

## AI and Technical Approach

Second Sight FC uses a machine learning model trained on historical international football results.

The model is built with:

- Python
- pandas
- scikit-learn
- RandomForestClassifier
- joblib
- Streamlit

The training script builds team-level historical features such as:

- Historical win rate
- Average goals scored
- Average goals allowed
- Recent form
- Neutral venue context
- Major tournament context

The Streamlit app loads the trained model and explains the model output in a fan-friendly way.

---

## IBM Tooling

IBM Bob was used as an AI coding assistant during the project process. It supported review and refinement of the prototype by helping identify ways to improve:

- Challenge alignment
- Explainability language
- Trust and transparency messaging
- README structure
- User experience priorities
- Final submission readiness

The project uses Bob as part of the development workflow while keeping the final application focused, lightweight, and understandable.

---

## Demo Scenario

A sample walkthrough:

1. Open the app.
2. Select Brazil vs Argentina.
3. Review the historical pattern reading.
4. Check the confidence label.
5. Read the explanation of which signals influenced the model.
6. Toggle New Fan Mode to see simpler explanations.
7. Review the Trust Check to understand what the AI does not know.
8. Try another matchup to see how uncertainty changes.

This shows that the app is not trying to be an oracle. It is helping the user understand how the model reads the matchup and where uncertainty remains.

---

## Known Limitations

Second Sight FC uses a historical international results dataset. This means the prototype is limited by the scope, quality, and age of that data.

The app does not currently include:

- Live match data
- Current injuries
- Official squad lists
- Starting lineups
- Player tracking
- Tactical formations
- Weather or venue conditions
- Real-time substitutions or red cards
- Emotional pressure or crowd environment

Because the model learns from past results, it can also reflect historical bias. Strong historical teams may appear stronger even when current conditions have changed.

---

## How to Run Locally

Clone the repository and install dependencies:

    pip install -r requirements.txt

Train or regenerate the model artifacts:

    python src/train_model.py

Run the Streamlit app:

    python -m streamlit run app/app.py

---

## Project Structure

    second-sight-fc/
    +-- app/
    |   +-- app.py
    +-- data/
    |   +-- results.csv
    +-- models/
    |   +-- feature_columns.joblib
    |   +-- second_sight_model.joblib
    |   +-- team_data.joblib
    +-- src/
    |   +-- train_model.py
    +-- DEMO_SCRIPT.md
    +-- LICENSE
    +-- requirements.txt
    +-- README.md

## Why It Matters

The World Cup is watched by people with different levels of football knowledge. Some fans understand tactics and statistics deeply. Others are newer to the sport and may struggle to interpret what numbers actually mean.

Second Sight FC makes the AI reasoning more accessible by turning model signals into plain-language explanations. It emphasizes uncertainty and limitations so fans can learn from the model without overtrusting it.

The project supports a more transparent, educational, and human-centered way to use AI in football.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contact and Feedback

Created by Zac for the IBM SkillsBuild AI Builders June Challenge.

Feedback is welcome through GitHub once the public repository is published.



