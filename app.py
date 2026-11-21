
import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Cricket Game Prediction",
    page_icon="🏏",
    layout="wide"
)

MODEL_DIR = "models"

@st.cache_resource
def load_models():
    next_model = joblib.load(
        os.path.join(MODEL_DIR, "next_ball_model.pkl")
    )
    vectorizer = joblib.load(
        os.path.join(MODEL_DIR, "vectorizer.pkl")
    )
    encoder = joblib.load(
        os.path.join(MODEL_DIR, "label_encoder.pkl")
    )
    winner_model = joblib.load(
        os.path.join(MODEL_DIR, "winner_model.pkl")
    )
    return next_model, vectorizer, encoder, winner_model


try:
    next_model, vectorizer, encoder, winner_model = load_models()
except Exception as e:
    st.error(f"Could not load models: {e}")
    st.stop()


st.title("🏏 Cricket Game Prediction System")
st.write(
    "Machine Learning project using cricket commentary "
    "and match statistics."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Home",
    "🏏 Next-Ball Prediction",
    "🏆 Match-Winner Prediction",
    "📊 Model Performance"
])


# ---------------- HOME ----------------
with tab1:
    st.header("Welcome to Cricket Game Prediction")

    st.write(
        "This project uses Machine Learning to analyze "
        "cricket commentary and predict outcomes."
    )

    st.subheader("Project Features")

    st.write("**1. Next-Ball Prediction**")
    st.write(
        "Predicts the outcome of the next ball using "
        "commentary from previous deliveries."
    )

    st.write("**2. Match-Winner Prediction**")
    st.write(
        "Predicts the likely winning side using "
        "first-innings commentary and statistics."
    )

    st.write("**3. Model Performance**")
    st.write(
        "Displays accuracy, baseline comparisons "
        "and confusion matrices."
    )

    st.info(
        "These are experimental predictions based "
        "on historical cricket data."
    )


# ------------- NEXT BALL -------------
with tab2:
    st.header("🏏 Next-Ball Outcome Prediction")

    st.write(
        "Enter commentary describing the previous "
        "three deliveries, in chronological order."
    )

    previous_commentary = st.text_area(
        "Previous three deliveries",
        height=180,
        placeholder=(
            "Ball 1: The batsman defends the delivery. "
            "Ball 2: A single is taken. "
            "Ball 3: The batsman plays a drive."
        )
    )

    if st.button("Predict Next Ball"):
        if not previous_commentary.strip():
            st.warning("Please enter commentary first.")
        else:
            try:
                features = vectorizer.transform(
                    [previous_commentary]
                )

                prediction = next_model.predict(features)[0]

                outcome = encoder.inverse_transform(
                    [int(prediction)]
                )[0]

                st.success(
                    f"Predicted next-ball outcome: {outcome}"
                )

                if hasattr(next_model, "predict_proba"):
                    probabilities = next_model.predict_proba(
                        features
                    )[0]

                    labels = encoder.inverse_transform(
                        next_model.classes_.astype(int)
                    )

                    results = pd.DataFrame({
                        "Outcome": labels,
                        "Probability": probabilities
                    }).sort_values(
                        "Probability", ascending=False
                    )

                    st.subheader("Model Probability Estimates")

                    st.dataframe(
                        results.style.format({
                            "Probability": "{:.2%}"
                        }),
                        hide_index=True,
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"Prediction error: {e}")

    st.caption(
        "The next-ball model was trained using historical "
        "commentary and may struggle with rare outcomes."
    )


# ------------- MATCH WINNER -------------
with tab3:
    st.header("🏆 Match-Winner Prediction")

    st.write(
        "Enter the teams, match format and completed "
        "first-innings statistics."
    )

    col1, col2 = st.columns(2)

    with col1:
        team1 = st.text_input(
            "Team 1",
            value="India"
        )

        batting_side = st.selectbox(
            "Which team batted first?",
            ["Team 1", "Team 2"]
        )

        runs = st.number_input(
            "First-Innings Runs",
            min_value=0,
            value=180
        )

        wickets = st.number_input(
            "First-Innings Wickets",
            min_value=0,
            max_value=10,
            value=6
        )

    with col2:
        team2 = st.text_input(
            "Team 2",
            value="Australia"
        )

        match_format = st.selectbox(
            "Match Format",
            [
                "ODI",
                "T20I",
                "TEST_MATCH",
                "IPL",
                "BBL",
                "PSL"
            ]
        )

        balls = st.number_input(
            "First-Innings Balls",
            min_value=1,
            value=120
        )

        ground = st.text_input(
            "Ground / Venue",
            value="Unknown"
        )

    run_rate = float(runs) / (float(balls) / 6)

    st.write(f"**Calculated Run Rate:** {run_rate:.2f}")

    commentary = st.text_area(
        "First-Innings Commentary",
        height=180,
        placeholder="Paste the first-innings commentary here..."
    )

    if st.button("Predict Match Winner"):
        if not team1.strip() or not team2.strip():
            st.warning("Please enter both team names.")

        elif team1.strip().casefold() == team2.strip().casefold():
            st.warning("Please enter two different teams.")

        elif not commentary.strip():
            st.warning(
                "Please enter first-innings commentary."
            )

        else:
            first_batting_side = (
                1 if batting_side == "Team 1" else 2
            )

            batting_team = (
                team1 if first_batting_side == 1 else team2
            )

            bowling_team = (
                team2 if first_batting_side == 1 else team1
            )

            match_text = (
                f"Team one {team1} "
                f"Team two {team2} "
                f"Batting first {batting_team} "
                f"Bowling first {bowling_team} "
                f"Batting side {first_batting_side} "
                f"Match format {match_format} "
                f"Ground {ground} "
                f"Commentary {commentary}"
            )

            input_data = pd.DataFrame([{
                "Match_Text": match_text,
                "First_Batting_Side": first_batting_side,
                "First_Innings_Runs": float(runs),
                "First_Innings_Wickets": float(wickets),
                "First_Innings_Balls": float(balls),
                "First_Innings_Run_Rate": run_rate,
                "Match_Format": match_format
            }])

            try:
                prediction = winner_model.predict(
                    input_data
                )[0]

                predicted_team = (
                    team1 if prediction == "Team 1" else team2
                )

                st.success(
                    f"Predicted Winning Team: {predicted_team}"
                )

                if hasattr(winner_model, "predict_proba"):
                    probabilities = winner_model.predict_proba(
                        input_data
                    )[0]

                    classes = winner_model.classes_

                    results = pd.DataFrame({
                        "Team": [
                            team1 if c == "Team 1" else team2
                            for c in classes
                        ],
                        "Probability": probabilities
                    })

                    st.subheader("Model Probability Estimates")

                    st.dataframe(
                        results.style.format({
                            "Probability": "{:.2%}"
                        }),
                        hide_index=True,
                        use_container_width=True
                    )

                st.info(
                    "This prediction is experimental. "
                    "The winner model's test accuracy is 53.36%, "
                    "below its majority baseline of 57.46%. "
                    "Probabilities are not guaranteed to be calibrated."
                )

            except Exception as e:
                st.error(f"Prediction error: {e}")


# ------------- PERFORMANCE -------------
with tab4:
    st.header("📊 Model Performance")

    st.subheader("Next-Ball Prediction Model")

    st.metric("Test Accuracy", "55.08%")
    st.write("Majority baseline: approximately 53.41%")
    st.write("Algorithm: Multinomial Naive Bayes")
    st.write("Features: Previous three deliveries' commentary")

    next_cm = os.path.join(
        MODEL_DIR, "confusion_matrix.png"
    )

    if os.path.exists(next_cm):
        st.image(
            next_cm,
            caption="Next-Ball Confusion Matrix"
        )

    st.divider()

    st.subheader("Match-Winner Prediction Model")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Model Accuracy", "53.36%")

    with col2:
        st.metric("Majority Baseline", "57.46%")

    st.write("Training matches: 1,070")
    st.write("Testing matches: 268")
    st.write(
        "Algorithm: TF-IDF + StandardScaler + "
        "OneHotEncoder + Logistic Regression"
    )

    winner_cm = os.path.join(
        MODEL_DIR, "winner_confusion_matrix.png"
    )

    if os.path.exists(winner_cm):
        st.image(
            winner_cm,
            caption="Match-Winner Confusion Matrix"
        )

    st.warning(
        "The match-winner model does not outperform "
        "the majority baseline on its current test set."
    )

st.divider()
st.caption("Cricket Game Prediction | Machine Learning Project")
