import streamlit as st
import pandas as pd
import joblib 
import os
from sklearn.preprocessing import LabelEncoder


MODEL_DIR = "C:/Users/omerf/Desktop/Match-Result-Prediction/model/models"
LEAGUES = ["cl", "bundesliga", "eredivise","seriea", "ligue1", "laliga", "premier_league"]

DATASETS = {
    "cl": "C:/Users/omerf/Desktop/Match-Result-Prediction/train_data/data/cl_train.csv",
    "bundesliga": "C:/Users/omerf/Desktop/Match-Result-Prediction/train_data/data/bundesliga_train.csv",
    "eredivise": "C:/Users/omerf/Desktop/Match-Result-Prediction/train_data/data/eredivise_train.csv",
    "seriea": "C:/Users/omerf/Desktop/Match-Result-Prediction/train_data/data/seriea_train.csv",
    "ligue1": "C:/Users/omerf/Desktop/Match-Result-Prediction/train_data/data/ligue1_train.csv",
    "premier_league": "C:/Users/omerf/Desktop/Match-Result-Prediction/train_data/data/premier_league_train.csv",
    "laliga": "C:/Users/omerf/Desktop/Match-Result-Prediction/train_data/data/laliga_train.csv"
}

st.set_page_config(page_title="⚽ Score Guess", layout="centered")
st.title("⚽ Match Score Prediction")
st.markdown("#### Choose Two Teams!")

selected_league = st.selectbox("Choose League:", LEAGUES)

home_team = st.text_input("🏠 Home:", placeholder="Liverpool")
away_team = st.text_input("🚀 Away:", placeholder="Real Madrid")

def load_model(league):
    home_model_path = os.path.join(MODEL_DIR, f"{league}_home.pkl")
    away_model_path = os.path.join(MODEL_DIR, f"{league}_away.pkl")

    if os.path.exists(home_model_path) and os.path.exists(away_model_path):
        home_model = joblib.load(home_model_path)
        away_model = joblib.load(away_model_path)
        return home_model, away_model
    else:
        return None, None

def get_team_features(league, team_name):
    dataset_path = DATASETS.get(league)
    if dataset_path is None:
        return None
    
    
    df = pd.read_csv(dataset_path)
    relevant_columns = [col for col in df.columns if col not in ["Home Score", "Away Score"]]
    label_encoder = LabelEncoder()
    df["Home Team"] = label_encoder.fit_transform(df["Home Team"])
    df["Away Team"] = label_encoder.fit_transform(df["Away Team"])

    team_stats = df[(df["Home Team"] == label_encoder.transform([team_name])[0]) | (df["Away Team"] == label_encoder.transform([team_name])[0])]

    if team_stats.empty:
        return None
    else:
        team_stats = team_stats.iloc[-1]
        return team_stats[relevant_columns].values
    

if st.button("🔮 Predict"):
    if home_team and away_team:
        home_model, away_model = load_model(selected_league)

        if home_model and away_model:
            home_features = get_team_features(selected_league, home_team)
            away_features = get_team_features(selected_league, away_team)

            if home_features is not None and away_features is not None:
    
                home_score = round(home_model.predict([home_features])[0])
                away_score = round(away_model.predict([away_features])[0])

                st.success(f"📊 Guessed Score: **{home_team} {home_score} - {away_score} {away_team}**")
            else:
                st.error("Team stats not found!")
        else:
            st.error("Model not found!")
    else:
        st.warning("Please enter two team names.")