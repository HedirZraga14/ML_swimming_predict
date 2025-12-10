import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path


BASE = Path(__file__).resolve().parent
ART = BASE / "artifacts"

# Load artifacts
scaler_cluster = joblib.load(ART / "scaler_cluster.joblib")
kmeans = joblib.load(ART / "kmeans.joblib")
le_gender = joblib.load(ART / "le_gender.joblib")
le_country = joblib.load(ART / "le_country.joblib")
agg_ref = pd.read_csv(ART / "agg_reference.csv")

scaler_perf = joblib.load(ART / "scaler_perf.joblib")
rf_medal = joblib.load(ART / "rf_medal.joblib")
le_medal = joblib.load(ART / "le_medal.joblib")
le_sex = joblib.load(ART / "le_sex.joblib")
le_injury = joblib.load(ART / "le_injury.joblib")

rf_reg_100m = joblib.load(ART / "rf_reg_100m.joblib")
scaler_reg_100m = joblib.load(ART / "scaler_reg_100m.joblib")
svr_100m = joblib.load(ART / "svr_100m.joblib")
scaler_svr_100m = joblib.load(ART / "scaler_svr_100m.joblib")


def _match_encoder(value: str, classes) -> str:
    v = value.strip().lower()
    for cls in classes:
        if v == str(cls).strip().lower():
            return cls
    return ""


def normalize_sex(val: str) -> str:
    mapped = _match_encoder(val, le_sex.classes_)
    if mapped:
        return mapped
    aliases = {"m": "Male", "male": "Male", "f": "Female", "female": "Female"}
    return aliases.get(val.strip().lower(), val)


def normalize_injury(val: str) -> str:
    mapped = _match_encoder(val, le_injury.classes_)
    if mapped:
        return mapped
    aliases = {
        "none": "None",
        "no": "None",
        "no injury": "None",
        "na": "None",
        "nil": "None",
        "minor": "Minor",
        "moderate": "Moderate",
        "severe": "Severe",
    }
    return aliases.get(val.strip().lower(), val)


def predict_cluster(inputs: dict):
    g = le_gender.transform([inputs["gender"]])[0]
    c = le_country.transform([inputs["country"]])[0]
    vec = np.array(
        [
            inputs["mean_time"],
            inputs["best_time"],
            inputs["std_time"],
            inputs["improvement"],
            inputs["n_competitions"],
            inputs["age"],
            g,
            c,
        ]
    ).reshape(1, -1)
    Xs = scaler_cluster.transform(vec)
    label = int(kmeans.predict(Xs)[0])
    # similarity
    ref_scaled = scaler_cluster.transform(
        agg_ref[
            [
                "mean_time",
                "best_time",
                "std_time",
                "improvement",
                "n_competitions",
                "age",
                "gender_enc",
                "country_enc",
            ]
        ]
    )
    sims = (ref_scaled @ Xs.T).flatten()
    top_idx = sims.argsort()[::-1][:5]
    sim_norm = (sims - sims.min()) / (sims.max() - sims.min() + 1e-9)
    neighbors = agg_ref.iloc[top_idx][["Athlete Full Name", "cluster_kmeans"]]
    neighbors = neighbors.assign(similarity=sim_norm[top_idx])
    return label, neighbors


def predict_medal(inputs: dict):
    sex = normalize_sex(inputs["Sex"])
    inj = normalize_injury(inputs["Injury_History"])
    try:
        sex_enc = le_sex.transform([sex])[0]
    except Exception:
        sex_enc = le_sex.transform([le_sex.classes_[0]])[0]
    try:
        inj_enc = le_injury.transform([inj])[0]
    except Exception:
        inj_enc = le_injury.transform([le_injury.classes_[0]])[0]

    vec = np.array(
        [
            inputs["Age"],
            inputs["Height"],
            inputs["Weight"],
            inputs["Nutrition_Quality_Score"],
            inputs["Sleep_Hours"],
            inputs["_50m"],
            inputs["_100m"],
            inputs["_200m"],
            inputs["_400m"],
            inputs["_800m"],
            inputs["_1500m"],
            sex_enc,
            inj_enc,
        ]
    ).reshape(1, -1)
    xs = scaler_perf.transform(vec)
    pred = rf_medal.predict(xs)[0]
    return le_medal.inverse_transform([pred])[0]


def predict_time(inputs: dict):
    sex = normalize_sex(inputs["Sex"])
    inj = normalize_injury(inputs["Injury_History"])
    try:
        sex_enc = le_sex.transform([sex])[0]
    except Exception:
        sex_enc = le_sex.transform([le_sex.classes_[0]])[0]
    try:
        inj_enc = le_injury.transform([inj])[0]
    except Exception:
        inj_enc = le_injury.transform([le_injury.classes_[0]])[0]

    vec = np.array(
        [
            inputs["Age"],
            inputs["Height"],
            inputs["Weight"],
            inputs["Nutrition_Quality_Score"],
            inputs["Sleep_Hours"],
            inputs["_50m"],
            inputs["_200m"],
            inputs["_400m"],
            inputs["_800m"],
            inputs["_1500m"],
            sex_enc,
            inj_enc,
        ]
    ).reshape(1, -1)
    xs = scaler_reg_100m.transform(vec)
    pred = float(rf_reg_100m.predict(xs)[0])
    return pred


def predict_time_svr(age: float):
    X = np.array([[age]])
    Xs = scaler_svr_100m.transform(X)
    pred = float(svr_100m.predict(Xs)[0])
    return pred


def feature_importance():
    feat_names = [
        "Age",
        "Height",
        "Weight",
        "Nutrition Quality Score",
        "Sleep Hours",
        "50m Freestyle Time",
        "100m Freestyle Time",
        "200m Freestyle Time",
        "400m Freestyle Time",
        "800m Freestyle Time",
        "1500m Freestyle Time",
        "Sex_enc",
        "Injury_enc",
    ]
    importances = rf_medal.feature_importances_
    pairs = list(zip(feat_names, importances))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


def main():
    st.set_page_config(page_title="Swimming ML (Streamlit)", layout="wide")
    st.title("Swimming ML Portal — Streamlit")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Cluster & Recommendation", "Medal Prediction", "100m Time Prediction", "Performance Factors"]
    )

    with tab1:
        st.subheader("Cluster & Recommendation")
        col1, col2 = st.columns(2)
        mean_time = col1.number_input("Mean time (s)", value=47.5, step=0.001)
        best_time = col2.number_input("Best time (s)", value=46.9, step=0.001)
        std_time = col1.number_input("Std time", value=0.4, step=0.001)
        improvement = col2.number_input("Improvement ratio", value=0.012, step=0.0001, format="%.4f")
        n_competitions = col1.number_input("# competitions", value=5, step=1)
        age = col2.number_input("Age", value=21.4, step=0.1)
        gender = col1.selectbox("Gender", options=list(le_gender.classes_))
        country = col2.selectbox("Country", options=list(le_country.classes_))

        if st.button("Predict cluster"):
            c, neighbors = predict_cluster(
                dict(
                    mean_time=mean_time,
                    best_time=best_time,
                    std_time=std_time,
                    improvement=improvement,
                    n_competitions=n_competitions,
                    age=age,
                    gender=gender,
                    country=country,
                )
            )
            st.success(f"Cluster: {c}")
            reco = {
                0: "🏊 Programme intensif",
                1: "💪 Endurance + technique",
                2: "⚙️ Perfectionnement technique",
                3: "🎯 Stratégie de course",
            }.get(c, "🔄 Suivi individuel")
            st.info(f"Recommendation: {reco}")
            st.subheader("Similar athletes")
            st.bar_chart(neighbors.set_index("Athlete Full Name")["similarity"])
            st.dataframe(neighbors)

    with tab2:
        st.subheader("Medal Prediction")
        cols = st.columns(3)
        Age = cols[0].number_input("Age", value=22.0, step=0.1)
        Height = cols[1].number_input("Height (m)", value=1.85, step=0.001)
        Weight = cols[2].number_input("Weight (kg)", value=75.0, step=0.1)
        Nutrition_Quality_Score = cols[0].number_input("Nutrition quality", value=7.0, step=0.1)
        Sleep_Hours = cols[1].number_input("Sleep hours", value=8.0, step=0.1)
        _50m = cols[2].number_input("50m time", value=23.5, step=0.001)
        _100m = cols[0].number_input("100m time", value=50.1, step=0.001)
        _200m = cols[1].number_input("200m time", value=112.0, step=0.001)
        _400m = cols[2].number_input("400m time", value=230.0, step=0.001)
        _800m = cols[0].number_input("800m time", value=470.0, step=0.001)
        _1500m = cols[1].number_input("1500m time", value=900.0, step=0.001)
        Sex = cols[2].selectbox("Sex", options=list(le_sex.classes_))
        Injury_History = cols[0].selectbox("Injury history", options=list(le_injury.classes_))

        if st.button("Predict medal"):
            medal = predict_medal(
                dict(
                    Age=Age,
                    Height=Height,
                    Weight=Weight,
                    Nutrition_Quality_Score=Nutrition_Quality_Score,
                    Sleep_Hours=Sleep_Hours,
                    _50m=_50m,
                    _100m=_100m,
                    _200m=_200m,
                    _400m=_400m,
                    _800m=_800m,
                    _1500m=_1500m,
                    Sex=Sex,
                    Injury_History=Injury_History,
                )
            )
            st.success(f"Predicted medal: {medal}")

    with tab3:
        st.subheader("100m Time Prediction")
        # Only Age as input for SVR
        age = st.number_input("Age", value=22.0, step=0.1, key="svr_age")
        if st.button("Predict 100m (SVR)", key="btn_svr_pred"):
            pred = predict_time_svr(age)
            st.success(f"Predicted 100m Freestyle (SVR): {pred:.2f} s")

    with tab4:
        st.subheader("Performance Factors")
        pairs = feature_importance()
        labels = [p[0] for p in pairs[:8]]
        values = [p[1] for p in pairs[:8]]
        st.bar_chart(pd.DataFrame({"importance": values}, index=labels))
        st.write("Full ranking:")
        st.dataframe(pd.DataFrame(pairs, columns=["feature", "importance"]))


if __name__ == "__main__":
    main()

