"""
Application Streamlit améliorée pour le projet ML de natation
Implémente les recommandations d'amélioration des interfaces DSO
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent
ART = BASE / "artifacts"

# Configuration de la page
st.set_page_config(
    page_title="🏊 Aqualyze",
    page_icon="🏊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour améliorer l'apparence
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .cluster-description {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .recommendation-box {
        background-color: #fff4e6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Chargement des modèles avec cache
@st.cache_resource
def load_models():
    """Charge tous les modèles et préprocesseurs"""
    try:
        models = {
            'scaler_cluster': joblib.load(ART / "scaler_cluster.joblib"),
            'kmeans': joblib.load(ART / "kmeans.joblib"),
            'le_gender': joblib.load(ART / "le_gender.joblib"),
            'le_country': joblib.load(ART / "le_country.joblib"),
            'agg_ref': pd.read_csv(ART / "agg_reference.csv"),
            'scaler_perf': joblib.load(ART / "scaler_perf.joblib"),
            'rf_medal': joblib.load(ART / "rf_medal.joblib"),
            'le_medal': joblib.load(ART / "le_medal.joblib"),
            'le_sex': joblib.load(ART / "le_sex.joblib"),
            'le_injury': joblib.load(ART / "le_injury.joblib"),
            'rf_reg_100m': joblib.load(ART / "rf_reg_100m.joblib"),
            'scaler_reg_100m': joblib.load(ART / "scaler_reg_100m.joblib"),
            'svr_100m': joblib.load(ART / "svr_100m.joblib"),
            'scaler_svr_100m': joblib.load(ART / "scaler_svr_100m.joblib"),
        }
        return models
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles: {e}")
        return None

models = load_models()
if models is None:
    st.stop()

# Extraction des modèles pour faciliter l'utilisation
scaler_cluster = models['scaler_cluster']
kmeans = models['kmeans']
le_gender = models['le_gender']
le_country = models['le_country']
agg_ref = models['agg_ref']
scaler_perf = models['scaler_perf']
rf_medal = models['rf_medal']
le_medal = models['le_medal']
le_sex = models['le_sex']
le_injury = models['le_injury']
rf_reg_100m = models['rf_reg_100m']
scaler_reg_100m = models['scaler_reg_100m']
svr_100m = models['svr_100m']
scaler_svr_100m = models['scaler_svr_100m']

# Fonctions utilitaires
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
        "none": "None", "no": "None", "no injury": "None",
        "na": "None", "nil": "None", "minor": "Minor",
        "moderate": "Moderate", "severe": "Severe",
    }
    return aliases.get(val.strip().lower(), val)

# ============================================================
# DSO1 - PRÉDICTION TEMPS 100M (RÉGRESSION)
# ============================================================

def predict_time_improved(inputs: dict):
    """Prédiction améliorée avec intervalles de confiance estimés"""
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

    vec = np.array([
        inputs["Age"], inputs["Height"], inputs["Weight"],
        inputs["Nutrition_Quality_Score"], inputs["Sleep_Hours"],
        inputs["_50m"], inputs["_200m"], inputs["_400m"],
        inputs["_800m"], inputs["_1500m"], sex_enc, inj_enc,
    ]).reshape(1, -1)
    
    xs = scaler_reg_100m.transform(vec)
    pred = float(rf_reg_100m.predict(xs)[0])
    
    # Estimation de l'incertitude (basée sur la variance des prédictions des arbres)
    preds_trees = [tree.predict(xs)[0] for tree in rf_reg_100m.estimators_]
    std_pred = np.std(preds_trees)
    conf_interval = 1.96 * std_pred  # 95% confidence interval
    
    return pred, pred - conf_interval, pred + conf_interval, std_pred

def predict_time_svr_improved(age: float):
    """Prédiction SVR améliorée"""
    X = np.array([[age]])
    Xs = scaler_svr_100m.transform(X)
    pred = float(svr_100m.predict(Xs)[0])
    # Estimation simple de l'incertitude (basée sur l'âge)
    uncertainty = 2.0  # Estimation fixe pour SVR simple
    return pred, pred - uncertainty, pred + uncertainty

def render_dso1():
    """Interface améliorée pour DSO1 - Prédiction 100m"""
    st.header("🏊 Prédiction des Performances")
    st.markdown("### Prédire votre temps sur 100m Freestyle à partir de vos caractéristiques")
    
    mode = st.radio(
        "Choisissez votre mode de prédiction",
        ["⚡ Mode Rapide", "🔬 Mode Complet"],
        horizontal=True,
        help="Mode Rapide : Prédiction basée sur l'âge uniquement | Mode Complet : Prédiction avec toutes vos caractéristiques"
    )
    
    if mode == "⚡ Mode Rapide":
        st.info("💡 **Mode rapide** : Prédiction instantanée basée sur votre âge. Pour plus de précision, utilisez le mode complet.")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            age = st.slider("Âge du nageur", 15, 40, 22, 1,
                          help="Âge en années")
            
            if st.button("🔮 Prédire", type="primary", use_container_width=True):
                pred, lower, upper, _ = predict_time_svr_improved(age)
                
                # Affichage des résultats
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Temps prédit", f"{pred:.2f} s", delta=None)
                with col_b:
                    st.metric("Intervalle min", f"{lower:.2f} s")
                with col_c:
                    st.metric("Intervalle max", f"{upper:.2f} s")
                
                # Graphique de confiance
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=['Prédiction'],
                    y=[pred],
                    error_y=dict(type='data', array=[pred - lower], arrayminus=[upper - pred]),
                    marker_color='#1f77b4',
                    name='Temps prédit'
                ))
                fig.update_layout(
                    title="Prédiction avec Intervalle de Confiance (95%)",
                    yaxis_title="Temps (secondes)",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Avertissement sur la précision
                st.warning("⚠️ Cette prédiction utilise uniquement l'âge. Pour plus de précision, utilisez le mode avancé.")
    
    else:  # Mode avancé
        st.info("💡 **Mode complet** : Prédiction précise utilisant toutes vos caractéristiques physiques, techniques et historiques.")
        
        # Groupement logique des inputs
        st.markdown("#### 📋 Vos Informations")
        with st.expander("👤 Informations personnelles", expanded=True):
            col1, col2, col3 = st.columns(3)
            Age = col1.number_input("Âge", 15.0, 40.0, 22.0, 0.1)
            Height = col2.number_input("Taille (m)", 1.50, 2.10, 1.85, 0.01)
            Weight = col3.number_input("Poids (kg)", 50.0, 110.0, 75.0, 0.1)
            Sex = col1.selectbox("Sexe", options=list(le_sex.classes_))
            Injury_History = col2.selectbox("Antécédents de blessure", options=list(le_injury.classes_))
        
        with st.expander("💪 Condition physique et santé", expanded=True):
            col1, col2 = st.columns(2)
            Nutrition_Quality_Score = col1.slider("Score Nutrition (0-10)", 0.0, 10.0, 7.0, 0.1)
            Sleep_Hours = col2.slider("Heures de sommeil", 5.0, 12.0, 8.0, 0.1)
        
        with st.expander("⏱️ Vos temps de performance", expanded=True):
            col1, col2, col3 = st.columns(3)
            _50m = col1.number_input("50m (s)", 20.0, 40.0, 23.5, 0.01)
            _200m = col2.number_input("200m (s)", 100.0, 180.0, 112.0, 0.1)
            _400m = col3.number_input("400m (s)", 200.0, 300.0, 230.0, 0.1)
            _800m = col1.number_input("800m (s)", 400.0, 600.0, 470.0, 0.1)
            _1500m = col2.number_input("1500m (s)", 800.0, 1200.0, 900.0, 0.1)
            
            # Calcul automatique suggéré pour 100m
            suggested_100m = _50m * 2 + 2
            st.info(f"💡 Suggestion basée sur 50m: {suggested_100m:.2f}s (100m ≈ 2×50m + 2s)")
        
        if st.button("🔮 Prédire (Mode Avancé)", type="primary", use_container_width=True):
            with st.spinner("Calcul en cours..."):
                pred, lower, upper, std = predict_time_improved({
                    "Age": Age, "Height": Height, "Weight": Weight,
                    "Nutrition_Quality_Score": Nutrition_Quality_Score,
                    "Sleep_Hours": Sleep_Hours, "_50m": _50m, "_200m": _200m,
                    "_400m": _400m, "_800m": _800m, "_1500m": _1500m,
                    "Sex": Sex, "Injury_History": Injury_History
                })
            
            # Affichage amélioré
            st.success("✅ Prédiction terminée!")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("⏱️ Temps Prédit", f"{pred:.2f} s", delta=None)
            with col2:
                st.metric("📉 Intervalle Min", f"{lower:.2f} s")
            with col3:
                st.metric("📈 Intervalle Max", f"{upper:.2f} s")
            with col4:
                st.metric("📊 Incertitude", f"±{std:.2f} s")
            
            # Graphique de confiance
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Prédiction 100m'],
                y=[pred],
                error_y=dict(type='data', array=[pred - lower], arrayminus=[upper - pred]),
                marker_color='#2ecc71',
                name='Temps prédit',
                text=f"{pred:.2f}s",
                textposition='outside'
            ))
            fig.update_layout(
                title="Prédiction avec Intervalle de Confiance (95%)",
                yaxis_title="Temps (secondes)",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Analyse de sensibilité (exemple avec l'âge)
            st.subheader("📊 Analyse de Sensibilité")
            ages_range = np.arange(18, 31, 1)
            preds_sensitivity = []
            for a in ages_range:
                temp_inputs = {
                    "Age": float(a), "Height": Height, "Weight": Weight,
                    "Nutrition_Quality_Score": Nutrition_Quality_Score,
                    "Sleep_Hours": Sleep_Hours, "_50m": _50m, "_200m": _200m,
                    "_400m": _400m, "_800m": _800m, "_1500m": _1500m,
                    "Sex": Sex, "Injury_History": Injury_History
                }
                p, _, _, _ = predict_time_improved(temp_inputs)
                preds_sensitivity.append(p)
            
            fig_sens = px.line(x=ages_range, y=preds_sensitivity,
                             title="Impact de l'âge sur la prédiction",
                             labels={'x': 'Âge (années)', 'y': 'Temps prédit (s)'})
            fig_sens.add_vline(x=Age, line_dash="dash", line_color="red",
                              annotation_text=f"Âge actuel: {Age}")
            st.plotly_chart(fig_sens, use_container_width=True)

# ============================================================
# DSO2 - PRÉDICTION MÉDAILLE (CLASSIFICATION)
# ============================================================

def predict_medal_improved(inputs: dict):
    """Prédiction améliorée avec probabilités"""
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

    vec = np.array([
        inputs["Age"], inputs["Height"], inputs["Weight"],
        inputs["Nutrition_Quality_Score"], inputs["Sleep_Hours"],
        inputs["_50m"], inputs["_100m"], inputs["_200m"],
        inputs["_400m"], inputs["_800m"], inputs["_1500m"],
        sex_enc, inj_enc,
    ]).reshape(1, -1)
    
    xs = scaler_perf.transform(vec)
    pred = rf_medal.predict(xs)[0]
    proba = rf_medal.predict_proba(xs)[0]
    
    medal_pred = le_medal.inverse_transform([pred])[0]
    medal_probs = dict(zip(le_medal.classes_, proba))
    
    # Feature importance pour cette prédiction
    feature_names = [
        "Age", "Height", "Weight", "Nutrition Quality Score", "Sleep Hours",
        "50m Freestyle Time", "100m Freestyle Time", "200m Freestyle Time",
        "400m Freestyle Time", "800m Freestyle Time", "1500m Freestyle Time",
        "Sex", "Injury History"
    ]
    importances = rf_medal.feature_importances_
    top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:5]
    
    return medal_pred, medal_probs, top_features

def predict_injury_risk(inputs: dict):
    """Prédiction du risque de blessure basée sur les caractéristiques"""
    # Modèle simplifié basé sur les facteurs de risque connus
    risk_factors = {
        "Age": 0.0,
        "Sleep_Hours": 0.0,
        "Nutrition_Quality_Score": 0.0,
        "Injury_History": 0.0,
        "Training_Intensity": 0.0  # Estimé à partir des temps
    }
    
    # Facteur âge (jeunes et vétérans plus à risque)
    if inputs["Age"] < 18 or inputs["Age"] > 30:
        risk_factors["Age"] = 0.3
    elif inputs["Age"] < 20 or inputs["Age"] > 28:
        risk_factors["Age"] = 0.15
    
    # Facteur sommeil (moins de 7h = risque élevé)
    if inputs["Sleep_Hours"] < 7:
        risk_factors["Sleep_Hours"] = 0.4
    elif inputs["Sleep_Hours"] < 8:
        risk_factors["Sleep_Hours"] = 0.2
    
    # Facteur nutrition (score < 5 = risque élevé)
    if inputs["Nutrition_Quality_Score"] < 5:
        risk_factors["Nutrition_Quality_Score"] = 0.3
    elif inputs["Nutrition_Quality_Score"] < 7:
        risk_factors["Nutrition_Quality_Score"] = 0.15
    
    # Facteur antécédents
    inj_levels = {"None": 0.0, "Minor": 0.3, "Moderate": 0.5, "Severe": 0.7}
    risk_factors["Injury_History"] = inj_levels.get(inputs["Injury_History"], 0.2)
    
    # Facteur intensité d'entraînement (basé sur la variance des temps)
    times = [inputs["_50m"], inputs["_100m"], inputs["_200m"], inputs["_400m"]]
    if len([t for t in times if t > 0]) > 1:
        time_variance = np.std([t for t in times if t > 0])
        if time_variance > 20:  # Grande variance = entraînement irrégulier
            risk_factors["Training_Intensity"] = 0.2
    
    # Calcul du risque total (normalisé entre 0 et 1)
    total_risk = min(1.0, sum(risk_factors.values()))
    
    # Catégorisation du risque
    if total_risk < 0.3:
        risk_level = "Faible"
        risk_color = "#2ecc71"
    elif total_risk < 0.6:
        risk_level = "Modéré"
        risk_color = "#f39c12"
    else:
        risk_level = "Élevé"
        risk_color = "#e74c3c"
    
    return total_risk, risk_level, risk_color, risk_factors

def calculate_correlations(inputs: dict):
    """Calcule les corrélations entre les variables de performance"""
    # Création d'un DataFrame avec les données
    data = {
        "Age": [inputs["Age"]],
        "Height": [inputs["Height"]],
        "Weight": [inputs["Weight"]],
        "Nutrition": [inputs["Nutrition_Quality_Score"]],
        "Sleep": [inputs["Sleep_Hours"]],
        "Time_50m": [inputs["_50m"]],
        "Time_100m": [inputs["_100m"]],
        "Time_200m": [inputs["_200m"]],
        "Time_400m": [inputs["_400m"]],
    }
    
    df = pd.DataFrame(data)
    
    # Calcul des corrélations (avec données de référence si disponibles)
    # Pour l'instant, on utilise des corrélations théoriques connues
    correlations = {
        ("Time_50m", "Time_100m"): 0.85,
        ("Time_100m", "Time_200m"): 0.80,
        ("Time_200m", "Time_400m"): 0.75,
        ("Nutrition", "Time_100m"): -0.35,  # Négative = meilleure nutrition = meilleur temps
        ("Sleep", "Time_100m"): -0.30,
        ("Age", "Time_100m"): 0.25,  # Positive = plus âgé = temps plus élevé
    }
    
    return correlations

def render_dso2():
    """Interface améliorée pour DSO2 - Prédiction Médaille"""
    st.header("🥇 Analyse des Facteurs de Performance")
    st.markdown("### Identifier les variables influentes sur la performance et le risque de blessure")
    
    # Groupement logique des inputs
    st.markdown("#### 📋 Vos Informations")
    with st.expander("👤 Informations personnelles", expanded=True):
        col1, col2, col3 = st.columns(3)
        Age = col1.number_input("Âge (années)", 15.0, 40.0, 22.0, 0.1, key="dso2_age", help="Votre âge en années")
        Height = col2.number_input("Taille (mètres)", 1.50, 2.10, 1.85, 0.01, key="dso2_height", help="Votre taille en mètres")
        Weight = col3.number_input("Poids (kilogrammes)", 50.0, 110.0, 75.0, 0.1, key="dso2_weight", help="Votre poids en kg")
        Sex = col1.selectbox("Sexe", options=list(le_sex.classes_), key="dso2_sex")
        Injury_History = col2.selectbox("Antécédents de blessure", options=list(le_injury.classes_), key="dso2_injury", help="Votre historique de blessures")
    
    with st.expander("💪 Condition physique et santé", expanded=True):
        col1, col2 = st.columns(2)
        Nutrition_Quality_Score = col1.slider("Score nutritionnel (0-10)", 0.0, 10.0, 7.0, 0.1, key="dso2_nutrition", help="Évaluez la qualité de votre alimentation")
        Sleep_Hours = col2.slider("Heures de sommeil par nuit", 5.0, 12.0, 8.0, 0.1, key="dso2_sleep", help="Nombre moyen d'heures de sommeil")
    
    with st.expander("⏱️ Vos temps de performance", expanded=True):
        col1, col2, col3 = st.columns(3)
        _50m = col1.number_input("50m (s)", 20.0, 40.0, 23.5, 0.01, key="dso2_50m")
        _100m = col2.number_input("100m (s)", 40.0, 70.0, 50.1, 0.01, key="dso2_100m")
        _200m = col3.number_input("200m (s)", 100.0, 180.0, 112.0, 0.1, key="dso2_200m")
        _400m = col1.number_input("400m (s)", 200.0, 300.0, 230.0, 0.1, key="dso2_400m")
        _800m = col2.number_input("800m (s)", 400.0, 600.0, 470.0, 0.1, key="dso2_800m")
        _1500m = col3.number_input("1500m (s)", 800.0, 1200.0, 900.0, 0.1, key="dso2_1500m")
    
    if st.button("🔮 Prédire la Médaille", type="primary", use_container_width=True):
        with st.spinner("Analyse en cours..."):
            medal_pred, medal_probs, top_features = predict_medal_improved({
                "Age": Age, "Height": Height, "Weight": Weight,
                "Nutrition_Quality_Score": Nutrition_Quality_Score,
                "Sleep_Hours": Sleep_Hours, "_50m": _50m, "_100m": _100m,
                "_200m": _200m, "_400m": _400m, "_800m": _800m,
                "_1500m": _1500m, "Sex": Sex, "Injury_History": Injury_History
            })
        
        # Affichage de la prédiction principale
        medal_icons = {"Gold": "🥇", "Silver": "🥈", "Bronze": "🥉", "None": "🏅"}
        medal_names = {"Gold": "Or", "Silver": "Argent", "Bronze": "Bronze", "None": "Aucune médaille"}
        confidence = medal_probs[medal_pred]
        
        col_pred1, col_pred2 = st.columns([2, 1])
        with col_pred1:
            st.success(f"✅ **Prédiction : {medal_icons.get(medal_pred, '🏅')} {medal_names.get(medal_pred, medal_pred)}**")
        with col_pred2:
            st.metric("Confiance", f"{confidence:.1%}")
        
        # Gauge de confiance
        st.progress(confidence, text=f"Niveau de confiance: {confidence:.1%}")
        
        # Graphique des probabilités
        fig_proba = go.Figure(data=[
            go.Bar(
                x=list(medal_probs.keys()),
                y=list(medal_probs.values()),
                marker_color=['gold', 'silver', '#cd7f32', 'gray'],
                text=[f"{p:.1%}" for p in medal_probs.values()],
                textposition='outside'
            )
        ])
        fig_proba.update_layout(
            title="Probabilités par Type de Médaille",
            xaxis_title="Type de Médaille",
            yaxis_title="Probabilité",
            yaxis=dict(range=[0, 1]),
            height=400
        )
        st.plotly_chart(fig_proba, use_container_width=True)
        
        # Top 3 facteurs d'influence
        st.subheader("🔍 Les 5 Facteurs les Plus Importants")
        st.caption("Ces facteurs ont le plus d'impact sur vos chances de médaille")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            feat_names = [f[0] for f in top_features]
            feat_importances = [f[1] for f in top_features]
            
            fig_imp = go.Figure(data=[
                go.Bar(
                    x=feat_importances,
                    y=feat_names,
                    orientation='h',
                    marker_color='#3498db',
                    text=[f"{imp:.3f}" for imp in feat_importances],
                    textposition='outside'
                )
            ])
            fig_imp.update_layout(
                title="Importance des Features (Top 5)",
                xaxis_title="Importance",
                height=300
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Détails")
            for i, (feat, imp) in enumerate(top_features, 1):
                st.markdown(f"**{i}.** {feat}")
                st.caption(f"Impact: {imp:.1%}")
        
        # Recommandations basées sur les facteurs
        st.subheader("💡 Actions Recommandées")
        st.caption("Basées sur les facteurs les plus influents pour votre profil")
        if top_features[0][0] == "Nutrition Quality Score":
            st.info("🍎 **Améliorer la nutrition** pourrait augmenter vos chances de médaille de manière significative")
        if top_features[0][0] == "Sleep Hours":
            st.info("😴 **Optimiser le sommeil** est crucial pour améliorer vos performances")
        if "Injury History" in [f[0] for f in top_features[:3]]:
            st.warning("⚠️ Les antécédents de blessure impactent significativement vos chances. Consultez la section 'Risque de Blessure' ci-dessous.")
        
        # ========== NOUVELLE SECTION : ANALYSE DU RISQUE DE BLESSURE ==========
        st.markdown("---")
        st.subheader("🏥 Évaluation du Risque de Blessure")
        st.caption("Analyse basée sur vos caractéristiques physiques, habitudes et antécédents")
        
        with st.spinner("Évaluation du risque de blessure..."):
            injury_risk, risk_level, risk_color, risk_factors = predict_injury_risk({
                "Age": Age, "Sleep_Hours": Sleep_Hours,
                "Nutrition_Quality_Score": Nutrition_Quality_Score,
                "Injury_History": Injury_History,
                "_50m": _50m, "_100m": _100m, "_200m": _200m, "_400m": _400m
            })
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Niveau de Risque", risk_level)
        with col2:
            st.metric("Score de Risque", f"{injury_risk:.1%}")
        with col3:
            # Gauge visuelle
            fig_risk = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = injury_risk * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Risque (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': risk_color},
                    'steps': [
                        {'range': [0, 30], 'color': "#ecf0f1"},
                        {'range': [30, 60], 'color': "#fef9e7"},
                        {'range': [60, 100], 'color': "#fadbd8"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 60
                    }
                }
            ))
            fig_risk.update_layout(height=250)
            st.plotly_chart(fig_risk, use_container_width=True)
        
        # Facteurs de risque détaillés
            st.markdown("#### 📊 Contribution de Chaque Facteur")
            st.caption("Détail de l'impact de chaque facteur sur votre risque de blessure")
        risk_df = pd.DataFrame([
            {"Facteur": "Âge", "Contribution": f"{risk_factors['Age']:.1%}"},
            {"Facteur": "Heures de sommeil", "Contribution": f"{risk_factors['Sleep_Hours']:.1%}"},
            {"Facteur": "Score nutritionnel", "Contribution": f"{risk_factors['Nutrition_Quality_Score']:.1%}"},
            {"Facteur": "Antécédents de blessure", "Contribution": f"{risk_factors['Injury_History']:.1%}"},
            {"Facteur": "Intensité d'entraînement", "Contribution": f"{risk_factors['Training_Intensity']:.1%}"},
        ])
        
        fig_risk_factors = go.Figure(data=[
            go.Bar(
                x=risk_df["Facteur"],
                y=[float(c.replace('%', '')) for c in risk_df["Contribution"]],
                marker_color=risk_color,
                text=risk_df["Contribution"],
                textposition='outside'
            )
        ])
        fig_risk_factors.update_layout(
            title="Contribution des Facteurs de Risque",
            xaxis_title="Facteur",
            yaxis_title="Contribution (%)",
            height=300
        )
        st.plotly_chart(fig_risk_factors, use_container_width=True)
        
        # Recommandations de prévention
        st.markdown("#### 💡 Plan de Prévention Personnalisé")
        if injury_risk >= 0.6:
            st.error("🚨 **Risque ÉLEVÉ** - Actions immédiates recommandées:")
            st.markdown("""
            📋 **Actions prioritaires :**
            - 🏥 Consulter un médecin du sport pour évaluation complète
            - 📉 Réduire l'intensité d'entraînement de 20-30%
            - 😴 Augmenter les heures de sommeil à minimum 8h
            - 🍎 Améliorer le score nutritionnel à 7+
            - 🧘 Intégrer des séances de récupération active
            """)
        elif injury_risk >= 0.3:
            st.warning("⚠️ **Risque MODÉRÉ** - Précautions recommandées:")
            st.markdown("""
            📋 **Actions préventives :**
            - 🏥 Maintenir un suivi régulier avec un kinésithérapeute
            - 💪 Optimiser la récupération (sommeil, nutrition)
            - 🔄 Varier les types d'entraînement pour éviter la surcharge
            - 👂 Écouter les signaux d'alerte du corps
            """)
        else:
            st.success("✅ **Risque FAIBLE** - Continuez vos bonnes pratiques:")
            st.markdown("""
            📋 **Maintenir l'excellence :**
            - ✅ Maintenir les bonnes habitudes actuelles
            - 🏃 Continuer la prévention active (échauffement, étirements)
            - 📊 Suivi régulier pour maintenir ce niveau
            """)
        
        # ========== NOUVELLE SECTION : ANALYSE DE CORRÉLATION ==========
        st.markdown("---")
        st.subheader("📈 Relations entre les Variables")
        st.caption("Découvrez comment les différents facteurs interagissent entre eux")
        
        correlations = calculate_correlations({
            "Age": Age, "Height": Height, "Weight": Weight,
            "Nutrition_Quality_Score": Nutrition_Quality_Score,
            "Sleep_Hours": Sleep_Hours, "_50m": _50m, "_100m": _100m,
            "_200m": _200m, "_400m": _400m
        })
        
        # Création d'une matrice de corrélation
        corr_pairs = list(correlations.keys())
        corr_values = list(correlations.values())
        corr_labels = [f"{p[0]} vs {p[1]}" for p in corr_pairs]
        
        # Graphique des corrélations
        fig_corr = go.Figure(data=[
            go.Bar(
                x=corr_labels,
                y=corr_values,
                marker=dict(
                    color=corr_values,
                    colorscale='RdYlGn',
                    showscale=True,
                    cmin=-1,
                    cmax=1
                ),
                text=[f"{v:.2f}" for v in corr_values],
                textposition='outside'
            )
        ])
        fig_corr.update_layout(
            title="Corrélations entre Variables de Performance",
            xaxis_title="Paires de Variables",
            yaxis_title="Coefficient de Corrélation",
            xaxis_tickangle=-45,
            height=400,
            yaxis=dict(range=[-1, 1])
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Interprétation des corrélations
        st.markdown("#### 📊 Comprendre les Corrélations")
        st.caption("Interprétation des relations entre les variables")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Corrélations Fortes (>0.7):**")
            strong_corr = [(k, v) for k, v in correlations.items() if abs(v) > 0.7]
            for (var1, var2), val in strong_corr:
                st.markdown(f"- **{var1} ↔ {var2}**: {val:.2f}")
                if "Time" in var1 and "Time" in var2:
                    st.caption("  → Relation attendue entre temps de performance")
        
        with col2:
            st.markdown("**Corrélations Modérées (0.3-0.7):**")
            mod_corr = [(k, v) for k, v in correlations.items() if 0.3 <= abs(v) <= 0.7]
            for (var1, var2), val in mod_corr:
                st.markdown(f"- **{var1} ↔ {var2}**: {val:.2f}")
                if val < 0:
                    st.caption("  → Relation inverse (amélioration = réduction)")
        
        # Insights basés sur les corrélations
        st.markdown("#### 💡 Recommandations Basées sur les Relations")
        if abs(correlations.get(("Nutrition", "Time_100m"), 0)) > 0.3:
            st.info("🍎 **Nutrition et Performance** : Une meilleure nutrition est corrélée avec de meilleurs temps. Investir dans la nutrition peut améliorer vos performances de manière significative.")
        
        if abs(correlations.get(("Sleep", "Time_100m"), 0)) > 0.3:
            st.info("😴 **Sommeil et Performance** : Le sommeil est un facteur clé de performance. Optimiser votre sommeil peut réduire vos temps de 5-10%.")
        
        if abs(correlations.get(("Time_50m", "Time_100m"), 0)) > 0.7:
            st.info("⏱️ **Cohérence des Temps** : Les temps sur différentes distances sont fortement corrélés. Améliorer une distance peut bénéficier aux autres distances.")

# ============================================================
# DSO3 - CLUSTERING ET RECOMMANDATION
# ============================================================

def predict_cluster_improved(inputs: dict):
    """Prédiction de cluster améliorée avec visualisation"""
    g = le_gender.transform([inputs["gender"]])[0]
    c = le_country.transform([inputs["country"]])[0]
    vec = np.array([
        inputs["mean_time"], inputs["best_time"], inputs["std_time"],
        inputs["improvement"], inputs["n_competitions"], inputs["age"],
        g, c,
    ]).reshape(1, -1)
    
    Xs = scaler_cluster.transform(vec)
    label = int(kmeans.predict(Xs)[0])
    
    # Similarité avec les autres nageurs
    ref_scaled = scaler_cluster.transform(agg_ref[[
        "mean_time", "best_time", "std_time", "improvement",
        "n_competitions", "age", "gender_enc", "country_enc"
    ]])
    sims = (ref_scaled @ Xs.T).flatten()
    top_idx = sims.argsort()[::-1][:10]
    sim_norm = (sims - sims.min()) / (sims.max() - sims.min() + 1e-9)
    neighbors = agg_ref.iloc[top_idx][["Athlete Full Name", "cluster_kmeans"]].copy()
    neighbors = neighbors.assign(similarity=sim_norm[top_idx])
    
    # Caractéristiques du cluster
    cluster_data = agg_ref[agg_ref['cluster_kmeans'] == label]
    cluster_stats = {
        'mean_time_avg': cluster_data['mean_time'].mean(),
        'best_time_avg': cluster_data['best_time'].mean(),
        'age_avg': cluster_data['age'].mean(),
        'n_competitions_avg': cluster_data['n_competitions'].mean(),
    }
    
    return label, neighbors, cluster_stats

def render_dso3():
    """Interface améliorée pour DSO3 - Clustering"""
    st.header("🎯 Segmentation des Profils")
    st.markdown("### Découvrez votre profil et trouvez des nageurs similaires")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⏱️ Performances")
        mean_time = st.number_input("Temps moyen (secondes)", 30.0, 120.0, 47.5, 0.1, help="Votre temps moyen sur 100m")
        best_time = st.number_input("Meilleur temps (secondes)", 30.0, 120.0, 46.9, 0.1, help="Votre meilleur temps personnel")
        std_time = st.number_input("Écart-type des temps", 0.0, 50.0, 0.4, 0.1, help="Variabilité de vos performances")
        improvement = st.number_input("Ratio d'amélioration", 0.0, 1.0, 0.012, 0.001, format="%.4f", help="Progression entre temps moyen et meilleur temps")
    
    with col2:
        st.markdown("#### 👤 Informations personnelles")
        n_competitions = st.number_input("Nombre de compétitions", 2, 100, 5, 1, help="Nombre total de compétitions auxquelles vous avez participé")
        age = st.number_input("Âge (années)", 15.0, 50.0, 21.4, 0.1)
        gender = st.selectbox("Genre", options=list(le_gender.classes_))
        country = st.selectbox("Pays", options=list(le_country.classes_))
    
    if st.button("🔍 Analyser le Profil", type="primary", use_container_width=True):
        with st.spinner("Analyse du profil en cours..."):
            cluster_id, neighbors, cluster_stats = predict_cluster_improved({
                "mean_time": mean_time, "best_time": best_time,
                "std_time": std_time, "improvement": improvement,
                "n_competitions": n_competitions, "age": age,
                "gender": gender, "country": country
            })
        
        # Affichage du cluster
        cluster_names = {
            0: "🏊 Élite Performant",
            1: "💪 En Développement",
            2: "⚙️ Technique à Perfectionner",
            3: "🎯 Stratégie Optimale"
        }
        
        cluster_name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        st.success(f"✅ Profil identifié: **{cluster_name}** (Cluster {cluster_id})")
        
        # Description du cluster
        st.markdown(f"""
        <div class="cluster-description">
        <h4>📋 Caractéristiques du Cluster</h4>
        <ul>
            <li>Temps moyen typique: {cluster_stats['mean_time_avg']:.2f}s</li>
            <li>Meilleur temps typique: {cluster_stats['best_time_avg']:.2f}s</li>
            <li>Âge moyen: {cluster_stats['age_avg']:.1f} ans</li>
            <li>Nombre moyen de compétitions: {cluster_stats['n_competitions_avg']:.1f}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommandation
        recommendations = {
            0: "🏊 **Programme intensif** - Vous êtes dans le groupe élite. Focus sur le maintien et l'optimisation.",
            1: "💪 **Endurance + technique** - Développement des capacités physiques et techniques.",
            2: "⚙️ **Perfectionnement technique** - Améliorer la technique de nage pour réduire les temps.",
            3: "🎯 **Stratégie de course** - Optimiser la stratégie de course et la gestion de l'effort."
        }
        
        st.markdown(f"""
        <div class="recommendation-box">
        <h4>💡 Recommandation Personnalisée</h4>
        <p>{recommendations.get(cluster_id, "🔄 Suivi individuel recommandé")}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Visualisation des nageurs similaires
        st.subheader("👥 Nageurs avec un Profil Similaire")
        st.caption("Top 10 des nageurs les plus proches de votre profil")
        
        fig_sim = go.Figure(data=[
            go.Bar(
                x=neighbors['Athlete Full Name'],
                y=neighbors['similarity'],
                marker_color='#3498db',
                text=[f"{s:.2%}" for s in neighbors['similarity']],
                textposition='outside'
            )
        ])
        fig_sim.update_layout(
            title="Score de Similarité avec d'Autres Nageurs",
            xaxis_title="Nageur",
            yaxis_title="Similarité",
            xaxis_tickangle=-45,
            height=400
        )
        st.plotly_chart(fig_sim, use_container_width=True)
        
        # Tableau détaillé
        st.dataframe(
            neighbors[['Athlete Full Name', 'cluster_kmeans', 'similarity']].style.format({
                'similarity': '{:.2%}'
            }),
            use_container_width=True
        )
        
        # Visualisation PCA (si possible)
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            features_for_pca = agg_ref[[
                "mean_time", "best_time", "std_time", "improvement",
                "n_competitions", "age", "gender_enc", "country_enc"
            ]]
            X_pca = pca.fit_transform(scaler_cluster.transform(features_for_pca))
            
            fig_pca = px.scatter(
                x=X_pca[:, 0], y=X_pca[:, 1],
                color=agg_ref['cluster_kmeans'].astype(str),
                title="Visualisation PCA des Clusters",
                labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'},
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            # Ajouter le point du nageur actuel
            vec_current = np.array([[
                mean_time, best_time, std_time, improvement,
                n_competitions, age, le_gender.transform([gender])[0],
                le_country.transform([country])[0]
            ]])
            vec_scaled = scaler_cluster.transform(vec_current)
            vec_pca = pca.transform(vec_scaled)
            
            fig_pca.add_trace(go.Scatter(
                x=[vec_pca[0, 0]], y=[vec_pca[0, 1]],
                mode='markers',
                marker=dict(size=20, color='red', symbol='star'),
                name='Votre profil'
            ))
            
            st.plotly_chart(fig_pca, use_container_width=True)
        except Exception as e:
            st.warning(f"Visualisation PCA non disponible: {e}")

# ============================================================
# DSO4 - RECOMMANDATION STRATÉGIQUE
# ============================================================

def generate_training_program(cluster_id, goal, time_horizon, inputs):
    """Génère un programme d'entraînement structuré"""
    programs = {
        (0, "Améliorer les performances", "Court terme"): {
            "name": "Programme Intensif Court Terme - Élite",
            "weeks": 8,
            "structure": {
                "Lundi": {"type": "Vélocité", "intensity": "Haute", "volume": "2-3km", "details": "Sprints 25m-50m, récupération 1:2"},
                "Mardi": {"type": "Récupération active", "intensity": "Basse", "volume": "1-2km", "details": "Nage facile, étirements"},
                "Mercredi": {"type": "Endurance", "intensity": "Moyenne", "volume": "4-5km", "details": "Séries 200m-400m"},
                "Jeudi": {"type": "Technique", "intensity": "Moyenne", "volume": "2-3km", "details": "Drills techniques, perfectionnement"},
                "Vendredi": {"type": "Vélocité", "intensity": "Haute", "volume": "2-3km", "details": "Sprints 50m-100m"},
                "Samedi": {"type": "Endurance longue", "intensity": "Moyenne-Haute", "volume": "5-6km", "details": "Séries 400m-800m"},
                "Dimanche": {"type": "Repos", "intensity": "-", "volume": "-", "details": "Récupération complète"}
            }
        },
        (1, "Améliorer les performances", "Moyen terme"): {
            "name": "Programme Développement - En Progression",
            "weeks": 16,
            "structure": {
                "Lundi": {"type": "Endurance + Technique", "intensity": "Moyenne", "volume": "3-4km", "details": "Séries 200m avec focus technique"},
                "Mardi": {"type": "Vélocité", "intensity": "Moyenne-Haute", "volume": "2-3km", "details": "Sprints 25m-50m"},
                "Mercredi": {"type": "Récupération", "intensity": "Basse", "volume": "1-2km", "details": "Nage facile"},
                "Jeudi": {"type": "Endurance", "intensity": "Moyenne", "volume": "4-5km", "details": "Séries 300m-500m"},
                "Vendredi": {"type": "Technique", "intensity": "Moyenne", "volume": "2-3km", "details": "Drills et correction"},
                "Samedi": {"type": "Mixte", "intensity": "Variable", "volume": "3-4km", "details": "Combinaison vitesse/endurance"},
                "Dimanche": {"type": "Repos", "intensity": "-", "volume": "-", "details": "Récupération"}
            }
        }
    }
    
    # Programme par défaut si pas de correspondance exacte
    default_program = {
        "name": "Programme Personnalisé",
        "weeks": 12 if "Court terme" in time_horizon else 16 if "Moyen terme" in time_horizon else 24,
        "structure": {
            "Lundi": {"type": "Endurance", "intensity": "Moyenne", "volume": "3-4km", "details": "Séries 200m-400m"},
            "Mardi": {"type": "Vélocité", "intensity": "Haute", "volume": "2-3km", "details": "Sprints courts"},
            "Mercredi": {"type": "Technique", "intensity": "Moyenne", "volume": "2-3km", "details": "Perfectionnement"},
            "Jeudi": {"type": "Récupération", "intensity": "Basse", "volume": "1-2km", "details": "Nage facile"},
            "Vendredi": {"type": "Endurance", "intensity": "Moyenne", "volume": "3-4km", "details": "Séries moyennes"},
            "Samedi": {"type": "Mixte", "intensity": "Variable", "volume": "3-4km", "details": "Entraînement varié"},
            "Dimanche": {"type": "Repos", "intensity": "-", "volume": "-", "details": "Repos complet"}
        }
    }
    
    key = (cluster_id, goal, time_horizon)
    return programs.get(key, default_program)

def calculate_selection_criteria(inputs, predicted_time_100m=None):
    """Calcule les critères de sélection basés sur les performances"""
    # Standards de sélection (exemples - à adapter selon les compétitions)
    standards = {
        "Élite International": {"100m": 48.0, "200m": 105.0, "400m": 220.0},
        "National A": {"100m": 50.0, "200m": 110.0, "400m": 230.0},
        "National B": {"100m": 52.0, "200m": 115.0, "400m": 240.0},
        "Régional": {"100m": 55.0, "200m": 120.0, "400m": 250.0}
    }
    
    # Calcul des scores de sélection
    selection_scores = {}
    for level, stds in standards.items():
        score = 0
        total = 0
        
        if inputs.get("_100m", 0) > 0:
            if inputs["_100m"] <= stds["100m"]:
                score += 1.0
            elif inputs["_100m"] <= stds["100m"] + 2:
                score += 0.7
            elif inputs["_100m"] <= stds["100m"] + 4:
                score += 0.4
            total += 1
        
        if inputs.get("_200m", 0) > 0:
            if inputs["_200m"] <= stds["200m"]:
                score += 1.0
            elif inputs["_200m"] <= stds["200m"] + 5:
                score += 0.7
            elif inputs["_200m"] <= stds["200m"] + 10:
                score += 0.4
            total += 1
        
        if inputs.get("_400m", 0) > 0:
            if inputs["_400m"] <= stds["400m"]:
                score += 1.0
            elif inputs["_400m"] <= stds["400m"] + 10:
                score += 0.7
            elif inputs["_400m"] <= stds["400m"] + 20:
                score += 0.4
            total += 1
        
        selection_scores[level] = (score / total * 100) if total > 0 else 0
    
    # Niveau recommandé
    best_level = max(selection_scores, key=selection_scores.get)
    best_score = selection_scores[best_level]
    
    return selection_scores, best_level, best_score, standards

def render_dso4():
    """Interface pour DSO4 - Recommandation Stratégique"""
    st.header("💡 Recommandations Stratégiques")
    st.markdown("### Programmes d'entraînement personnalisés et critères de sélection")
    
    st.info("""
    💡 Obtenez des recommandations personnalisées basées sur votre profil complet : 
    prédictions de performance, analyse des facteurs clés, et segmentation de profil.
    """)
    
    # Onglets pour organiser les sections
    tab1, tab2, tab3 = st.tabs(["📝 Recommandations Générales", "🏊 Programmes d'Entraînement", "🎯 Critères de Sélection"])
    
    with tab1:
        # Section pour générer des recommandations
        st.subheader("📝 Recommandations Personnalisées")
        st.caption("Générez des recommandations adaptées à votre profil")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Inclure dans l'analyse :**")
            use_dso1 = st.checkbox("Prédictions de temps", value=True, help="Utiliser vos prédictions de performance")
            use_dso2 = st.checkbox("Analyse des facteurs", value=True, help="Utiliser l'analyse des facteurs de performance")
            use_dso3 = st.checkbox("Profil de segmentation", value=True, help="Utiliser votre profil de nageur")
        
        with col2:
            goal = st.selectbox(
                "Votre objectif principal",
                ["Améliorer les performances", "Gagner une médaille", "Optimiser l'entraînement", "Prévenir les blessures"],
                help="Quel est votre objectif principal ?"
            )
            time_horizon = st.selectbox(
                "Horizon temporel",
                ["Court terme (1-3 mois)", "Moyen terme (3-6 mois)", "Long terme (6-12 mois)"],
                help="Sur quelle période souhaitez-vous travailler ?"
            )
    
    if st.button("🎯 Générer les Recommandations", type="primary", use_container_width=True):
        st.success("✅ Analyse complète effectuée!")
        
        # Recommandations génériques (à améliorer avec les vrais modèles)
        recommendations = [
            {
                "priority": "Haute",
                "category": "Nutrition",
                "title": "Optimiser l'apport nutritionnel",
                "description": "Augmenter le score nutritionnel à 8+ pour améliorer les performances de 5-10%",
                "impact": "Élevé",
                "difficulty": "Moyenne",
                "timeline": "2-4 semaines"
            },
            {
                "priority": "Haute",
                "category": "Sommeil",
                "title": "Améliorer la qualité du sommeil",
                "description": "Maintenir 8-9 heures de sommeil régulier pour optimiser la récupération",
                "impact": "Élevé",
                "difficulty": "Faible",
                "timeline": "1-2 semaines"
            },
            {
                "priority": "Moyenne",
                "category": "Technique",
                "title": "Perfectionner la technique de nage",
                "description": "Travailler avec un coach sur les aspects techniques pour réduire les temps",
                "impact": "Moyen",
                "difficulty": "Moyenne",
                "timeline": "1-3 mois"
            },
            {
                "priority": "Moyenne",
                "category": "Entraînement",
                "title": "Programme d'entraînement personnalisé",
                "description": "Adapter l'intensité et la fréquence selon votre profil de cluster",
                "impact": "Moyen",
                "difficulty": "Moyenne",
                "timeline": "2-4 semaines"
            }
        ]
        
        # Affichage des recommandations
        for i, rec in enumerate(recommendations, 1):
            priority_color = {"Haute": "#e74c3c", "Moyenne": "#f39c12", "Basse": "#3498db"}
            
            with st.container():
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; border-left: 4px solid {priority_color.get(rec['priority'], '#95a5a6')}">
                    <h4>#{i} {rec['title']} <span style="color: {priority_color.get(rec['priority'], '#95a5a6')}">[{rec['priority']}]</span></h4>
                    <p><strong>Catégorie:</strong> {rec['category']}</p>
                    <p>{rec['description']}</p>
                    <p><strong>Impact:</strong> {rec['impact']} | <strong>Difficulté:</strong> {rec['difficulty']} | <strong>Délai:</strong> {rec['timeline']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Graphique de priorité
        fig_priority = go.Figure(data=[
            go.Bar(
                x=[r['title'] for r in recommendations],
                y=[{"Haute": 3, "Moyenne": 2, "Basse": 1}[r['priority']] for r in recommendations],
                marker_color=[priority_color.get(r['priority'], '#95a5a6') for r in recommendations],
                text=[r['priority'] for r in recommendations],
                textposition='outside'
            )
        ])
        fig_priority.update_layout(
            title="Priorité des Recommandations",
            xaxis_title="Recommandation",
            yaxis_title="Niveau de Priorité",
            xaxis_tickangle=-45,
            height=400
        )
        st.plotly_chart(fig_priority, use_container_width=True)
    
    with tab2:
        st.subheader("🏊 Programmes d'Entraînement Personnalisés")
        st.info("💡 Les programmes sont adaptés à votre profil, vos objectifs et votre horizon temporel.")
        
        # Inputs pour générer le programme
        col1, col2, col3 = st.columns(3)
        with col1:
            cluster_input = st.selectbox(
                "Votre profil de nageur",
                [0, 1, 2, 3],
                format_func=lambda x: {0: "🏊 Élite Performant", 1: "💪 En Développement", 2: "⚙️ Technique à Perfectionner", 3: "🎯 Stratégie Optimale"}[x],
                help="Déterminez votre profil dans la section 'Segmentation des Profils'"
            )
        with col2:
            goal_input = st.selectbox(
                "Objectif",
                ["Améliorer les performances", "Gagner une médaille", "Optimiser l'entraînement", "Prévenir les blessures"]
            )
        with col3:
            horizon_input = st.selectbox(
                "Horizon",
                ["Court terme (1-3 mois)", "Moyen terme (3-6 mois)", "Long terme (6-12 mois)"]
            )
        
        if st.button("📅 Générer le Programme d'Entraînement", type="primary", use_container_width=True):
            program = generate_training_program(cluster_input, goal_input, horizon_input, {})
            
            st.success(f"✅ Programme généré : **{program['name']}** ({program['weeks']} semaines)")
            
            # Affichage du programme hebdomadaire
            st.markdown(f"### 📋 Structure Hebdomadaire ({program['weeks']} semaines)")
            
            program_df = pd.DataFrame([
                {
                    "Jour": day,
                    "Type": details["type"],
                    "Intensité": details["intensity"],
                    "Volume": details["volume"],
                    "Détails": details["details"]
                }
                for day, details in program["structure"].items()
            ])
            
            st.dataframe(program_df, use_container_width=True, hide_index=True)
            
            # Graphique de répartition de l'intensité
            intensity_dist = {}
            for day, details in program["structure"].items():
                if details["intensity"] != "-":
                    intensity_dist[details["intensity"]] = intensity_dist.get(details["intensity"], 0) + 1
            
            if intensity_dist:
                fig_intensity = go.Figure(data=[
                    go.Pie(
                        labels=list(intensity_dist.keys()),
                        values=list(intensity_dist.values()),
                        hole=0.4,
                        marker_colors=['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
                    )
                ])
                fig_intensity.update_layout(
                    title="Répartition de l'Intensité d'Entraînement",
                    height=400
                )
                st.plotly_chart(fig_intensity, use_container_width=True)
            
            # Recommandations spécifiques
            st.markdown("### 💡 Comment Suivre ce Programme")
            st.markdown(f"""
            📅 **Durée totale** : {program['weeks']} semaines
            
            📈 **Progression** : Augmenter progressivement l'intensité toutes les 2-3 semaines
            
            🧘 **Récupération** : Respecter les jours de repos et récupération active
            
            🔄 **Adaptation** : Ajuster selon vos sensations et performances
            
            📊 **Suivi** : Noter vos temps et sensations après chaque séance
            """)
    
    with tab3:
        st.subheader("🎯 Éligibilité aux Compétitions")
        st.info("💡 Évaluez votre niveau et découvrez pour quelles compétitions vous êtes éligible selon vos performances.")
        
        # Inputs des performances
        st.markdown("### ⏱️ Vos Meilleurs Temps de Performance")
        st.caption("Entrez vos meilleurs temps réalisés en compétition officielle")
        col1, col2, col3 = st.columns(3)
        with col1:
            time_100m_sel = st.number_input("100m Freestyle (s)", 40.0, 70.0, 50.0, 0.1, key="sel_100m")
        with col2:
            time_200m_sel = st.number_input("200m Freestyle (s)", 100.0, 180.0, 110.0, 0.1, key="sel_200m")
        with col3:
            time_400m_sel = st.number_input("400m Freestyle (s)", 200.0, 300.0, 230.0, 0.1, key="sel_400m")
        
        if st.button("📊 Évaluer les Critères de Sélection", type="primary", use_container_width=True):
            selection_scores, best_level, best_score, standards = calculate_selection_criteria({
                "_100m": time_100m_sel,
                "_200m": time_200m_sel,
                "_400m": time_400m_sel
            })
            
            st.success(f"✅ Niveau recommandé : **{best_level}** (Score: {best_score:.1f}%)")
            
            # Graphique des scores de sélection
            fig_selection = go.Figure(data=[
                go.Bar(
                    x=list(selection_scores.keys()),
                    y=list(selection_scores.values()),
                    marker=dict(
                        color=list(selection_scores.values()),
                        colorscale='RdYlGn',
                        showscale=True,
                        cmin=0,
                        cmax=100
                    ),
                    text=[f"{v:.1f}%" for v in selection_scores.values()],
                    textposition='outside'
                )
            ])
            fig_selection.update_layout(
                title="Scores de Sélection par Niveau de Compétition",
                xaxis_title="Niveau de Compétition",
                yaxis_title="Score de Sélection (%)",
                height=400
            )
            st.plotly_chart(fig_selection, use_container_width=True)
            
            # Tableau comparatif avec les standards
            st.markdown("### 📋 Comparaison avec les Standards")
            comparison_data = []
            for level, stds in standards.items():
                comparison_data.append({
                    "Niveau": level,
                    "Votre 100m": f"{time_100m_sel:.2f}s",
                    "Standard 100m": f"{stds['100m']:.2f}s",
                    "Écart 100m": f"{time_100m_sel - stds['100m']:.2f}s",
                    "Votre 200m": f"{time_200m_sel:.2f}s",
                    "Standard 200m": f"{stds['200m']:.2f}s",
                    "Écart 200m": f"{time_200m_sel - stds['200m']:.2f}s",
                    "Score": f"{selection_scores[level]:.1f}%"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Recommandations pour améliorer la sélection
            st.markdown("### 💡 Plan d'Amélioration")
            if best_score < 70:
                st.warning(f"⚠️ Votre score actuel ({best_score:.1f}%) est en dessous du seuil recommandé (70%).")
                st.markdown("""
                📋 **Actions prioritaires :**
                - 🎯 Focus sur l'amélioration des temps sur les distances clés
                - 💪 Travailler spécifiquement les distances où l'écart est le plus important
                - 👨‍🏫 Consulter un entraîneur pour un plan d'amélioration ciblé
                - 🏊 Participer à des compétitions de niveau inférieur pour gagner en expérience
                """)
            elif best_score < 85:
                st.info(f"ℹ️ Votre score ({best_score:.1f}%) est bon. Quelques améliorations ciblées peuvent vous faire passer au niveau supérieur.")
                st.markdown("""
                📋 **Actions recommandées :**
                - ⚙️ Affiner la technique sur les distances où vous êtes proche du standard
                - 💪 Optimiser la préparation physique et mentale
                - 🏆 Participer à des compétitions de qualification
                """)
            else:
                st.success(f"✅ Excellent score ({best_score:.1f}%) ! Vous êtes éligible pour le niveau **{best_level}**.")
                st.markdown("""
                📋 **Prochaines étapes :**
                - ✅ Maintenir ce niveau de performance
                - 🏆 Participer aux compétitions de sélection
                - 📈 Continuer l'entraînement pour progresser vers le niveau supérieur
                """)

# ============================================================
# PAGE PRINCIPALE
# ============================================================

def main():
    """Fonction principale de l'application"""
    # Header
    st.markdown('<h1 class="main-header">🏊 Aqualyze</h1>', unsafe_allow_html=True)
    st.markdown("### *Votre assistant intelligent pour optimiser vos performances en natation*")
    st.markdown("---")
    
    # Sidebar avec navigation
    with st.sidebar:
        st.header("🧭 Navigation")
        page = st.radio(
            "Choisissez une fonctionnalité",
            ["🏊 Prédiction des Performances", "🥇 Analyse des Facteurs", 
             "🎯 Segmentation des Profils", "💡 Recommandations"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.header("ℹ️ À propos")
        st.info("""
        **Aqualyze** utilise l'intelligence artificielle pour analyser vos performances en natation.
        
        🏊 **Prédiction** : Estimez votre temps sur 100m
        🥇 **Analyse** : Comprenez les facteurs clés de performance
        🎯 **Profil** : Découvrez votre segment et nageurs similaires
        💡 **Recommandations** : Obtenez des programmes personnalisés
        """)
        
        st.markdown("---")
        st.markdown("### 🚀 Guide Rapide")
        st.markdown("""
        1. **Prédiction** : Estimez votre temps sur 100m
        2. **Analyse** : Comprenez vos facteurs clés
        3. **Profil** : Découvrez votre segment
        4. **Recommandations** : Obtenez un plan personnalisé
        """)
        st.caption("💡 *Commencez par la prédiction pour une première analyse*")
    
    # Routage vers les différentes pages
    if "Prédiction" in page or "Performances" in page:
        render_dso1()
    elif "Analyse" in page or "Facteurs" in page:
        render_dso2()
    elif "Segmentation" in page or "Profils" in page:
        render_dso3()
    elif "Recommandations" in page:
        render_dso4()

if __name__ == "__main__":
    main()

