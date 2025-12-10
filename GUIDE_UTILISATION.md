# 📖 Guide d'Utilisation - Application ML Natation Améliorée

## 🚀 Démarrage Rapide

### Installation

1. **Installer les dépendances**:
```bash
pip install -r requirements.txt
```

2. **Lancer l'application améliorée**:
```bash
streamlit run streamlit_app_improved.py
```

3. **Ou lancer l'ancienne version pour comparaison**:
```bash
streamlit run streamlit_app.py
```

## 🎯 Navigation dans l'Application

L'application améliorée utilise une **sidebar de navigation** pour basculer entre les 4 DSO :

1. **🏊 DSO1 - Prédiction 100m** : Prédiction du temps sur 100m freestyle
2. **🥇 DSO2 - Prédiction Médaille** : Prédiction de la médaille
3. **🎯 DSO3 - Clustering** : Segmentation des profils de nageurs
4. **💡 DSO4 - Recommandation** : Recommandations stratégiques

## 📊 Utilisation de Chaque DSO

### DSO1 - Prédiction 100m

#### Mode Rapide
1. Sélectionner "⚡ Rapide (SVR - Age uniquement)"
2. Ajuster l'âge avec le slider
3. Cliquer sur "🔮 Prédire"
4. Consulter les résultats avec intervalle de confiance

#### Mode Avancé
1. Sélectionner "🔬 Avancé (Random Forest - Toutes les features)"
2. Remplir les informations dans les sections :
   - **Informations Personnelles** : Âge, Taille, Poids, Sexe, Blessures
   - **Condition Physique** : Nutrition, Sommeil
   - **Temps de Performance** : 50m, 200m, 400m, 800m, 1500m
3. Cliquer sur "🔮 Prédire (Mode Avancé)"
4. Consulter :
   - Temps prédit avec intervalle de confiance
   - Graphique de confiance
   - Analyse de sensibilité (impact de l'âge)

### DSO2 - Prédiction Médaille

1. Remplir les informations dans les sections :
   - **Informations Personnelles** : Âge, Taille, Poids, Sexe, Blessures
   - **Condition Physique** : Nutrition, Sommeil
   - **Temps de Performance** : Tous les temps (50m à 1500m)
2. Cliquer sur "🔮 Prédire la Médaille"
3. Consulter :
   - Prédiction principale avec icône de médaille
   - Gauge de confiance
   - Graphique des probabilités pour toutes les classes
   - Top 5 facteurs d'influence
   - Recommandations personnalisées

### DSO3 - Clustering

1. Remplir les caractéristiques du nageur :
   - **Caractéristiques** : Temps moyen, Meilleur temps, Écart-type, Ratio d'amélioration
   - **Informations Personnelles** : Nombre de compétitions, Âge, Genre, Pays
2. Cliquer sur "🔍 Analyser le Profil"
3. Consulter :
   - Profil identifié (nom du cluster)
   - Caractéristiques du cluster
   - Recommandation personnalisée
   - Top 10 nageurs similaires avec scores de similarité
   - Visualisation PCA des clusters (position du nageur)

### DSO4 - Recommandation

1. Configurer les options :
   - Cocher les DSO à inclure (DSO1, DSO2, DSO3)
   - Sélectionner l'objectif principal
   - Choisir l'horizon temporel
2. Cliquer sur "🎯 Générer les Recommandations"
3. Consulter :
   - Liste priorisée des recommandations
   - Détails de chaque recommandation (impact, difficulté, délai)
   - Graphique de priorité

## 🎨 Fonctionnalités Améliorées

### Visualisations Interactives
- **Graphiques Plotly** : Zoom, pan, hover pour plus de détails
- **Graphiques de confiance** : Intervalles de confiance visuels
- **Graphiques de similarité** : Comparaison avec autres nageurs
- **Visualisations PCA** : Position dans l'espace des clusters

### Feedback Utilisateur
- **Messages de succès/erreur** : Feedback clair sur les actions
- **Indicateurs de chargement** : Spinners pendant les calculs
- **Tooltips** : Aide contextuelle sur les champs
- **Suggestions automatiques** : Calculs suggérés (ex: 100m ≈ 2×50m + 2s)

### Organisation de l'Interface
- **Groupement logique** : Inputs organisés par catégorie
- **Expandeurs** : Sections repliables pour réduire l'encombrement
- **Colonnes** : Layout optimisé pour l'utilisation
- **Design cohérent** : CSS personnalisé pour une meilleure apparence

## 💡 Conseils d'Utilisation

1. **Commencez par le mode rapide** (DSO1) pour une première impression
2. **Utilisez le mode avancé** pour des prédictions plus précises
3. **Consultez les visualisations** pour mieux comprendre les résultats
4. **Explorez les recommandations** pour des actions concrètes
5. **Comparez avec d'autres nageurs** (DSO3) pour le contexte

## ⚠️ Notes Importantes

- Les performances des modèles (DSO1, DSO2) sont actuellement limitées
- Les intervalles de confiance sont des estimations
- Les recommandations sont génériques et doivent être adaptées
- L'application nécessite tous les fichiers dans le dossier `artifacts/`

## 🔧 Dépannage

### Erreur de chargement des modèles
- Vérifier que tous les fichiers `.joblib` sont dans `artifacts/`
- Vérifier que `agg_reference.csv` existe dans `artifacts/`

### Erreur d'import Plotly
- Installer avec : `pip install plotly>=5.0.0`

### Application lente
- Les modèles sont mis en cache avec `@st.cache_resource`
- La première exécution peut être plus lente

## 📚 Documentation Complète

Pour plus de détails :
- **`ANALYSE_ET_RECOMMANDATIONS_DSO.md`** : Analyse complète et recommandations détaillées
- **`RESUME_EXECUTIF.md`** : Résumé exécutif du projet

## 🆘 Support

En cas de problème :
1. Vérifier les dépendances installées
2. Consulter les messages d'erreur dans la console
3. Vérifier la structure des fichiers dans `artifacts/`
4. Consulter la documentation complète

