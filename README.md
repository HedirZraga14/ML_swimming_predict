# 🏊 Aqualyze

**Aqualyze** est une application d'intelligence artificielle pour l'analyse et l'optimisation des performances en natation.

## 🎯 Fonctionnalités

- 🏊 **Prédiction des Performances** : Prédisez votre temps sur 100m Freestyle à partir de vos caractéristiques physiques, techniques et historiques
- 🥇 **Analyse des Facteurs** : Identifiez les variables les plus influentes sur la performance et le risque de blessure
- 🎯 **Segmentation des Profils** : Découvrez votre profil de nageur et trouvez des athlètes similaires
- 💡 **Recommandations Stratégiques** : Obtenez des programmes d'entraînement personnalisés et des critères de sélection

## 🚀 Installation

### Prérequis
- Python 3.8 ou supérieur
- pip

### Installation des dépendances

```bash
pip install -r requirements.txt
```

## 📖 Utilisation

### Lancer l'application

```bash
streamlit run streamlit_app_improved.py
```

L'application sera accessible sur `http://localhost:8501`

## 📁 Structure du Projet

```
Aqualyze/
├── streamlit_app_improved.py    # Application principale (interface améliorée)
├── streamlit_app.py             # Version originale
├── requirements.txt             # Dépendances Python
├── artifacts/                   # Modèles ML (non inclus dans Git)
│   ├── *.joblib
│   └── *.csv
├── GUIDE_*.md                   # Guides et documentation
├── .gitignore                   # Fichiers à exclure de Git
└── README.md                    # Ce fichier
```

## 🛠️ Technologies Utilisées

- **Streamlit** : Interface utilisateur web
- **Scikit-learn** : Machine Learning (régression, classification, clustering)
- **Pandas** : Manipulation de données
- **NumPy** : Calculs numériques
- **Plotly** : Visualisations interactives
- **Joblib** : Sauvegarde des modèles

## 📊 Modèles ML

L'application utilise plusieurs modèles de machine learning :

- **Régression** : Random Forest, SVR pour la prédiction de temps sur 100m
- **Classification** : Random Forest pour la prédiction de médaille
- **Clustering** : KMeans pour la segmentation des profils de nageurs
- **Analyse** : Feature importance, corrélations, risque de blessure

## 🎨 Interface Utilisateur

L'interface est conçue pour être :
- ✅ Intuitive et conviviale
- ✅ Sans jargon technique
- ✅ Avec visualisations interactives
- ✅ Guide rapide intégré

## 📚 Documentation

- `GUIDE_DEPLOIEMENT_GITHUB.md` : Guide complet pour déployer sur GitHub
- `GUIDE_UTILISATION.md` : Guide d'utilisation de l'application
- `ANALYSE_ET_RECOMMANDATIONS_DSO.md` : Analyse technique du projet
- `VERIFICATION_OBJECTIFS_DSO.md` : Vérification de conformité aux objectifs

## 🔧 Développement

### Structure des DSO (Decision Support Objects)

1. **Prédiction des Performances** : Modèle de régression pour prédire le temps sur 100m
2. **Analyse des Facteurs** : Classification et analyse statistique des variables influentes
3. **Segmentation des Profils** : Clustering pour regrouper les nageurs similaires
4. **Recommandations Stratégiques** : Système de recommandation pour programmes d'entraînement

## 👥 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
- Ouvrir une issue pour signaler un bug
- Proposer de nouvelles fonctionnalités
- Créer une pull request

## 📄 Licence

[Spécifiez votre licence ici]

## 👤 Auteur

[Votre nom]

## 🙏 Remerciements

Merci à tous les contributeurs et à la communauté open source.

---

**🏊 Fait avec ❤️ pour la communauté de la natation**

