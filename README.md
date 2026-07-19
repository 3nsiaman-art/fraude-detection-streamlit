# Détection de fraude bancaire — Streamlit

Application de détection de fraude appliquée à la base `Bank_transaction_scenario1.csv`
(5 382 transactions, cible **Target** à 3 classes : `Normal`, `Suspect`, `Fraude`).

## Structure

```
fraude-detection-streamlit/
├── data/
│   └── transactions.csv          # base de transactions (sép. ';')
├── model/
│   ├── train_model.py            # script d'entraînement
│   ├── fraud_model.pkl           # RandomForestClassifier entraîné
│   ├── scaler.pkl                # StandardScaler (variables numériques)
│   ├── encoders.pkl              # LabelEncoders (variables catégorielles)
│   ├── target_encoder.pkl        # LabelEncoder de la cible
│   └── metadata.pkl              # listes de features + localisation habituelle par client
├── app.py                        # application Streamlit
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation locale

```bash
python -m venv venv
source venv/bin/activate      # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## Entraînement du modèle

```bash
python model/train_model.py
```

Génère `fraud_model.pkl`, `scaler.pkl`, `encoders.pkl`, `target_encoder.pkl`, `metadata.pkl`.

## Lancer l'application

```bash
streamlit run app.py
```

Ouvre automatiquement http://localhost:8501.

## Variables utilisées par le modèle

| Variable | Origine |
|---|---|
| Montant | colonne `Montant` |
| Heure | extraite de `Date` |
| JourSemaine | extraite de `Date` |
| Type de transaction | colonne `Type de transaction` |
| Status operation | colonne `Status operation` |
| Localisation | colonne `Localisation` |
| Localisation_habituelle | dérivée : la localisation correspond-elle à la ville la plus fréquente historiquement pour ce client (`ID Clients`) ? |

## Performance (test set, 20%)

- Accuracy globale : ~0.88
- F1-score classe **Fraude** (minoritaire, ~3.7% des cas) : ~0.69, grâce à `class_weight="balanced"`
- Variables les plus importantes : Montant, Statut de l'opération, Localisation

## Déploiement sur Streamlit Community Cloud

1. Pousser le dépôt sur GitHub (voir guide de déploiement fourni).
2. Se rendre sur https://share.streamlit.io, se connecter avec GitHub.
3. "New app" → sélectionner le dépôt, la branche `main`, le fichier `app.py`.
4. Déployer.

⚠️ Vérifier que `data/transactions.csv` ne contient pas de données clients réelles
avant de rendre le dépôt public (anonymiser si nécessaire).
