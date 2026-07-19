"""
Script d'entraînement du modèle de détection de fraude bancaire.
Adapté à la base réelle : Bank_transaction_scenario1.csv
(séparateur ';', cible 'Target' à 3 classes : Normal / Suspect / Fraude)
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------------------------------------------
# 1. Chargement des données
# ------------------------------------------------------------------
df = pd.read_csv("data/transactions.csv", sep=";")

# ------------------------------------------------------------------
# 2. Ingénierie des variables (feature engineering)
# ------------------------------------------------------------------
df["Date"] = pd.to_datetime(df["Date"])
df["Heure"] = df["Date"].dt.hour
df["JourSemaine"] = df["Date"].dt.dayofweek  # 0 = lundi

# Localisation habituelle du client : la transaction se fait-elle dans
# la ville la plus fréquente historiquement pour ce client ?
loc_habituelle = (
    df.groupby("ID Clients")["Localisation"]
    .agg(lambda s: s.mode().iloc[0])
    .rename("Localisation_frequente")
)
df = df.merge(loc_habituelle, on="ID Clients", how="left")
df["Localisation_habituelle"] = np.where(
    df["Localisation"] == df["Localisation_frequente"], "Oui", "Non"
)

# ------------------------------------------------------------------
# 3. Sélection des features et encodage
# ------------------------------------------------------------------
features_num = ["Montant", "Heure", "JourSemaine"]
features_cat = ["Type de transaction", "Status operation", "Localisation", "Localisation_habituelle"]

encoders = {}
df_encoded = df.copy()
for col in features_cat:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    encoders[col] = le

X = df_encoded[features_num + features_cat]

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df["Target"])  # Fraude / Normal / Suspect -> 0/1/2

# ------------------------------------------------------------------
# 4. Normalisation des variables numériques
# ------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[features_num] = scaler.fit_transform(X[features_num])

# ------------------------------------------------------------------
# 5. Split train/test (stratifié, car classes déséquilibrées)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------------
# 6. Entraînement
# ------------------------------------------------------------------
# Jeu de données déséquilibré (Fraude ~3.7%, Suspect ~20%, Normal ~76%)
# -> class_weight="balanced" pour compenser
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
)
model.fit(X_train, y_train)

# ------------------------------------------------------------------
# 7. Évaluation
# ------------------------------------------------------------------
y_pred = model.predict(X_test)
print("=== Rapport de classification ===")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))
print("=== Matrice de confusion ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== Importance des variables ===")
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importances)

# ------------------------------------------------------------------
# 8. Sauvegarde du modèle, du scaler, des encodeurs et des métadonnées
# ------------------------------------------------------------------
joblib.dump(model, "model/fraud_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(encoders, "model/encoders.pkl")
joblib.dump(target_encoder, "model/target_encoder.pkl")
joblib.dump(
    {
        "features_num": features_num,
        "features_cat": features_cat,
        "localisation_frequente": loc_habituelle.to_dict(),
    },
    "model/metadata.pkl",
)

print("\nModèle et artefacts sauvegardés dans model/")
