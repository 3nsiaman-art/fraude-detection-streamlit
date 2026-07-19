import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------
st.set_page_config(
    page_title="Détection de Fraude Bancaire",
    page_icon="🏦",
    layout="wide",
)

# ---------------------------------------------------------
# Chargement du modèle et des artefacts (mis en cache)
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model/fraud_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    encoders = joblib.load("model/encoders.pkl")
    target_encoder = joblib.load("model/target_encoder.pkl")
    metadata = joblib.load("model/metadata.pkl")
    return model, scaler, encoders, target_encoder, metadata


model, scaler, encoders, target_encoder, metadata = load_artifacts()
features_num = metadata["features_num"]
features_cat = metadata["features_cat"]
loc_frequente = metadata["localisation_frequente"]

# ---------------------------------------------------------
# En-tête
# ---------------------------------------------------------
st.title("🏦 Système de Détection de Fraude Bancaire")
st.markdown(
    "Analysez une transaction ou un lot de transactions pour détecter un "
    "risque de fraude (classes : **Normal**, **Suspect**, **Fraude**)."
)

mode = st.sidebar.radio("Mode d'analyse", ["Transaction unique", "Fichier CSV (lot)"])


def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Reproduit l'ingénierie des variables faite à l'entraînement."""
    df = df_raw.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Heure"] = df["Date"].dt.hour
    df["JourSemaine"] = df["Date"].dt.dayofweek

    df["Localisation_frequente"] = df["ID Clients"].map(loc_frequente)
    df["Localisation_habituelle"] = np.where(
        df["Localisation"] == df["Localisation_frequente"], "Oui", "Non"
    )

    for col in features_cat:
        le = encoders[col]
        # gère les valeurs jamais vues à l'entraînement
        df[col] = df[col].apply(lambda v: v if v in le.classes_ else le.classes_[0])
        df[col] = le.transform(df[col])

    X = df[features_num + features_cat].copy()
    X[features_num] = scaler.transform(X[features_num])
    return X


# ===========================================================
# MODE 1 : Transaction unique
# ===========================================================
if mode == "Transaction unique":
    st.subheader("Saisie manuelle d'une transaction")

    id_clients_options = sorted(loc_frequente.keys())

    col1, col2 = st.columns(2)
    with col1:
        id_client = st.selectbox("ID Client", id_clients_options)
        montant = st.number_input(
            "Montant de la transaction (FCFA)", min_value=0.0, value=50000.0, step=1000.0
        )
        date_transaction = st.date_input("Date de la transaction")
        heure_transaction = st.slider("Heure de la transaction (0-23)", 0, 23, 12)
    with col2:
        type_transaction = st.selectbox("Type de transaction", list(encoders["Type de transaction"].classes_))
        status_operation = st.selectbox("Statut de l'opération", list(encoders["Status operation"].classes_))
        localisation = st.selectbox("Localisation de la transaction", list(encoders["Localisation"].classes_))

    if st.button("Analyser la transaction", type="primary"):
        date_complete = pd.Timestamp.combine(date_transaction, pd.Timestamp("00:00").time()) + pd.Timedelta(
            hours=heure_transaction
        )
        transaction = pd.DataFrame(
            [
                {
                    "ID Clients": id_client,
                    "Montant": montant,
                    "Date": date_complete,
                    "Type de transaction": type_transaction,
                    "Status operation": status_operation,
                    "Localisation": localisation,
                }
            ]
        )

        X = build_features(transaction)
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        classe_predite = target_encoder.inverse_transform([prediction])[0]

        st.divider()
        if classe_predite == "Fraude":
            st.error(f"⚠️ **Transaction frauduleuse** — probabilité : {proba[prediction]:.1%}")
        elif classe_predite == "Suspect":
            st.warning(f"🟠 **Transaction suspecte** — probabilité : {proba[prediction]:.1%}")
        else:
            st.success(f"✅ **Transaction normale** — probabilité : {proba[prediction]:.1%}")

        proba_df = pd.DataFrame(
            {"Classe": target_encoder.classes_, "Probabilité": proba}
        ).sort_values("Probabilité", ascending=False)
        st.dataframe(proba_df, hide_index=True, use_container_width=True)
        st.bar_chart(proba_df.set_index("Classe"))

# ===========================================================
# MODE 2 : Fichier CSV (lot)
# ===========================================================
else:
    st.subheader("Analyse par lot (fichier CSV)")
    st.caption("Le fichier doit avoir le même format que la base d'entraînement (séparateur ';').")

    fichier = st.file_uploader("Déposez un fichier CSV de transactions", type=["csv"])

    if fichier is not None:
        df_raw = pd.read_csv(fichier, sep=";")
        st.write("Aperçu des données :", df_raw.head())

        if st.button("Lancer l'analyse du lot"):
            X = build_features(df_raw)
            predictions = model.predict(X)
            probas = model.predict_proba(X)

            df_result = df_raw.copy()
            df_result["prediction"] = target_encoder.inverse_transform(predictions)
            df_result["probabilite_fraude"] = probas[:, list(target_encoder.classes_).index("Fraude")]

            nb_fraudes = (df_result["prediction"] == "Fraude").sum()
            nb_suspects = (df_result["prediction"] == "Suspect").sum()
            st.warning(
                f"**{nb_fraudes}** transaction(s) frauduleuse(s) et "
                f"**{nb_suspects}** transaction(s) suspecte(s) détectée(s) sur {len(df_result)}."
            )

            def highlight(row):
                if row["prediction"] == "Fraude":
                    return ["background-color: #ffcccc"] * len(row)
                elif row["prediction"] == "Suspect":
                    return ["background-color: #fff3cd"] * len(row)
                return [""] * len(row)

            st.dataframe(df_result.style.apply(highlight, axis=1), use_container_width=True)

            csv_export = df_result.to_csv(index=False, sep=";").encode("utf-8")
            st.download_button(
                "Télécharger les résultats", csv_export, "resultats_analyse.csv", "text/csv"
            )

# ---------------------------------------------------------
# Pied de page
# ---------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption("Projet ONEF / DIT — Détection de fraude bancaire par IA")
