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
# Style CSS personnalisé (cartes métriques, boutons, tableaux)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Cartes de métriques */
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 14px 18px;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        color: #555;
    }
    /* Bouton principal */
    div.stButton > button[kind="primary"] {
        border-radius: 8px;
        font-weight: 600;
    }
    /* Onglets */
    button[data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 600;
    }
    /* Titre */
    h1 {
        padding-bottom: 0px;
    }
    </style>
    """,
    unsafe_allow_html=True,
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

# Couleurs cohérentes utilisées dans toute l'application
CLASS_COLORS = {
    "Normal": "#2e7d32",
    "Suspect": "#e6a700",
    "Fraude": "#c62828",
}
CLASS_BG = {
    "Normal": "background-color: #e8f5e9; color: #1b5e20;",
    "Suspect": "background-color: #fff8e1; color: #8a6d00;",
    "Fraude": "background-color: #ffebee; color: #b71c1c;",
}

# ---------------------------------------------------------
# En-tête
# ---------------------------------------------------------
st.title("🏦 Système de Détection de Fraude Bancaire")
st.markdown(
    "Analysez une transaction ou un lot de transactions pour détecter un "
    "risque de fraude (classes : **Normal**, **Suspect**, **Fraude**)."
)
st.divider()


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


def render_class_badge(classe: str, proba: float):
    """Affiche un bandeau de résultat cohérent avec le code couleur global."""
    if classe == "Fraude":
        st.error(f"⚠️ **Transaction frauduleuse** — probabilité : {proba:.1%}")
    elif classe == "Suspect":
        st.warning(f"🟠 **Transaction suspecte** — probabilité : {proba:.1%}")
    else:
        st.success(f"✅ **Transaction normale** — probabilité : {proba:.1%}")


# ===========================================================
# NAVIGATION PAR ONGLETS
# ===========================================================
tab_unique, tab_lot = st.tabs(["🔎 Transaction unique", "📂 Fichier CSV (lot)"])

# ===========================================================
# ONGLET 1 : Transaction unique
# ===========================================================
with tab_unique:
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
        with st.spinner("Analyse de la transaction en cours..."):
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
        render_class_badge(classe_predite, proba[prediction])

        proba_df = pd.DataFrame(
            {"Classe": target_encoder.classes_, "Probabilité": proba}
        ).sort_values("Probabilité", ascending=False)

        col_table, col_chart = st.columns([1, 1.4])
        with col_table:
            st.dataframe(
                proba_df.style.format({"Probabilité": "{:.1%}"}),
                hide_index=True,
                use_container_width=True,
            )
        with col_chart:
            st.bar_chart(
                proba_df.set_index("Classe"),
                color=[CLASS_COLORS.get(c, "#888888") for c in proba_df["Classe"]][:1] or None,
            )

# ===========================================================
# ONGLET 2 : Fichier CSV (lot)
# ===========================================================
with tab_lot:
    st.subheader("Analyse par lot (fichier CSV)")

    with st.expander("ℹ️ Format de fichier attendu", expanded=False):
        st.write(
            "Le fichier doit être un CSV avec séparateur `;` et contenir les mêmes "
            "colonnes que la base d'entraînement : `ID Clients`, `Montant`, `Date`, "
            "`Type de transaction`, `Status operation`, `Localisation`."
        )

    fichier = st.file_uploader("Déposez un fichier CSV de transactions", type=["csv"])

    if fichier is not None:
        df_raw = pd.read_csv(fichier, sep=";")
        st.write("**Aperçu des données :**")
        st.dataframe(df_raw.head(), use_container_width=True)

        if st.button("Lancer l'analyse du lot", type="primary"):
            with st.spinner(f"Analyse de {len(df_raw)} transaction(s) en cours..."):
                X = build_features(df_raw)
                predictions = model.predict(X)
                probas = model.predict_proba(X)

                df_result = df_raw.copy()
                df_result["prediction"] = target_encoder.inverse_transform(predictions)
                df_result["probabilite_fraude"] = probas[:, list(target_encoder.classes_).index("Fraude")]

            st.divider()

            # --- Cartes de synthèse ---
            total = len(df_result)
            nb_normal = (df_result["prediction"] == "Normal").sum()
            nb_suspects = (df_result["prediction"] == "Suspect").sum()
            nb_fraudes = (df_result["prediction"] == "Fraude").sum()
            taux_risque = (nb_suspects + nb_fraudes) / total * 100 if total else 0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total analysé", f"{total}")
            c2.metric("✅ Normal", f"{nb_normal}")
            c3.metric("🟠 Suspect", f"{nb_suspects}")
            c4.metric("⚠️ Fraude", f"{nb_fraudes}", delta=f"{taux_risque:.1f}% à risque", delta_color="inverse")

            st.write("")

            col_dist, col_table = st.columns([1, 2])

            with col_dist:
                st.caption("Répartition par classe")
                repartition = df_result["prediction"].value_counts().reindex(
                    ["Normal", "Suspect", "Fraude"]
                ).fillna(0)
                st.bar_chart(repartition)

            with col_table:
                st.caption("Détail des transactions")

                def highlight(row):
                    style = CLASS_BG.get(row["prediction"], "")
                    return [style] * len(row)

                st.dataframe(
                    df_result.style.apply(highlight, axis=1).format(
                        {"probabilite_fraude": "{:.1%}"}
                    ),
                    use_container_width=True,
                    height=350,
                )

            csv_export = df_result.to_csv(index=False, sep=";").encode("utf-8")
            st.download_button(
                "⬇️ Télécharger les résultats",
                csv_export,
                "resultats_analyse.csv",
                "text/csv",
                type="primary",
            )

# ---------------------------------------------------------
# Pied de page
# ---------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption("Projet DIT — Détection de fraude bancaire par IA")
