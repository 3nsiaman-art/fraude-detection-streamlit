import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from streamlit_option_menu import option_menu

# ---------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------
st.markdown("""
<div style="
    background: linear-gradient(90deg, #0F172A, #1E3A8A);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 20px;
">

<h1 style="
    color: white;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 8px;">
🏦 Application de détection de Fraude Bancaire
</h1>

<p style="
    color: #E2E8F0;
    font-size: 18px;
    margin:0;">
Détection intelligente des transactions frauduleuses grâce au Machine Learning
</p>

</div>
""", unsafe_allow_html=True)

REFERENCE_DATA_PATH = "transactions_reference.csv"  # <- adapte ce chemin si besoin

CLASS_COLORS = {"Normal": "#1e8e3e", "Suspect": "#f2a900", "Fraude": "#d93025"}
CLASS_BG = {
    "Normal": "background-color: #e8f5e9; color: #1b5e20;",
    "Suspect": "background-color: #fff8e1; color: #8a6d00;",
    "Fraude": "background-color: #ffebee; color: #b71c1c;",
}

# ---------------------------------------------------------
# CSS personnalisé — cartes KPI, badges, pill toggle
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    #MainMenu, header, footer {visibility: hidden;}

    .kpi-card {
        border-radius: 16px;
        padding: 20px 22px;
        background: #ffffff;
        border: 1px solid #eef0f3;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        position: relative;
        overflow: hidden;
    }
    .kpi-card.blue   { background: linear-gradient(135deg, #ffffff 55%, #e8f0fe 100%); }
    .kpi-card.red    { background: linear-gradient(135deg, #ffffff 55%, #fce8e6 100%); }
    .kpi-card.purple { background: linear-gradient(135deg, #ffffff 55%, #f3e8fd 100%); }

    .kpi-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        color: #6b7280;
        text-transform: uppercase;
    }
    .kpi-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #111827;
        margin: 4px 0 2px 0;
    }
    .kpi-sub {
        font-size: 0.82rem;
        color: #9aa0a6;
    }
    .kpi-icon {
        position: absolute;
        top: 16px;
        right: 16px;
        width: 34px;
        height: 34px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
    }
    .icon-blue   { background: #e8f0fe; }
    .icon-red    { background: #fce8e6; }
    .icon-purple { background: #f3e8fd; }

    .panel-title {
        font-weight: 700;
        font-size: 1.05rem;
        color: #111827;
    }
    .panel-sub {
        font-size: 0.85rem;
        color: #9aa0a6;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi_card(label, value, sub, icon, color):
    st.markdown(
        f"""
        <div class="kpi-card {color}">
            <div class="kpi-icon icon-{color}">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# Chargement des données de référence (pour Données / Modèle)
# ---------------------------------------------------------
@st.cache_data
def load_reference_data(path):
    df = pd.read_csv(path, sep=";")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


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


def build_features(df_raw, encoders, scaler, features_num, features_cat, loc_frequente):
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
        df[col] = df[col].apply(lambda v: v if v in le.classes_ else le.classes_[0])
        df[col] = le.transform(df[col])

    X = df[features_num + features_cat].copy()
    X[features_num] = scaler.transform(X[features_num])
    return X


def to_binaire(label):
    return "Fraude" if label == "Fraude" else "Non-fraude"

# ---------------------------------------------------------
# Barre de navigation supérieure
# ---------------------------------------------------------
col_nav, col_toggle = st.columns([4, 1])

with col_nav:
    selected_tab = option_menu(
        menu_title=None,
        options=["Données", "Modèle", "Analyse individuelle", "Analyse groupée"],
        icons=["database", "cpu", "search", "cloud-upload"],
        orientation="horizontal",
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "#fafafa"},
            "icon": {"font-size": "16px"},
            "nav-link": {"font-size": "15px", "font-weight": "600", "text-align": "left"},
            "nav-link-selected": {"background-color": "#111827", "color": "white"},
        },
    )

with col_toggle:
    mode_classes = option_menu(
        menu_title=None,
        options=["Binaire", "3 classes"],
        orientation="horizontal",
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "#f0f1f3", "border-radius": "10px"},
            "nav-link": {"font-size": "13px", "font-weight": "600"},
            "nav-link-selected": {"background-color": "white", "color": "#111827"},
        },
    )

st.write("")

#  ===========================================================
# ONGLET : DONNÉES
# ===========================================================
if selected_tab == "Données":
    try:
        df_ref = load_reference_data(REFERENCE_DATA_PATH)
    except FileNotFoundError:
        st.warning(
            f"Fichier de référence introuvable à `{REFERENCE_DATA_PATH}`. "
            "Dépose-le ici pour afficher le dashboard :"
        )
        uploaded = st.file_uploader("Base de données de référence (CSV, séparateur ';')", type=["csv"])
        if uploaded is None:
            st.stop()
        df_ref = pd.read_csv(uploaded, sep=";")
        df_ref["Date"] = pd.to_datetime(df_ref["Date"])

    total = len(df_ref)
    nb_fraudes = (df_ref["Target"] == "Fraude").sum()
    pct_fraudes = nb_fraudes / total * 100 if total else 0
    nb_clients = df_ref["ID Clients"].nunique()
    nb_villes = df_ref["Localisation"].nunique()
    montant_median = df_ref["Montant"].median()
    date_min = df_ref["Date"].min().strftime("%d/%m/%Y")
    date_max = df_ref["Date"].max().strftime("%d/%m/%Y")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Transactions", f"{total:,}".replace(",", " "), f"{date_min} → {date_max}", "📊", "blue")
    with c2:
        kpi_card("Fraudes", f"{nb_fraudes:,}".replace(",", " "), f"{pct_fraudes:.1f} % du total", "⚠️", "red")
    with c3:
        kpi_card("Clients", f"{nb_clients}", f"{nb_villes} villes couvertes", "👥", "purple")
    with c4:
        kpi_card("Montant médian", f"{montant_median:,.0f}".replace(",", " "), "FCFA par transaction", "🏦", "blue")

    st.write("")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="panel-title">📊 Répartition des transactions</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-sub">Étiquette attribuée à chaque opération</div>', unsafe_allow_html=True)

        repartition = df_ref["Target"].value_counts().reindex(["Normal", "Suspect", "Fraude"]).fillna(0).reset_index()
        repartition.columns = ["Classe", "Nombre"]
        fig = px.bar(
            repartition, x="Classe", y="Nombre", text="Nombre",
            color="Classe", color_discrete_map=CLASS_COLORS,
        )
        fig.update_traces(textposition="outside")
        fig.update_xaxes(type="category", tickmode="array",
                          tickvals=repartition["Classe"], ticktext=repartition["Classe"])
        fig.update_layout(showlegend=False, height=400, margin=dict(t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="panel-title">📍 Top 10 des localisations</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-sub">Nombre de transactions par ville</div>', unsafe_allow_html=True)

        top10 = df_ref["Localisation"].value_counts().head(10).sort_values().reset_index()
        top10.columns = ["Ville", "Nombre"]
        fig2 = px.bar(top10, x="Nombre", y="Ville", orientation="h", text="Nombre")
        fig2.update_traces(marker_color="#1a73e8", textposition="outside")
        fig2.update_layout(height=400, margin=dict(t=20, b=0))
        st.plotly_chart(fig2, use_container_width=True)

# ===========================================================
# ONGLET : MODÈLE
# ===========================================================
elif selected_tab == "Modèle":
    st.markdown('<div class="panel-title">🧠 Performance du modèle</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-sub">Métriques calculées en appliquant le modèle déployé sur la base de référence</div>',
        unsafe_allow_html=True,
    )

    try:
        model, scaler, encoders, target_encoder, metadata = load_artifacts()
        df_ref = load_reference_data(REFERENCE_DATA_PATH)
    except FileNotFoundError as e:
        st.error(f"Fichier manquant : {e}. Vérifie que `model/` et `{REFERENCE_DATA_PATH}` existent dans le repo.")
        st.stop()

    features_num = metadata["features_num"]
    features_cat = metadata["features_cat"]
    loc_frequente = metadata["localisation_frequente"]

    X = build_features(df_ref, encoders, scaler, features_num, features_cat, loc_frequente)
    y_true_labels = df_ref["Target"].values
    y_pred_encoded = model.predict(X)
    y_pred_labels = target_encoder.inverse_transform(y_pred_encoded)

    if mode_classes == "Binaire":
        y_true = [to_binaire(v) for v in y_true_labels]
        y_pred = [to_binaire(v) for v in y_pred_labels]
        labels_order = ["Non-fraude", "Fraude"]
    else:
        y_true = y_true_labels
        y_pred = y_pred_labels
        labels_order = ["Normal", "Suspect", "Fraude"]

    acc = accuracy_score(y_true, y_pred)

    c1, c2 = st.columns([1, 2])
    with c1:
        kpi_card("Accuracy", f"{acc*100:.1f} %", f"Sur {len(df_ref)} transactions", "🎯", "blue")

        report = classification_report(y_true, y_pred, labels=labels_order, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).T.loc[labels_order, ["precision", "recall", "f1-score", "support"]]
        st.caption("Détail par classe")
        st.dataframe(
            report_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}"}),
            use_container_width=True,
        )

    with c2:
        cm = confusion_matrix(y_true, y_pred, labels=labels_order)
        fig_cm = px.imshow(
            cm, x=labels_order, y=labels_order, text_auto=True,
            color_continuous_scale="Blues", labels=dict(x="Prédit", y="Réel", color="Nombre"),
        )
        fig_cm.update_layout(height=380, margin=dict(t=20, b=0))
        st.caption("Matrice de confusion")
        st.plotly_chart(fig_cm, use_container_width=True)

    st.info(
        "ℹ️ Si cette base de référence a servi à l'entraînement du modèle, ces scores "
        "peuvent être optimistes (pas de vraie généralisation). Idéalement, utilise un "
        "jeu de test séparé, non vu à l'entraînement."
    ) 
# ===========================================================
# ONGLET : ANALYSE (transaction unique)
# ===========================================================
elif selected_tab == "Analyse individuelle":
    try:
        model, scaler, encoders, target_encoder, metadata = load_artifacts()
    except FileNotFoundError as e:
        st.error(f"Fichier de modèle manquant : {e}")
        st.stop()

    features_num = metadata["features_num"]
    features_cat = metadata["features_cat"]
    loc_frequente = metadata["localisation_frequente"]

    st.markdown('<div class="panel-title">🔎 Analyse d\'une transaction</div>', unsafe_allow_html=True)
    st.write("")

    id_clients_options = sorted(loc_frequente.keys())

    col1, col2 = st.columns(2)
    with col1:
        id_client = st.selectbox("ID Client", id_clients_options)
        montant = st.number_input("Montant de la transaction (FCFA)", min_value=0.0, value=50000.0, step=1000.0)
        date_transaction = st.date_input("Date de la transaction")
        heure_transaction = st.slider("Heure de la transaction (0-23)", 0, 23, 12)
    with col2:
        type_transaction = st.selectbox("Type de transaction", list(encoders["Type de transaction"].classes_))
        status_operation = st.selectbox("Statut de l'opération", list(encoders["Status operation"].classes_))
        localisation = st.selectbox("Localisation de la transaction", list(encoders["Localisation"].classes_))

    if st.button("Analyser la transaction", type="primary"):
        with st.spinner("Analyse en cours..."):
            date_complete = pd.Timestamp.combine(date_transaction, pd.Timestamp("00:00").time()) + pd.Timedelta(
                hours=heure_transaction
            )
            transaction = pd.DataFrame([{
                "ID Clients": id_client, "Montant": montant, "Date": date_complete,
                "Type de transaction": type_transaction, "Status operation": status_operation,
                "Localisation": localisation,
            }])

            X = build_features(transaction, encoders, scaler, features_num, features_cat, loc_frequente)
            prediction = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            classe_predite = target_encoder.inverse_transform([prediction])[0]

        st.divider()

        if mode_classes == "Binaire":
            classes = list(target_encoder.classes_)
            proba_fraude = proba[classes.index("Fraude")] if "Fraude" in classes else 0.0
            proba_non_fraude = 1 - proba_fraude
            classe_bin = "Fraude" if classe_predite == "Fraude" else "Non-fraude"

            if classe_bin == "Fraude":
                st.error(f"⚠️ **Transaction frauduleuse** — probabilité : {proba_fraude:.1%}")
            else:
                st.success(f"✅ **Transaction non frauduleuse** — probabilité : {proba_non_fraude:.1%}")

            bin_df = pd.DataFrame({"Classe": ["Non-fraude", "Fraude"], "Probabilité": [proba_non_fraude, proba_fraude]})
            st.dataframe(bin_df.style.format({"Probabilité": "{:.1%}"}), hide_index=True, use_container_width=True)
        else:
            if classe_predite == "Fraude":
                st.error(f"⚠️ **Transaction frauduleuse** — probabilité : {proba[prediction]:.1%}")
            elif classe_predite == "Suspect":
                st.warning(f"🟠 **Transaction suspecte** — probabilité : {proba[prediction]:.1%}")
            else:
                st.success(f"✅ **Transaction normale** — probabilité : {proba[prediction]:.1%}")

            proba_df = pd.DataFrame(
                {"Classe": target_encoder.classes_, "Probabilité": proba}
            ).sort_values("Probabilité", ascending=False)
            st.dataframe(proba_df.style.format({"Probabilité": "{:.1%}"}), hide_index=True, use_container_width=True)

# ===========================================================
# ONGLET : IMPORT CSV (lot)
# ===========================================================
else:
    try:
        model, scaler, encoders, target_encoder, metadata = load_artifacts()
    except FileNotFoundError as e:
        st.error(f"Fichier de modèle manquant : {e}")
        st.stop()

    features_num = metadata["features_num"]
    features_cat = metadata["features_cat"]
    loc_frequente = metadata["localisation_frequente"]

    st.markdown('<div class="panel-title">📂 Analyse par lot (fichier CSV)</div>', unsafe_allow_html=True)

    with st.expander("ℹ️ Format de fichier attendu"):
        st.write(
            "CSV avec séparateur `;` et les colonnes : `ID Clients`, `Montant`, `Date`, "
            "`Type de transaction`, `Status operation`, `Localisation`."
        )

    fichier = st.file_uploader("Déposez un fichier CSV de transactions", type=["csv"])

    if fichier is not None:
        df_raw = pd.read_csv(fichier, sep=";")
        st.write("**Aperçu des données :**")
        st.dataframe(df_raw.head(), use_container_width=True)

        if st.button("Lancer l'analyse du lot", type="primary"):
            with st.spinner(f"Analyse de {len(df_raw)} transaction(s)..."):
                X = build_features(df_raw, encoders, scaler, features_num, features_cat, loc_frequente)
                predictions = model.predict(X)
                probas = model.predict_proba(X)

                df_result = df_raw.copy()
                df_result["prediction"] = target_encoder.inverse_transform(predictions)
                classes = list(target_encoder.classes_)
                if "Fraude" in classes:
                    df_result["probabilite_fraude"] = probas[:, classes.index("Fraude")]
                df_result["prediction_binaire"] = df_result["prediction"].apply(to_binaire)

            st.divider()
            total = len(df_result)

            if mode_classes == "Binaire":
                nb_fraudes = (df_result["prediction_binaire"] == "Fraude").sum()
                nb_non_fraudes = total - nb_fraudes
                c1, c2, c3 = st.columns(3)
                c1.metric("Total analysé", total)
                c2.metric("✅ Non-fraude", nb_non_fraudes)
                c3.metric("⚠️ Fraude", nb_fraudes, delta=f"{nb_fraudes/total*100:.1f}%", delta_color="inverse")

                st.bar_chart(df_result["prediction_binaire"].value_counts().reindex(["Non-fraude", "Fraude"]).fillna(0))

                def highlight_bin(row):
                    style = "background-color: #ffebee; color: #b71c1c;" if row["prediction_binaire"] == "Fraude" else ""
                    return [style] * len(row)

                st.dataframe(df_result.style.apply(highlight_bin, axis=1), use_container_width=True, height=350)
            else:
                nb_normal = (df_result["prediction"] == "Normal").sum()
                nb_suspects = (df_result["prediction"] == "Suspect").sum()
                nb_fraudes = (df_result["prediction"] == "Fraude").sum()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total analysé", total)
                c2.metric("✅ Normal", nb_normal)
                c3.metric("🟠 Suspect", nb_suspects)
                c4.metric("⚠️ Fraude", nb_fraudes, delta=f"{(nb_suspects+nb_fraudes)/total*100:.1f}% à risque", delta_color="inverse")

                st.bar_chart(df_result["prediction"].value_counts().reindex(["Normal", "Suspect", "Fraude"]).fillna(0))

                def highlight(row):
                    return [CLASS_BG.get(row["prediction"], "")] * len(row)

                st.dataframe(df_result.style.apply(highlight, axis=1), use_container_width=True, height=350)

            csv_export = df_result.to_csv(index=False, sep=";").encode("utf-8")
            st.download_button("⬇️ Télécharger les résultats", csv_export, "resultats_analyse.csv", "text/csv", type="primary")

# ---------------------------------------------------------
# Pied de page
# ---------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption("Projet DIT — Détection de fraude bancaire par l'IA")
# ---------------------------------------------------------
# Bandeau de contexte
# ---------------------------------------------------------
st.markdown(
    """
    <div class="context-banner" style="text-align: justify;">
        🏦 <b><i>Application développée par N'faly SIAMAN</i></b>,
        pour la détection automatisée de fraude bancaire par apprentissage automatique.
    </div>
    """,
    unsafe_allow_html=True,
)

 
