import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import pickle

# Modèles avec information sur les colonnes numériques et catégoriques, ainsi que les colonnes d'anomalies
models_info = {
        '7002': {
        'type': 'pickle',
        'model': './modèles/7002.pkl',
        'numeric_cols': ['7002Taux 2'],
        'categorical_cols': [],
        'anomaly_cols': ['7002Taux 2'],
        'target_col': '7002 Fraud'
    },
    '7030': {
        'type': 'joblib',
        'model': './modèles/7030.pkl',
        'numeric_cols': ['Absences par Jour', 'Absences par Heure', 'PLAFOND CUM', 'ASSIETTE CU', 'ALL FAM CUM', '7030Base', '7030Taux 2', '7030Montant Pat.'],
        'categorical_cols': ['Catégorie salariés', 'Statut de salariés'],
        'anomaly_cols': ['7030Taux 2'],
        'target_col': '7030 Fraud'
    },
    '7025': {
        'type': 'joblib',
        'model': './modèles/7025.pkl',
        'numeric_cols': ['Matricule', 'Absences par Jour', 'Absences par Heure', 'PLAFOND CUM', 'ASSIETTE CU', 'ALL FAM CUM', '7025Base', '7025Taux 2', '7025Montant Pat.'],
        'categorical_cols': [],
        'anomaly_cols': ['7025Taux 2'],
        'target_col': '7025 Fraud'
    },
    '6080': {
        'type': 'joblib',
        'model': './modèles/6080.pkl',
        'numeric_cols': ['Absences par Heure', '6080Base', '6080Taux', '6080Montant Sal.'],
        'categorical_cols': ['Statut de salariés'],
        'anomaly_cols': ['6080Taux'],
        'target_col': '6080 Fraud'
    },
    '6084': {
        'type': 'joblib',
        'model': './modèles/6084.pkl',
        'numeric_cols': ['Absences par Heure', '6084Base', '6084Taux', '6084Montant Sal.'],
        'categorical_cols': ['Statut de salariés'],
        'anomaly_cols': ['6084Taux'],
        'target_col': '6084 Fraud'
    },
    '7010': {
        'type': 'pickle',
        'model': './modèles/7010.pkl',
        'numeric_cols': ['7010Taux', '7010Montant Sal.', '7010Taux 2', '7010Montant Pat.', '7010Base'],
        'categorical_cols': [],
        'anomaly_cols': ['7010Taux', '7010Taux 2'],
        'target_col': '7010 Fraud'
    },
    '7020': {
        'type': 'joblib',
        'model': './modèles/7020.pkl',
        'numeric_cols': ['Effectif', 'ASSIETTE CU', 'PLAFOND CUM', '7020Taux 2', '7020Montant Pat.', 'Absences par Jour'],
        'categorical_cols': ['Catégorie salariés', 'Statut de salariés'],
        'anomaly_cols': ['7020Taux 2'],
        'target_col': '7020 Fraud'
    },
    #'7050': {
    #    'type': 'joblib',
    #    'model': './modèles/7050.pkl',
    #    'numeric_cols': ['7050Base', '7050Taux 2', '7050Montant Pat.'],
    #    'categorical_cols': [],
    #    'anomaly_cols': ['7050Taux 2'],
    #    'target_col': '7050 Fraud'
    #},
    '7001': {
            'type' : 'joblib',
            'model': './modèles/7001.pkl',
            'numeric_cols': ['Matricule', 'Absences par Jour', 'Absences par Heure', 'PLAFOND CUM', 'ASSIETTE CU', 'MALADIE CUM', '7001Base', '7001Taux 2', '7001Montant Pat.'],
            'categorical_cols': ['Catégorie salariés', 'Statut de salariés'],
            'target_col': '7001 Fraud'
        },
    '7035': {
        'type': 'joblib',
        'model': './modèles/7035.pkl',
        'numeric_cols': ['7035Taux 2', '7035Montant Pat.', '7035Base'],
        'categorical_cols': ['Catégorie salariés', 'Statut de salariés'],
        'anomaly_cols': ['7035Taux 2'],
        'target_col': '7035 Fraud'
    },
    '7040': {
        'type': 'joblib',
        'model': './modèles/7040.pkl',
        'numeric_cols': ['7040Taux 2', '7040Base', '7040Montant Pat.'],
        'categorical_cols': [],
        'anomaly_cols': ['7040Taux 2'],
        'target_col': '7040 Fraud'
    }
}
def process_7010(df, model_name, info, anomalies_report, model_anomalies):
    model_dict = load_model(info)

    # Charger les modèles et scalers
    iso_forest_1 = model_dict.get('iso_forest_1')
    iso_forest_2 = model_dict.get('iso_forest_2')
    lof_1 = model_dict.get('lof_1')
    lof_2 = model_dict.get('lof_2')
    scaler_1 = model_dict.get('scaler_1')
    scaler_2 = model_dict.get('scaler_2')

    if not all([iso_forest_1, iso_forest_2, lof_1, lof_2, scaler_1, scaler_2]):
        st.warning(f"Modèles ou scalers manquants pour {model_name}")
        return

    # Colonnes à traiter
    colonne1 = '7010Taux'
    colonne2 = '7010Taux 2'

    # Créer des indicateurs pour les NaN dans chaque colonne
    df['is_nan_1'] = df[colonne1].isna().astype(int)
    df['is_nan_2'] = df[colonne2].isna().astype(int)

    # Filtrer les lignes sans NaN dans les deux colonnes
    df_non_nan_1 = df[df[colonne1].notna()]
    df_non_nan_2 = df[df[colonne2].notna()]

    # Standardiser les données non-NaN en utilisant les scalers sauvegardés
    X_scaled_test_1 = scaler_1.transform(df_non_nan_1[[colonne1, 'is_nan_1']])
    X_scaled_test_2 = scaler_2.transform(df_non_nan_2[[colonne2, 'is_nan_2']])

    # Appliquer Isolation Forest pour détecter les anomalies
    df_non_nan_1['Anomaly_IF_1'] = iso_forest_1.predict(X_scaled_test_1)
    df_non_nan_1['Anomaly_IF_1'] = df_non_nan_1['Anomaly_IF_1'].map({1: 0, -1: 1})  # -1 = anomalie, 1 = normal

    df_non_nan_2['Anomaly_IF_2'] = iso_forest_2.predict(X_scaled_test_2)
    df_non_nan_2['Anomaly_IF_2'] = df_non_nan_2['Anomaly_IF_2'].map({1: 0, -1: 1})  # -1 = anomalie, 1 = normal

    # Appliquer LOF pour détecter les anomalies
    df_non_nan_1['Anomaly_LOF_1'] = lof_1.fit_predict(X_scaled_test_1)
    df_non_nan_1['Anomaly_LOF_1'] = df_non_nan_1['Anomaly_LOF_1'].map({1: 0, -1: 1})  # -1 = anomalie, 1 = normal

    df_non_nan_2['Anomaly_LOF_2'] = lof_2.fit_predict(X_scaled_test_2)
    df_non_nan_2['Anomaly_LOF_2'] = df_non_nan_2['Anomaly_LOF_2'].map({1: 0, -1: 1})  # -1 = anomalie, 1 = normal

    # Combiner les résultats pour chaque colonne : une anomalie est détectée si l'un ou l'autre modèle la marque comme telle
    df_non_nan_1['Combined_Anomaly_1'] = np.where(
        (df_non_nan_1['Anomaly_IF_1'] == 1) | (df_non_nan_1['Anomaly_LOF_1'] == 1), 1, 0
    )

    df_non_nan_2['Combined_Anomaly_2'] = np.where(
        (df_non_nan_2['Anomaly_IF_2'] == 1) | (df_non_nan_2['Anomaly_LOF_2'] == 1), 1, 0
    )

    # Réintégrer les résultats d'anomalies dans le DataFrame original, en laissant les NaN intacts
    df['Anomaly_IF_1'] = np.nan
    df['Anomaly_LOF_1'] = np.nan
    df['Combined_Anomaly_1'] = np.nan

    df['Anomaly_IF_2'] = np.nan
    df['Anomaly_LOF_2'] = np.nan
    df['Combined_Anomaly_2'] = np.nan

    df.loc[df_non_nan_1.index, 'Anomaly_IF_1'] = df_non_nan_1['Anomaly_IF_1']
    df.loc[df_non_nan_1.index, 'Anomaly_LOF_1'] = df_non_nan_1['Anomaly_LOF_1']
    df.loc[df_non_nan_1.index, 'Combined_Anomaly_1'] = df_non_nan_1['Combined_Anomaly_1']

    df.loc[df_non_nan_2.index, 'Anomaly_IF_2'] = df_non_nan_2['Anomaly_IF_2']
    df.loc[df_non_nan_2.index, 'Anomaly_LOF_2'] = df_non_nan_2['Anomaly_LOF_2']
    df.loc[df_non_nan_2.index, 'Combined_Anomaly_2'] = df_non_nan_2['Combined_Anomaly_2']

    # Ajouter les anomalies dans le rapport
    for index in df_non_nan_1.index:
        if df_non_nan_1.loc[index, 'Combined_Anomaly_1'] == 1:
            anomalies_report.setdefault(index, set()).add(model_name)
            model_anomalies[model_name] = model_anomalies.get(model_name, 0) + 1

    for index in df_non_nan_2.index:
        if df_non_nan_2.loc[index, 'Combined_Anomaly_2'] == 1:
            anomalies_report.setdefault(index, set()).add(model_name)
            model_anomalies[model_name] = model_anomalies.get(model_name, 0) + 1
# Fonction pour charger les modèles IF et LOF
def load_model(model_info):
    model_path = model_info['model']
    try:
        model_dict = joblib.load(model_path)
        if isinstance(model_dict, dict):
            return model_dict
        else:
            raise ValueError("Le modèle chargé n'est pas un dictionnaire")
    except Exception as e:
        raise ValueError(f"Erreur lors du chargement du modèle joblib : {str(e)}")

# Fonction pour appliquer IF et LOF sur les colonnes contenant des anomalies
def process_model(df, model_name, info, anomalies_report, model_anomalies):
    df_filtered = df.copy()

    if df_filtered.empty:
        st.write(f"Aucune donnée à traiter pour le modèle {model_name}.")
        return

    required_columns = info['numeric_cols'] + info['categorical_cols']
    missing_columns = [col for col in required_columns if col not in df_filtered.columns]

    # Créer des colonnes manquantes avec des NaN
    for col in missing_columns:
        df_filtered[col] = np.nan

    # Charger le modèle (IF et LOF)
    try:
        model_dict = load_model(info)
        iso_forest = model_dict.get('iso_forest', None)
        lof = model_dict.get('lof', None)
        scaler = model_dict.get('scaler', None)

        # Gérer l'absence des modèles
        if iso_forest is None and lof is None:
            st.warning(f"Aucun modèle 'iso_forest' ou 'lof' n'est disponible pour le modèle {model_name}.")
            return  # Passer à l'étape suivante si aucun modèle n'est disponible

        if scaler is None:
            st.error(f"Erreur : 'scaler' manquant dans le modèle {model_name}.")
            return
    except ValueError as e:
        st.error(str(e))
        return

    # Utiliser les colonnes d'anomalies si elles existent, sinon utiliser une seule colonne
    anomaly_cols = info.get('anomaly_cols', [info['numeric_cols'][0]])

    for anomaly_col in anomaly_cols:
        # Vérifier si la colonne d'anomalie existe dans le DataFrame
        if anomaly_col not in df_filtered.columns:
            st.write(f"La colonne {anomaly_col} n'existe pas dans le DataFrame pour le modèle {model_name}.")
            continue

        # Créer un indicateur pour les NaN
        df_filtered['is_nan'] = df_filtered[anomaly_col].isna().astype(int)

        # Filtrer les lignes sans NaN
        df_filtered_non_nan = df_filtered[df_filtered[anomaly_col].notna()]

        # Vérifier les colonnes que le scaler a vues lors de l'entraînement
        scaler_features = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else []
        missing_features = [col for col in scaler_features if col not in df_filtered_non_nan.columns]

        # Ajouter les colonnes manquantes avec des valeurs par défaut (par exemple, 0 ou NaN)
        for feature in missing_features:
            df_filtered_non_nan[feature] = 0  # ou np.nan selon ce qui est approprié

        # Vérifier si 'is_nan' doit être inclus dans la transformation
        features_to_scale = [anomaly_col]
        if 'is_nan' in scaler_features:
            features_to_scale.append('is_nan')

        # Appliquer la transformation sur X_scaled_test
        try:
            X_scaled_test = scaler.transform(df_filtered_non_nan[features_to_scale])
        except ValueError as e:
            st.error(f"Erreur de transformation avec scaler pour {model_name}: {str(e)}")
            continue

        # Appliquer Isolation Forest pour détecter les anomalies si iso_forest est présent
        if iso_forest is not None:
            df_filtered_non_nan[f'Anomaly_IF_{anomaly_col}'] = iso_forest.predict(X_scaled_test)
            df_filtered_non_nan[f'Anomaly_IF_{anomaly_col}'] = df_filtered_non_nan[f'Anomaly_IF_{anomaly_col}'].map({1: 0, -1: 1})
        else:
            st.write(f"iso_forest est manquant pour {model_name}, anomalie IF non calculée.")

        # Appliquer LOF pour détecter les anomalies si lof est présent
        if lof is not None:
            df_filtered_non_nan[f'Anomaly_LOF_{anomaly_col}'] = lof.fit_predict(X_scaled_test)
            df_filtered_non_nan[f'Anomaly_LOF_{anomaly_col}'] = df_filtered_non_nan[f'Anomaly_LOF_{anomaly_col}'].map({1: 0, -1: 1})
        else:
            st.write(f"lof est manquant pour {model_name}, anomalie LOF non calculée.")

        # Combiner les résultats des deux modèles (uniquement si l'un des modèles a produit un résultat)
        if f'Anomaly_IF_{anomaly_col}' in df_filtered_non_nan.columns or f'Anomaly_LOF_{anomaly_col}' in df_filtered_non_nan.columns:
            df_filtered_non_nan[f'Combined_Anomaly_{anomaly_col}'] = np.where(
                (df_filtered_non_nan.get(f'Anomaly_IF_{anomaly_col}', 0) == 1) | 
                (df_filtered_non_nan.get(f'Anomaly_LOF_{anomaly_col}', 0) == 1), 1, 0
            )

            for index in df_filtered_non_nan.index:
                if df_filtered_non_nan.loc[index, f'Combined_Anomaly_{anomaly_col}'] == 1:
                    anomalies_report.setdefault(index, set()).add(model_name)
                    model_anomalies[model_name] = model_anomalies.get(model_name, 0) + 1

            # Réintégrer les résultats dans le DataFrame original
            df.loc[df_filtered_non_nan.index, f'Combined_Anomaly_{anomaly_col}'] = df_filtered_non_nan[f'Combined_Anomaly_{anomaly_col}']


def process_7001(df, model_name, info, anomalies_report, model_anomalies):
    # Charger les modèles et scalers
    model_dict = load_model(info)
    iso_forest = model_dict.get('iso_forest', None)
    scaler = model_dict.get('scaler', None)

    if iso_forest is None or scaler is None:
        st.warning(f"Modèles ou scaler manquants pour {model_name}")
        return

    # Colonne à traiter : '7001Taux 2'
    colonne = '7001Taux 2'

    # Créer une colonne 'is_nan' pour signaler les NaN, mais NE PAS remplacer les NaN par 0
    df['is_nan'] = df[colonne].isna().astype(int)

    # Filtrer les lignes où la colonne '7001Taux 2' n'est pas NaN (ignorer les lignes avec NaN)
    df_non_nan = df[df[colonne].notna()].copy()

    if df_non_nan.empty:
        st.write(f"Aucune donnée valide à traiter pour le modèle {model_name}.")
        return

    # Vérifier les colonnes que le scaler a vues lors de l'entraînement
    scaler_features = scaler.feature_names_in_

    # Ajouter les colonnes manquantes dans les données test si elles sont présentes dans le scaler
    for feature in scaler_features:
        if feature not in df_non_nan.columns:
            df_non_nan[feature] = 0  # ou np.nan selon ce qui est approprié

    # Filtrer les colonnes de df_non_nan pour correspondre exactement aux colonnes vues par le scaler
    df_scaled = df_non_nan[scaler_features]

    try:
        # Transformer les données avec le scaler
        X_scaled_test = scaler.transform(df_scaled)

        # Appliquer Isolation Forest pour prédire les anomalies
        df_non_nan['Anomaly_IF'] = iso_forest.predict(X_scaled_test)
        df_non_nan['Anomaly_IF'] = df_non_nan['Anomaly_IF'].map({1: 0, -1: 1})  # -1 = anomalie, 1 = normal

        # Réintégrer les résultats dans le DataFrame original
        df['Anomaly_IF'] = np.nan  # Initialiser avec NaN
        df.loc[df_non_nan.index, 'Anomaly_IF'] = df_non_nan['Anomaly_IF']

        # Ajouter les anomalies dans le rapport
        for index in df_non_nan.index:
            if df_non_nan.loc[index, 'Anomaly_IF'] == 1:
                anomalies_report.setdefault(index, set()).add(model_name)
                model_anomalies[model_name] = model_anomalies.get(model_name, 0) + 1

    except ValueError as e:
        st.error(f"Erreur lors de la transformation des données avec le scaler : {e}")

def process_7002(df, model_name, info, anomalies_report, model_anomalies):
    # Charger les modèles et scalers
    model_dict = load_model(info)
    iso_forest = model_dict.get('iso_forest', None)
    lof = model_dict.get('lof', None)
    scaler = model_dict.get('scaler', None)

    if iso_forest is None or lof is None or scaler is None:
        st.warning(f"Modèles ou scaler manquants pour {model_name}")
        return

    # Colonne à traiter : '7002Taux 2'
    colonne = '7002Taux 2'

    # Créer une colonne 'is_nan' pour signaler les NaN, mais NE PAS remplacer les NaN par 0
    df['is_nan'] = df[colonne].isna().astype(int)

    # Filtrer les lignes où la colonne '7002Taux 2' n'est pas NaN (ignorer les lignes avec NaN)
    df_non_nan = df[df[colonne].notna()].copy()

    if df_non_nan.empty:
        st.write(f"Aucune donnée valide à traiter pour le modèle {model_name}.")
        return

    # Vérifier les colonnes que le scaler a vues lors de l'entraînement
    scaler_features = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else []

    # Construire les colonnes à passer au scaler
    features_to_scale = [colonne]  # Commencer avec la colonne de données
    if 'is_nan' in scaler_features:
        features_to_scale.append('is_nan')

    # Standardiser les données non-NaN en utilisant le scaler sauvegardé
    try:
        X_scaled_test = scaler.transform(df_non_nan[features_to_scale])

        # Appliquer Isolation Forest pour détecter les anomalies
        df_non_nan['Anomaly_IF'] = iso_forest.predict(X_scaled_test)
        df_non_nan['Anomaly_IF'] = df_non_nan['Anomaly_IF'].map({1: 0, -1: 1})  # -1 = anomalie, 1 = normal

        # Appliquer LOF pour détecter les anomalies
        df_non_nan['Anomaly_LOF'] = lof.fit_predict(X_scaled_test)
        df_non_nan['Anomaly_LOF'] = df_non_nan['Anomaly_LOF'].map({1: 0, -1: 1})  # -1 = anomalie, 1 = normal

        # Combiner les résultats : une anomalie est détectée si l'un ou l'autre modèle la marque comme telle
        df_non_nan['Combined_Anomaly'] = np.where(
            (df_non_nan['Anomaly_IF'] == 1) | (df_non_nan['Anomaly_LOF'] == 1), 1, 0
        )

        # Réintégrer les résultats d'anomalies dans le DataFrame original, en laissant les NaN intacts
        df['Anomaly_IF'] = np.nan
        df['Anomaly_LOF'] = np.nan
        df['Combined_Anomaly'] = np.nan

        df.loc[df_non_nan.index, 'Anomaly_IF'] = df_non_nan['Anomaly_IF']
        df.loc[df_non_nan.index, 'Anomaly_LOF'] = df_non_nan['Anomaly_LOF']
        df.loc[df_non_nan.index, 'Combined_Anomaly'] = df_non_nan['Combined_Anomaly']

        # Ajouter les anomalies dans le rapport
        for index in df_non_nan.index:
            if df_non_nan.loc[index, 'Combined_Anomaly'] == 1:
                anomalies_report.setdefault(index, set()).add(model_name)
                model_anomalies[model_name] = model_anomalies.get(model_name, 0) + 1

    except ValueError as e:
        st.error(f"Erreur lors de la transformation des données avec le scaler : {e}")



def detect_anomalies(df):
    anomalies_report = {}
    model_anomalies = {}

    for model_name, info in models_info.items():
        if model_name == '7010':
            process_7010(df, model_name, info, anomalies_report, model_anomalies)
        elif model_name == '7001':
            process_7001(df, model_name, info, anomalies_report, model_anomalies)
        elif model_name == '7002':  # Ajout de 7002
            process_7002(df, model_name, info, anomalies_report, model_anomalies)
        else:
            process_model(df, model_name, info, anomalies_report, model_anomalies)

    st.write("**Rapport d'anomalies détectées :**")
    total_anomalies = len(anomalies_report)
    st.write(f"**Total des lignes avec des anomalies :** {total_anomalies}")

    report_content = []
    report_content.append("Rapport d'anomalies détectées :\n\n")
    report_content.append(f"Total des lignes avec des anomalies : {total_anomalies}\n")
    for model_name, count in model_anomalies.items():
        report_content.append(f"Un nombre de {int(count)} anomalies a été détecté pour la cotisation {model_name}.\n")

    report_content.append("\n")

    for line_index, models in anomalies_report.items():
        matricule = df.loc[line_index, 'Matricule'] if 'Matricule' in df.columns else 'Inconnu'
        report_content.append(f"Matricule {matricule} : anomalie dans les cotisations {', '.join(sorted(models))}\n")

    return "\n".join(report_content)


# Exemple d'application avec un fichier CSV uploadé
csv_upload = st.file_uploader("Entrez votre bulletin de paie (Format csv)", type=['csv'])

if csv_upload:
    try:
        df = pd.read_csv(csv_upload, encoding='utf-8', on_bad_lines='skip')
    except pd.errors.EmptyDataError:
        st.write("Le fichier CSV est vide ou mal formaté.")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_upload, encoding='ISO-8859-1', on_bad_lines='skip')
        except Exception as e:
            st.write(f"Erreur lors de la lecture du fichier CSV : {e}")
    except Exception as e:
        st.write(f"Erreur inattendue lors de la lecture du fichier CSV : {e}")
    else:
        df.columns = df.columns.str.strip()  # Nettoyer les colonnes
        report_content = detect_anomalies(df)
        st.write(report_content)