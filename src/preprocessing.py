"""Nettoyage et ingenierie de variables

"""
import numpy as np
import pandas as pd


COLS_FUITE = ["nb_jours_arret_12_mois", "intervention_prevue",
              "cout_reparation_estime_fcfa"]


SENTINELLES = {
    "latitude": [0.0], "longitude": [0.0],          # 214 lignes : au large du golfe
    "annee_construction": [0],                       # 168 : donnerait un age de 2025 ans
    "population_desservie": [0],                     # 95
    "debit_essai_m3_h": [-1],                        # 64
}

COLS_CAT = ["departement", "commune", "type_ouvrage", "type_pompe", "qualite_eau",
            "mode_gestion", "mode_paiement", "maitre_ouvrage", "installateur"]

SYNONYMES = {"type_pompe": {"India mark ii": "India mark 2"}}

SEUIL_PROFONDEUR_CM = 200
CM_PAR_METRE = 100

ANNEE_RELEVE = 2025


NORME_PERSONNES_PAR_POINT = 300
SEUIL_MAINTENANCE_RECENTE = 18


def normaliser_texte(serie: pd.Series) -> pd.Series:

    return (serie.astype(str)
                 .str.strip()
                 .str.replace(r"\s+", " ", regex=True)
                 .str.normalize("NFKD")
                 .str.encode("ascii", "ignore").str.decode("utf-8")
                 .str.capitalize())


def nettoyer(df: pd.DataFrame, garder_fuite: bool = False) -> pd.DataFrame:

    df = df.copy()


    for col in COLS_CAT:
        if col in df.columns:
            df[col] = normaliser_texte(df[col])
            if col in SYNONYMES:
                df[col] = df[col].replace(SYNONYMES[col])


    if "date_releve" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date_releve"]):
        df["date_releve"] = pd.to_datetime(df["date_releve"], format="mixed", dayfirst=True)


    df = df.drop_duplicates().reset_index(drop=True)


    for col, valeurs in SENTINELLES.items():
        if col in df.columns:
            df[col] = df[col].replace(valeurs, np.nan)


    if "profondeur_forage_m" in df.columns:
        en_cm = df["profondeur_forage_m"] > SEUIL_PROFONDEUR_CM
        df.loc[en_cm, "profondeur_forage_m"] /= CM_PAR_METRE

    if {"niveau_statique_m", "profondeur_forage_m"} <= set(df.columns):
        incoherent = df["niveau_statique_m"] > df["profondeur_forage_m"]
        df.loc[incoherent, "niveau_statique_m"] = np.nan

    if garder_fuite:
        return df
    return df.drop(columns=[c for c in COLS_FUITE if c in df.columns])


def construire_capacite_entretien(df: pd.DataFrame) -> pd.Series:

    maintenance_recente = (df["mois_depuis_derniere_maintenance"]
                           <= SEUIL_MAINTENANCE_RECENTE).astype(int)
    cotisation_percue = (df["cotisation_mensuelle_fcfa"] > 0).astype(int)
    gestion_structuree = (~df["mode_gestion"].isin(["Aucune gestion formelle"])).astype(int)
    paiement_effectif = (df["mode_paiement"] != "Gratuit").astype(int)

    return (maintenance_recente + cotisation_percue
            + gestion_structuree + paiement_effectif)


def ajouter_variables_derivees(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()


    df["age_ans"] = ANNEE_RELEVE - df["annee_construction"]

    df["pression_usage"] = (df["population_desservie"]
                            / (NORME_PERSONNES_PAR_POINT
                               * df["nb_points_eau_village"].clip(lower=1)))

    df["ratio_niveau_statique"] = (df["niveau_statique_m"]
                                   / df["profondeur_forage_m"].replace(0, np.nan))


    df["capacite_entretien"] = construire_capacite_entretien(df)

    for col, nom in [("latitude", "coordonnees_manquantes"),
                     ("annee_construction", "annee_inconnue"),
                     ("population_desservie", "population_inconnue"),
                     ("debit_essai_m3_h", "debit_inconnu")]:
        if col in df.columns:
            df[nom] = df[col].isna().astype(int)

    df["pannes_par_an"] = df["nb_pannes_12_mois"] / df["age_ans"].clip(lower=1)

    df["distance_atelier_relative"] = (df["distance_atelier_km"]
                                       / df["distance_atelier_km"].median())

    df["appui_technique"] = (df["technicien_forme_village"].fillna(0)
                             + df["stock_pieces_rechange_commune"].fillna(0))

    df["debit_par_personne"] = (df["debit_essai_m3_h"]
                                / df["population_desservie"].clip(lower=1))

    if "date_releve" in df.columns:
        df["mois_releve"] = df["date_releve"].dt.month
        df["trimestre_releve"] = df["date_releve"].dt.quarter

    return df.replace([np.inf, -np.inf], np.nan)


def preparer(df: pd.DataFrame, garder_fuite: bool = False) -> pd.DataFrame:
    return ajouter_variables_derivees(nettoyer(df, garder_fuite=garder_fuite))
