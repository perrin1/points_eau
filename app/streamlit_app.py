"""Tableau de bord de diagnostic des points d'eau partie F.

Lancement :  streamlit run app/streamlit_app.py

Le pipeline serialise et le module de preparation sont partages avec
l'entrainement (src/preprocessing.py) : les variables derivees sont donc
construites de facon strictement identique ici et dans src/train.py.
"""
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from preprocessing import COLS_FUITE, ajouter_variables_derivees, nettoyer

st.set_page_config(page_title="Diagnostic des points d'eau", page_icon="💧", layout="wide")

CHEMIN_MODELE = RACINE / "models" / "pipeline_points_eau.joblib"
CHEMIN_DONNEES = RACINE / "data" / "points_eau.csv"
REPORTS = RACINE / "reports"

BLEU, ORANGE, VERT, JAUNE, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#52514e"
COULEUR_CLASSE = {"en panne": ORANGE, "fonctionnel": VERT, "fonctionnel a reparer": JAUNE}
CLASSE_CIBLE = "fonctionnel a reparer"

# F4  bornes observees : hors de ces plages, on avertit sans bloquer.
DOMAINE = {
    "annee_construction": (1982, 2024), "profondeur_forage_m": (5, 162),
    "niveau_statique_m": (0.5, 60), "debit_essai_m3_h": (0.1, 20),
    "population_desservie": (20, 2500), "nb_menages": (5, 400),
    "cotisation_mensuelle_fcfa": (0, 3250), "distance_atelier_km": (1, 200),
    "mois_depuis_derniere_maintenance": (0, 96),
}


# ---------------------------------------------------------------------------
@st.cache_resource
def charger_modele():
    return joblib.load(CHEMIN_MODELE) if CHEMIN_MODELE.exists() else None


@st.cache_data
def charger_donnees():
    if not CHEMIN_DONNEES.exists():
        return None
    return ajouter_variables_derivees(nettoyer(pd.read_csv(CHEMIN_DONNEES)))


@st.cache_data
def charger_gabarit():
    """Une ligne brute qui sert de squelette au formulaire."""
    return pd.read_csv(CHEMIN_DONNEES).iloc[0].copy() if CHEMIN_DONNEES.exists() else None


bundle = charger_modele()
df = charger_donnees()

if bundle is None:
    st.error("**Modele introuvable**  `models/pipeline_points_eau.joblib` est absent.")
    st.code("python src/train.py", language="bash")
    st.stop()
if df is None:
    st.error(f"**Donnees introuvables**  `{CHEMIN_DONNEES}` est absent.")
    st.stop()

modele = bundle["pipeline"]
CLASSES = bundle["classes"]
COLS_MODELE = bundle["cols_num"] + bundle["cols_cat"]


proba = modele.predict_proba(df[COLS_MODELE])
df = df.assign(**{f"proba_{c}": proba[:, k] for k, c in enumerate(CLASSES)},
               etat_predit=modele.predict(df[COLS_MODELE]))


def modalites(colonne):
    """Modalites triees, sans NaN : sorted() ne compare pas un float a une chaine."""
    return sorted(df[colonne].dropna().unique())


st.title(" Diagnostic des points d'eau")
st.caption(f"Modele : **{bundle['modele']}** · exactitude {bundle['test_exactitude']:.3f} "
           f"contre {bundle['exactitude_triviale']:.3f} pour la reference triviale · "
           f"F1 macro {bundle['test_f1_macro']:.3f} contre {bundle['f1_macro_trivial']:.3f}")

onglets = st.tabs([" Exploration", "️ Carte", " Performance",
                   " Priorisation des tournees", " Diagnostic d'un ouvrage"])


# F1  Exploration

with onglets[0]:
    parts = df["etat_fonctionnement"].value_counts(normalize=True).mul(100)
    k = st.columns(4)
    k[0].metric("Ouvrages", f"{len(df):,}".replace(",", " "))
    for i, classe in enumerate(["fonctionnel", "en panne", CLASSE_CIBLE]):
        k[i + 1].metric(classe.capitalize(), f"{parts.get(classe, 0):.1f} %")

    st.subheader("Etat constate par segment")
    segment = st.selectbox("Segment", ["departement", "type_pompe", "mode_gestion",
                                       "mode_paiement", "type_ouvrage", "qualite_eau",
                                       "capacite_entretien"])
    croise = (pd.crosstab(df[segment], df["etat_fonctionnement"], normalize="index")
                .mul(100).round(1))
    croise["n"] = df[segment].value_counts()
    croise = croise.sort_values("en panne", ascending=False)

    g, d = st.columns([2, 1])
    g.bar_chart(croise[[c for c in CLASSES if c in croise]], height=360,
                color=[COULEUR_CLASSE[c] for c in CLASSES if c in croise],
                y_label="repartition (%)", stack=True)
    d.dataframe(croise, width="stretch", height=360)

    st.caption("`capacite_entretien` (indice de gouvernance construit en partie B) fait "
               "passer le taux de panne de **74,2 % a 18,9 %** entre ses extremes : "
               "c'est le gradient le plus net du jeu.")


# F2  Carte

with onglets[1]:
    st.subheader("Repartition geographique")
    c1, c2, c3 = st.columns([1, 1, 2])
    source = c1.radio("Afficher", ["Etat constate", "Etat predit",
                                   "Probabilite 'a reparer'"], index=0)
    departement = c2.selectbox("Departement", ["Tous"] + modalites("departement"))

    carte = df.dropna(subset=["latitude", "longitude"]).copy()
    if departement != "Tous":
        carte = carte[carte.departement == departement]

    if source == "Probabilite 'a reparer'":
        agrege = (carte.groupby("commune")
                       .agg(lat=("latitude", "mean"), lon=("longitude", "mean"),
                            n=("id_point_eau", "size"),
                            valeur=(f"proba_{CLASSE_CIBLE}", "mean")))
        c3.metric("Communes affichees", len(agrege))
        st.scatter_chart(agrege, x="lon", y="lat", size="n", color="valeur", height=520)
        st.caption("Un point par commune : la taille represente le nombre d'ouvrages, "
                   "la couleur la probabilite moyenne d'etre rattrapable.")
    else:
        colonne = "etat_fonctionnement" if source == "Etat constate" else "etat_predit"
        c3.metric("Ouvrages affiches", f"{len(carte):,}".replace(",", " "))
        st.scatter_chart(carte, x="longitude", y="latitude", color=colonne, height=520)
        st.caption("B5 : aucun regroupement geographique exploitable. L'ecart-type des "
                   "taux de panne entre communes (4,59) est compatible avec le hasard "
                   "(4,01 ± 0,32, soit z = +1,79).")


# F3  Performance

with onglets[2]:
    st.subheader("Performance du modele")
    k = st.columns(4)
    k[0].metric("Exactitude", f"{bundle['test_exactitude']:.3f}",
                delta=f"{bundle['test_exactitude'] - bundle['exactitude_triviale']:+.3f} vs trivial")
    k[1].metric("F1 macro", f"{bundle['test_f1_macro']:.3f}",
                delta=f"{bundle['test_f1_macro'] - bundle['f1_macro_trivial']:+.3f} vs trivial")
    k[2].metric("F1 'a reparer'", f"{bundle['f1_par_classe'][CLASSE_CIBLE]:.3f}")
    k[3].metric("Rappel 'a reparer'", f"{bundle['rappel_par_classe'][CLASSE_CIBLE]:.3f}")

    st.warning("**L'exactitude est trompeuse ici.** Un modele qui predirait toujours "
               "'fonctionnel' obtiendrait 54,8 % d'exactitude pour un F1 macro de 0,236. "
               "C'est le F1 par classe qu'il faut lire.")

    g, d = st.columns(2)
    g.markdown("**F1 par classe**")
    f1_classes = pd.Series(bundle["f1_par_classe"]).sort_values()
    g.bar_chart(f1_classes, height=280, color=BLEU, y_label="F1")
    g.caption("La classe minoritaire (8 % des ouvrages) reste la plus difficile : "
              "0,38 contre 0,73 et 0,79.")

    d.markdown("**Matrice de confusion**")
    chemin_cm = REPORTS / "e1_confusion.csv"
    if chemin_cm.exists():
        d.dataframe(pd.read_csv(chemin_cm, index_col=0), width="stretch")
        d.caption("115 ouvrages 'a reparer' non detectes sur 182, dont 79 predits "
                  "'fonctionnel' donc jamais visités.")
    else:
        d.info("Lance `python src/evaluate.py` pour generer la matrice.")

    for nom, legende in [("e5_importance.png", "Importance par permutation (F1 macro)"),
                         ("e6_priorisation.png", "Courbe de gain des inspections"),
                         ("e4_courbes.png", "ROC et precision-rappel par classe")]:
        if (REPORTS / nom).exists():
            st.image(str(REPORTS / nom), caption=legende, width=900)


# F4  Priorisation des tournees

with onglets[3]:
    st.subheader("Quels ouvrages inspecter en priorite ?")
    c1, c2, c3 = st.columns([1, 1, 1])
    budget = c1.number_input("Nombre d'inspections possibles", 10, 5000, 500, 50)
    dep = c2.selectbox("Departement", ["Tous"] + modalites("departement"), key="prio_dep")
    critere = c3.selectbox("Trier par", [f"Probabilite '{CLASSE_CIBLE}'",
                                         "Probabilite 'en panne'",
                                         "Probabilite d'etre degrade"])

    parc = df if dep == "Tous" else df[df.departement == dep]
    parc = parc.copy()
    parc["proba_degrade"] = parc[f"proba_en panne"] + parc[f"proba_{CLASSE_CIBLE}"]
    colonne_tri = {f"Probabilite '{CLASSE_CIBLE}'": f"proba_{CLASSE_CIBLE}",
                   "Probabilite 'en panne'": "proba_en panne",
                   "Probabilite d'etre degrade": "proba_degrade"}[critere]

    liste = parc.nlargest(min(budget, len(parc)), colonne_tri)

    prevalence = (parc["etat_fonctionnement"] == CLASSE_CIBLE).mean()
    trouves = (liste["etat_fonctionnement"] == CLASSE_CIBLE).sum()
    attendus = len(liste) * prevalence

    m = st.columns(4)
    m[0].metric("Ouvrages selectionnes", len(liste))
    m[1].metric(f"'{CLASSE_CIBLE}' dans la liste", int(trouves))
    m[2].metric("Au hasard, on en trouverait", f"{attendus:.0f}")
    m[3].metric("Gain", f"x{trouves / max(attendus, 1e-9):.2f}")

    st.success(f"Sur {len(liste)} inspections, cette liste identifie **{int(trouves)} "
               f"ouvrages rattrapables** contre {attendus:.0f} par tirage au hasard.")

    colonnes = ["id_point_eau", "departement", "commune", "type_pompe", "mode_gestion",
                "age_ans", "capacite_entretien", f"proba_{CLASSE_CIBLE}",
                "proba_en panne", "etat_predit"]
    colonnes = [c for c in colonnes if c in liste.columns]
    st.dataframe(liste[colonnes].round(3), width="stretch", height=380)

    st.download_button("Telecharger la liste de tournee",
                       liste[colonnes].to_csv(index=False).encode("utf-8"),
                       f"tournee_{dep.lower().replace(' ', '_')}_{len(liste)}.csv", "text/csv")

    st.caption("⚠$️ La performance du modele n'est pas homogene : le F1 macro varie de "
               "0,571 (Plateau) a 0,669 (Atacora), et la correlation entre taux de "
               "donnees manquantes et performance vaut **−0,61**. Une liste etablie sur "
               "un departement mal documente est moins fiable, elle doit etre relue "
               "par un agent qui connait le terrain.")


# diagnostic d'un ouvrage saisi au formulaire

with onglets[4]:
    st.subheader("Diagnostiquer un ouvrage")
    gabarit = charger_gabarit()

    with st.form("ouvrage"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Ouvrage**")
            departement = st.selectbox("Departement", modalites("departement"))
            type_ouvrage = st.selectbox("Type d'ouvrage", modalites("type_ouvrage"))
            type_pompe = st.selectbox("Type de pompe", modalites("type_pompe"))
            annee = st.number_input("Annee de construction", 1960, 2025, 2005)
            profondeur = st.number_input("Profondeur du forage (m)", 3.0, 200.0, 55.0, 1.0)
            niveau = st.number_input("Niveau statique (m)", 0.0, 100.0, 20.0, 0.5)
        with c2:
            st.markdown("**Usage et desserte**")
            debit = st.number_input("Debit d'essai (m3/h)", 0.0, 30.0, 2.5, 0.1)
            qualite = st.selectbox("Qualite de l'eau", modalites("qualite_eau"))
            population = st.number_input("Population desservie", 10, 5000, 440, 10)
            menages = st.number_input("Nombre de menages", 1, 800, 80, 5)
            points_village = st.number_input("Points d'eau dans le village", 1, 15, 3)
            distance_village = st.number_input("Distance au village (m)", 0, 5000, 500, 50)
        with c3:
            st.markdown("**Gouvernance et entretien**")
            gestion = st.selectbox("Mode de gestion", modalites("mode_gestion"))
            paiement = st.selectbox("Mode de paiement", modalites("mode_paiement"))
            cotisation = st.number_input("Cotisation mensuelle (FCFA)", 0, 5000, 300, 50)
            maintenance = st.number_input("Mois depuis la derniere maintenance", 0, 120, 15)
            pannes = st.number_input("Pannes sur 12 mois", 0, 15, 1)
            distance_atelier = st.number_input("Distance a l'atelier (km)", 0.0, 300.0, 35.0, 5.0)

        c4, c5, c6 = st.columns(3)
        technicien = c4.selectbox("Technicien forme au village", [0, 1],
                                  format_func=lambda v: "Oui" if v else "Non")
        stock = c5.selectbox("Stock de pieces a la commune", [0, 1],
                             format_func=lambda v: "Oui" if v else "Non")
        maitre = c6.selectbox("Maitre d'ouvrage", modalites("maitre_ouvrage"))

        envoyer = st.form_submit_button("Diagnostiquer", type="primary")

    if envoyer:
        saisie = {
            "departement": departement, "type_ouvrage": type_ouvrage,
            "type_pompe": type_pompe, "annee_construction": annee,
            "profondeur_forage_m": profondeur, "niveau_statique_m": niveau,
            "debit_essai_m3_h": debit, "qualite_eau": qualite,
            "population_desservie": population, "nb_menages": menages,
            "nb_points_eau_village": points_village,
            "distance_village_m": distance_village, "mode_gestion": gestion,
            "mode_paiement": paiement, "cotisation_mensuelle_fcfa": cotisation,
            "mois_depuis_derniere_maintenance": maintenance,
            "nb_pannes_12_mois": pannes, "distance_atelier_km": distance_atelier,
            "technicien_forme_village": technicien,
            "stock_pieces_rechange_commune": stock, "maitre_ouvrage": maitre,
        }

        # F4 valeurs hors domaine : on avertit sans bloquer
        for col, (bas, haut) in DOMAINE.items():
            if col in saisie and not (bas <= saisie[col] <= haut):
                st.warning(f"`{col}` = {saisie[col]:,.0f} hors de la plage observee "
                           f"[{bas:,.0f} – {haut:,.0f}] : prediction peu fiable.")
        if niveau > profondeur:
            st.warning("Le niveau statique depasse la profondeur du forage : "
                       "incoherence physique, le niveau sera ignore.")

        ligne = gabarit.copy()
        for cle, valeur in saisie.items():
            ligne[cle] = valeur
        try:
            prepare = ajouter_variables_derivees(nettoyer(pd.DataFrame([ligne])))
            manquantes = [c for c in COLS_MODELE if c not in prepare.columns]
            if manquantes:
                st.error(f"Colonnes absentes apres preparation : {manquantes}")
                st.stop()
            probabilites = modele.predict_proba(prepare[COLS_MODELE])[0]
        except Exception as e:
            st.error(f"Prediction impossible : {e}")
            st.stop()

        st.divider()
        resultats = pd.Series(dict(zip(CLASSES, probabilites))).sort_values(ascending=False)
        etat = resultats.index[0]

        r = st.columns([1, 1, 2])
        for i, (classe, p) in enumerate(resultats.items()):
            if i < 2:
                r[i].metric(classe.capitalize(), f"{p*100:.1f} %")
        if etat == "en panne":
            r[2].error(f"###  {etat.upper()}\nRehabilitation lourde probable")
        elif etat == CLASSE_CIBLE:
            r[2].warning(f"###  {etat.upper()}\nUne intervention legere evite la panne totale")
        else:
            r[2].success(f"### {etat.upper()}")

        st.bar_chart(resultats, height=220, color=BLEU, y_label="probabilite")

        st.markdown("**Variables de gouvernance calculees**")
        cles = ["capacite_entretien", "appui_technique", "age_ans", "pression_usage",
                "ratio_niveau_statique", "pannes_par_an"]
        cles = [c for c in cles if c in prepare.columns]
        st.dataframe(prepare[cles].T.rename(columns={0: "valeur"}).round(3), width="stretch")
        st.info("`capacite_entretien` va de 0 a 4 : a 0, 74,2 % des ouvrages sont en "
                "panne ; a 4, seulement 18,9 %. C'est le levier le plus actionnable.")
        st.caption("Ce diagnostic assiste la decision, il ne la remplace pas. "
                   "Le modele ne detecte que 37 % des ouvrages rattrapables : "
                   "une absence d'alerte ne garantit pas qu'un ouvrage va bien.")
