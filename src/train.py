"""Encodage, modelisation et serialisation.
Usage : python src/train.py
"""
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from preprocessing import preparer

warnings.filterwarnings("ignore")

RACINE = Path(__file__).resolve().parent.parent
CSV = RACINE / "data" / "points_eau.csv"
RANDOM_STATE = 42
CIBLE = "etat_fonctionnement"

EXCLURE = [CIBLE, "id_point_eau", "date_releve", "periode_releve"]

COLS_CARDINALITE = ["commune", "installateur"]

COLS_RETIREES = []


def charger():
    return preparer(pd.read_csv(CSV))


def colonnes(df, retirer=()):
    exclues = set(EXCLURE) | set(retirer)
    num = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclues]
    cat = [c for c in df.select_dtypes(include=["object", "string", "category"]).columns
           if c not in exclues]
    return num, cat


def decouper(df, retirer=()):

    num, cat = colonnes(df, retirer)
    X, y = df[num + cat], df[CIBLE]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    return X_train, y_train, X_test, y_test, num, cat


def prepro_onehot(cols_num, cols_cat, cols_cible=()):

    cols_cat = [c for c in cols_cat if c not in cols_cible]
    blocs = [
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), cols_num),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore",
                                            sparse_output=False)), cols_cat),
    ]
    if cols_cible:
        blocs.append(("cible", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            TargetEncoder(target_type="auto", random_state=RANDOM_STATE)), list(cols_cible)))
    return ColumnTransformer(blocs)


class FrequenceEncoder:

    def __init__(self):
        self.frequences_ = {}

    def fit(self, X, y=None):
        for col in X.columns:
            self.frequences_[col] = X[col].value_counts(normalize=True)
        return self

    def transform(self, X):
        return pd.DataFrame({c: X[c].map(self.frequences_[c]).fillna(0)
                             for c in X.columns}, index=X.index)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, noms=None):
        return np.asarray(list(self.frequences_))


def modele_reference():
    return HistGradientBoostingClassifier(
        class_weight="balanced", random_state=RANDOM_STATE, max_iter=200)


def poids_renforces(y, facteur=2.0):
    """Poids 'balanced' avec la classe minoritaire multipliee par `facteur`.

    Les cles sont les POSITIONS des classes triees, pas leurs libelles :
    HistGradientBoosting encode y en 0..n-1 avant d'appliquer class_weight.

    Justification (D5) : ce reglage porte le rappel sur 'fonctionnel a reparer'
    de 0,330 a 0,368 pour 1,3 point d'exactitude en moins. Sur un outil de
    priorisation de tournees, detecter les ouvrages rattrapables prime.
    """
    classes = sorted(y.unique())
    effectifs = y.value_counts()
    poids = {i: len(y) / (len(classes) * effectifs[c]) for i, c in enumerate(classes)}
    poids[classes.index(effectifs.idxmin())] *= facteur
    return poids


def evaluer(X_train, y_train, X_test, y_test, prepro, etiquette):
    pipe = Pipeline([("pre", prepro), ("clf", modele_reference())])
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    f1_classes = f1_score(y_test, pred, average=None,
                          labels=sorted(y_test.unique()))
    return {
        "strategie": etiquette,
        "cv_f1_macro": round(scores.mean(), 4),
        "cv_ecart_type": round(scores.std(), 4),
        "test_f1_macro": round(f1_score(y_test, pred, average="macro"), 4),
        "test_f1_pondere": round(f1_score(y_test, pred, average="weighted"), 4),
        **{f"f1_{c}": round(v, 4) for c, v in zip(sorted(y_test.unique()), f1_classes)},
        "n_colonnes": pipe.named_steps["pre"].transform(X_train.head(50)).shape[1],
    }


def comparer_encodages(df):
    lignes = []

    # 1. Exclusion pure et simple
    X_tr, y_tr, X_te, y_te, num, cat = decouper(df, retirer=COLS_CARDINALITE)
    lignes.append(evaluer(X_tr, y_tr, X_te, y_te,
                          prepro_onehot(num, cat), "1. exclusion des deux variables"))

    X_tr, y_tr, X_te, y_te, num, cat = decouper(df)

    # 2. One-hot brut : 76 + 52 colonnes supplementaires
    lignes.append(evaluer(X_tr, y_tr, X_te, y_te,
                          prepro_onehot(num, cat), "2. one-hot brut"))

    # 3. Regroupement des modalites rares dans 'Autre'
    seuil = 0.01
    X_tr3, X_te3 = X_tr.copy(), X_te.copy()
    for col in COLS_CARDINALITE:
        frequentes = X_tr[col].value_counts(normalize=True)
        gardees = frequentes[frequentes >= seuil].index
        X_tr3[col] = X_tr[col].where(X_tr[col].isin(gardees), "Autre")
        X_te3[col] = X_te[col].where(X_te[col].isin(gardees), "Autre")
    lignes.append(evaluer(X_tr3, y_tr, X_te3, y_te,
                          prepro_onehot(num, cat), f"3. regroupement des rares (<{seuil:.0%})"))

    # 4. Encodage par frequence : une colonne au lieu de 76
    X_tr4, X_te4 = X_tr.copy(), X_te.copy()
    encodeur = FrequenceEncoder().fit(X_tr[COLS_CARDINALITE])
    freq_tr = encodeur.transform(X_tr[COLS_CARDINALITE])
    freq_te = encodeur.transform(X_te[COLS_CARDINALITE])
    for col in COLS_CARDINALITE:
        X_tr4[f"freq_{col}"] = freq_tr[col]
        X_te4[f"freq_{col}"] = freq_te[col]
    X_tr4 = X_tr4.drop(columns=COLS_CARDINALITE)
    X_te4 = X_te4.drop(columns=COLS_CARDINALITE)
    num4 = num + [f"freq_{c}" for c in COLS_CARDINALITE]
    cat4 = [c for c in cat if c not in COLS_CARDINALITE]
    lignes.append(evaluer(X_tr4, y_tr, X_te4, y_te,
                          prepro_onehot(num4, cat4), "4. encodage par frequence"))

    # 5. Target encoding, calcule par validation croisee interne
    lignes.append(evaluer(X_tr, y_tr, X_te, y_te,
                          prepro_onehot(num, cat, cols_cible=COLS_CARDINALITE),
                          "5. target encoding (CV interne)"))

    tableau = pd.DataFrame(lignes).sort_values("cv_f1_macro", ascending=False)
    tableau.to_csv(RACINE / "reports" / "c2_encodages.csv", index=False)
    print(tableau.to_string(index=False))
    return tableau


def comparer_geographie(df):
    configurations = [
        ("les deux", []),
        ("commune seule", ["departement"]),
        ("departement seul", ["commune"]),
        ("aucun des deux", ["departement", "commune"]),
    ]
    lignes = []
    for nom, retirer in configurations:
        X_tr, y_tr, X_te, y_te, num, cat = decouper(df, retirer=retirer)
        lignes.append(evaluer(X_tr, y_tr, X_te, y_te, prepro_onehot(num, cat), nom))
    tableau = pd.DataFrame(lignes).sort_values("cv_f1_macro", ascending=False)
    tableau.to_csv(RACINE / "reports" / "c3_geographie.csv", index=False)
    print(tableau.to_string(index=False))
    return tableau


def experience_fuite():
    lignes = []
    for garder in (True, False):
        df = preparer(pd.read_csv(CSV), garder_fuite=garder)
        X_tr, y_tr, X_te, y_te, num, cat = decouper(df)
        pipe = Pipeline([("pre", prepro_onehot(num, cat)),
                         ("clf", modele_reference())]).fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        lignes.append({
            "configuration": "AVEC les colonnes post-constat" if garder
                             else "SANS (modele retenu)",
            "exactitude": round((pred == y_te).mean(), 4),
            "f1_macro": round(f1_score(y_te, pred, average="macro"), 4),
        })
    tableau = pd.DataFrame(lignes)
    tableau.to_csv(RACINE / "reports" / "experience_fuite.csv", index=False)
    print(tableau.to_string(index=False))
    return tableau


def comparer_modeles(df, retirer=()):
    X_tr, y_tr, X_te, y_te, num, cat = decouper(df, retirer)
    candidats = {
        "Reference triviale": DummyClassifier(strategy="most_frequent"),
        "Regression logistique": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "Foret aleatoire": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=RANDOM_STATE, max_iter=200),
    }
    lignes = []
    for nom, estimateur in candidats.items():
        pipe = Pipeline([("pre", prepro_onehot(num, cat)), ("clf", estimateur)])
        cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1_macro")
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        f1_classes = f1_score(y_te, pred, average=None, labels=sorted(y_te.unique()))
        lignes.append({
            "modele": nom,
            "cv_f1_macro": round(scores.mean(), 4),
            "cv_ecart_type": round(scores.std(), 4),
            "test_exactitude": round((pred == y_te).mean(), 4),
            "test_f1_macro": round(f1_score(y_te, pred, average="macro"), 4),
            **{f"f1_{c}": round(v, 4) for c, v in zip(sorted(y_te.unique()), f1_classes)},
        })
    tableau = pd.DataFrame(lignes).sort_values("cv_f1_macro", ascending=False)
    tableau.to_csv(RACINE / "reports" / "comparaison_modeles.csv", index=False)
    print(tableau.to_string(index=False))
    return tableau


# D5  traitement du desequilibre
def comparer_desequilibre(df, retirer=()):
    """Cinq approches, a modele egal.
    ImbPipeline (imbalanced-learn) et non Pipeline (scikit-learn) : le
    reechantillonnage ne doit s'appliquer qu'aux blocs d'entrainement, jamais
    aux blocs de validation, sinon on evalue sur des individus synthetiques.
    """
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.pipeline import Pipeline as ImbPipeline

    X_tr, y_tr, X_te, y_te, num, cat = decouper(df, retirer)
    classes = sorted(y_tr.unique())
    minoritaire = y_tr.value_counts().idxmin()

    poids_manuels = poids_renforces(y_tr, facteur=2.0)

    def hgb(**kw):
        return HistGradientBoostingClassifier(random_state=RANDOM_STATE, max_iter=200, **kw)

    approches = {
        "1. aucun traitement": Pipeline([
            ("pre", prepro_onehot(num, cat)), ("clf", hgb())]),
        "2. class_weight='balanced'": Pipeline([
            ("pre", prepro_onehot(num, cat)), ("clf", hgb(class_weight="balanced"))]),
        "3. poids manuels (x2 sur la minoritaire)": Pipeline([
            ("pre", prepro_onehot(num, cat)), ("clf", hgb(class_weight=poids_manuels))]),
        "4. sur-echantillonnage aleatoire": ImbPipeline([
            ("pre", prepro_onehot(num, cat)),
            ("sampler", RandomOverSampler(random_state=RANDOM_STATE)), ("clf", hgb())]),
        "5. SMOTE": ImbPipeline([
            ("pre", prepro_onehot(num, cat)),
            ("sampler", SMOTE(random_state=RANDOM_STATE)), ("clf", hgb())]),
    }

    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    lignes = []
    for nom, pipe in approches.items():
        scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1_macro")
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        f1_classes = f1_score(y_te, pred, average=None, labels=classes)
        rappels = [((pred == c) & (y_te == c)).sum() / (y_te == c).sum() for c in classes]
        lignes.append({
            "approche": nom,
            "cv_f1_macro": round(scores.mean(), 4),
            "cv_ecart_type": round(scores.std(), 4),
            "test_exactitude": round((pred == y_te).mean(), 4),
            "test_f1_macro": round(f1_score(y_te, pred, average="macro"), 4),
            **{f"f1_{c}": round(v, 4) for c, v in zip(classes, f1_classes)},
            **{f"rappel_{c}": round(v, 4) for c, v in zip(classes, rappels)},
        })
    tableau = pd.DataFrame(lignes).sort_values("cv_f1_macro", ascending=False)
    tableau.to_csv(RACINE / "reports" / "d5_desequilibre.csv", index=False)
    print(tableau.to_string(index=False))
    return tableau


# D7  reglage des hyperparametres

def regler_hyperparametres(df, retirer=()):
    """Grille documentee sur HistGradientBoosting, scoring f1_macro."""
    from sklearn.model_selection import RandomizedSearchCV

    X_tr, y_tr, X_te, y_te, num, cat = decouper(df, retirer)
    grille = {
        "clf__max_iter": [200, 400, 600],             # nombre d'arbres
        "clf__learning_rate": [0.03, 0.05, 0.1],      # pas d'apprentissage
        "clf__max_depth": [None, 4, 6, 8],            # profondeur
        "clf__min_samples_leaf": [10, 20, 40],        # regularisation
        "clf__l2_regularization": [0.0, 0.1, 1.0],
    }
    recherche = RandomizedSearchCV(
        Pipeline([("pre", prepro_onehot(num, cat)),
                  ("clf", HistGradientBoostingClassifier(
                      class_weight="balanced", random_state=RANDOM_STATE))]),
        grille, n_iter=25, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring="f1_macro", random_state=RANDOM_STATE, refit=True)
    recherche.fit(X_tr, y_tr)

    pred = recherche.predict(X_te)
    classes = sorted(y_te.unique())
    print(f"  grille : {grille}")
    print(f"  meilleurs parametres : {recherche.best_params_}")
    print(f"  CV f1_macro {recherche.best_score_:.4f} | "
          f"test {f1_score(y_te, pred, average='macro'):.4f}")
    print(f"  F1 par classe : "
          f"{dict(zip(classes, f1_score(y_te, pred, average=None, labels=classes).round(4)))}")
    pd.DataFrame(recherche.cv_results_).to_csv(
        RACINE / "reports" / "d7_hyperparametres.csv", index=False)
    return recherche



# D8    : approche ordinale en cascade

def approche_ordinale(df, retirer=()):
    """Les trois classes sont ordonnees : fonctionnel > a reparer > en panne.

    Deux classifieurs binaires en cascade au lieu d'un multi-classes :
        etage 1 : degrade (a reparer + en panne) contre fonctionnel
        etage 2 : parmi les degrades, en panne contre a reparer

    L'interet attendu : l'etage 2 ne voit que 5 149 ouvrages degrades, ou la
    classe minoritaire pese 17,7 % au lieu de 8 %. Le desequilibre y est donc
    deux fois moins severe.
    """
    X_tr, y_tr, X_te, y_te, num, cat = decouper(df, retirer)
    classes = sorted(y_te.unique())

    def hgb():
        return HistGradientBoostingClassifier(class_weight="balanced",
                                              random_state=RANDOM_STATE, max_iter=200)

    # Etage 1 : degrade ou non
    y_tr_1 = (y_tr != "fonctionnel").astype(int)
    etage1 = Pipeline([("pre", prepro_onehot(num, cat)), ("clf", hgb())]).fit(X_tr, y_tr_1)

    # Etage 2 : parmi les degrades seulement
    degrade_tr = y_tr != "fonctionnel"
    y_tr_2 = (y_tr[degrade_tr] == "en panne").astype(int)
    etage2 = Pipeline([("pre", prepro_onehot(num, cat)),
                       ("clf", hgb())]).fit(X_tr[degrade_tr], y_tr_2)

    # Assemblage sur le test
    pred = np.array(["fonctionnel"] * len(X_te), dtype=object)
    est_degrade = etage1.predict(X_te) == 1
    if est_degrade.any():
        en_panne = etage2.predict(X_te[est_degrade]) == 1
        pred[np.where(est_degrade)[0]] = np.where(en_panne, "en panne", "fonctionnel a reparer")

    f1_classes = f1_score(y_te, pred, average=None, labels=classes)
    ligne = {
        "approche": "cascade ordinale (2 binaires)",
        "test_exactitude": round((pred == y_te).mean(), 4),
        "test_f1_macro": round(f1_score(y_te, pred, average="macro"), 4),
        **{f"f1_{c}": round(v, 4) for c, v in zip(classes, f1_classes)},
    }
    print(pd.DataFrame([ligne]).to_string(index=False))
    print(f"\n  Part de la minoritaire a l'etage 2 : "
          f"{(y_tr[degrade_tr] == 'fonctionnel a reparer').mean()*100:.1f} % "
          f"(contre {(y_tr == 'fonctionnel a reparer').mean()*100:.1f} % en multi-classes)")
    pd.DataFrame([ligne]).to_csv(RACINE / "reports" / "d8_cascade.csv", index=False)
    return etage1, etage2, ligne


def configuration_finale(df):
    """Verifie que retirer les trois variables geographiques ne coute rien.

    """
    lignes = []
    for nom, retirer in [("toutes les variables", []),
                         ("sans commune", ["commune"]),
                         ("sans commune + departement", ["commune", "departement"]),
                         ("sans installateur", ["installateur"]),
                         ("sans les trois", ["commune", "departement", "installateur"])]:
        X_tr, y_tr, X_te, y_te, num, cat = decouper(df, retirer)
        lignes.append(evaluer(X_tr, y_tr, X_te, y_te, prepro_onehot(num, cat), nom))
    tableau = pd.DataFrame(lignes)
    tableau.to_csv(RACINE / "reports" / "c_configuration_finale.csv", index=False)
    print(tableau.to_string(index=False))
    return tableau


def serialiser(df, retirer=COLS_RETIREES):
    """Entraine le modele final et le sauvegarde pour l'application.

    Modele retenu : HistGradientBoosting avec poids renforces sur la classe
    minoritaire (D5), sur le jeu de variables issu de C2 et C3.
    """
    X_tr, y_tr, X_te, y_te, num, cat = decouper(df, retirer)
    final = Pipeline([
        ("pre", prepro_onehot(num, cat)),
        ("clf", HistGradientBoostingClassifier(
            class_weight=poids_renforces(y_tr), random_state=RANDOM_STATE, max_iter=200)),
    ]).fit(X_tr, y_tr)

    pred = final.predict(X_te)
    classes = sorted(y_te.unique())
    print(classification_report(y_te, pred, digits=3))

    rappels = {c: float(((pred == c) & (y_te == c)).sum() / (y_te == c).sum())
               for c in classes}
    joblib.dump({
        "pipeline": final,
        "modele": "HistGradientBoosting (poids renforces x2 sur la minoritaire)",
        "cols_num": num,
        "cols_cat": cat,
        "classes": classes,
        "colonnes_retirees": list(retirer),
        "test_exactitude": float((pred == y_te).mean()),
        "test_f1_macro": float(f1_score(y_te, pred, average="macro")),
        "f1_par_classe": {c: float(v) for c, v in
                          zip(classes, f1_score(y_te, pred, average=None, labels=classes))},
        "rappel_par_classe": rappels,
        "f1_macro_trivial": 0.2361,
        "exactitude_triviale": 0.5484,
    }, RACINE / "models" / "pipeline_points_eau.joblib")
    print("-> models/pipeline_points_eau.joblib")
    return final


def main():
    (RACINE / "reports").mkdir(exist_ok=True)
    (RACINE / "models").mkdir(exist_ok=True)
    df = charger()
    print(f"{len(df)} ouvrages | {df[CIBLE].value_counts(normalize=True).mul(100).round(1).to_dict()}")

    print("\n" + "=" * 78); print("C2. strategies d'encodage a haute cardinalite"); print("=" * 78)
    comparer_encodages(df)

    print("\n" + "=" * 78); print("C3. departement et commune"); print("=" * 78)
    comparer_geographie(df)

    print("\n" + "=" * 78); print("Experience de fuite"); print("=" * 78)
    experience_fuite()

    print("\n" + "=" * 78); print("D2/D4/D6 : modeles"); print("=" * 78)
    comparer_modeles(df)

    print("\n" + "=" * 78); print("D5 : traitement du desequilibre"); print("=" * 78)
    comparer_desequilibre(df)

    print("\n" + "=" * 78); print("D7 : reglage des hyperparametres"); print("=" * 78)
    regler_hyperparametres(df)

    print("\n" + "=" * 78); print("D8 : bonus, approche ordinale en cascade"); print("=" * 78)
    approche_ordinale(df)

    print("\n" + "=" * 78); print("Configuration finale des variables"); print("=" * 78)
    configuration_finale(df)

    print("\n" + "=" * 78); print("Modele final et serialisation"); print("=" * 78)
    serialiser(df)


if __name__ == "__main__":
    main()
