"""Evaluation et exploitation operationnelle  partie E.

Usage : python src/evaluate.py   (apres python src/train.py)

Produit :
    reports/e1_confusion.png / .csv     matrice 3x3, effectifs et pourcentages
    reports/e2_rapport.csv              precision, rappel, F1 par classe + agregats
    reports/e4_courbes.png              ROC et precision-rappel une-contre-toutes
    reports/e5_importance.png / .csv    importance par permutation (f1_macro)
    reports/e6_priorisation.png / .csv  les 500 ouvrages a inspecter
    reports/e7_equite.csv               performance par departement
"""
from pathlib import Path
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (auc, average_precision_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, roc_curve)

from preprocessing import preparer
from train import CIBLE, COLS_RETIREES, RACINE, decouper

warnings.filterwarnings("ignore")

REPORTS = RACINE / "reports"
CSV = RACINE / "data" / "points_eau.csv"
BLEU, ORANGE, VERT, JAUNE, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#52514e"
COULEUR_CLASSE = {"en panne": ORANGE, "fonctionnel": VERT, "fonctionnel a reparer": JAUNE}

# E6  capacite d'inspection du service technique
BUDGET_INSPECTIONS = 500


def habiller(ax, titre="", x="", y=""):
    ax.set(title=titre, xlabel=x, ylabel=y)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.2); ax.set_axisbelow(True)


def e1_confusion(y_vrai, pred, classes):
    """E1  matrice 3x3 en effectifs et en pourcentages par ligne."""
    cm = confusion_matrix(y_vrai, pred, labels=classes)
    cm_pct = confusion_matrix(y_vrai, pred, labels=classes, normalize="true") * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, matrice, titre, fmt in [(axes[0], cm, "Effectifs", "{:.0f}"),
                                    (axes[1], cm_pct, "% par ligne (reel)", "{:.1f}")]:
        ax.imshow(matrice, cmap="Blues")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, fmt.format(matrice[i, j]), ha="center", va="center",
                        fontsize=13,
                        color="white" if matrice[i, j] > matrice.max() / 2 else "black")
        ax.set_xticks(range(len(classes)), classes, rotation=20, ha="right", fontsize=9)
        ax.set_yticks(range(len(classes)), classes, fontsize=9)
        ax.set(title=titre, xlabel="predit", ylabel="reel")
        ax.grid(False)
    fig.suptitle("E1  Matrice de confusion", fontsize=12)
    fig.tight_layout(); fig.savefig(REPORTS / "e1_confusion.png", dpi=150, bbox_inches="tight")

    tableau = pd.DataFrame(cm, index=classes, columns=classes)
    tableau.to_csv(REPORTS / "e1_confusion.csv")
    print(tableau.to_string())

    # La confusion la plus frequente, hors diagonale
    hors_diagonale = cm.copy()
    np.fill_diagonal(hors_diagonale, 0)
    i, j = np.unravel_index(hors_diagonale.argmax(), cm.shape)
    print(f"\nConfusion la plus frequente : {classes[i]} predit {classes[j]} "
          f"({cm[i, j]} ouvrages, {cm_pct[i, j]:.1f} % de la classe)")

    rare = "fonctionnel a reparer"
    ir = classes.index(rare)
    manques = cm[ir].sum() - cm[ir, ir]
    print(f"Confusion la plus COUTEUSE : {manques} ouvrages '{rare}' non detectes "
          f"({manques / cm[ir].sum() * 100:.1f} % de la classe) — chacun bascule en "
          "panne totale faute d'intervention legere.")
    return cm


def e2_rapport(y_vrai, pred, classes):
    """E2  rapport complet, plus les trois agregats et ce qu'ils cachent."""
    rapport = pd.DataFrame(classification_report(
        y_vrai, pred, labels=classes, output_dict=True, zero_division=0)).T.round(4)
    rapport.to_csv(REPORTS / "e2_rapport.csv")
    print(rapport.to_string())

    agregats = {
        "exactitude": (pred == y_vrai).mean(),
        "f1_macro": f1_score(y_vrai, pred, average="macro"),
        "f1_pondere": f1_score(y_vrai, pred, average="weighted"),
        "exactitude_equilibree": balanced_accuracy_score(y_vrai, pred),
    }
    print("\nAgregats :")
    for nom, valeur in agregats.items():
        print(f"  {nom:<24} {valeur:.4f}")
    return rapport, agregats


def e4_courbes(modele, X_test, y_test, classes):
    """E4  ROC et precision-rappel en une-contre-toutes."""
    proba = modele.predict_proba(X_test)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for k, classe in enumerate(classes):
        binaire = (y_test == classe).astype(int)
        fpr, tpr, _ = roc_curve(binaire, proba[:, k])
        axes[0].plot(fpr, tpr, lw=2.5, color=COULEUR_CLASSE[classe],
                     label=f"{classe} (AUC {auc(fpr, tpr):.3f})")

        prec, rapp, _ = precision_recall_curve(binaire, proba[:, k])
        ap = average_precision_score(binaire, proba[:, k])
        axes[1].plot(rapp, prec, lw=2.5, color=COULEUR_CLASSE[classe],
                     label=f"{classe} (AP {ap:.3f})")
        axes[1].axhline(binaire.mean(), ls=":", lw=1, color=COULEUR_CLASSE[classe])

    axes[0].plot([0, 1], [0, 1], "--", color=GRIS, lw=1, label="hasard")
    habiller(axes[0], "Courbes ROC (une contre toutes)", "taux de faux positifs", "rappel")
    habiller(axes[1], "Courbes precision-rappel", "rappel", "precision")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("E4  Performance par classe", fontsize=12)
    fig.tight_layout(); fig.savefig(REPORTS / "e4_courbes.png", dpi=150, bbox_inches="tight")

    print("Les lignes pointillees marquent la prevalence de chaque classe : "
          "c'est la ligne de base de la precision-rappel, pas 0,5.")


def e5_importance(modele, X_test, y_test):
    """E5  importance par permutation, scoring f1_macro."""
    imp = permutation_importance(modele, X_test, y_test, n_repeats=10,
                                 random_state=42, scoring="f1_macro")
    top = (pd.DataFrame({"variable": X_test.columns,
                         "importance": imp.importances_mean,
                         "ecart_type": imp.importances_std})
           .sort_values("importance", ascending=False))
    top.round(4).to_csv(REPORTS / "e5_importance.csv", index=False)

    quinze = top.head(15)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(quinze["variable"][::-1], quinze["importance"][::-1],
            xerr=quinze["ecart_type"][::-1], color=BLEU, height=.7,
            error_kw={"ecolor": GRIS, "capsize": 2})
    habiller(ax, "E5  Importance par permutation (chute du F1 macro)",
             "degradation du F1 macro", "")
    fig.tight_layout(); fig.savefig(REPORTS / "e5_importance.png", dpi=150, bbox_inches="tight")

    print(top.head(12).round(4).to_string(index=False))
    gouvernance = ["capacite_entretien", "mode_gestion", "mode_paiement",
                   "cotisation_mensuelle_fcfa", "mois_depuis_derniere_maintenance"]
    rangs = {v: int(top.index.get_indexer([top[top.variable == v].index[0]])[0]) + 1
             for v in gouvernance if v in top.variable.values}
    print(f"\nRang des variables de gouvernance : "
          f"{ {v: int(top.reset_index(drop=True).query('variable == @v').index[0]) + 1 for v in gouvernance if v in top.variable.values} }")
    return top


def e6_priorisation(modele, X_test, y_test, classes, budget=BUDGET_INSPECTIONS):
    """E6  les N ouvrages a inspecter en priorite.

    Le service peut inspecter 500 points d'eau. On classe les ouvrages par
    probabilite predite d'etre 'fonctionnel a reparer' — la classe dont la
    detection est la plus rentable : une intervention legere y evite une
    panne totale.
    """
    rare = "fonctionnel a reparer"
    k = classes.index(rare)
    proba = modele.predict_proba(X_test)

    classement = pd.DataFrame({
        "proba_a_reparer": proba[:, k],
        "proba_en_panne": proba[:, classes.index("en panne")],
        "reel": y_test.values,
    }, index=X_test.index).sort_values("proba_a_reparer", ascending=False)

    prevalence = (y_test == rare).mean()
    lignes = []
    for n in [100, 250, 500, 1000]:
        detectes = (classement.head(n)["reel"] == rare).sum()
        hasard = n * prevalence
        lignes.append({
            "budget_inspections": n,
            "detectes_a_reparer": int(detectes),
            "attendu_au_hasard": round(hasard, 1),
            "gain": round(detectes / hasard, 2),
            "precision_%": round(detectes / n * 100, 1),
            "part_de_la_classe_couverte_%": round(detectes / (y_test == rare).sum() * 100, 1),
        })
    tableau = pd.DataFrame(lignes)
    tableau.to_csv(REPORTS / "e6_priorisation.csv", index=False)
    print(tableau.to_string(index=False))

    # Courbe de gain cumule
    trie = (classement["reel"] == rare).astype(int).values
    cumul = np.cumsum(trie)
    rangs = np.arange(1, len(trie) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rangs, cumul, color=BLEU, lw=2.5, label="modele (par probabilite decroissante)")
    ax.plot(rangs, rangs * prevalence, "--", color=GRIS, lw=1.5, label="tirage au hasard")
    ax.axvline(budget, ls=":", color=ORANGE, lw=2,
               label=f"budget de {budget} inspections")
    detectes = int(cumul[budget - 1])
    ax.annotate(f"{detectes} detectes\ncontre {budget * prevalence:.0f} au hasard",
                xy=(budget, detectes), xytext=(budget + 120, detectes - 20),
                fontsize=9, color=GRIS,
                arrowprops={"arrowstyle": "->", "color": GRIS})
    habiller(ax, "E6  Courbe de gain : ouvrages 'a reparer' detectes",
             "nombre d'ouvrages inspectes", "ouvrages 'a reparer' trouves")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(REPORTS / "e6_priorisation.png", dpi=150, bbox_inches="tight")
    return tableau, classement


def e7_equite(df_test, y_test, pred, classes):
    """E7  la performance est-elle homogene entre departements ?"""
    d = df_test.copy()
    d["pred"] = pred
    d["reel"] = y_test.values

    lignes = []
    for departement, groupe in d.groupby("departement"):
        f1_classes = f1_score(groupe.reel, groupe.pred, average=None,
                              labels=classes, zero_division=0)
        lignes.append({
            "departement": departement,
            "n": len(groupe),
            "taux_manquants_%": round(groupe.isna().mean().mean() * 100, 2),
            "exactitude": round((groupe.pred == groupe.reel).mean(), 4),
            "f1_macro": round(f1_score(groupe.reel, groupe.pred, average="macro",
                                       zero_division=0), 4),
            **{f"f1_{c}": round(v, 4) for c, v in zip(classes, f1_classes)},
        })
    tableau = pd.DataFrame(lignes).sort_values("f1_macro")
    tableau.to_csv(REPORTS / "e7_equite.csv", index=False)
    print(tableau.to_string(index=False))

    ecart = tableau.f1_macro.max() - tableau.f1_macro.min()
    correlation = tableau["taux_manquants_%"].corr(tableau["f1_macro"])
    print(f"\nEcart de F1 macro entre departements : {ecart:.4f}")
    print(f"Correlation entre taux de manquants et F1 macro : {correlation:+.3f}")
    print("  -> " + ("le modele sous-performe la ou les donnees sont incompletes"
                     if correlation < -0.4 else
                     "pas de lien net entre qualite des donnees et performance"))
    return tableau


# ---------------------------------------------------------------------------
def main():
    REPORTS.mkdir(exist_ok=True)
    bundle = joblib.load(RACINE / "models" / "pipeline_points_eau.joblib")
    modele, classes = bundle["pipeline"], bundle["classes"]

    df = preparer(pd.read_csv(CSV))
    X_train, y_train, X_test, y_test, _, _ = decouper(df, COLS_RETIREES)
    df_test = df.loc[X_test.index]
    pred = modele.predict(X_test)

    print("=" * 78); print("E1  matrice de confusion"); print("=" * 78)
    e1_confusion(y_test, pred, classes)

    print("\n" + "=" * 78); print("E2  rapport de classification"); print("=" * 78)
    e2_rapport(y_test, pred, classes)

    print("\n" + "=" * 78); print("E4  courbes une-contre-toutes"); print("=" * 78)
    e4_courbes(modele, X_test, y_test, classes)

    print("\n" + "=" * 78); print("E5  importance par permutation"); print("=" * 78)
    e5_importance(modele, X_test, y_test)

    print("\n" + "=" * 78); print("E6  priorisation des inspections"); print("=" * 78)
    e6_priorisation(modele, X_test, y_test, classes)

    print("\n" + "=" * 78); print("E7  equite geographique"); print("=" * 78)
    e7_equite(df_test, y_test, pred, classes)

    print(f"\n-> figures et tableaux ecrits dans {REPORTS}")


if __name__ == "__main__":
    main()
