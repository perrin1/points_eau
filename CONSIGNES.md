# TP 3 — Diagnostic de l'état des points d'eau

**Domaine :** eau et infrastructures publiques · **Tâche :** classification multi-classes déséquilibrée
**Données :** `points_eau.csv` — 11 472 lignes × 32 colonnes
**Durée estimée :** 8 à 12 h · **Livrable :** dépôt GitHub + app Streamlit

---

## 1. Contexte

Un service national de l'eau dispose de l'inventaire de **11 472 points d'eau**
(forages, puits modernes, adductions villageoises) relevés sur le terrain en 2025. Pour
chacun : localisation, caractéristiques techniques de l'ouvrage, population desservie,
mode de gestion, historique d'entretien — et l'**état de fonctionnement constaté**.

Envoyer une équipe technique inspecter un point d'eau coûte cher. L'objectif est de
prédire l'état d'un ouvrage à partir des données d'inventaire, pour **prioriser les
tournées de maintenance** sans avoir à tout inspecter.

La cible a trois modalités :

| Classe | Part | Enjeu opérationnel |
|---|---|---|
| `fonctionnel` | 54,8 % | Rien à faire |
| `en panne` | 37,2 % | Réhabilitation lourde, coûteuse |
| `fonctionnel a reparer` | **8,0 %** | **Une intervention légère évite la panne totale** |

> ️ Les données sont **synthétiques**. La structure du problème s'inspire des
> inventaires nationaux de points d'eau ; les noms de communes sont réels mais les
> coordonnées sont des tirages aléatoires autour du centre de chaque département.
> Aucune conclusion réelle ne doit en être tirée.

---

## 2. Objectifs pédagogiques

1. Nettoyer un inventaire de terrain : sentinelles, erreurs d'unité, incohérences
   physiques, libellés non harmonisés.
2. **Identifier une fuite de données** dans un problème multi-classes.
3. Comprendre pourquoi **l'exactitude (*accuracy*) est une métrique trompeuse** et
   savoir choisir entre F1 macro, F1 pondéré et exactitude équilibrée.
4. Traiter une **classe minoritaire à 8 %** dans un cadre multi-classes.
5. Gérer des variables **catégorielles à haute cardinalité** (76 communes, 52 installateurs).
6. Lire une **matrice de confusion 3×3** et en tirer des conclusions opérationnelles.
7. Traduire un modèle en **règle de priorisation** utile à un service technique.

---

## 3. Le piège central : la classe majoritaire

Un modèle qui prédit **toujours** `fonctionnel` obtient :

- **exactitude = 54,8 %** — ce qui a l'air d'un résultat ;
- **F1 macro = 0,236** — ce qui révèle qu'il est inutile.

Pire : les modèles les plus performants en exactitude ont tendance à **ne jamais prédire
la classe minoritaire**. Sur ce jeu de données, une forêt aléatoire sans pondération
atteint environ 75 % d'exactitude en n'annonçant *jamais* `fonctionnel a reparer` — donc
en ratant exactement les ouvrages qu'il était le plus rentable de détecter.

**Conséquence pour votre travail : l'exactitude ne doit pas être votre métrique de
sélection.** Choisissez `f1_macro` ou `balanced_accuracy`, justifiez votre choix, et
rapportez **systématiquement le F1 par classe**, pas seulement la moyenne.

## 4. Le second piège : la fuite de données

Trois colonnes sont renseignées après le constat d'état :
`nb_jours_arret_12_mois`, `intervention_prevue`, `cout_reparation_estime_fcfa`.

Avec elles, tous les modèles atteignent **100 % d'exactitude**. Le modèle est parfait et
sans aucune valeur : si vous connaissez déjà le devis de réparation, vous n'avez plus
besoin de prédire la panne.

**Expérience demandée** : entraînez le même modèle avec puis sans ces colonnes, et
documentez l'écart (100 % contre environ 73 %).

Attention au faux ami : `nb_pannes_12_mois` **n'est pas** une fuite. C'est un historique
de maintenance connu avant l'inspection, donc une variable légitime. Savoir distinguer
les deux cas est précisément la compétence évaluée — expliquez votre raisonnement
colonne par colonne dans votre rendu.

**Toute soumission dont le modèle final utilise les trois colonnes post-constat est
notée 0 sur la partie modélisation.**

---

## 5. Consignes

### Partie A — Audit et nettoyage (4 points)

- A1. Tableau des valeurs manquantes (7 colonnes concernées, jusqu'à 14,6 %).
- A2. Doublons exacts (65 lignes) : détecter et traiter.
- A3. **Sentinelles** — il y en a quatre à trouver :
  `latitude`/`longitude` à `0.0` (214 lignes), `annee_construction` à `0` (168 lignes),
  `population_desservie` à `0` (95 lignes), `debit_essai_m3_h` à `-1` (64 lignes).
  Que se passe-t-il si vous tracez une carte sans les traiter ? Si vous calculez un âge
  moyen ?
- A4. **Erreur d'unité** : 106 profondeurs saisies en centimètres. Détectez-les par un
  raisonnement sur les ordres de grandeur, pas en codant un seuil arbitraire sans le
  justifier.
- A5. **Incohérence physique** : 78 lignes où `niveau_statique_m` dépasse
  `profondeur_forage_m`. Une nappe ne peut pas être plus profonde que le forage qui la
  capte. Que faites-vous de ces lignes ?
- A6. Harmonisez les libellés : casse des communes, espaces parasites sur les
  départements, variantes `india mark 2` / `India Mark II`, `Immergee Solaire `.
  Combien de modalités réelles pour `type_pompe` après nettoyage ?
- A7. **Les valeurs manquantes sont-elles informatives ?** Créez des indicateurs
  binaires (`coordonnees_manquantes`, `annee_inconnue`) et vérifiez si le taux de panne
  y diffère. Un point d'eau dont personne n'a relevé les coordonnées est peut-être un
  point d'eau que personne ne suit.

### Partie B — Analyse exploratoire (4 points)

- B1. Distribution des trois classes. Calculez les métriques d'un modèle trivial
  (« toujours `fonctionnel` ») : c'est votre plancher.
- B2. Taux de panne par `type_pompe`, `mode_gestion`, `mode_paiement`, `qualite_eau`,
  `departement`. Quels facteurs ressortent ?
- B3. Taux de panne en fonction de l'**âge** de l'ouvrage, par tranches. La dégradation
  est-elle linéaire ?
- B4. **Exercice central** — comparez les profils de `en panne` et de
  `fonctionnel a reparer` **entre eux**, en ignorant la classe `fonctionnel`. Ces deux
  groupes d'ouvrages dégradés se ressemblent beaucoup sur les caractéristiques
  techniques (âge, pompe, qualité de l'eau). **Ils se distinguent sur un autre type de
  variables : lesquelles ?** Trouvez-les, et vous saurez comment attaquer la classe
  minoritaire. Construisez ensuite une variable `capacite_entretien` qui les combine.
- B5. Analyse spatiale : taux de panne par commune, sur une carte
  (`st.map`, `folium` ou un simple nuage de points latitude/longitude). Y a-t-il des
  regroupements géographiques ?
- B6. `installateur` : le taux de panne varie-t-il d'une entreprise à l'autre plus que
  ne l'expliquerait le hasard ? Attention aux entreprises à faible effectif.
- B7. Quatre graphiques minimum, chacun avec une phrase d'interprétation.

### Partie C — Préparation des variables (3 points)

- C1. Créez au moins **six** variables dérivées, dont `age_ans`, `pression_usage`
  (population par point d'eau du village, rapportée à la norme de 300 personnes),
  `ratio_niveau_statique`, votre `capacite_entretien` de la partie B4, et les indicateurs
  de valeur manquante de A7.
- C2. **Haute cardinalité** : `commune` (76 modalités) et `installateur` (52 modalités).
  Testez et comparez **au moins deux** stratégies : encodage one-hot brut, regroupement
  des modalités rares dans « Autre », encodage par fréquence, encodage par la cible
  (*target encoding*) **calculé dans la validation croisée**, ou exclusion pure et
  simple. Documentez l'effet de chaque stratégie sur le F1 macro.
  ⚠️ Un encodage par la cible calculé sur l'ensemble des données avant la découpe est une
  fuite : il vous fera gagner en validation et perdre en test.
- C3. Justifiez le traitement de `departement` et `commune` ensemble : la commune contient
  déjà l'information du département. Faut-il les deux ?

### Partie D — Modélisation (5 points)

- D1. Découpe **stratifiée** train/test (80/20). Pourquoi la stratification est-elle
  indispensable ici, et pas seulement souhaitable ?
- D2. Établissez la **référence triviale** (`DummyClassifier(strategy="most_frequent")`) :
  exactitude 0,548 et F1 macro 0,236. Tout doit être comparé à ça.
- D3. Construisez un `Pipeline` avec `ColumnTransformer` : imputation et encodage appris
  **à l'intérieur**, jamais sur le jeu complet avant la découpe.
- D4. Comparez au moins **trois familles** de modèles : régression logistique
  multinomiale, forêt aléatoire, *gradient boosting*.
- D5. **Traitez le déséquilibre** — comparez au moins trois approches :
  `class_weight="balanced"`, poids de classes définis manuellement, sur-échantillonnage
  de la classe minoritaire, SMOTE (`imbalanced-learn`). Mesurez l'effet sur le F1 de
  **chaque** classe, pas seulement sur la moyenne.
- D6. Validation croisée stratifiée 5 blocs, scoring `f1_macro`. Rapportez moyenne
  **et écart-type** : avec 8 % de minoritaires, la variance entre blocs n'est pas
  négligeable.
- D7. Réglage des hyperparamètres du meilleur modèle, grille documentée.
- D8. **Bonus conceptuel** : les trois classes sont naturellement ordonnées
  (fonctionnel > à réparer > en panne). Testez une approche ordinale — deux
  classifieurs binaires en cascade — et comparez-la à la classification multi-classes
  directe.

### Partie E — Évaluation et exploitation (4 points)

- E1. **Matrice de confusion 3×3**, en effectifs et en pourcentages par ligne.
  Quelle confusion est la plus fréquente ? Laquelle est la plus coûteuse pour le service
  technique ?
- E2. Rapport de classification complet : précision, rappel et F1 **par classe**, plus
  F1 macro, F1 pondéré et exactitude équilibrée. Expliquez ce que chaque agrégat cache.
- E3. **Discussion attendue** : votre meilleur modèle en exactitude est-il le même que
  votre meilleur modèle en F1 macro ? Si oui, montrez-le ; si non — c'est le cas le plus
  probable — expliquez lequel vous retenez et pourquoi.
- E4. Courbes ROC et précision-rappel en *une-contre-toutes* pour les trois classes.
- E5. Importance par permutation avec `scoring="f1_macro"`. Vos variables de gouvernance
  ressortent-elles ? Bonus : SHAP.
- E6. **Exploitation opérationnelle** : le service peut inspecter **500 points d'eau**.
  Construisez la liste des 500 à visiter en priorité à partir des probabilités prédites,
  et estimez combien d'ouvrages `fonctionnel a reparer` vous détectez, comparé à un
  tirage au hasard de 500 ouvrages. C'est le seul chiffre qui intéressera votre
  commanditaire.
- E7. **Équité géographique** : la performance du modèle est-elle homogène entre
  départements ? Un modèle qui ne marche que dans les zones bien documentées aggrave
  l'inégalité d'accès au service.

### Partie F — Application Streamlit (3 points)

Construisez un tableau de bord destiné à un service technique. Sérialisez votre pipeline
(`joblib.dump`) et chargez-le avec `@st.cache_resource`.

Attendu au minimum :

- F1. un onglet **exploration** : état des ouvrages par département, par type de pompe,
  par mode de gestion ;
- F2. une **carte** des points d'eau colorée par état prédit ou constaté ;
- F3. un onglet **performance** : matrice de confusion, F1 par classe, importance des
  variables ;
- F4. une **liste de priorisation** : les *N* ouvrages à inspecter en priorité, filtrable
  par département, exportable en CSV.

Bonus valorisés : diagnostic d'un ouvrage saisi au formulaire avec les trois
probabilités, comparaison de deux modèles côte à côte, filtre par commune.

### Partie G — Livraison GitHub (3 points)

- G1. Arborescence claire (`data/`, `src/`, `notebooks/`, `app/`, `models/`, `reports/`).
- G2. `README.md` : problème, données, démarche, **tableau de résultats avec le F1 par
  classe**, capture d'écran de l'app, installation, lancement, limites.
- G3. `requirements.txt` avec versions, `.gitignore`, commits atomiques.
- G4. `random_state` fixé partout ; exécution possible depuis un clone propre.
- G5. Section **« Limites et éthique »** : que se passe-t-il si les tournées de
  maintenance sont priorisées par ce modèle et qu'il sous-performe systématiquement dans
  les zones où les données sont les plus incomplètes ? Quelle supervision humaine ?

---

## 6. Barème (/26, ramené sur 20)

| Partie | Points |
|---|---|
| A — Audit et nettoyage | 4 |
| B — Analyse exploratoire | 4 |
| C — Préparation des variables | 3 |
| D — Modélisation | 5 |
| E — Évaluation et exploitation | 4 |
| F — Application Streamlit | 3 |
| G — Livraison GitHub | 3 |
| **Total** | **26** |

**Pénalités :** colonnes post-constat utilisées → 0 à la partie D.
Sélection du modèle sur l'exactitude seule → −3.
Absence de F1 par classe → −2.
Encodage par la cible calculé avant la découpe → −3.
Prétraitement appris hors pipeline → −3.

---

## 7. Performances de référence

Mesurées lors de la conception du jeu de données : découpe stratifiée 80/20, colonnes de
fuite retirées, imputation (médiane / mode) et encodage one-hot appris dans le pipeline,
`installateur` exclu.

| Modèle | Exactitude | F1 macro | Exactitude équilibrée | **F1 sur « à réparer »** |
|---|---|---|---|---|
| Toujours `fonctionnel` | 0,548 | 0,236 | 0,333 | 0,000 |
| Régression logistique (`balanced`) | 0,682 | **0,609** | **0,667** | **0,364** |
| Forêt aléatoire (`balanced_subsample`) | **0,734** | 0,602 | 0,592 | 0,288 |
| HistGradientBoosting (sans pondération) | 0,733 | 0,588 | 0,575 | 0,252 |

**Lisez ce tableau attentivement : la forêt aléatoire gagne en exactitude (0,734 contre
0,682) et perd sur tout ce qui compte.** La régression logistique pondérée détecte
nettement mieux la classe utile. C'est le résultat central du TP — commentez-le dans
votre README.

À noter, contrairement au TP 2 : ici l'ingénierie de variables apporte peu (environ
0,01 de F1 macro). Le gain vient du **choix de la métrique et du traitement du
déséquilibre**. Chaque jeu de données a son levier ; une partie du travail consiste à
trouver lequel.

**Vos objectifs :**

- F1 macro **≥ 0,62** (facile à atteindre en soignant la pondération) ;
- F1 sur `fonctionnel a reparer` **≥ 0,40** — c'est le vrai défi ;
- exactitude ≥ 0,70 **maintenue en même temps** que les deux précédents.

Une exactitude supérieure à 0,90 est le signe presque certain d'une fuite : vérifiez vos
colonnes.

---

## 8. Livrables attendus

| Fichier | Contenu |
|---|---|
| `notebooks/01_exploration.ipynb` | Parties A et B, graphiques commentés |
| `src/preprocessing.py` | Nettoyage et variables dérivées, réutilisable |
| `src/train.py` | Entraînement, validation croisée, comparaison des stratégies de déséquilibre |
| `src/evaluate.py` | Matrice de confusion, F1 par classe, liste de priorisation |
| `app/streamlit_app.py` | Le tableau de bord |
| `models/` | Pipeline sérialisé |
| `README.md` | Avec le tableau de résultats détaillé par classe |
| `requirements.txt` | Versions figées |

Environnement minimal : `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `joblib`,
`streamlit`. Optionnels : `imbalanced-learn` (SMOTE), `xgboost`, `lightgbm`, `shap`,
`folium` / `streamlit-folium` pour la carte.

## 9. Bonus (+2 max)

- **Analyse coût-bénéfice** : une réparation légère coûte de l'ordre de 90 000 FCFA,
  une réhabilitation lourde plus d'un million. Construisez une matrice de coûts de
  confusion 3×3 et optimisez les seuils de décision pour minimiser le coût total plutôt
  que maximiser le F1.
- Classification **ordinale** (cascade de deux classifieurs binaires) comparée à
  l'approche directe.
- Modèle par département, ou variable régionale hiérarchique.
- Auto-corrélation spatiale : le voisin le plus proche d'un ouvrage en panne est-il plus
  souvent en panne ? Construisez une variable de voisinage à partir des coordonnées —
  attention à ne l'utiliser qu'avec les données d'entraînement.
- Calibration des probabilités multi-classes (`CalibratedClassifierCV`) : indispensable
  si votre liste de priorisation repose sur les probabilités prédites.
- SHAP, MLflow, API FastAPI, `Dockerfile`.
