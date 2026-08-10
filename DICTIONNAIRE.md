# Dictionnaire des données — `points_eau.csv`

11 472 lignes (dont 65 doublons exacts) × 32 colonnes. Une ligne = un point d'eau
inventorié lors d'une campagne de relevé en 2025. Données **synthétiques** générées par
`generateurs/generer_tp3.py`.

## Identification et localisation

| Colonne | Type | Description | Pièges |
|---|---|---|---|
| `id_point_eau` | texte | Identifiant (`PE-XXXXXX`) | Non unique (doublons). À exclure des variables. |
| `date_releve` | texte | Date du relevé de terrain (2025) | **Deux formats mélangés** : `AAAA-MM-JJ` et `JJ/MM/AAAA`. |
| `departement` | catégorie | 11 départements | Espaces parasites en début de chaîne sur 260 lignes. |
| `commune` | catégorie | 76 communes | 420 lignes en majuscules. **Haute cardinalité** : attention au one-hot brut. |
| `latitude` | réel | Latitude décimale | **214 lignes à `0.0`** — coordonnée non relevée, pas le golfe de Guinée. |
| `longitude` | réel | Longitude décimale | Idem, mêmes lignes. |
| `altitude_m` | réel | Altitude en mètres | 6,1 % manquants. |

## Caractéristiques de l'ouvrage

| Colonne | Type | Description | Pièges |
|---|---|---|---|
| `annee_construction` | entier | Année de construction (1982-2024) | **168 lignes à `0`** (année inconnue). Sert à calculer l'âge. |
| `type_ouvrage` | catégorie | Forage équipé PMH, Puits moderne, Poste d'eau autonome, Adduction d'eau villageoise, Puits traditionnel amélioré | — |
| `type_pompe` | catégorie | India Mark II, Vergnet, Volanta, Kardia, Immergée solaire / électrique / thermique, Aucune | Variantes `india mark 2`, `Immergee Solaire ` (150 lignes). Le modèle de pompe est un facteur majeur. |
| `profondeur_forage_m` | réel | Profondeur en mètres (5-165) | **106 lignes saisies en centimètres** (×100). |
| `niveau_statique_m` | réel | Niveau statique de la nappe (m) | 14,6 % manquants. **78 lignes supérieures à la profondeur du forage** : physiquement impossible. |
| `debit_essai_m3_h` | réel | Débit mesuré à l'essai de pompage | 9,2 % manquants, **64 lignes à `-1`**. |
| `qualite_eau` | catégorie | Potable, Ferrugineuse, Saumâtre, Fluorée, Turbide | 3,5 % manquants. Les eaux agressives usent la pompe. |

## Usage et desserte

| Colonne | Type | Description | Pièges |
|---|---|---|---|
| `nb_menages` | entier | Ménages desservis (5-900) | 4,2 % manquants. |
| `population_desservie` | entier | Population desservie | **95 lignes à `0`**. |
| `distance_village_m` | réel | Distance du point d'eau au village (m) | — |
| `nb_points_eau_village` | entier | Nombre de points d'eau dans le village | Sert à calculer la pression d'usage par ouvrage. |

## Gouvernance et entretien

| Colonne | Type | Description | Pièges |
|---|---|---|---|
| `mode_gestion` | catégorie | Comité de gestion villageois, Délégataire privé, Gestion communale, Aucune gestion formelle | Variantes de casse et double espace sur 190 lignes. **Très prédictif.** |
| `mode_paiement` | catégorie | Au volume, Forfait mensuel, Cotisation annuelle, Gratuit | Le paiement au volume finance l'entretien. |
| `cotisation_mensuelle_fcfa` | réel | Recette mensuelle moyenne | 2,6 % manquants ; vaut 0 si gratuit. |
| `technicien_forme_village` | binaire | 1 si un technicien formé réside au village | — |
| `stock_pieces_rechange_commune` | binaire | 1 si des pièces de rechange sont disponibles au chef-lieu | Regardez-la de près lors de l'exploration. |
| `distance_atelier_km` | réel | Distance à l'atelier de réparation le plus proche (1-190) | — |
| `maitre_ouvrage` | catégorie | Commune, État, ONG internationale, Coopération bilatérale, Association villageoise, Privé | — |
| `installateur` | catégorie | Entreprise ayant réalisé l'ouvrage (`ENT-001` … `ENT-052`) | **52 modalités.** Certaines entreprises travaillent mieux que d'autres, mais l'encodage one-hot brut ajoute 52 colonnes : justifiez votre choix (regroupement, encodage par fréquence, encodage par la cible avec validation croisée, ou exclusion). |
| `nb_pannes_12_mois` | entier | Pannes survenues sur les 12 derniers mois | **N'est pas une fuite** : c'est un historique de maintenance connu avant l'inspection. Variable légitime et utile. |
| `mois_depuis_derniere_maintenance` | entier | Mois écoulés depuis la dernière maintenance | 7,8 % manquants. |

## ⛔ Colonnes post-constat — FUITE DE DONNÉES

Ces trois colonnes sont renseignées **après** que l'état de l'ouvrage a été constaté.
Les inclure donne une exactitude de **100 %** et un modèle sans aucune valeur : si vous
disposez déjà du devis de réparation, vous n'avez pas besoin de prédire la panne.

| Colonne | Type | Pourquoi il faut la supprimer |
|---|---|---|
| `nb_jours_arret_12_mois` | réel | Un ouvrage à l'arrêt 300 jours est un ouvrage en panne, par définition. |
| `intervention_prevue` | catégorie | Réhabilitation lourde / Abandon / Aucune : c'est la conséquence directe du diagnostic. |
| `cout_reparation_estime_fcfa` | réel | Vaut exactement 0 pour tous les ouvrages fonctionnels. |

## 🎯 Variable cible

| Colonne | Type | Description |
|---|---|---|
| `etat_fonctionnement` | catégorie | **3 classes** : `fonctionnel` (54,8 %), `en panne` (37,2 %), `fonctionnel a reparer` (**8,0 %**) |

La classe `fonctionnel a reparer` désigne un ouvrage qui débite encore mais nécessite une
intervention. C'est **la classe la plus utile en pratique** — c'est là qu'une réparation
peu coûteuse évite une panne totale — et **la plus difficile à prédire**. Tout l'enjeu du
TP tient là.

---

## Variables dérivées suggérées

| Variable | Formule | Intuition |
|---|---|---|
| `age_ans` | `2025 − annee_construction` | Facteur d'usure principal |
| `pression_usage` | `population_desservie / (nb_points_eau_village × 300)` | Sur-sollicitation par rapport à la norme de 300 personnes par point d'eau |
| `ratio_niveau_statique` | `niveau_statique_m / profondeur_forage_m` | Marge de rabattement disponible |
| `coordonnees_manquantes` | `1` si latitude vaut 0 | Le défaut de relevé est-il informatif en soi ? |
| `annee_inconnue` | `1` si `annee_construction` vaut 0 | Idem |
| `capacite_entretien` | À construire vous-même à partir des variables de gouvernance | Voir la partie B de l'énoncé |

## Relations présentes dans les données

| Facteur | Effet |
|---|---|
| Âge de l'ouvrage | Dégradation, accélérée après une vingtaine d'années |
| Modèle de pompe | Écarts de fiabilité importants entre modèles ; les pompes thermiques sont les plus fragiles |
| Mode de gestion et de paiement | Effet fort : un ouvrage sans gestion formelle et gratuit se dégrade vite |
| Qualité de l'eau | Les eaux saumâtres et ferrugineuses abrègent la vie de la pompe |
| Pression d'usage | Pénalité au-delà de 300 personnes par point d'eau |
| Installateur | Effet réel mais modeste, dispersé sur 52 entreprises |
| Contexte hydrogéologique | Varie selon le département |
| Capacité d'entretien locale | **Détermine si un ouvrage dégradé est « à réparer » plutôt qu'« en panne »** |
