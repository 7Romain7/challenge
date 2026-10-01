# Challenge 1 — résultats bruts, toutes méthodes (v2)

> Synthèse et figures commentées : [`../README.md`](../README.md). Dans la synthèse, **M5_min** est appelé « régression logistique » et **M2_matched** « filtre adapté ».

Cinq détecteurs légers, sans torch, sur CPU, et leurs ablations. Le code est dans [`challenge1/detection/baselines.py`](../detection/baselines.py).

```bash
uv run python -m detection.baselines                     # ~4 min (le 1er run génère les jeux figés, ~15 min, une seule fois)
uv run python -m detection.baselines --select dice --out challenge1/results/selection_dice
uv run python challenge1/results/make_figs.py
uv run python -m detection.baselines --quick             # smoke test ~40 s
```

## Ce qui change par rapport à la v1 ([`../archive/v1_methodes_faciles/`](../archive/v1_methodes_faciles/README.md))

La v1 avait trois défauts :
- son test (seed 2025) a été **regardé**, il est donc brûlé ;
- son seuil était choisi par tol-F1 seul ;
- son OOD n'était qu'un léger ajout de bruit.

La v2 ajoute les garde-fous suivants.

| # | garde-fou | piège évité |
|---|---|---|
| S1 | fit sur `data/train` (seed 0) ; val, test et OOD sur les seeds de `challenge1/detection/data_gen.py` (1e7+k, 2e7+k, 3e7+…) ; images hachées pour vérifier qu'aucune n'apparaît dans deux splits | fuite entre splits, test déjà vu |
| S2 | un seul paramètre libre par méthode, le seuil, choisi **sur val** par (obj F1 + tol F1)/2 ; tous les autres hyperparamètres sont fixés a priori par la physique ; alerte si le seuil tombe en bord de grille | sur-réglage sur val ; blobs gras qui trichent obj F1 ; largeur du masque qui triche les métriques pixel |
| S3 | test, OOD et null scorés une seule fois avec le seuil figé ; les labels du stage 2 viennent des internes du simulateur et ne servent **qu'à l'évaluation** | sélection sur test ou sur OOD ; usage de l'état caché |
| S4 | IC 95 % par bootstrap sur les images, bootstrap apparié contre M2, 3 seeds de fit pour M5 | différences non significatives |
| S5 | ratio de largeur, blobs par stick trouvé, fausses alarmes sur des scènes sans sticks | triche sur les métriques |
| S6 | parité exacte de nos comptes objet avec `detection.metrics.object_scores` | bug de métrique |

Les variantes de M5 sont fixées avant de voir les résultats :
- **M5_logreg** : sans le pixel brut, choisie a priori ;
- **M5_full** : avec le pixel brut, donc avec un passe-haut z − s1 ;
- **M5_min** : matched + s2 seulement.

## Résultats (test, 400 scènes)

Le tableau complet est dans [`baselines.md`](baselines.md). Les mêmes métriques avec le seuil choisi par le Dice sont dans [`selection_dice/baselines.md`](selection_dice/baselines.md).

| méthode | obj F1 [IC 95 %] | tol F1 | Dice global | Dice / image | frames stage 2 (obj F1) |
|---|---|---|---|---|---|
| M1 smooth | 0,868 [0,853 ; 0,881] | 0,723 | 0,283 | 0,323 | 0,37 |
| M2 matched | 0,863 [0,845 ; 0,881] | 0,868 | 0,430 | 0,431 | 0,14 |
| M3 ridge | 0,868 [0,854 ; 0,881] | 0,742 | 0,320 | 0,361 | 0,34 |
| M4 hyst | 0,865 [0,850 ; 0,878] | 0,647 | 0,213 | 0,238 | 0,26 |
| M5 logreg | 0,865 [0,850 ; 0,880] | 0,916 | 0,726 | 0,697 | 0,57 |
| M5 full | 0,904 [0,887 ; 0,919] | 0,920 | **0,892** | **0,847** | 0,33 |
| M5 min | **0,924** [0,909 ; 0,937] | **0,954** | 0,555 | 0,542 | **0,62** |

## Lecture

1. **En obj F1, les filtres M1 à M4 sont indiscernables** : tous sont autour de 0,865, et l'IC de la différence avec M2 contient 0. Ils trouvent les mêmes interdots. Seule la largeur de leurs blobs diffère : le ratio de largeur va de 2,9 à 7,5.
2. **Le Dice récompense surtout la copie de la largeur du masque.** M5_full a le meilleur Dice (0,89, ratio de largeur 0,95) mais s'effondre sur les frames du stage 2 (0,33). Son passe-haut est réglé sur le flou du simulateur.
3. **Choisir le seuil par le Dice dégrade la détection.** Les obj F1 des filtres tombent de 0,86 à 0,49–0,75, et le stage 2 passe sous 0,12 pour toutes les méthodes sauf M5_full (`fig3_metriques.png`). Le Dice ne doit donc pas servir de critère de sélection.
4. **M5_min est meilleur en obj F1 et en tol F1, et on aurait pu le choisir sans regarder le test** : il est aussi le meilleur sur val (sel 0,933). Mais il produit 0,28 fausse alarme par scène vide, contre 0 pour les filtres.
5. **La vraie limite est le transfert au stage 2.** Le seuil appris sur U(1, 33) ne convient plus à |i| ≈ 3. Il faudra recalibrer le seuil sur des frames du stage 2, sans utiliser leurs labels, par exemple via le taux de fausses alarmes sur des zones vides.

## Figures

- `fig1_masques_test.png` : image, masque du générateur, puis le masque binaire de chaque méthode.
- `fig2_masques_stage2.png` : la même chose sur des frames du challenge 2.
- `fig3_metriques.png` : obj F1, tol F1 et Dice avec leurs IC, pour les deux critères de seuil.
- `fig4_robustesse.png` : obj F1 sur chaque jeu décalé.
- `fig5_rappel_amplitude.png` : la limite de détection en fonction de |i|.

## Limites

- Tous les décalages sont **simulés**.
- Les jeux OOD font 150 scènes : leurs IC, non calculés, sont plus larges.
- Le ms/img inclut toute la pile de features, partagée entre méthodes : c'est une borne haute.
- `csd.new_experiment` plante sur les dispositifs tirés sans aucun stick. Ces seeds sont sautées et listées dans `data/eval_light/ood_stage2/meta.json`.
