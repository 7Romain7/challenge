# Challenge 1 : détecter les interdots

**Tâche.** Une image de CSD (150 × 150 px) en entrée, un masque binaire des pixels d'interdot en sortie.

**Réponse courte.** Une **régression logistique** sur deux cartes de features (filtre adapté et lissage large) : trois poids et un seuil, sans GPU ni torch, entraînée en moins d'une minute sur CPU. Sur le test, elle trouve **92 % des interdots** (obj F1 0,924) contre 0,863 pour le meilleur filtre classique, le **filtre adapté**. L'écart est significatif. Elle reste aussi la seule méthode utilisable sur les frames bruitées du challenge 2.

| | Régression logistique | Filtre adapté |
|---|---|---|
| obj F1 (test) [IC 95 %] | **0,924** [0,909 ; 0,937] | 0,863 [0,845 ; 0,881] |
| tol F1 (pixel ±1 px) | **0,954** | 0,868 |
| Dice global | **0,555** | 0,430 |
| précision / rappel objet | **0,94 / 0,90** | 0,90 / 0,83 |
| obj F1 sur les frames du challenge 2 | **0,62** | 0,14 |
| fausses alarmes par scène vide | 0,28 | **0** |
| paramètres appris | 3 poids + 1 seuil | 1 seuil |

![Pipeline](figures/fig1_pipeline_reglog.png)

---

## 1. Démarche

On s'est fixé une règle : n'utiliser que ce qu'un expérimentateur voit, c'est-à-dire l'image. L'état caché du simulateur sert uniquement à **noter** les détecteurs. On a donc commencé par des filtres classiques, dont chaque hypothèse se justifie physiquement, avant d'ajouter un peu d'apprentissage, et seulement là où il apporte quelque chose.

### Prétraitement commun

Toutes les méthodes reçoivent la même image normalisée :

1. On soustrait la médiane de chaque ligne. Cela enlève les stries horizontales de l'axe de scan rapide, et comme les interdots couvrent peu de pixels, la médiane estime bien le fond.
2. On inverse le signe : les interdots, qui sont des creux, deviennent des pics.
3. On divise par le MAD × 1,4826. L'image est alors en **unités de σ du bruit**. Un z-score ferait l'affaire en apparence, mais les sticks brillants le gonfleraient et le SNR absolu serait perdu. Avec le MAD, un même seuil se transporte d'une scène à l'autre.

### Les détecteurs essayés

Chaque méthode produit une carte de score en σ, puis applique un seuil. Les a priori se cumulent d'une méthode à la suivante. Le détail est dans [`detection/README.md`](detection/README.md).

| | méthode | a priori |
|---|---|---|
| M1 | lissage gaussien | aucun : un interdot est un excès local |
| **M2** | **filtre adapté** | **forme : un bâtonnet court orienté vers π/4 ± 0,2** |
| M3 | crête hessienne | géométrie : une vallée fine |
| M4 | hystérésis | connexité |
| **M5** | **régression logistique** | **combinaison linéaire apprise de ces cartes** |

Trois variantes de M5 ont été fixées **avant** de regarder les résultats : 6 features sans le pixel brut, 7 avec le pixel brut, ou 2 seulement (matched + s2). La version retenue, appelée ici simplement **régression logistique**, est la plus petite : `M5_min` dans le code et dans les tableaux bruts.

### Pourquoi comparer au filtre adapté

Parmi les filtres sans apprentissage, M1 à M4 ont des obj F1 statistiquement indiscernables (≈ 0,865, l'IC de leur différence contient 0). Pour les départager, on prend le critère de sélection du protocole, (obj F1 + tol F1)/2 sur val : le **filtre adapté** arrive en tête (0,845, contre 0,79 au plus pour les autres). C'est aussi la référence du bootstrap apparié. La comparaison reste honnête, puisque la régression logistique **contient** le filtre adapté et ne lui ajoute qu'une seule feature.

---

## 2. Ce que la régression logistique apprend

Le modèle tient en une ligne (features standardisées, fit sur 500 images de `data/train`, seed 0) :

```
logit = +8,01 · matched  −4,57 · s2  −4,96      pixel d'interdot si logit > −1,35
```

- **matched** est la réponse du filtre adapté : *« ça ressemble à un bâtonnet orienté »*.
- **s2** est l'image lissée à σ = 2 px : *« il y a beaucoup de signal dans un grand voisinage »*.

Le poids de s2 est **négatif**. À réponse de filtre égale, le modèle se méfie des zones où il y a trop de signal à grande échelle, c'est-à-dire des **lignes de charge**. Elles sont longues et brillantes et répondent au filtre adapté, mais elles ne sont pas des interdots. La figure ci-dessous le montre : le filtre adapté coupe à la verticale (`matched > 8,74`), alors que la régression logistique coupe en oblique. Elle accepte des interdots faibles (matched ≈ 5) tant que s2 reste bas, et rejette le nuage gris du haut, qui vient des lignes de charge.

![Frontière de décision](figures/fig4_frontiere_decision.png)

---

## 3. Masques prédits

Couleurs : **vert** = pixel prédit correct à ±1 px, **rose** = fausse alarme, **jaune** = pixel d'interdot manqué.

![Masques test](figures/fig2_masques_test.png)

Ce qu'on voit :

- Les deux méthodes trouvent presque tous les interdots isolés.
- Le filtre adapté s'allume en **rose** sur les lignes de charge (scènes #7, #21, #30). La régression logistique les rejette en grande partie, grâce au poids négatif de s2.
- Le Dice reste autour de 0,5 même quand le résultat est visuellement très bon : on explique pourquoi en section 4.

Sur des frames du challenge 2, où le contraste est faible (|i| ≈ 3σ), l'écart devient énorme. Le filtre adapté ne voit presque plus rien, tandis que la régression logistique en retrouve la moitié avec le même seuil :

![Masques challenge 2](figures/fig2b_masques_challenge2.png)

---

## 4. Les métriques, et pourquoi on n'en garde pas qu'une

Le masque officiel vient d'un rectangle flou de 1 à 2 px seuillé à 0,5. Il est donc **fragmenté** (environ 2,5 morceaux par interdot) et sa largeur est arbitraire. Une métrique pixel stricte mesure surtout la chance de rastérisation. On rapporte donc trois niveaux :

![Métriques expliquées](figures/fig3_metriques_expliquees.png)

| métrique | définition | question à laquelle elle répond |
|---|---|---|
| **Dice** (= F1 pixel strict) | 2·\|P∩V\| / (\|P\|+\|V\|), sans tolérance. *Global* : pixels cumulés sur tout le jeu. *Par image* : moyenne des Dice de chaque image | ai-je copié le masque au pixel près ? |
| **tol F1** | précision : part des pixels prédits ayant un pixel vrai dans leur voisinage 3×3 ; rappel : l'inverse ; F1 = moyenne harmonique | ai-je dessiné le trait au bon endroit, à 1 px près ? |
| **obj F1** | rappel : un interdot est *trouvé* si un pixel prédit tombe à moins de len/2 + 1 px de son centre ; précision : un blob prédit (fragments fusionnés par dilatation 3×3) est *correct* si son centroïde est à moins de len/2 + 2 px d'un interdot | **ai-je trouvé les interdots ?** C'est ce qui compte pour l'expérimentateur |

Garde-fous complémentaires :

- **ratio de largeur** = pixels prédits / pixels vrais. Il détecte un masque trop gras qui gonflerait le rappel pixel.
- **blobs par interdot trouvé**. Il détecte une détection fragmentée qui gonflerait la précision objet.
- **fausses alarmes sur des scènes vides** (sans interdot). C'est le taux de faux positifs pur.
- **IC 95 % par bootstrap** sur les images, et bootstrap **apparié** contre le filtre adapté pour savoir si un écart est réel.

**Pourquoi on ne sélectionne pas sur le Dice.** Dans les variantes, le meilleur Dice (0,89, `M5_full`) appartient au modèle qui copie le mieux la largeur du masque. C'est aussi celui qui s'effondre au challenge 2 (0,33). Choisir le seuil par le Dice fait chuter l'obj F1 des filtres de 0,86 à 0,5–0,75 ([`results/selection_dice/`](results/selection_dice/baselines.md)). Le seuil est donc choisi par **(obj F1 + tol F1)/2 sur val**, et le Dice n'est rapporté qu'à titre indicatif.

![Métriques test](figures/fig5_metriques.png)

---

## 5. Robustesse et limite de détection

Le seuil choisi sur val est **figé**, puis appliqué tel quel à des jeux décalés (orientation, bruit, zoom, frames du challenge 2). Ces jeux servent uniquement au rapport, jamais à une sélection.

![Robustesse](figures/fig6_robustesse.png)

La régression logistique est au-dessus ou à égalité partout. La seule exception est le zoom fin à 1 mV, où les interdots deviennent plus gros que le gabarit du filtre. L'écart le plus net concerne les frames du challenge 2 : **0,62 contre 0,14**.

![Rappel vs amplitude](figures/fig7_rappel_amplitude.png)

La limite de détection apparaît clairement. En dessous de 2σ, aucune méthode ne voit rien, ce qui est attendu. Entre 2σ et 8σ, la régression logistique récupère beaucoup plus d'interdots (0,86 contre 0,55 sur [4, 8)). C'est précisément le régime du challenge 2.

---

## 6. Vers le challenge 2 : le seuil doit être recalibré

Le seuil appris sur des images contrastées (|i| ~ U(1, 33)) n'est plus optimal sur les frames du challenge 2 (|i| ≈ 3). Un balayage du seuil sur le logit le montre ([`../challenge2/transfer_m5/`](../challenge2/transfer_m5/sweep.py)) :

![Transfert du seuil](figures/fig8_transfert_seuil.png)

- Au challenge 1, le plateau est large, de −1,5 à +1. Le seuil −1,35 est bon.
- Au challenge 2, le maximum (0,66) est à −1,75, et l'obj F1 retombe vite au-delà.
- On ne peut pas régler ce seuil avec les labels du challenge 2. On le fixe donc par le **taux de fausses alarmes sur des scènes vides**, une quantité mesurable sur une vraie puce : −2,0 donne ≈ 4,5 blobs parasites par scène et −2,5 en donne ≈ 19. Ce sont les variantes `bo_m5_fa5` et `bo_m5_fa20` du challenge 2.
- Le fond du logit a la même loi sur tous les jeux (médiane −5,3, σ ≈ 1). Le seuil n'est donc pas « mal calibré » : c'est le SNR des interdots au point de départ du challenge 2 qui limite. Une fois branchée dans l'optimiseur, la régression logistique ne bat pas le filtre du challenge 2. Le détail est dans [`../challenge2/transfer_m5/README.md`](../challenge2/transfer_m5/README.md).

---

## 7. Garde-fous du protocole

| # | garde-fou | piège évité |
|---|---|---|
| S1 | fit sur `data/train` (seed 0) ; val, test et OOD sur des seeds disjointes, vérifiées par hachage des images | fuite entre splits |
| S2 | un seul paramètre libre par méthode, le seuil, choisi **sur val** ; tous les autres hyperparamètres sont fixés a priori par la physique | sur-réglage |
| S3 | test, OOD et scènes vides scorés **une seule fois** ; les labels du simulateur ne servent qu'à l'évaluation | sélection sur le test |
| S4 | IC bootstrap, bootstrap apparié, 3 seeds de fit pour la régression logistique (écart-type < 0,001) | différences non significatives |
| S5 | ratio de largeur, blobs par interdot, fausses alarmes sur scènes vides | triche sur les métriques |
| S6 | parité exacte avec `detection.metrics.object_scores` | bug de métrique |

Le test d'une première version (seed 2025) avait été regardé : il est **brûlé**, et cette version est archivée dans [`archive/v1_methodes_faciles/`](archive/v1_methodes_faciles/README.md).

**Limites.** Tous les décalages sont simulés. Les jeux OOD (150 scènes) n'ont pas d'IC. La régression logistique produit 0,28 blob parasite par scène vide, contre 0 pour les filtres. Les temps (≈ 10 ms/image) incluent toute la pile de features.

---

## Reproduire

Depuis la racine du repo :

```bash
uv sync
uv run python hackathon/starter/stage1_detection/generate_data.py --n 2000 --out data/train
uv run python -m detection.baselines                     # ~4 min ; le 1er run génère data/eval_light (~15 min)
uv run python -m detection.baselines --select dice --out challenge1/results/selection_dice
uv run python -m detection.export_m5                      # fige la régression logistique pour le challenge 2
uv run python challenge1/figures/make_figures.py          # figures de ce README
```

## Contenu du dossier

| chemin | contenu |
|---|---|
| [`detection/`](detection/) | code : `baselines.py` (5 détecteurs, protocole, métriques), `metrics.py`, `export_m5.py`, et la piste transformers (`models.py`, `train.py`, [`PROTOCOL.md`](detection/PROTOCOL.md), pas encore lancée) |
| [`results/`](results/README.md) | résultats bruts des 7 variantes (`baselines.md` / `.json`), paramètres figés `m5_min_params.json`, figures toutes méthodes |
| [`figures/`](figures/) | figures de ce README (régression logistique contre filtre adapté) et le script qui les génère |
| [`archive/`](archive/) | v1 (test brûlé), première démo M5 |

Correspondance des noms : **régression logistique = `M5_min`**, **filtre adapté = `M2_matched`**.
