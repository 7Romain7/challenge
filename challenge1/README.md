# Challenge 1 : détecter les interdots

**Tâche.** Une image de CSD (150 × 150 px) en entrée, un masque binaire des pixels d'interdot en sortie.

**Le fil.** Trois étapes, chacune motivée par la limite de la précédente :

1. **Régression logistique** sur deux cartes physiques : simple, robuste, mais plafonne à bas SNR.
2. **U-Net** entraîné sur un générateur infini : bien meilleur dans le simulateur, mais il s'effondre sur des artefacts de mesure qu'il n'a jamais vus.
3. **U-Net + familles d'artefacts**, évalué en **leave-one-family-out** : est-ce qu'apprendre K artefacts protège contre un artefact inconnu ?

| | Filtre adapté | Régression logistique | U-Net (lowsnr) |
|---|---|---|---|
| obj F1 (test, 400 scènes) | 0,863 | 0,924 | **0,979** |
| tol F1 (pixel ±1 px) | 0,868 | 0,954 | **0,989** |
| obj F1 sur les frames du challenge 2 | 0,14 | 0,62 | **0,96** |
| fausses alarmes par scène vide | **0** | 0,28 | 1,61 |
| robustesse aux artefacts (1 = insensible) | 0,934 | **0,935** | 0,872 |
| paramètres appris | 1 seuil | 3 poids + 1 seuil | 1,9 M |

Règles communes : seuil et checkpoint choisis **sur val**, test lu une seule fois, jeux décalés et artefacts **de rapport uniquement**. On sélectionne sur **(obj F1 + tol F1)/2**. Le Dice est rapporté mais jamais utilisé pour choisir, parce qu'il récompense la copie de la largeur arbitraire du masque (§ Métriques).

---

## 1. Régression logistique

Toutes les méthodes reçoivent la même image : on soustrait la médiane de chaque ligne (contre les rayures), on inverse le signe (les interdots deviennent des pics), puis on divise par MAD × 1,4826 (l'image est en **unités de σ du bruit**). On a essayé quatre filtres classiques (lissage, **filtre adapté** à un bâtonnet orienté à π/4, crête hessienne, hystérésis), puis une régression logistique sur leurs cartes. La version retenue est la plus petite (`M5_min`) :

```
logit = +8,01 · matched  −4,57 · s2  −4,96      pixel d'interdot si logit > −1,35
```

Le poids **négatif** du lissage large s2 apprend à rejeter les **lignes de charge**, qui sont longues et brillantes et répondent au filtre adapté. La frontière devient oblique au lieu de verticale :

![Frontière de décision](figures/fig4_frontiere_decision.png)

| test | Régression logistique | Filtre adapté |
|---|---|---|
| obj F1 [IC 95 %] | **0,924** [0,909 ; 0,937] | 0,863 [0,845 ; 0,881] |
| précision / rappel objet | **0,94 / 0,90** | 0,90 / 0,83 |
| frames du challenge 2 | **0,62** | 0,14 |

![Masques test](figures/fig2_masques_test.png)

**Limite.** Entre 2σ et 4σ, le régime du challenge 2, elle ne retrouve que 37 % des interdots. Ses trois paramètres ne permettent pas de faire mieux. Détails : [`detection/README.md`](detection/README.md), [`results/`](results/README.md).

---

## 2. U-Net sur un générateur infini

**Données.** Le flou est linéaire, donc `image = i · template + flou(σ_pix · blanc + σ_h · rayures)`. On pré-calcule 30k géométries (template et label soft), puis on tire **sur GPU** l'intensité et le bruit à chaque batch : aucune image n'est vue deux fois. Le bruit synthétique reproduit celui du générateur officiel (écart-type 0,807, rayures 0,565, autocorrélations identiques). On lit les paramètres publics de `GeneratorConfig`, sans rétro-ingénierie.

**Entraînement.** On apprend le label soft (pas le masque fragmenté), avec une perte BCE + Dice. Seules les symétries exactes sont autorisées, appliquées avant le bruit. EMA, checkpoint et seuil sont choisis sur val. Variante **lowsnr** : |i| est tiré log-uniforme, ce qui donne 40 % de scènes sous 4σ. Le TransUNet (5,9 M) fait exactement pareil que le U-Net (0,979), donc l'attention globale n'apporte rien : on garde le **U-Net**. Pas de mémorisation des géométries : l'écart pool − val reste sous 0,008.

![Masques test DL](figures/fig15_masques_test_dl.png)

Le gain se concentre entre 2σ et 8σ : sur [2, 4), le rappel passe de 0,37 à 0,87–0,89. Sur les frames du challenge 2, l'obj F1 passe de 0,62 à **0,96**.

![Rappel vs amplitude DL](figures/fig14_rappel_amplitude_dl.png)

**Mais** le Dice vaut 1,00 : le réseau a appris la règle de rastérisation du générateur. Premier signe qu'il est **spécialisé**.

---

## 3. Robustesse : U-Net + familles d'artefacts, une famille laissée de côté

### 3.1 La suite d'artefacts

[`detection/robustness.py`](detection/robustness.py) prend les 200 premières scènes du test et dégrade chacune avec **un** artefact de laboratoire, à 3 niveaux. Le générateur officiel ne produit aucun de ces artefacts.

| famille | origine physique | niveaux |
|---|---|---|
| `white` bruit blanc | intégration plus courte | × 0,5 / 1 / 2 σ_pix |
| `pink` bruit 1/f le long du scan | bruit de charge | × 0,5 / 1 / 2 σ_pix |
| `drift` dérive du fond | dérive du point capteur | crête-crête 1,5 / 3 / 6 |
| `jumps` sauts télégraphiques | piège de charge | 1,5 / 3 / 6 |
| `stripes` rayures | gigue ligne à ligne | × 0,5 / 1 / 2 σ_h |
| `lowpass` passe-bas 1 pôle | constante de temps du lock-in | τ = 0,5 / 1 / 2 px |
| `saturate` c·tanh(x/c) | flanc fini du pic de Coulomb | c = 10 / 5 / 2,5 |
| `spikes` pixels aberrants ±10 | glitchs | 0,1 / 0,5 / 2 % |
| `polarity` signe inversé | capteur sur l'autre flanc | — |

![Robustesse par artefact](figures/fig12_robustesse_dl.png)

**Constat.** Le U-Net reste meilleur en absolu sur 6 familles sur 9, mais il **s'effondre** sur ce qu'il n'a jamais vu : 0,1 % de pixels aberrants le fait passer de 0,98 à 0,66, et un bruit blanc ×2 de 0,98 à 0,60, contre 0,85 pour la régression logistique. Sans augmentation, la polarité est fatale (0,04 à 0,62). Score de robustesse : **0,872** pour le U-Net contre **0,935** pour la régression logistique, dont le prétraitement physique encaisse ces cas.

### 3.2 La parade : prétraitement physique + générateur randomisé

Deux changements, sans toucher à l'architecture :

- **`PhysInput`** ([`models.py`](detection/models.py)), une couche sans paramètre placée devant le U-Net, donc appliquée aussi à l'inférence. Elle soustrait la médiane de chaque ligne, renormalise par le MAD et fait un **dé-spiking** : un pixel au-dessus de 6σ dont les 8 voisins restent faibles est remplacé par la médiane 3×3. Mesuré : 90 % des spikes ±10 sont retirés, et seulement 0,4 % des pixels de sticks sont touchés. Le signe reste appris (polarité augmentée).
- **Générateur randomisé** ([`synth.py`](detection/synth.py), `--artifacts`) : les 8 familles ci-dessus sont tirées sur GPU, chacune avec une probabilité de 0,2 et une sévérité uniforme. Les plages couvrent la suite de test et la dépassent légèrement.

### 3.3 Le test honnête : leave-one-family-out

Si on entraîne sur toutes les familles puis qu'on teste sur ces mêmes familles, on mesure seulement l'apprentissage de la suite. Sur une vraie puce, il y aura toujours un artefact qu'on n'a pas prévu. D'où le protocole [`detection/lofo.py`](detection/lofo.py) : pour chaque famille f, on entraîne un U-Net **sur toutes les familles sauf f**, puis on le note **uniquement sur f**. Deux références avec la même recette :

| run | familles vues à l'entraînement | rôle |
|---|---|---|
| `none` | aucune (PhysInput seul) | plancher : ce que donne le prétraitement |
| **`no-f`** | toutes sauf f | **la question : f est inconnue** |
| `all` | toutes, f comprise | plafond : f est dans la distribution |

On lit deux quantités :

- **transfert** = LOFO − none, ce que les autres familles apportent face à une famille inconnue ;
- **écart** = all − LOFO, ce qu'on perd à ne pas avoir vu f.

On fait tout cela à deux capacités, `unet16_robust` (0,5 M) et `unet_robust` (1,9 M), pour voir si un réseau plus petit se spécialise moins. Cela fait 2 × 11 = 22 runs de 8000 steps, avec seuils sur val et suite de robustesse en rapport seulement.

**Statut : runs en cours** sur le cluster. Le tableau `results/lofo/lofo.md` (obj F1 moyen et pire niveau par famille, transfert, écart, test propre) sera produit par `python -m detection.lofo report`.

**Lecture attendue.** Un transfert positif sur les familles « proches » (white ↔ pink, drift ↔ jumps ↔ stripes) et un transfert nul sur les familles orthogonales (spikes, saturate, polarity) voudraient dire que la randomisation généralise par **ressemblance**, pas par principe. Dans ce cas, le garde-fou reste nécessaire : basculer sur la régression logistique quand les statistiques du résidu sortent du domaine d'entraînement.

---

## Métriques

Le masque officiel vient d'un rectangle flou de 1 à 2 px seuillé à 0,5. Il est **fragmenté** (environ 2,5 morceaux par interdot) et sa largeur est arbitraire. On rapporte donc trois niveaux :

| métrique | question |
|---|---|
| **Dice** (pixel strict) | ai-je copié le masque au pixel près ? |
| **tol F1** (voisinage 3×3) | ai-je dessiné le trait au bon endroit, à 1 px près ? |
| **obj F1** (centre à len/2 + 1 px) | **ai-je trouvé les interdots ?** |

Garde-fous complémentaires : ratio de largeur, blobs par interdot, fausses alarmes sur des scènes vides, IC bootstrap et bootstrap apparié, splits disjoints vérifiés par hachage. La sélection par le Dice fait chuter l'obj F1 des filtres à 0,5–0,75 ([`results/selection_dice/`](results/selection_dice/baselines.md)). Le test d'une première version a été regardé : il est brûlé et archivé dans [`archive/`](archive/v1_methodes_faciles/README.md).

**Limites.** Un seul seed par run DL : les écarts de ~0,005 ne sont pas significatifs, alors que les effondrements de robustesse le sont. Tous les artefacts sont synthétiques. Le U-Net produit 1,6 fausse alarme par scène vide : avant de l'utiliser au challenge 2, il faut recalibrer son seuil sur le taux de fausses alarmes, comme pour la régression logistique ([`../challenge2/transfer_m5/`](../challenge2/transfer_m5/README.md)).

---

## Reproduire

```bash
uv sync
uv run python hackathon/starter/stage1_detection/generate_data.py --n 2000 --out data/train
uv run python -m detection.baselines                  # 1. régression logistique et filtres (~4 min CPU)
uv run python -m detection.export_m5
```

Parties 2 et 3 (GPU, `uv sync --extra train`) :

```bash
uv run python -m detection.data_gen pool --n 30000 --out data/pool --workers 32
uv run python -m detection.data_gen sets --out data/eval --workers 32
uv run python -m detection.train --arch unet --seed 0 --steps 14000 --warmup 300 --eval-every 1000 --compile --intensity-law loguniform --out runs/fast_unet_lowsnr
uv run python -m detection.export_dl runs/fast_unet_lowsnr challenge1/models/unet_lowsnr.pt
uv run python -m detection.robustness make                                  # suite d'artefacts
uv run python -m detection.evaluate --runs "runs/fast_unet*" --eval-dir data/robustness --out challenge1/results/robustness
uv run python -m detection.lofo jobs                                        # 22 lignes "nom|args" ; chacune -> bash detection/lofo_job.sh nom args
uv run python -m detection.lofo report --out challenge1/results/lofo
```

```python
from detection.export_dl import load, predict_mask
model, thr = load("challenge1/models/unet_lowsnr.pt", device="cuda")
masks = predict_mask(model, thr, images, device="cuda")   # images : (N, H, W) CSD bruts
```

| chemin | contenu |
|---|---|
| [`detection/`](detection/) | code : `baselines.py` (partie 1), `data_gen.py`, `synth.py`, `models.py`, `train.py`, `evaluate.py` (partie 2), `robustness.py`, `lofo.py` (partie 3), [`PROTOCOL.md`](detection/PROTOCOL.md) |
| [`models/`](models/) | U-Net et TransUNet lowsnr figés (float16 + seuil) |
| [`results/`](results/README.md) | tableaux bruts : `baselines.md`, `dl_p4/`, `robustness/`, `p4_m5/` |
| [`figures/`](figures/) | figures et leurs scripts |
| [`archive/`](archive/) | v1 (test brûlé), première démo |

Correspondance des noms : régression logistique = `M5_min`, filtre adapté = `M2_matched`.
