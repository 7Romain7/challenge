# Protocole — détection d'interdots par transformers sur générateur infini

**Question scientifique.** Avec un générateur capable de produire une infinité de CSD
étiquetés, quel biais inductif (localité vs attention globale) détecte le mieux des
interdots sous-pixel, et lequel **survit à un décalage de distribution** (autre
dispositif, autre résolution, autre bruit, frames du challenge 2) ?

On compare 3 transformers qui parcourent l'axe « biais de localité », plus une
référence CNN. Sans cette référence, on ne peut pas justifier « pourquoi un transformer ».

---

## Démarrage rapide (machine GPU, depuis la racine du repo)

| ordre | commande | durée | contrôle avant de passer à la suite |
|---|---|---|---|
| 0 | `uv sync --extra train` | 1–2 min | `torch` est installé |
| 1 | `GPUS="0" WORKERS=32 bash detection/run_protocol.sh data` | ~10 min (CPU) | `verify` : stats du bruit identiques (§6, P0) |
| 2 | `GPUS="0 1 2 3" bash detection/run_protocol.sh sanity` | ~5 min | `grep sanity runs/sanity_*/stdout.log` donne tol_f1 > 0,95 |
| 3 | `GPUS="0 1 2 3" bash detection/run_protocol.sh main` | selon GPU | `val_tol_f1` monte, `geom_gap` < 0,02 |
| 4 | `GPUS="0 1 2 3" bash detection/run_protocol.sh scaling` | selon GPU | — |
| 5 | `GPUS="0 1 2 3" bash detection/run_protocol.sh sandbox` | selon GPU | — |
| 6 | `bash detection/run_protocol.sh eval` | ~10 min | tableaux et figures dans `challenge1/results/transformers/` |

Adapte `GPUS` à `nvidia-smi` et `WORKERS` à `nproc`. Les jobs tournent en `nohup` : tu peux fermer la session SSH. Relancer une phase reprend les runs interrompus. Le détail de chaque phase est au §6.

---

## 1. Les pièges (mesurés sur le générateur, pas supposés)

| # | Piège | Mesure | Parade dans le code |
|---|---|---|---|
| 1 | **Objets sous-pixel.** Un stick fait 4–8 px de long et **0,4–1,6 px de large**. Les positifs ne représentent que **0,35 %** des pixels. | 150×150, pas de 2 mV ; longueur = 0,1–0,2 × 80 mV | Pas de patch 16. Chaque modèle finit par une **tête pleine résolution** nourrie par l'image brute. Perte BCE + Dice. |
| 2 | **Masque officiel fragmenté.** Il est obtenu en seuillant à 0,5 un rectangle de 1 px flouté, donc il se casse en morceaux. | **32 composantes pour 13 sticks**, 2,4 px par composante | L'entraînement utilise des **cibles soft** (le label flou), pas le masque binaire. L'IoU stricte mesure surtout la chance de rastérisation : on rapporte en plus des métriques **tolérantes à 1 px** et **par objet**. |
| 3 | **Générateur lent.** 0,17 s par scène, à cause des boucles Python sur un réseau 100×100. En génération à la volée, le GPU attend. | ~350 scènes/min/cœur | **Pool de géométries** pré-calculé, puis intensité et bruit **retirés à l'infini sur GPU** (voir §2). |
| 4 | **« Infini » ne veut pas dire « pas d'overfit ».** Le pool de géométries est fini, et le simulateur n'est qu'une distribution parmi d'autres. | — | `geom_gap` loggé à chaque éval. La phase P3 réduit le pool (1k, 10k). Des jeux **OOD** testent la robustesse. |
| 5 | **Plancher de SNR.** `i_mean ~ U(-33, -1)` contre un bruit σ = 0,9 en blanc + 0,7 en rayures : certaines scènes sont physiquement indétectables. | — | On ne rapporte jamais seulement une moyenne : la courbe **recall vs amplitude** donne la limite de détection. |
| 6 | **Augmentations non physiques.** Un flip H ou V envoie θ = π/4 sur −π/4, et une rotation de 90° sur 3π/4 : c'est un autre dispositif. | — | On utilise le seul groupe **exact** : {id, transposée, rot180, anti-transposée}. Il préserve θ ≈ π/4 et échange les deux familles de lignes de charge. On l'applique **avant** le bruit, pour que les rayures restent horizontales (axe de scan rapide). |
| 7 | **Normalisation.** Le z-score est gonflé par les sticks brillants et efface le SNR absolu. | — | Normalisation **médiane/MAD** : l'image est exprimée en unités de σ du bruit. |
| 8 | **Fuite de sélection.** Choisir le seuil ou le checkpoint sur test ou sur OOD. | — | Seuil et checkpoint sont choisis sur **val uniquement**. Test et OOD ne sont touchés qu'à la fin. *Les jeux OOD servent à rapporter, jamais à sélectionner.* |
| 9 | **Comparaison injuste entre architectures.** | — | Pour toutes : même nombre de steps, même batch, mêmes données, même éval. **3 seeds**, avec moyenne ± écart-type. On rapporte aussi le nombre de paramètres et le débit. |
| 10 | **Distracteurs.** Les lignes de charge (`p_line` = 0,2, intensité 0,5 × i_mean) ne sont **pas** labellisées. | — | On analyse les faux positifs le long des lignes sur la figure `examples.png`. |
| 11 | **Transfert vers le stage 2.** Les sticks dérivent et sortent du cadre, et le contraste vient des régions. | intensités de −3 à −28, donc dans la plage d'entraînement | Jeu `ood_stage2` : frames de `new_experiment`. Les labels sont tirés des internes du simulateur, **uniquement pour l'évaluation**. |
| 12 | **Le stage 2 vit à faible SNR.** Loin de l'optimum, le contraste vaut `base` = 3, d'où une amplitude médiane d'environ **3,1** sur `ood_stage2` contre 19 sur test. Avec la loi uniforme, seulement ~9 % des scènes ont \|i\| < 4. | test du pipeline : recall ≈ 0 dans le bin [2, 4) | P4b `--intensity-law loguniform` : même support, mais ~40 % des scènes sous \|i\| = 4. C'est précisément le régime où le détecteur sert d'objectif au stage 2. |
| 13 | **Embeddings de position figés** sur une seule taille d'image. | — | Sin-cos 2D (TransUNet, ViT) ou aucun embedding (SegFormer) : le modèle accepte toute résolution. |

---

## 2. Données

**Pool d'entraînement** (`data/pool`, seeds dans [0, 1e7)) : pour chaque géométrie, on stocke
- le **template** : le rendu propre et flouté, divisé par l'intensité de scène, en float16 ;
- le **label soft**, en uint8.

À chaque batch, sur GPU :

```
image = i · T(template) + blur(σ_pix · blanc + σ_h · rayures)     i ~ U(-33, -1)
```

Ce mélange est **exact** parce que le flou est linéaire. On le vérifie avec
`data_gen verify`, qui compare les statistiques du bruit synthétique à celles du résidu
officiel (image − rendu propre) :

```
official : pixel_std 0.809, row_mean_std 0.568, lag1_h 0.619, lag1_v 0.259
synthetic: pixel_std 0.809, row_mean_std 0.568, lag1_h 0.619, lag1_v 0.256
```

Au final, on a N géométries × 4 symétries, avec intensité et bruit infinis. Le seul
overfit possible porte sur les géométries, et la phase P3 le mesure.

**Jeux figés** (`data/eval`), tous rendus par le **générateur officiel** :

| jeu | n | rôle | décalage |
|---|---|---|---|
| `val` | 1000 | sélection du checkpoint et du seuil | aucun |
| `test` | 2000 | chiffre final, mesuré une seule fois | aucun |
| `ood_theta_shift` | 500 | rapport | pente des sticks π/4 + 0,35 (autres bras de levier) |
| `ood_theta_wide` | 500 | rapport | jitter de pente 0,1 → 0,3 |
| `ood_noise_up` | 500 | rapport | bruit × 1,5 |
| `ood_zoom_in` | 500 | rapport | pas de 1 mV (sticks 2× plus grands en pixels) |
| `ood_zoom_out` | 500 | rapport | pas de 3 mV (sticks 1,5× plus petits) |
| `ood_stage2` | 500 | rapport | frames du challenge 2 (dérive, contraste par région) |

Les seeds sont disjointes par construction : val = 1e7 + k, test = 2e7 + k, OOD = 3e7 + 1e6·s + k.

---

## 3. Architectures

| nom | type | biais de localité | params | pourquoi |
|---|---|---|---|---|
| `unet` | CNN | maximal | 1,9 M | référence : un transformer doit la battre pour se justifier |
| `transunet` | encodeur CNN jusqu'à /8, puis 6 blocs d'attention globale (19×19 tokens), puis décodeur CNN avec skips | fort | 5,9 M | les détails sont locaux, mais l'attention globale voit **le réseau** d'interdots (périodicité ≈ 80 mV) |
| `segformer` | Mix Transformer hiérarchique, attention à réduction spatiale, Mix-FFN, sans embedding de position | moyen | 3,5 M | robuste aux changements de résolution (`ood_zoom_*`) |
| `vit` | ViT plat, patch 4 (38×38 tokens), attention globale, décodeur SETR-PUP | minimal | 3,8 M | test du cas où l'on n'impose aucune localité |

Les trois transformers terminent par `FullResHead` : une fusion avec des features
pleine résolution de l'image brute, rendue nécessaire par le piège n°1.

---

## 4. Entraînement (identique pour tous)

| paramètre | valeur |
|---|---|
| optimiseur | AdamW, β = (0,9 ; 0,999), wd 0,05 (pas de wd sur les normes et biais) |
| learning rate | 1e-3 pour unet, 5e-4 pour les transformers ; warmup 1k steps puis cosine jusqu'à 1 % |
| volume | **40k steps × batch 32**, soit 1,28 M images ≈ 11 passages sur 30k géométries × 4 symétries |
| perte | BCE (cibles soft) + Dice global sur le batch |
| précision et stabilité | autocast bf16, gradient clip 1,0 |
| poids évalués | EMA 0,999 ; ce sont eux qu'on évalue et sauvegarde |
| éval | tous les 2k steps, sur val : `val_tol_f1` (critère de sélection), `val_obj_f1`, `geom_gap` |

---

## 5. Métriques

- **Pixel strict** : F1 et IoU contre le masque officiel. C'est ce que demande le challenge, mais il est bruité (piège n°2).
- **Pixel tolérant (`tol_f1`)** : on accepte un décalage de 1 px dans les deux sens. **Critère principal de sélection.**
- **Objet** : un stick est trouvé si un pixel prédit tombe à moins de len/2 + 1 px de son centre. Un blob prédit est correct si son centroïde est à moins de len/2 + 2 px d'un stick. On en tire `obj_precision`, `obj_recall` et `obj_f1`. **C'est la métrique « physicien »**, et celle que le stage 2 utilise.
- **Recall par amplitude** du stick, en bins |i| ∈ [0, 2, 4, 8, 16, ∞) : la limite de détection.
- **Seuil** : choisi sur val pour maximiser `tol_f1`, puis figé pour test et OOD.

---

## 6. Phases

### P0 — Données (CPU, ~15 min avec 32 cœurs)

> **À faire en premier : les données ne sont pas fournies dans le repo.** `data/pool` et `data/eval` n'existent qu'après cette phase.
> Les dossiers `data/train` et `data/val` du starter ne servent **pas** ici : le pipeline a besoin de templates sans bruit, pas d'images bruitées.

**Lancer sur la machine GPU distante, pas en local.** C'est du CPU uniquement : la commande se lance une seule fois, quel que soit le nombre de GPU.
Un PC à 8 cœurs met environ 50 min pour le pool de 30k, puis il faut transférer environ 3 Go.

```bash
uv sync --extra train
```
```bash
GPUS="0" WORKERS=32 bash detection/run_protocol.sh data
```
Fixe `WORKERS` au nombre de cœurs de la machine (`nproc`).

Ce que la commande produit :

| dossier | contenu | taille |
|---|---|---|
| `data/pool` | 30k templates sans bruit + labels soft | ~2,0 Go |
| `data/eval` | val 1000, test 2000, 6 jeux OOD de 500 | ~1 Go |

Le pool de référence fait 30k géométries (`POOL_N`, défaut du script). Essai plus court : `POOL_N=5000 GPUS="0" WORKERS=32 bash detection/run_protocol.sh data`.

**Contrôle** : la commande se termine par `verify`.
- Les statistiques du bruit officiel et synthétique doivent coïncider (`pixel_std` ≈ 0,809).
- La fraction de pixels positifs doit être d'environ 0,0035, sur le pool comme sur val.

Si ce n'est pas le cas, ne lance pas la suite.

### P1 — Sanity (5 min)
```bash
GPUS="0 1 2 3" bash detection/run_protocol.sh sanity
grep "sanity" runs/sanity_*/stdout.log
```
Chaque modèle doit **sur-apprendre un batch fixe**, avec un `tol_f1` supérieur à 0,95 sur ce batch.
La perte ne descend pas à 0, et c'est normal : avec des cibles soft, la BCE et le Dice ont un plancher non nul.
Si le `tol_f1` n'atteint pas 0,95, le pipeline est cassé : il ne faut pas lancer P2. Note aussi le `img_per_s` de chaque architecture pour estimer la durée de P2.

### P2 — Comparaison principale (4 architectures × 3 seeds = 12 runs)
```bash
GPUS="0 1 2 3" bash detection/run_protocol.sh main
```
**À surveiller** dans `runs/*/log.jsonl` :
- `val_tol_f1` doit monter puis plafonner ;
- `geom_gap` doit rester proche de 0 (inférieur à 0,02) ; s'il grandit, il y a overfit des géométries et il faut plus de pool ;
- `gnorm` ne doit pas exploser.

### P3 — Overfit au pool et rôle des symétries (12 runs, seed 0)
```bash
GPUS="0 1 2 3" bash detection/run_protocol.sh scaling
```
On entraîne avec un pool de 1k, puis de 10k, puis de 10k sans symétries, et on compare à 30k (P2).
**Attendu** : `geom_gap` grandit quand le pool rétrécit, et le transformer le moins local (ViT) est le plus affecté.
C'est la réponse chiffrée à « comment éviter l'overfit au générateur ».

### P4 — Sandbox : physique décalée (P4a) et faible SNR (P4b), 24 runs
```bash
GPUS="0 1 2 3" bash detection/run_protocol.sh sandbox
```
On ajoute `--affine --polarity --noise-jitter 0.3` :
- échelle anisotrope et cisaillement, qui représentent d'autres bras de levier et d'autres résolutions ;
- inversion de signe, qui correspond à un capteur sur l'autre flanc du pic de Coulomb ;
- niveau de bruit variable.

**Attendu** : on perd un peu sur `test` mais on gagne sur `ood_*`. Ce compromis est le résultat à montrer.

P4b (`_lowsnr`) change un seul facteur, `--intensity-law loguniform`.
**Attendu** : le recall progresse dans les bins [0, 2) et [2, 4), et `ood_stage2` progresse.
C'est l'argument qui relie le détecteur au challenge 2.
⚠️ Le baseline officiel reste P2 (config par défaut). P4 est une expérience sandbox documentée.

### P5 — Évaluation finale (mesurée une seule fois)
```bash
bash detection/run_protocol.sh eval
```
La commande produit dans `challenge1/results/transformers/` :
- `summary.md`, tableau moyenne ± écart-type par groupe et par jeu ;
- `recall_vs_amplitude.png` ;
- `robustness.png` ;
- `examples.png`.

---

## 7. Ce qu'on montre dans les slides
1. Les pièges 1, 2 et 6, avec une image : un stick de 1 px, le masque fragmenté, et ce que fait un flip à θ.
2. Le pool de géométries plus le bruit exact sur GPU, avec la vérification statistique.
3. Le tableau P2 : est-ce qu'un transformer bat le U-Net, et à quel coût ?
4. La courbe de limite de détection.
5. P3 : la taille du pool contre `geom_gap`, c'est-à-dire l'overfit au générateur rendu mesurable.
6. P4 : le compromis test / OOD, et l'argument du transfert vers une vraie machine.

## 8. Limites connues
- Tous les décalages OOD restent **simulés**. Seules de vraies données diraient si les augmentations couvrent la réalité (bruit 1/f, sauts de charge, dérive lente pendant le scan, tous absents du générateur).
- Le masque officiel (seuil 0,5) sous-estime la largeur réelle : les métriques strictes sont plafonnées.
- `ood_stage2` utilise les internes du simulateur pour l'étiquetage. C'est légitime pour l'évaluation, jamais dans l'algorithme.
