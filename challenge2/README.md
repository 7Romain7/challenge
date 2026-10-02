# Challenge 2 : trouver le réglage qui maximise le contraste d'un interdot

> **Statut.** Méthode retenue confirmée sur 100 appareils neufs (dev 900–999) puis sur le split **val** (200 appareils × 3 graines). Le split **test** (10000–10199) n'a jamais été lancé : il servira une seule fois, sur un commit gelé.

**La tâche.** Un appareil simulé inconnu a cinq grilles. Les barrières g1, g3, g5 règlent le contraste des interdots et font aussi glisser l'image ; les plungers g2, g4 déplacent la fenêtre de mesure. Il faut rendre le réglage (g1…g5) où un interdot a le plus fort contraste, en mesurant le moins possible et sans jamais lire l'état caché du simulateur.

**Le livrable.**

```python
from csd import new_experiment
from solve import optimize

exp = new_experiment()        # appareil neuf, optimum caché
gates = optimize(exp)         # {"g1": ..., "g2": ..., "g3": ..., "g4": ..., "g5": ...}
```

[`solve.py`](solve.py) lance la méthode retenue avec le budget de référence (1 M pixels, 300 mesures) et renvoie le réglage engagé.

## Sommaire

1. [Résultat principal](#1-résultat-principal)
2. [Mesurer le succès : les métriques et pourquoi celles-là](#2-mesurer-le-succès--les-métriques-et-pourquoi-celles-là)
3. [Progression des architectures](#3-progression-des-architectures)
4. [Robustesse hors du simulateur par défaut](#4-robustesse-hors-du-simulateur-par-défaut)
5. [Que vaut plus de budget ?](#5-que-vaut-plus-de-budget-)
6. [Un run en images](#6-un-run-en-images)
7. [Résultats négatifs](#7-résultats-négatifs)
8. [Protocole, données et reproductibilité](#8-protocole-données-et-reproductibilité)
9. [Limites et suites](#9-limites-et-suites)
10. [Lancer et contenu du dossier](#10-lancer-et-contenu-du-dossier)

---

## 1. Résultat principal

Budget fixé avant toute méthode : **1 M pixels et 300 mesures par appareil**, soit environ 44 photos entières de 150 × 150 px, environ 7 fois moins que la baseline fournie par les organisateurs.

**Val : 200 appareils jamais utilisés pour décider, 3 graines d'algorithme, 600 runs par méthode.**

| méthode | regret médian R | gain récupéré, médiane | à 5 % de l'optimum | à 10 % | ratés (R > 0,3) | ΔR apparié vs avant [IC 95 %] | p (Holm) |
|---|---|---|---|---|---|---|---|
| `bo_roi_dlf` (méthode d'avant) | 0,150 | 85 % | 22 % | 38 % | 22 % | | |
| **`bo_roi_auto_ucb_dlf`** (retenue) | **0,112** | **89 %** | **28 %** | **46 %** | **12 %** | **−0,018 [−0,038 ; −0,010]** | **2·10⁻⁷** |
| `bo_coarse_ucb_dlf` | 0,128 | 87 % | 25 % | 41 % | 13 % | −0,037 [−0,067 ; −0,008] | 1·10⁻⁴ |
| `bo_region_ucb_dlf` | 0,128 | 87 % | 25 % | 42 % | 16 % | −0,017 [−0,027 ; −0,000] | 0,002 |
| `bo_roi_meta_dlf` | 0,145 | 86 % | 26 % | 40 % | 20 % | +0,003 [−0,016 ; +0,011] | 0,43 |

La méthode retenue récupère en médiane **89 % du gain de contraste possible**, rend un réglage à moins de 10 % de l'optimum sur près d'un appareil sur deux, et **divise par deux les ratés**. Elle est la meilleure dans 7 des 9 configurations décalées du simulateur (section 4), ce qui a départagé les deux finalistes.

![Distribution des regrets en val](figures/fig3_val_distribution.png)

La courbe se lit ainsi : pour chaque valeur de R en abscisse, la part des runs qui font au moins aussi bien. Plus la courbe monte tôt, mieux c'est. L'écart entre la méthode d'avant (gris) et les méthodes UCB se voit surtout à droite : les ratés lourds ont presque disparu.

---

## 2. Mesurer le succès : les métriques et pourquoi celles-là

### Ce qu'on veut mesurer

Le contraste vrai d'un réglage b est le facteur f(b) que le simulateur applique au meilleur interdot. Il vaut au moins un **plancher** (3 dans la configuration par défaut : tout interdot reste un peu visible) et au plus **f\***, le pic de la meilleure région (entre 23 et 33 selon l'appareil). L'algorithme ne voit jamais f ; seul le code d'évaluation le lit, une fois le run terminé.

### Première idée écartée : la distance à l'optimum en tensions

Mesurer ‖b̂ − b\*‖ semble naturel, mais un appareil a quatre régions dont les pics sont proches : sur 39 % des appareils, les deux meilleures diffèrent de moins de 5 %. Rendre le pic de la deuxième région coûte alors très peu de contraste mais une énorme distance en tensions. La distance punirait une réponse presque parfaite. On mesure donc **en contraste**, pas en tensions.

### Deuxième idée écartée : le contraste brut f(b̂) ou le rapport f(b̂)/f\*

Le rapport f(b̂)/f\* est intuitif, mais il flatte le résultat : comme f ne descend jamais sous le plancher, un réglage quelconque vaut déjà environ 10 % de f\*. Et f\* change d'un appareil à l'autre, ce qui rend les moyennes difficiles à comparer.

### La métrique principale : le regret normalisé

![Définition du regret normalisé](figures/fig1_metrique.png)

$$R = \frac{f^* - f(\hat b)}{f^* - \text{plancher}}$$

- **R = 0** : on a rendu l'optimum exact. **R = 1** : aucun progrès par rapport au plancher.
- **1 − R** se lit comme la **part du gain de contraste possible** effectivement obtenue.
- La normalisation par (f\* − plancher) rend les appareils comparables entre eux.
- R est calculé au **point engagé** b̂, celui que l'algorithme rend à la fin et qu'il a re-mesuré sur des frames fraîches. Jamais au meilleur point visité par chance : on évalue ce que l'algorithme affirmerait, pas ce qu'il a croisé.

### Les indicateurs dérivés de R

| indicateur | définition | ce qu'il dit |
|---|---|---|
| **regret médian** | médiane de R sur les runs | la performance typique, insensible à quelques runs extrêmes |
| **regret moyen** | moyenne de R | pénalise les ratés, complète la médiane |
| **succès** | part des runs avec R ≤ 0,05 | « on a trouvé l'optimum », à 5 % du gain près |
| **à 10 %** | part des runs avec R ≤ 0,1 | critère plus souple, utile vu les quasi-égalités entre régions |
| **ratés** | part des runs avec R > 0,3 | les runs où l'on rend un réglage clairement mauvais ; c'est souvent ce qui compte le plus pour un expérimentateur |
| **courbe au fil du budget** | R du point recommandé après 250 k, 500 k, 1 M pixels | la vitesse : quand l'algorithme commence-t-il à être utile ? |
| **région correcte** (diagnostic) | la région qui domine en b̂ est-elle la meilleure ? | sert à expliquer les échecs, jamais à classer |

### Le coût : le budget

Chaque mesure coûte du temps de manipulation. L'énoncé propose les **pixels** comme mesure de ce temps. Nous comptons aussi les **mesures**, parce que chaque changement de réglage demande d'attendre que l'appareil se stabilise : sans cette seconde limite, mille petites images de 100 pixels passeraient pour gratuites. Les deux limites (1 M pixels, 300 mesures) sont imposées par un intermédiaire, [`BlindExperiment`](optimization/blind.py), qui compte, refuse toute mesure qui dépasserait, et ne laisse jamais voir l'état caché. Les valeurs sont des ordres de grandeur fixés dans le [protocole](PROTOCOL.md) avant d'écrire les méthodes ; la section 5 montre ce que donne un budget plus large.

### Comparer deux méthodes sans se tromper

- **Appariement** : toutes les méthodes passent sur les mêmes appareils. On compare appareil par appareil (ΔR = R_méthode − R_référence), ce qui élimine la variance entre appareils faciles et difficiles.
- **Intervalle de confiance** du ΔR médian par bootstrap (5 000 tirages).
- **Test de Wilcoxon** signé, puis **correction de Holm** sur toutes les comparaisons d'un même tableau : plus on essaie de méthodes, plus il faut de preuves pour en déclarer une meilleure.
- Outil : [`evaluation/compare.py`](evaluation/compare.py).

---

## 3. Progression des architectures

Le fil conducteur est simple : à chaque étape, un diagnostic mesuré dit où se trouve la perte, et l'étape suivante vise cette perte et rien d'autre.

![Progression des briques sur dev 0–99](figures/fig2_progression.png)

### Étape 0 : la baseline fournie et ses pièges

La baseline des organisateurs fait une montée par coordonnées sur `img.std()`. Elle échoue presque toujours (gain médian 2 %), pour deux raisons physiques que nous avons mesurées avant d'écrire la moindre méthode ([PROTOCOL.md](PROTOCOL.md), section 1) :

- **la dérive** : bouger une barrière fait glisser les interdots de 0,66 V en médiane, pour une fenêtre de 0,3 V. Sans recentrage, la région intéressante sort de l'image et le score chute, ce qui ressemble à tort à « mauvais contraste » ;
- **le score** : `img.std()` sur toute l'image mélange le fond, le bruit et les stries, alors que le contraste se joue sur quelques pixels d'interdots.

### Étape 1 : le socle commun, sans lequel rien ne marche

Toutes les méthodes suivantes partagent ce socle ([`optimization/session.py`](optimization/session.py)) :

1. **Pare-feu et budget** : l'algorithme ne voit l'appareil qu'à travers `BlindExperiment`.
2. **Perception** : repérer les interdots dans chaque image (U-Net du challenge 1, ou banc de filtres orientés) et lire leur amplitude.
3. **Suivi de la dérive** : une frame de référence et trois petites sondes (une par barrière) mesurent comment l'image glisse ; un modèle de dérive est ensuite affiné à chaque image recalée. Les plungers g2, g4 ne sont plus des variables de décision : ils sont calculés pour garder les interdots au centre. Le problème passe de 5 à 3 dimensions, comme avec les « virtual gates » en laboratoire.
4. **Région de confiance** : on ne mesure que là où la dérive est prévisible à 25 mV près ; la zone explorable grandit avec le modèle.
5. **Score par interdot** : l'amplitude de chaque interdot est rapportée à son propre plancher, puis on prend le maximum, diminué d'un écart-type de bruit contre la malédiction du gagnant.

### Étape 2 : quel algorithme de décision ?

Avec ce socle, il reste à choisir où mesurer ensuite dans l'espace des trois barrières. Nous avons comparé, au même budget, des représentants de chaque famille classique de l'optimisation sans dérivée :

| famille | méthode | gain médian | succès |
|---|---|---|---|
| hasard | recherche aléatoire, on garde le meilleur | 32 % | 1 % |
| recherche directe | CMA-ES | 39 % | 2 % |
| gradient estimé | SPSA + Adam | 56 % | 3 % |
| modèle probabiliste | **optimisation bayésienne (GP + EI)** | **80 %** | **15 %** |

L'optimisation bayésienne gagne nettement, ce qui était attendu : avec une quarantaine de mesures coûteuses et bruitées, il faut un modèle qui se souvient de chaque mesure et qui sait où il ne sait pas. Un processus gaussien (noyau Matérn 5/2, une longueur de corrélation par barrière, bruit propre à chaque mesure) modélise le score ; une règle d'acquisition choisit le point suivant. Rien n'est entraîné à l'avance : le GP s'ajuste en ligne sur les mesures de l'appareil, et repart de zéro sur le suivant.

### Étape 3 : la perception n'est pas le goulot

Notre détecteur du challenge 1 (U-Net LOFO, F1 objet 0,98) remplace le filtre classique pour repérer les interdots. Résultat : regret identique en frames entières (0,193 contre 0,194). Le filtre repère déjà assez bien les interdots pour suivre la dérive, et l'amplitude est lue par le même filtre dans les deux cas. Le U-Net divise par deux les décrochages du suivi (0,3 contre 0,6 par run), ce qui justifie de le garder, mais la perte est ailleurs.

### Étape 4 : les zooms (mesure active)

**Constat** : une photo entière coûte 22 500 pixels alors qu'une fois la bonne région trouvée, seuls trois interdots comptent. **Idée**, empruntée à l'imagerie adaptative et à l'acquisition par rayons des boîtes quantiques (Lennon et al. 2019, Zwolak et al. 2021) : après la phase de repérage, ne mesurer que trois petits patchs de 25 × 25 px centrés sur les positions prédites, au même pas pour que les amplitudes restent comparables. Un réglage coûte alors 12 fois moins cher, et le même budget achète environ 250 réglages fins au lieu de quelques-uns. **Effet** : succès de 15 % à 25 %, ratés de 31 % à 23 %.

### Étape 5 : répartir le budget par le calcul

**Constat** : la version précédente basculait vers les zooms à 40 % du budget en pixels. Les zooms épuisaient alors les 300 mesures en laissant 37 % des pixels inutilisés. **Idée** : avant chaque nouvelle photo entière, vérifier si les pixels qui resteront dépassent ce que les zooms pourront dépenser avec les mesures restantes ; si oui, une photo d'exploration de plus ne coûte rien. La bascule n'est plus un paramètre réglé à la main mais la conséquence des deux limites. **Effet** : environ 35 photos d'exploration au lieu de 18, succès 29 %, ratés 17 %.

### Étape 6 : explorer davantage (UCB)

**Diagnostic.** Le code d'évaluation sait décomposer le regret d'un run en deux parts : la perte due au **choix de la région** et la perte due à **l'affinage** dans la région choisie.

![Origine des échecs](figures/fig7_diagnostic.png)

Sur 100 appareils, 47 perdent plus de 5 % à cause du choix de région, et dans 26 cas la meilleure région n'a **jamais** été allumée pendant l'exploration. La perte vient de l'exploration, pas de l'affinage. Un contrôle le confirme : même avec 4 fois plus de pixels, la meilleure région n'est visitée que dans 30 % des runs. Le problème n'est pas le nombre de photos mais la manière de les placer.

**Explication.** La règle EI (amélioration espérée) favorise les endroits qui ont une vraie chance de battre le record actuel. Sur un paysage plat presque partout avec quatre pics étroits, elle reste autour du premier pic trouvé. La règle **UCB** (borne haute de confiance, μ + 2σ, Srinivas et al. 2010) traite une zone inconnue comme potentiellement excellente tant qu'elle n'a pas été vue : c'est le principe d'optimisme face à l'incertitude. Le coefficient 2 est la valeur standard, fixée avant le test.

**Effet** : succès 37 %, ratés 9 % sur dev 0–99 (p Holm 0,005), confirmé sur 100 appareils neufs (succès 39 % contre 22 %, p Holm 4·10⁻⁴) puis en val.

### Étape 7 : quatre pistes pour aller plus loin dans l'exploration

Chacune vise une partie précise de la perte diagnostiquée ; chacune est jugée sur dev 0–99 puis sur 100 appareils neufs.

| piste | idée | dev 0–99 : R / succès / ratés | dev 900–999 : R / succès / ratés | décision |
|---|---|---|---|---|
| **modèle par région** | un GP par groupe spatial d'interdots au lieu d'un GP sur le maximum, pour qu'une deuxième région qui s'allume ne soit pas masquée par la première (vise 11 appareils où la meilleure région s'est allumée sans être suivie) | 0,083 / 34 % / 11 % | 0,133 / 30 % / 14 % | pas mieux qu'UCB seul |
| **exploration grossière** | photos au pas ×2, 4 fois moins chères, pour environ 100 points d'exploration au lieu de 35 (vise les 26 appareils jamais allumés) | 0,128 / 27 % / 9 % | **0,064 / 43 % / 10 %** | meilleure sur appareils neufs, mais fragile au bruit (section 4) |
| **course entre deux régions** | pendant les zooms, suivre en alternance la meilleure région et une seconde, puis garder la gagnante | 0,099 / 38 % / 10 % | 0,092 / 36 % / 14 % | en cours de val |
| **politique méta-apprise** | remplacer la règle de choix par 8 poids appris par CMA-ES sur ~31 000 épisodes simulés, récompense observable uniquement | 0,105 / 26 % / 18 % | | ne transfère pas en val |

**Pourquoi la retenue est `bo_roi_auto_ucb_dlf`.** L'exploration grossière est la plus forte sur appareils neufs, mais elle repose sur des photos basse résolution qui perdent le signal quand le bruit double ou quand les pics rétrécissent : son regret médian passe alors à 0,91 et 0,95, contre 0,31 et 0,45 pour la méthode retenue (section 4). Sur une vraie puce, dont le bruit et la forme des pics ne sont pas ceux du simulateur, nous préférons la méthode qui dégrade le moins.

---

## 4. Robustesse hors du simulateur par défaut

Les méthodes ont été conçues sur la configuration par défaut. Pour estimer ce qui se passerait sur une vraie puce, nous modifions le simulateur selon les scénarios S1–S7 du protocole, sans rien réentraîner ni régler, 50 appareils par case.

![Robustesse](figures/fig5_robustesse.png)

- La méthode retenue est la meilleure dans 7 lignes sur 9. Dans les deux autres (pics larges, dérive très non linéaire), l'écart avec la meilleure reste de 0,03 en regret médian.
- Le bruit doublé (S1) et les pics deux fois plus étroits (S5a) restent difficiles pour toutes les méthodes : l'aiguille devient plus fine ou plus noyée, et le budget ne suffit plus pour la trouver. La méthode retenue y garde un regret médian de 0,31 et 0,45, là où les autres montent entre 0,75 et 0,99.
- L'exploration grossière s'effondre exactement dans ces deux cas, ce qui a guidé le choix final.

---

## 5. Que vaut plus de budget ?

Les pixels et les mesures sont augmentés ensemble, dans le même rapport (250 k pixels et 75 mesures, … , 4 M pixels et 1 200 mesures).

![Courbe de budget](figures/fig4_budget.png)

| budget | gain médian | succès | à 10 % | ratés |
|---|---|---|---|---|
| 250 k px, 75 mesures | 32 % | 9 % | 13 % | 65 % |
| 500 k px, 150 mesures | 85 % | 24 % | 39 % | 26 % |
| **1 M px, 300 mesures** | **91 %** | **37 %** | **52 %** | **9 %** |
| 2 M px, 600 mesures | 92 % | 40 % | 55 % | 8 % |
| 4 M px, 1 200 mesures | 93 % | 39 % | 63 % | 5 % |

(méthode retenue, dev 0–99)

Tout se joue entre 250 k et 1 M pixels, le temps de calibrer la dérive puis de trouver une région brillante. Au-delà, quadrupler le budget n'ajoute que 2 points de gain médian : la difficulté restante est de **départager quatre pics proches**, et il faudrait pour cela les trouver tous. Le choix de 1 M pixels comme budget de référence se situe juste après le coude.

---

## 6. Un run en images

![Un run complet](figures/fig6_exemple_run.png)

1. La frame de référence et les interdots repérés : c'est la carte qui sert ensuite à suivre la dérive.
2. La meilleure frame trouvée pendant l'exploration : une région s'allume, son contraste est multiplié par 10. Les carrés sont les trois zooms de la phase suivante.
3. Les trois zooms ROI : chacun coûte 625 pixels au lieu de 22 500.
4. La trajectoire dans l'espace des barrières. L'exploration couvre la boîte, puis les zooms se concentrent. Les étoiles (vérité, tracées pour la figure seulement) montrent que l'algorithme a choisi une région dont le pic est très proche de celui de la meilleure : regret final 0,07.
5. Le regret du point recommandé au fil du run : rien d'utile pendant la calibration, une chute brutale quand une région est trouvée, puis l'affinage.

---

## 7. Résultats négatifs

Ils font partie de la démarche ; chacun a appris quelque chose.

| essai | résultat | ce qu'on en retient |
|---|---|---|
| U-Net du challenge 1 au lieu du filtre | regret identique, suivi deux fois plus fiable | le goulot n'est pas la vision |
| PFN : un transformer pré-entraîné sur des paysages synthétiques remplace le GP | regret 0,68, presque le hasard, alors qu'il bat le GP hors ligne | un a priori appris mal spécifié coûte plus qu'il ne rapporte |
| politique méta-apprise (8 poids, CMA-ES, ~31 000 épisodes simulés, récompense observable) | énorme progrès sur sa propre récompense (+5 unités de score), aucun gain en val (p = 0,43) | optimiser un objectif intermédiaire (le score observable en fin de phase 1) ne suffit pas : l'écart entre cet objectif et le regret final a tout absorbé |
| CMA-ES au lieu du GP pendant les zooms | identique (p = 0,25) | une fois la région choisie, l'algorithme de décision importe peu |
| bascule vers les zooms « dès qu'une région s'allume » | identique | le bruit au plancher atteint déjà le seuil, le critère se déclenche toujours |
| exploration grossière avec moins de photos fines | pas significatif sur dev, moins bon sur appareils neufs | les photos fines servent au suivi, on ne peut pas trop les réduire |
| détecteur M5 du challenge 1 à son seuil d'origine | suivi qui décroche | voir [`transfer_m5/`](transfer_m5/README.md) |

---

## 8. Protocole, données et reproductibilité

- **Protocole écrit avant les méthodes** : [PROTOCOL.md](PROTOCOL.md) fixe les métriques, le budget, les pièges physiques et les critères d'abandon.
- **Splits fixes d'appareils** ([`evaluation/seeds.py`](evaluation/seeds.py)) :
  - dev 0–99 : conception et criblage ;
  - dev 100–999 : entraînement des méthodes apprises (PFN, politique méta) ;
  - dev 900–999 : confirmation sur appareils neufs, ajoutée pendant la phase d'amélioration ;
  - val 1000–1199 : comparaison finale des candidates, 3 graines ;
  - test 10000–10199 : intact, lancé une seule fois à la fin.
- **Règle de décision écrite avant de voir les résultats** ([`results/NIGHT_LOG.md`](results/NIGHT_LOG.md)) : une idée vient d'un diagnostic, ses paramètres sont fixés a priori (aucun balayage), elle doit gagner sur dev 0–99 puis sur 900–999, et toutes les tentatives, échecs compris, sont journalisées.
- **Honnêteté sur la val** : la val a servi à comparer plusieurs candidates. Elle joue donc le rôle d'un jeu de sélection, et seul le test donnera un chiffre sans biais de sélection.
- **Pare-feu** : le code d'optimisation n'importe pas le simulateur et n'accède ni à `reveal` ni à `_sim` (vérifié par les tests). Seul [`evaluation/`](evaluation/) lit la vérité, pour noter.
- **Aucune récompense cachée** : la politique méta est entraînée sur un score que l'algorithme peut mesurer lui-même, jamais sur le contraste vrai.
- **Un appareil par processus**, chaque algorithme avec son propre générateur aléatoire : le simulateur tire son bruit dans le générateur global de NumPy, et l'entrelacer fausserait silencieusement les mesures.

---

## 9. Limites et suites

- **Tout est simulé.** Le transfert est estimé par les scénarios S1–S7, pas mesuré sur une vraie puce.
- **Le budget est un ordre de grandeur.** Sur une vraie puce, il dépendrait du temps d'intégration par pixel et du temps de stabilisation des grilles.
- **Les quatre pics proches restent la difficulté.** Même avec un budget quadruplé, environ 40 % des appareils sont résolus à 5 % près. Les deux suites les plus prometteuses :
  - combiner l'exploration grossière et la course entre régions, en adaptant la résolution au bruit mesuré pour ne pas perdre la robustesse ;
  - une politique apprise qui choisit le **type** de mesure (photo fine, grossière ou zoom) et pas seulement l'endroit, entraînée directement sur un objectif plus proche du regret final.
- **Bruit doublé et pics étroits** (S1, S5a) restent des cas où aucune méthode ne trouve l'optimum avec ce budget.

---

## 10. Lancer et contenu du dossier

```bash
uv run python challenge2/solve.py                                     # un appareil, de bout en bout
uv run python -m evaluation.run --methods bo_roi_dlf bo_roi_auto_ucb_dlf --split dev --n 100 --workers 8 --out challenge2/results/run.jsonl
uv run python -m evaluation.compare --ref bo_roi_dlf challenge2/results/run.jsonl
uv run python -m evaluation.run --methods bo_roi_auto_ucb_dlf --split dev --n 50 --shift S1 --out challenge2/results/rob_S1.jsonl
PYTHONPATH=challenge2:hackathon:challenge1 python challenge2/figures/make_figures.py
```

Le U-Net du challenge 1 est utilisé si `C12_DL_CKPT` pointe vers son checkpoint (extra GPU `uv sync --extra train`) ; sinon, la perception sans apprentissage prend le relais, avec un regret équivalent (section 3, étape 3).

| chemin | contenu |
|---|---|
| [`solve.py`](solve.py) | point d'entrée `optimize(exp)` |
| [`PROTOCOL.md`](PROTOCOL.md) | protocole écrit avant les méthodes |
| [`optimization/`](optimization/) | pare-feu, perception, suivi, score, et toutes les méthodes (`methods/`) |
| [`evaluation/`](evaluation/) | seul code qui lit la vérité : regret, diagnostic, configurations décalées, comparaisons appariées |
| [`training/`](training/) | PFN et politique méta-apprise |
| [`figures/`](figures/) | figures de ce README et le script qui les refait |
| [`results/`](results/) | fichiers bruts `*.jsonl`, [`BENCH_C2.md`](results/BENCH_C2.md), journal des tentatives [`NIGHT_LOG.md`](results/NIGHT_LOG.md) |
| [`transfer_m5/`](transfer_m5/README.md) | transfert de la régression logistique du challenge 1 |
