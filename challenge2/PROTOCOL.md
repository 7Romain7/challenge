# Protocole expérimental — Challenge 2 (optimisation du contraste)

Périmètre : **challenge 2 uniquement**. Ce document fixe, *avant* d'écrire les méthodes, ce qu'on mesure, comment, et quels pièges on s'interdit. Il couvre 5 architectures GPU + les baselines.

Tout chiffre marqué **[pilote]** a été mesuré sur le simulateur par défaut (`csd/config.py` inchangé, 100–200 appareils, scripts jetables). À re-mesurer dans `eval/pilot.py` pour être citable.

---

## 0. Règles du jeu → contraintes d'ingénierie

| Règle (README / slides) | Traduction concrète |
|---|---|
| Pas de `reveal()` dans l'optimiseur | **Pare-feu de code** : le package `opt/` n'importe jamais `csd.config`, `csd.generator`, `csd.simulator`, ni `exp._sim` / `exp._optimum`. Il reçoit un `BlindExperiment` (wrapper) qui n'expose que `measure`, `scan_1d`, `start`, `extent` et les compteurs. Seul `eval/` touche `reveal()` / `_sim`. Test automatique (import-linter ou `grep` en CI). |
| Pas de rétro-ingénierie du bruit | Aucun σ, aucune valeur de `GeneratorConfig` codée en dur. Tout niveau de bruit est **estimé sur les données** (MAD après soustraction de la médiane de ligne ; répétitions au même point). |
| Pas de brute force pixel | Budget **dur** en pixels imposé par le wrapper (`BudgetExceeded`). Le budget de référence est bien inférieur au baseline fourni (≈ 7 M pixels, **[pilote]** calculé : 310 frames × 22 500 px). |
| Méthodes qui transfèrent au réel | Perception sans entraînement (pas de détecteur du challenge 1 requis), pas de connaissance de la forme exacte du paysage dans le run principal, **tests sous config décalée** (§6). |
| Baseline par défaut obligatoire | Résultats titres = `CHALLENGE` par défaut. Toute modification de config = expérience *séparée*, documentée. |
| Évalué sur la démarche | Chaque choix a une hypothèse, un test, un critère d'abandon (§5). Les échecs sont rapportés. |

Zone grise à décider explicitement (par défaut : **interdit**) : utiliser la vraie valeur du contraste comme récompense/label **pendant l'entraînement** d'un modèle appris (« critique privilégié »). Interdit dans le run principal ; si testé, c'est une variante étiquetée « privileged-training », jamais mélangée aux résultats titres.

---

## 1. Faits mesurés sur le simulateur **[pilote]**

1. Débit : **≈ 9,5 ms par frame 150×150** (≈ 100 frames/s, 1 cœur) — pas 17/s.
2. Dérive : sur la boîte de barrières [−0,5 ; 0,5]³, le décalage max des sticks vaut en médiane **0,66 V** (max 1,0 V) pour une fenêtre de **0,3 V**. Au barrières de l'optimum d'un appareil, les sticks étaient décalés de (−0,21 ; +0,18) V : sans re-centrage, la région optimale sort de la fenêtre.
3. Résolution : les sticks font ≈ 0,4–1,6 px de large. Le pixel le plus sombre au point optimal (facteur vrai ≈ 28–33) vaut ≈ **−25 à 2 mV/px**, **−13 à 5 mV/px**, **−4 à 10 mV/px**, avec une forte variabilité selon la graine (crénelage). Un « aperçu large basse résolution » **détruit le contraste**.
4. Bruit : après soustraction de la médiane de chaque ligne, σ ≈ 0,57 (contre 0,81 brut) : les stries horizontales sont corrélées le long d'une ligne.
5. Structure des appareils : 5 à 57 sticks (moy. 13) ; **médiane 3 sticks dans la région optimale**, parfois 1 ; 6 % des régions sont vides ; écart relatif d'amplitude entre les deux meilleures régions peuplées : médiane 6,7 %, **39 % des appareils < 5 %** (quasi-égalité).
6. Visibilité par point aléatoire de la boîte [−0,6 ; 0,6]³ : la meilleure région dépasse son plancher de ≥ 1 unité avec probabilité **6 %** (≥ 0,5 : 11 % ; ≥ 2 : 3 %). Un tirage aléatoire de n points la voit au moins une fois avec probabilité : n=16 → 61 %, 50 → 92 %, 100 → 98 %. **La recherche aléatoire est une baseline forte et obligatoire.**
7. RNG : le bruit est tiré dans le RNG **global** `np.random`, réinitialisé par `new_experiment(seed)`. Si l'utilisateur appelle `np.random.seed(0)`, les « nouveaux » frames ont **un bruit identique** (vérifié) — la moyenne sur n frames n'apporte alors rien, silencieusement.
8. Environnement : PyTorch **absent** de l'env `uv` ; `nvidia-smi` introuvable dans le PATH du shell (GPU non confirmé ici).

---

## 2. Pièges physiques et garde-fous

| # | Piège | Pourquoi c'est faux physiquement | Garde-fou |
|---|---|---|---|
| P1 | Mélanger **dérive** et **contraste** | Bouger une barrière décale les sticks (jusqu'à 1 V) ; si les plungers restent fixes, la région sort de la fenêtre → score ≈ 0 qui ressemble à « mauvais contraste ». | Les plungers ne sont **pas** des variables de décision : un module de suivi les recentre. Tout score est calculé sur des sticks *suivis*, jamais sur une fenêtre fixe. |
| P2 | Comparer des scores à **résolutions différentes** | Le contraste apparent dépend du pas (fait 3). | Pas fixe ≤ 3 mV pour tout score comparé ; les fenêtres de zoom changent la taille, pas le pas. Test T3 : monotonie du proxy à pas constant. |
| P3 | `img.std()` / score global | Dilué par le fond, dominé par le bruit ; un seul patch s'allume. | Score **par stick**, agrégé par un max/quantile haut (le contraste vrai est le *max* sur les sticks). |
| P4 | Agréger par **moyenne** des sticks | Un optimum = une région brillante, les autres au plancher : la moyenne pénalise le bon point. | Statistique d'ordre (max avec correction, §3.3). |
| P5 | **Malédiction du gagnant** | argmax de mesures bruitées surestime le maximum. | Le point recommandé est **re-mesuré avec des frames indépendants** ; l'estimation finale vient de la re-mesure, pas du pic observé. |
| P6 | Gain géométrique par stick | Intensité ±10 %, largeur/longueur différentes, crénelage : l'amplitude apparente d'un stick = g_i · f_région. | Normaliser chaque stick par **sa propre référence hors-résonance** (ratio a_i(b)/a_i(b_ref)) : g_i s'annule ; classement des régions par ce ratio. Résolution irréductible : les quasi-égalités (39 %) ne sont pas séparables → juger au **regret en contraste**, pas à la distance en barrières. |
| P7 | Stries horizontales | Bruit corrélé par ligne ⇒ pixels non i.i.d. ; un CNR naïf surestime la significativité. | Soustraction de la médiane de ligne ; σ robuste (MAD). L'augmentation par rotation/flip est **interdite** (les stries sont l'axe rapide). |
| P8 | Normaliser chaque image (z-score) | Le contraste *est* le signal ; diviser par l'écart-type d'une image très contrastée écrase le contraste. | Images **brutes** (comme `measure`), échelle de bruit fixe. |
| P9 | Observabilité partielle | Une image seule ne dit pas dans quelle direction monter : les centres des bosses sont propres à l'appareil. | Toute politique/modèle appris doit être **à mémoire** (historique des mesures), sinon il est non identifiable (cf. §5.3, 5.5). |
| P10 | Mesure fraîche vs RNG global | Cf. fait 7. | Test T1 : deux mesures identiques au même point doivent différer ; jamais de `np.random.seed` dans `opt/` ; un appareil par **processus** (pas de threads/entrelacement). |
| P11 | Sticks absents ≠ pas de contraste | Régions vides, régions à 1 stick sous le seuil de détection. | Le suivi garde des sticks « candidats » ; seuil de détection estimé sur le bruit, pas fixé. |
| P12 | Limites matérielles | Un vrai appareil a une plage de tensions sûre et subit des sauts de charge sur grands pas. | Boîte de recherche déclarée (±0,6 V, documentée comme limite matérielle, sensibilité testée à ±0,5/±0,8) ; **longueur de trajectoire** et **saut max** rapportés. |
| P13 | Artefacts du simulateur absents du réel | Ici la scène = fenêtre de départ : tous les sticks y sont visibles à t=0 et rien n'entre ensuite ; un vrai réseau de charge est étendu. | Documenté en limites. N'est **pas** exploité : on n'utilise pas « la scène fait 0,3 V » comme a priori. |
| P14 | Biais de sélection de l'évaluation | Mesurer l'optimiseur sur ce qu'il a visité, ou choisir le meilleur run a posteriori. | Les métriques portent sur la **recommandation engagée** de l'algorithme (§4). |
| P15 | Structure exacte du paysage (produit de Lorentziennes) | C'est exploiter le modèle caché. | Échelle de connaissance a priori : **P0** aucune forme (GP) · **P1** pics lisses unimodaux (générique) · **P2** produit de Lorentziennes exact (**interdit en titre** ; ablation « structure-informed » étiquetée). |

### 2bis. Solutions retenues et justifications

Principe directeur : **chaque solution doit avoir un équivalent en laboratoire** (ce qu'un expérimentateur ferait avec un appareil réel) **et un test qui la valide**. Une solution qui n'a que l'un des deux est rejetée.

| Piège | Solution | Justification | Validation |
|---|---|---|---|
| P1 dérive ↔ contraste | **Suivi des interdots** : une image de référence fixe des positions (en tension absolue de plunger) ; chaque déplacement de barrière est prédit par un modèle de dérive ajusté en ligne, puis corrigé par recalage de l'image mesurée sur la référence. (g2, g4) sont dérivés du suivi, pas optimisés. | En labo, on suit une transition en réajustant les plungers quand on bouge une barrière (« virtual gates »). Cela ramène un problème 5-D à 3-D et sépare la géométrie (dérive) de la physique cherchée (contraste). | T4 : erreur de suivi < 4 mV sur appareils dev. |
| Calibration de la dérive | 3 sondes de 0,08 V (une par barrière) → colonnes de la matrice de levier ; puis régression ridge (linéaire + termes quadratiques fortement régularisés) sur tous les points recalés. | Mesure directe de la matrice de couplage, comme on calibre des « virtual gates » ; les pas de 0,08 V gardent la courbure négligeable (≲ 3 mV). Aucune valeur lue dans la config. | Erreur de prédiction de dérive vs vérité (diagnostic eval). |
| P2 résolution | Pas natif (celui de la première image) pour tout score comparé ; on ne réduit que la **taille** de fenêtre. | Le contraste d'un trait de ~1 px dépend du pas (fait 3) ; comparer à pas égal est la seule comparaison valide. Le pas natif est *mesuré* (taille de la 1re image), pas lu dans la config. | Pilote : contraste apparent vs pas. |
| P3/P4 score global, moyenne | **Amplitude par interdot**, via un banc de filtres linéaires orientés (invariant à l'orientation), puis statistique d'ordre : max sur les interdots suivis. | Le contraste est défini sur *un* interdot ; un filtre adapté à un segment mince est l'estimateur à rapport signal/bruit maximal pour un trait dans un bruit blanc, et le banc d'orientations évite de coder la pente. | Spearman proxy ↔ facteur vrai (T3). |
| P5 malédiction du gagnant | (a) **rétrécissement** : on retranche z·σ à l'excès ; (b) recommandation = argmax de la **moyenne a posteriori** (GP) sur les points visités, pas du pic bruité ; (c) **re-mesure** de la recommandation sur frames frais avant engagement. | Un maximum d'estimations bruitées est biaisé vers le haut ; lisser (GP) et re-mesurer sont les remèdes standard (régression vers la moyenne). | Écart « estimé − vrai » au point engagé, rapporté. |
| P6 gain géométrique par interdot | **Normalisation par le plancher propre** à chaque interdot : (a_i(b) − plancher_i)/plancher_i, plancher_i = quantile bas de ses amplitudes sur les points visités. | Le gain (largeur, crénelage, intensité) est constant en b pour un interdot donné et s'élimine dans le rapport. C'est l'analogue d'une normalisation par la réponse « hors résonance », et ce rapport est ∝ f/base − 1, comparable entre régions. | Spearman rapport ↔ f vrai > amplitude brute. |
| Quasi-égalités | Régret en **contraste** comme métrique principale ; « région correcte » seulement diagnostique. | Entre deux régions à 3 % d'écart, se tromper coûte 3 % de contraste ; la distance en barrières punirait à tort. | Stratification dans l'évaluation. |
| P7/P8 stries et échelle | Soustraction de la **médiane de chaque ligne** ; σ par MAD ; images **brutes**, aucune augmentation par symétrie ou échelle. | Les stries sont du bruit basse fréquence sur l'axe rapide (typique des mesures de transport) ; une médiane de ligne est robuste aux interdots (ils couvrent peu de pixels par ligne). L'échelle de bruit est une propriété de l'instrument. | σ après / avant. |
| P9 observabilité partielle | Tout modèle appris reçoit l'**historique** (b_j, y_j, σ_j) ; les méthodes sans mémoire (image seule) sont exclues. | La direction vers l'optimum dépend de paramètres propres à l'appareil, inférables seulement à partir d'essais passés. | Ablation « avec / sans historique ». |
| P10 RNG | Aucun `np.random.seed` ; chaque algorithme a son `Generator` ; un appareil par **processus** ; test T1. | Le bruit du simulateur partage le RNG global : toute remise à zéro fige le bruit. | T1, T5. |
| P11 sticks faibles | L'ensemble de référence **croît** : un interdot détecté à un autre point de barrière est ajouté (position absolue = détection − dérive). Les amplitudes sont lues aux positions *attendues*, même sous le seuil de détection. | Un interdot peut être invisible au plancher et apparaître en résonance (c'est précisément le signal). Lire à la position attendue évite le biais de détection. | Rappel de détection vs vérité (diagnostic). |
| P12 limites matérielles | Boîte déclarée ±0,6 V, saturation par `clip`, trajectoire et saut max rapportés. | Un instrument a une plage de sécurité ; on ne triche pas en sortant du domaine plausible. | Sensibilité ±0,5/±0,8 (S3). |
| P13 artefacts du simulateur | Aucun a priori sur la taille de scène ; la 1re image est la référence. | Dans un vrai réseau, on prendrait une carte de départ de n'importe quelle taille. | — (limite documentée). |
| P14 biais d'évaluation | Évaluation sur la **recommandation engagée** à chaque pas, journalisée par l'algorithme ; la vérité n'est utilisée que par `evaluation/`. | On évalue ce que l'algorithme affirmerait en s'arrêtant ; jamais ce qu'il a visité par chance. | T6. |
| P15 structure du paysage | A priori **générique** (GP Matérn ARD, pics lisses) ; la forme Lorentzienne n'est jamais codée dans `optimization/`. Le prior du PFN/RL est volontairement plus large (§5.2). | Si la méthode connaît la forme exacte, elle exploite le simulateur et ne dit rien du réel. | Tests de transfert S1–S7. |

Limites assumées : la normalisation P6 suppose que le plancher est visité par au moins ~30 % des points (vrai ici : ~80 % des points sont hors de toute bosse, fait 6) ; la co-modulation entre interdots d'une même région n'est pas encore exploitée en v1 (variante GP par groupe prévue).

---

## 3. Cadre commun

### 3.1 Interface et budget
- `BlindExperiment(exp, pixel_cap, meas_cap)` : mêmes méthodes que `Experiment`, plus un journal `(n_pixels, n_meas, gates, window)` et une méthode `commit(point)` que l'algorithme appelle pour **engager** sa recommandation courante. L'évaluateur rejoue ce journal.
- Budgets : **B ∈ {100 k, 250 k, 500 k, 1 M, 2 M} pixels** (1 frame par défaut = 22 500 px). Budget titre = **500 k** (≈ 22 frames), à confirmer après pilote. Plafond de mesures : 300.
- Fenêtre de zoom = taille + pas natif (2 mV) : p. ex. 0,08 × 0,08 V = 40×40 = 1 600 px (absorbe l'erreur de prédiction de dérive).

### 3.2 Front-end de perception (commun, **sans entraînement**)
1. Soustraction de la médiane de ligne ; σ_bruit par MAD (jamais lu dans la config).
2. Détection de sticks : filtre adapté/DoG orienté + suppression des non-maxima ; seuil = k·σ estimé. Sortie : liste (position, amplitude a_i, σ_a).
3. **Calibration du levier de dérive en ligne** : 3 sondes (une par barrière) → décalage mesuré par corrélation de phase sur les cartes de réponse → régression (linéaire + terme quadratique si assez de points). Données = images, pas la config.
4. Suivi : association position prédite ↔ sticks détectés (hongrois) ; recentrage automatique de (g2, g4) sur les sticks suivis.
5. Sortie : vecteur d'amplitudes suivies {a_i(b)} avec incertitudes ; groupement des sticks par **co-modulation** (mêmes variations ⇒ même région), sans étiquette cachée.

Tests de perception (hors optimisation, avec la vérité **seulement pour valider**) : rappel/précision de détection, erreur de suivi (px), erreur sur le levier de dérive (%), corrélation de Spearman entre proxy et facteur vrai de la région du stick (doit être > 0,9 à pas constant).

### 3.3 Observable scalaire
y(b) = estimateur **régularisé du max** des amplitudes normalisées (P6) sur les sticks suivis, avec variance propagée (shrinkage empirique-bayésien contre P5). Variantes d'ablation : moyenne, `img.std()`, quantile 90 %.

---

## 4. Métriques et statistique

**Notation** : f(b) = facteur de contraste vrai au point engagé (calculé par `eval/` via le simulateur) ; f* = base + A_max (région peuplée la meilleure).

| Métrique | Définition | Rôle |
|---|---|---|
| **Regret normalisé R** | (f* − f(b̂)) / (f* − base) ∈ [0,1] | Métrique principale. Insensible aux quasi-égalités (contrairement à une distance en barrières). |
| **R visible** | idem, en ne comptant que les sticks **dans la fenêtre finale engagée** | Le contraste doit être *observable* au point rendu (P1). |
| Succès@ε | R ≤ 0,05 et R ≤ 0,10 ; IC de Wilson | Taux de réussite. |
| Région correcte | la région engagée = région optimale | Diagnostic (les quasi-égalités rendent un échec « faux » acceptable). |
| Coût | pixels, mesures, **longueur de trajectoire** Σ‖Δb‖, saut max | Efficacité, sécurité matérielle. |
| Anytime | R(b̂ engagé) en fonction des pixels cumulés | Courbe de Pareto R–pixels (compare les méthodes à budget égal). |
| Calibration | couverture des intervalles prédictifs (BO/PFN), ρ de Spearman proxy↔vrai | Valide l'incertitude et le proxy. |
| Pixels-pour-succès | pixels médians pour atteindre R ≤ 0,05 | Efficacité à qualité fixée. |

**Splits de graines** (fixés dans `eval/seeds.py`) :
- **dev** 0–999 : conception, entraînement des modèles appris, pilotes.
- **val** 1000–1199 : réglage des hyperparamètres, sélection de modèle.
- **test** 10000–10199 : gelé, **exécuté une seule fois** sur un commit étiqueté. Jamais utilisé pour décider quoi que ce soit.
- Aucune graine de test/val dans l'entraînement ; découpe **par appareil** (une scène = une seed ; pas de fuite entre images d'un même appareil).

**Stats** : N_test = 200 appareils × 3 graines d'algorithme pour les méthodes stochastiques ; comparaisons **appariées** (mêmes appareils) ; médiane + IQR de R ; IC bootstrap apparié du ΔR ; Wilcoxon signé-rangs avec correction de Holm sur l'ensemble des comparaisons ; résultats **stratifiés** (sticks dans la région optimale = 1 vs ≥ 2 ; écart top-2 < 5 % vs ≥ 5 % ; distance de départ). Rapporter aussi les échecs (R > 0,3) un par un.

---

## 5. Les méthodes

Baselines communes : **B0** point de départ ; **B1** coordonnée ascendante sur `std` fournie ; **B2** recherche aléatoire/Sobol sur y (budget égal) ; **B3** descente par gradient (Adam, différences finies) sur y ; **B4** BO sans suivi (fenêtre fixe) pour *quantifier P1*. Borne haute : **Oracle** (point optimal exact, regret 0, coût ∞) — pour l'échelle des graphiques seulement.

### 5.1 BO sur GPU (BoTorch) — référence solide
- **Espace** : (g1, g3, g5) ∈ boîte ; plungers dérivés du suivi.
- **Modèle** : GP Matérn-5/2 ARD, a priori sur les échelles (0,03–0,5 V), bruit hétéroscédastique fourni par le front-end ; variante A : un GP sur y ; variante B : un GP par groupe de sticks + max (Thompson).
- **Acquisition** : qLogNEI ; **multi-fidélité par coût en pixels** (EI / pixel) : zoom de 1 600 px par défaut, frame complet seulement pour la re-détection.
- **Init** : 3 sondes de dérive + 8–12 points Sobol.
- **Arrêt/engagement** : P(f(b̂) ≥ 0,95 · max) suffisamment grande ou budget ; vérification par 3 frames frais.
- **Pièges** : GP sur plateau plat (peu d'information avant d'avoir touché une bosse → exploration ≥ 16 pts, cf. fait 6) ; ne pas mettre g2, g4 dans l'espace de recherche ; ne pas faire dépendre l'a priori de γ ou du nombre de régions (apprendre les échelles).
- **Succès** : R médiane ≤ 0,05 à 500 k px et battre B2 de façon significative (Holm). **Abandon/réduction** si ne bat pas B2.
- **Ablations** : score (std / moyenne / max) ; suivi on/off ; GP par stick vs scalaire ; init Sobol vs aléatoire ; « structure-informed » (GP log-additif, P15) étiquetée.

### 5.2 PFN / transformer d'optimisation amortie
- **Idée** : transformer pré-entraîné à prédire la distribution postérieure de y(b) et à proposer le point suivant à partir de l'historique {(b_j, y_j, σ_j)}.
- **Entraînement** : sur des paysages **synthétiques** tirés par *notre* prior, volontairement **plus large** que le simulateur (somme de 1–8 bosses de formes variées — Lorentz, gaussienne, exponentielle —, largeurs log-uniformes 0,03–0,4 V, plancher et amplitudes variables, bruit hétéroscédastique) pour ne pas encoder le modèle exact (P15). Le simulateur reste le *test* « réel ».
- **Pièges** : prior trop proche du simulateur ⇒ mémorisation du modèle caché ; mesurer l'écart *in-distribution* (paysages synthétiques) vs *simulateur* vs config décalée. Séquences longues (≤ 100 obs.) ; normalisation des entrées par bornes de boîte uniquement.
- **Évaluation** : (i) NLL/calibration sur paysages synthétiques held-out ; (ii) R sur appareils test avec le même front-end que 5.1 ; (iii) comparaison à GP ajusté sur les mêmes historiques.
- **Courbe d'apprentissage** (§7).
- **Succès** : égale ou bat 5.1 en R à budget égal **ou** égale 5.1 avec moins de temps de calcul par décision. Sinon : résultat négatif documenté.

### 5.3 RL (PPO) à politique récurrente
- **Observabilité partielle (P9)** : politique GRU/transformer sur l'historique.
- **Actions** : Δb continu (3), type de mesure discret {zoom, frame complet, re-vérification}, `commit`. Le re-centrage est fait par le front-end (le RL ne gère pas la dérive).
- **Récompense** (observables seulement) : terminal = y vérifié par frames frais − λ·pixels ; pas de f vrai. Récompense dense par gain d'information optionnelle.
- **Environnement d'entraînement** : le simulateur est trop lent pour le RL en ligne ⇒ **environnement abstrait** (observations de contrastes suivis avec bruit *calibré par répétitions*, pas lu dans la config) générée par le même prior large que 5.2, puis validation finale sur le vrai simulateur (écart sim-abstrait → sim = diagnostic).
- **Pièges** : récompense creuse (aiguille dans une meule : fait 6) ⇒ curriculum (amplitudes plus fortes → réelles) ; hacking du coût en pixels (zoomer trop petit pour perdre le suivi) ⇒ pénalité d'échec de suivi ; mémorisation de graines ⇒ graines d'entraînement ≠ val/test.
- **Succès** : R(500 k) ≤ B2 et variance entre graines d'entraînement documentée (≥ 5 graines d'entraînement). Les RL instables doivent être rapportés comme tels.

### 5.4 JEPA n°1 — encodeur de représentation
- **Protocole** : (a) pré-entraînement auto-supervisé d'un petit ViT sur des frames (150×150, 1 canal, **brutes**, soustraction de médiane de ligne comme prétraitement unique) ; (b) l'embedding alimente un noyau GP (BO 5.1) ou l'état du RL.
- **Pièges propres au domaine** :
  - *Collapse sur le bruit* : l'objectif JEPA est satisfait en encodant ce qui est le plus **prédictible** — ici les stries et la périodicité du réseau, pas les sticks. Mitigation : médiane de ligne, patchs petits (4–8 px), masques en blocs, EMA ; **diagnostics** : écart-type des embeddings, rang effectif (RankMe), pas de collapse.
  - *Augmentations* : ni rotation/flip (axe rapide, pente π/4 physiques), ni mise à l'échelle d'intensité (détruit le contraste, P8).
  - *Transfert ImageNet* : un I-JEPA pré-entraîné (RGB 224, photos naturelles) est hors-domaine ; inclus seulement comme **baseline gelée** (attendue faible), pas comme méthode.
- **Test clé (sonde linéaire)** : régresser y (max suivi) et f vrai (sonde, *diagnostic uniquement*, jamais entraîné contre) depuis l'embedding gelé ; comparer à la feature artisanale du front-end. **Si la sonde ne bat pas la feature artisanale, la méthode est abandonnée** (le JEPA n'apporte alors rien).
- **Succès** : R(500 k) meilleur que 5.1 *sans* embedding avec IC excluant 0, ou dégradation moindre sous config décalée (§6).

### 5.5 JEPA n°2 — modèle du monde conditionné par les actions + planification
- **Architecture** : encodeur (5.4) + prédicteur ẑ_{t+1} = f(z_{≤t}, a_{≤t}) **à historique** (transformer causal) — un état d'image seul est non identifiable (P9) : la réponse à Δb dépend des centres cachés de l'appareil, donc le modèle fait de l'inférence en contexte.
- **Données** : transitions `(image, gates, image')` générées en parallèle sur N cœurs, graines dev, sauts d'actions de tailles variées (petits et grands) ; ≈ 100 frames/s/cœur ⇒ 50 k frames ≈ 8 min sur 1 cœur.
- **Planification** : CEM/MPC dans l'espace latent vers un but défini par la **tête de score** (régressée sur y observable), re-planification à chaque mesure réelle ; raffinement local par vraies mesures (Nelder–Mead) en fin de course.
- **Pièges** : le prédicteur doit apprendre une *translation rigide* (dérive) + modulation de luminosité par région ; en espace latent de patchs c'est difficile ⇒ **test intermédiaire** : erreur de prédiction du décalage et de l'amplitude suivie vs modèle linéaire simple ; erreurs cumulées en horizon long ⇒ horizon court (≤ 3) et re-mesure.
- **Critère d'abandon** : si l'erreur du modèle sur y(b + Δb) n'est pas meilleure qu'un GP ajusté sur les mêmes historiques, la planification latente est inutile.
- **Succès** : même critère que 5.4 ; résultat négatif acceptable et à présenter.

---

## 6. Robustesse et transfert (sandbox, tier 2 du README)

Appareils test regénérés avec `new_experiment(config=replace(CHALLENGE, …))`. Le résultat titre (config par défaut) est conservé à côté.

| Scénario | Changement | Question |
|---|---|---|
| S1 | bruit pixel et stries ×2 | marge de SNR |
| S2 | stries ×3 | robustesse à la corrélation par ligne |
| S3 | `optimum_range` 0,8 | boîte de recherche / hors-plage |
| S4 | `n_regions` 2 et 8 | indépendance du nombre de régions |
| S5 | `gamma` ×0,5 et ×2 | pics étroits (aiguille) / larges |
| S6 | `drift_jitter` 0,3, `drift_curvature` 1,5 | suivi non linéaire |
| S7 | `amplitude` 10 | contraste faible, détection limite |

Métrique : ΔR relatif au titre et taux d'échec. Les méthodes apprises (5.2–5.5) sont entraînées *uniquement* sur la config par défaut / le prior large, et jamais ré-entraînées par scénario : c'est le test de ce qui « transférerait ».

---

## 7. Étude de volume de données (JEPA / PFN / RL)

- Tailles **imbriquées** : 1 k ⊂ 5 k ⊂ 10 k ⊂ 20 k ⊂ 50 k (frames pour JEPA ; paysages pour PFN).
- Un modèle **par taille**, même nombre de pas de gradient *ou* arrêt anticipé sur val (le choix est fixé avant), ≥ 3 graines d'entraînement par taille.
- Métrique : **R sur les appareils test**, pas la perte d'entraînement ; en plus : sonde linéaire (5.4) et NLL (5.2).
- Découpe par appareil : plusieurs frames d'un même appareil restent dans le même sous-ensemble ; pour 50 k frames, ≥ 2 000 appareils distincts (sinon on mesure la diversité d'appareils, pas le nombre de frames — à rapporter séparément : frames/appareil vs appareils).
- Lecture : plateau précoce ⇒ inutile de générer plus ; pas d'amélioration ⇒ résultat négatif.

---

## 8. Tests automatiques (CI locale)

| Test | Vérifie |
|---|---|
| T0 | `opt/` n'importe aucun module interdit ; pas d'accès à `_sim`, `_optimum`, `reveal` |
| T1 | deux mesures consécutives au même point diffèrent ; aucun `np.random.seed` dans `opt/` |
| T2 | le wrapper lève `BudgetExceeded` au-delà du plafond ; compteurs = ceux d'`Experiment` |
| T3 | à pas constant, ρ de Spearman proxy ↔ facteur vrai de la région > 0,9 sur appareils dev |
| T4 | suivi : erreur de position < 1 px sur déplacements simulés de barrière connus |
| T5 | déterminisme : mêmes graines ⇒ mêmes recommandations (processus unique) |
| T6 | `eval/` n'utilise que la recommandation engagée (pas le meilleur point visité) |

---

## 9. Ordre de travail

1. `eval/` (wrapper, graines, métriques, T0–T2) + baselines B0–B2 → **chiffres de référence**.
2. Front-end de perception + T3/T4 (la brique dont tout dépend).
3. 5.1 BO.
4. Génération du jeu de données (parallèle) → 5.4 (sonde linéaire d'abord : décision aller/abandon).
5. 5.2 PFN → 5.3 RL → 5.5 monde.
6. Étude de volume, scénarios S1–S7, run test **unique**, figures.

## 10. Décisions ouvertes

- Budget titre (500 k px proposé) — à fixer après le pilote du front-end.
- PyTorch + CUDA à installer dans l'env `uv` ; GPU à confirmer (RTX 3090 annoncée).
- Faut-il aussi un front-end appris (détecteur du challenge 1) en ablation ? Par défaut **non** (hors périmètre).
- Variante « privileged-training » : non par défaut.

---

## 11. Statut d'implémentation et commandes (mis à jour)

| Brique | Fichier | État |
|---|---|---|
| Wrapper aveugle + budget dur | `optimization/blind.py` | fait, testé (T1, T2) |
| Perception sans entraînement | `optimization/perception.py` | fait |
| Modèle de dérive bayésien + recalage | `optimization/tracking.py` | fait, testé (T4) |
| Score par interdot (plancher propre, rétrécissement) | `optimization/scoring.py` | fait ; **T3 (ρ de Spearman) non atteint, voir ci-dessous** |
| Session (référence, sondes, région de confiance) | `optimization/session.py` | fait |
| B2 recherche aléatoire, **Méthode 1 BO** (GP numpy, CPU) | `optimization/methods/` | fait |
| Méthodes 2–5 (PFN, RL, JEPA encodeur, JEPA monde) | `optimization/methods/planned.py` | **interfaces seulement** (torch, GPU) |
| Prior synthétique large (PFN/RL) | `training/priors.py` | fait, testé |
| Jeu de données JEPA (observables seulement) | `training/make_dataset.py` | fait, vérifié sur 2 appareils |
| Évaluation (vérité cachée, R, anytime, routage multiprocessus) | `evaluation/` | fait |
| B1 (coordonnée sur std), B3 (Adam), B4 (sans suivi), S1–S7, bootstrap/Holm | — | **à faire** |

Pilote (dev, 30 appareils, 1 M pixels, 1 graine d'algorithme, **aucun réglage**) : regret médian au départ 0,97 ; recherche aléatoire 0,59 ; BO 0,20 (BO meilleure sur 80 % des appareils appariés ; succès R ≤ 0,05 : 13 % vs 3 %). Résultat préliminaire, non concluant statistiquement.

Constats de validation du front-end (dev) :
- Recalage : erreur ≈ 1 mV (médiane). Prédiction de dérive après quelques points : médiane 5,6 mV, p90 14 mV. Échecs de suivi : 4/168 mesures.
- Détection de la trame de référence : rappel 0,77, précision 0,52 (les faux positifs sont tolérés : le vote de recalage y est robuste et leur plancher est écrêté par `min_floor_snr`).
- **Écart au protocole** : la corrélation de Spearman entre le score y et le facteur vrai vaut 0,47 en médiane (cible 0,9) quand on échantillonne uniformément la région de confiance, car la plupart des points sont au plancher (rangs bruités). À redéfinir sur les points où l'excès vrai ≥ 1 avant de conclure sur la qualité du score.
- **Contrainte physique découverte** : la dérive est trop non linéaire (courbure ≈ 0,5 V/V²) pour un modèle linéaire (erreur 100–400 mV à |Δb| ≈ 0,4 V). D'où la **région de confiance** : on ne mesure que là où l'écart-type prédictif de la dérive ≤ 25 mV, et elle s'étend quand le modèle apprend. Conséquence : exploration progressive, trajectoire plus courte (10 V en BO vs 22 V en aléatoire).

Commandes (depuis la racine) :
```bash
uv run pytest -q                                                    # T0, T1, T2, T4 + priors
uv run python -m evaluation.diagnose_tracking --n 12 --pts 14       # diagnostic du front-end
uv run python -m evaluation.run --methods random bo --split dev --n 30 --budget 1000000 --workers 7 --out challenge2/results/pilot.jsonl
uv run python -m training.make_dataset --split dev --n-devices 200 --frames 40 --workers 6 --out data/jepa_dev
uv sync --extra train                                               # PyTorch, sur la machine GPU
```
Machine de travail actuelle : GPU Intel UHD 620 seulement (pas de RTX 3090) ; tout ce qui précède tourne sur CPU. Les méthodes 2–5 se lancent sur la machine à GPU.
