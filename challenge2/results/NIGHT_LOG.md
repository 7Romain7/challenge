# Journal de la nuit (challenge 2) — toutes les tentatives, y compris les échecs

Règles fixées avant la boucle : idée tirée du diagnostic, paramètres fixés a priori (aucun balayage) ;
criblage dev 0–99 (apparié contre `bo_roi_dlf`) ; confirmation sur appareils neufs dev 900–999 ;
Holm sur l'ensemble des tentatives ; gagnant final : val 1000–1199 × 3 graines + configurations décalées S1–S7.
Test 10000–10199 intact.

| # | idée (hypothèse) | dev 0–99 R@1M | succès | ΔR vs bo_roi_dlf [IC] | décision |
|---|---|---|---|---|---|
| 0 | `bo_roi_dlf` (référence) | 0,133 | 25 % | — | — |
| 1 | CMA-ES plein cadre (le GP sert-il ?) | 0,605 | 2 % | +0,37 [0,29 ; 0,48] | rejeté |
| 2 | SPSA + Adam (B3) | 0,442 | 3 % | +0,19 [0,13 ; 0,34] | rejeté |
| 3 | montée par coordonnées officielle (B1) | 0,977 | 0 % | +0,80 | baseline |
| 4 | CMA-ES en phase ROI (le GP sert-il en phase 2 ?) | 0,157 | 26 % | −0,003 [−0,012 ; 0,002] | équivalent → GP pas le goulot |
| 5 | bascule ROI adaptative (« rien d'allumé ») | 0,133 | 25 % | 0 | rejeté : le bruit au plancher atteint y−se ≈ 1,8 |
| 6 | acquisition méta-apprise, gén. 25 (récompense observable) | 0,115 | 29 % | −0,015 [−0,033 ; −0,003], p = 0,15 | à confirmer (entraînement en cours) |
| 7 | recherche aléatoire (témoin d'exploration) | 0,683 | 1 % | — | la BO explore déjà 3× mieux (meilleure région vue 32 % vs 9 %) |
| 8 | budget 2 M / 4 M (limite d'échantillons ?) | 0,129 / 0,094 | 28 % / 36 % | — | ×4 de budget → 36 % seulement : il faut localiser plusieurs pics, donc beaucoup plus de points d'exploration |
| 9 | partage des phases **calculé** depuis les deux plafonds (bo_roi laissait 37 % des pixels) | 0,102 | 29 % | −0,007 [−0,022 ; −0,002], p = 0,10 | gardé comme base |
| 10 | 9 + **UCB** (√β = 2, fixé a priori) en phase 1 | **0,094** | **37 %** | **−0,022 [−0,038 ; −0,007], p = 0,002 (Holm 0,005)** | **à confirmer sur dev 900–999** |
| 11 | 9 + moitié de la phase 1 en remplissage d'espace (maximin) | 0,091 | 32 % | −0,009 [−0,037 ; 0,007], p = 0,11 | non significatif |
| — | diagnostic : sur 47 appareils perdant > 5 % par le choix de région, 26 n'ont jamais allumé la meilleure région à 10 % (couverture), 11 l'ont allumée à ≥ 20 % sans la suivre (information perdue par le max) | | | | → pistes 12 (modèle par région) et 13 (exploration grossière ×2) |
