# Compte rendu : méthode 5 sur une image du générateur

M5 est une régression logistique sur six cartes (image en unités de σ, deux lissages, filtre adapté, crête hessienne, écart-type local). Le seuil décide ensuite quels pixels sont des interdots.

## Scène

- Générateur officiel `csd`, graine 4242 (distincte de train=0, val=999, test=2025).
- Grille 150×150 px, fenêtre 0.3 V, pas 0.002 V.
- 13 sticks, dont 13 ont leur centre dans le cadre.
- |amplitude| de 28.098 à 33.272.
- Masque officiel : 88 pixels positifs (seuil 0,5 sur l'étiquette floutée).

## Ajustement

- Fit sur les 500 premières images de `data/train` (27.8 s).
- Positifs rares (~0,35 %) : tous les pixels positifs, plus 20 négatifs tirés au hasard par positif. L'ordonnée à l'origine est donc mal calibrée. Le seuil, choisi sur la validation, absorbe ce biais.
- Seuil retenu : **2.5** (figé sur data/val dans le protocole des baselines (max tol-F1)).
- Poids sur variables standardisées : z=+11.758, smooth_s1=-12.073, smooth_s2=-1.311, matched=+17.140, ridge=-4.246, local_std=-3.480, intercept=-7.613.
- Temps de score sur cette image : 40.0 ms.

## Métriques sur cette image

| critère | valeur |
|---|---|
| F1 pixel strict | 0.967 |
| IoU strict | 0.936 |
| précision tolérante (1 px) | 0.989 |
| rappel tolérant (1 px) | 1.000 |
| F1 tolérant | 0.995 |
| précision objet | 1.000 |
| rappel objet | 1.000 (13/13) |
| F1 objet | 1.000 |
| blobs prédits | 13 |
| pixels prédits | 94 |

Rappel objet par |amplitude| (nombre de sticks du bin entre parenthèses) :

| bin | rappel | n |
|---|---|---|
| [0,2) | n/a | 0 |
| [2,4) | n/a | 0 |
| [4,8) | n/a | 0 |
| [8,16) | n/a | 0 |
| [16,inf) | 1.00 | 13 |

## Sticks

Un stick dont le centre est dans le cadre est compté trouvé si un pixel prédit tombe à moins de longueur/2 + 1 px de ce centre.

| # | ligne | colonne | longueur (px) | intensité | distance (px) | trouvé |
|---|---|---|---|---|---|---|
| 1 | 31.8 | 12.5 | 6.7 | -29.72 | 0.51 | oui |
| 2 | 20.4 | 47.3 | 6.1 | -30.94 | 0.54 | oui |
| 3 | 11.4 | 83.2 | 5.4 | -33.27 | 0.43 | oui |
| 4 | 53.0 | 64.9 | 6.2 | -28.10 | 0.12 | oui |
| 5 | 30.6 | 137.4 | 5.2 | -32.91 | 0.71 | oui |
| 6 | 106.2 | 11.9 | 5.6 | -30.66 | 0.19 | oui |
| 7 | 94.6 | 46.8 | 7.8 | -28.82 | 0.49 | oui |
| 8 | 72.3 | 119.8 | 4.6 | -29.69 | 0.39 | oui |
| 9 | 134.5 | 29.6 | 7.0 | -29.20 | 0.66 | oui |
| 10 | 119.8 | 73.6 | 8.0 | -29.75 | 0.48 | oui |
| 11 | 111.8 | 102.9 | 4.5 | -31.14 | 0.26 | oui |
| 12 | 97.9 | 145.1 | 6.0 | -29.74 | 0.11 | oui |
| 13 | 139.0 | 127.1 | 4.7 | -28.67 | 0.12 | oui |

## Lecture

Le vert marque un pixel prédit à ≤1 px d'un pixel du masque officiel. Le rouge est un faux positif (souvent une ligne de charge, non étiquetée). Le bleu est un pixel de vérité qu'aucun pixel prédit ne couvre à 1 px.

L'IoU strict reste sensible à la rastérisation du masque officiel (rectangle flou seuillé à 0,5, souvent fragmenté). Le F1 objet répond à la question physique : l'interdot est-il localisé ?
