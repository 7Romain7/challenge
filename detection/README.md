# Challenge 1 : cinq détecteurs (note théorique, temporaire)

Cinq méthodes classiques partagent le même prétraitement et la même interface : une carte de score en unités de σ du bruit, puis un seuil. Elles empilent des a priori de plus en plus forts. Cette note décrit les hypothèses. Elle ne reporte pas de chiffre.

Le code est `detection/baselines.py`. Les pièges du protocole (masque fragmenté, fuite de seuil, transfert) sont dans `detection/PROTOCOL.md`.

## Prétraitement commun

Un interdot est un creux fin (signal négatif) sur un fond strié horizontalement. Les stries suivent l'axe de scan rapide.

1. Médiane de chaque ligne, soustraite. Les sticks occupent peu de pixels, donc la médiane estime le fond de la ligne.
2. Inversion du signe. Le creux devient un pic.
3. Échelle médiane / MAD (facteur 1,4826). L'image est exprimée en σ du bruit de fond. Un z-score classique serait gonflé par les sticks brillants et effacerait le rapport signal sur bruit. Un seuil peut alors se transporter d'une scène à l'autre.

## M1. Lissage

Aucun a priori de forme. Gaussienne (σ ≈ 1 px), puis reblanchiment médiane / MAD de la carte filtrée.

Hypothèse : un interdot est un excès local une fois le bruit haute fréquence atténué.

Limite : tout excès lumineux franchit le seuil. Une ligne de charge, non étiquetée, répond comme un stick.

## M2. Filtre adapté

A priori de forme. Banc de gabarits gaussiens allongés, axe long autour de θ ≈ π/4 (pente typique des interdots sur ce double point quantique). Chaque gabarit est de moyenne nulle et de norme L2 unité. On garde le maximum du banc, puis on reblanchit : le max biaise la loi sous l'hypothèse nulle.

Dans un bruit blanc, le filtre adapté maximise le rapport signal sur bruit pour un motif connu.

Limite : une orientation hors du banc répond peu. Un trait qui n'est pas un bâtonnet court (ligne de charge longue, autre angle) est rejeté, ce qui est le but, tant que la pente réelle reste dans le banc.

## M3. Crête hessienne

A priori de géométrie locale, sans gabarit figé. Après lissage, on retient la valeur propre la plus négative de la hessienne (courbure la plus forte à travers le trait) et on ne garde que les crêtes.

Hypothèse : un interdot est une vallée fine, pas une tache.

Limite : une ligne de charge est aussi une crête. Le filtre ne sépare pas « interdot » et « ligne ».

## M4. Hystérésis

Même carte que M1, plus un a priori de cohérence spatiale. Le seuil haut germe les composantes. Un seuil bas (une fraction du haut) les prolonge. Les blobs plus petits qu'un minimum de pixels sont écartés.

Hypothèse : un interdot est un objet connexe, pas un pixel isolé de bruit.

Limite : une ligne de charge connexe est conservée, parfois épaissie par le seuil bas.

## M5. Régression logistique

Combinaison linéaire apprise des cartes précédentes. Six descripteurs par pixel : image en σ, lissage fin, lissage large, filtre adapté, crête, écart-type local. Ajustement par moindres carrés repondérés (IRLS). Environ sept poids, intercept compris.

Les positifs sont rares (de l'ordre de 0,3 % des pixels). L'ajustement prend tous les positifs et un multiple de négatifs tirés au hasard. L'ordonnée à l'origine est alors mal calibrée, volontairement. Le seuil, choisi plus tard sur la validation seule, absorbe ce biais.

Rôle : plancher appris, encore classique, avant d'engager un modèle à plus de capacité. Le détail de cette comparaison est théorique pour l'instant (`detection/PROTOCOL.md`).

## Séparation prévue, pas encore un livrable

- M5 s'ajuste sur le début du train uniquement.
- Le seuil se choisit sur la validation (F1 pixel tolérant à 1 px), puis il est figé.
- Le test est une graine fraîche, lue une seule fois.
- Le masque officiel vient d'un rectangle flou seuillé. Il se fragmente. L'IoU stricte mélange donc la physique et la rastérisation. Trois lectures sont prévues : pixel strict, pixel tolérant à 1 px, objet (le stick est-il localisé).
- Le rappel selon l'amplitude du stick donne la limite de rapport signal sur bruit. Sous quelques σ, aucune de ces méthodes ne peut voir le trait.
